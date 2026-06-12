"""spec_compile_probe.py — isolate the SPECTRUM jvp compile at production l_max
WITHOUT paying the ~10-min perturbation solve.

The spectrum jvp compile is shape-driven (the Wigner d-matrices d00/d1n/... are
param-INDEPENDENT, so forward-mode carries no tangent through them). So we can feed
get_Cl_batched a FAKE PerturbationTable (zeros of the right shape, built via
make_output_table_batched on a REAL background) and a fake PT tangent, and time the
jvp compile honestly. This answers: is the LENSED spectrum jvp compile at l2508
bounded? (the production gate; lensing=True => num_mu=ellmax+570, ells~ellmax+500).

Throwaway cache (cold). Run via srun, PYTHONPATH=$(pwd).
  SCP_LMAX(2508) SCP_B(2) SCP_LENSING(1)
"""
import os, time, tempfile
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_COMPILATION_CACHE_DIR"] = tempfile.mkdtemp(prefix="jaxcc_scp_")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from abcmb.main import Model, _recmodel_cpu
from abcmb.perturbations import make_lna_grid
from scan.batched_grad import raw_dict, derived_and_tangents, jvp_stage, _to_float

LMAX = int(os.environ.get("SCP_LMAX", 2508))
B = int(os.environ.get("SCP_B", 2))
LENSING = os.environ.get("SCP_LENSING", "1") != "0"


def peak_gb():
    try:
        return max(d.memory_stats()["peak_bytes_in_use"] for d in jax.devices('gpu')) / 1e9
    except Exception:
        return float('nan')


def main():
    print(f"devices={jax.devices()} lmax={LMAX} B={B} lensing={LENSING}", flush=True)
    model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=LENSING,
                  output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
                  rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
                  rtol_small_k_PE=1e-5, max_steps_PE=16384)
    PE = model.PE; SS = model.SS
    cpu = jax.devices('cpu')[0]; gpu = jax.devices('gpu')[0]
    raw_ps = [raw_dict(), raw_dict(h=0.68, omega_cdm=0.119, n_s=0.96)][:B]
    full_ps, params_dots = derived_and_tangents(model, raw_ps, ["omega_cdm"])
    pb = _to_float(jax.tree.map(lambda *xs: jnp.stack(xs), *full_ps))
    pdot = params_dots[0]

    # ---- real background (cheap stages only) + its tangent ----
    pre_BG = model._pre_recomb_batched(pb)
    ri_cpu = jax.device_put(pre_BG.recomb_inputs, cpu); p_cpu = jax.device_put(pb, cpu)
    hy = _recmodel_cpu(model.RecModel, True)
    recomb = jax.device_put(hy((ri_cpu, p_cpu)), gpu)
    BG = model._get_BG_batched(pb, pre_BG, recomb)
    _, pre_BG_dot = jvp_stage(model._pre_recomb_batched, (pb,), (pdot,))
    ri_dot = jax.device_put(pre_BG_dot.recomb_inputs, cpu); p_dot = jax.device_put(pdot, cpu)
    _, recomb_dot = jvp_stage(lambda ri, p: hy((ri, p)), (ri_cpu, p_cpu), (ri_dot, p_dot))
    recomb_dot = jax.device_put(recomb_dot, gpu)
    _, BG_dot = jvp_stage(model._get_BG_batched, (pb, pre_BG, recomb),
                          (pdot, pre_BG_dot, recomb_dot))
    print("real background + tangent built", flush=True)

    # ---- FAKE modes tensor of the right shape -> a valid PT (shapes only) ----
    n_k = len(PE.k_axis_perturbations)
    Ny = 1 + sum(int(s.num_equations) for s in PE.species_list)
    n_lna = model.specs["n_lna_PE"]
    print(f"  n_k={n_k} Ny={Ny} n_lna={n_lna}  (fake modes {(B,Ny,n_lna,n_k)})", flush=True)
    lna_batch = jax.vmap(
        lambda bg, p: make_lna_grid(bg, p, n_lna, model.specs))(BG, pb)
    rng = np.random.default_rng(0)
    modes = jnp.asarray(1e-5 * rng.standard_normal((B, Ny, n_lna, n_k)))
    modes_dot = jnp.asarray(1e-6 * rng.standard_normal((B, Ny, n_lna, n_k)))
    PT = PE.make_output_table_batched(lna_batch, modes, (BG, pb))
    _, PT_dot = jvp_stage(lambda lb, m: PE.make_output_table_batched(lb, m, (BG, pb)),
                          (lna_batch, modes), (jnp.zeros_like(lna_batch), modes_dot))
    jax.block_until_ready((PT, PT_dot))
    print("fake PT + PT tangent built", flush=True)

    # ---- TIMED: spectrum jvp (compile+run) ----
    t0 = time.perf_counter()
    cl, cl_dot = jvp_stage(SS.get_Cl_batched, (PT, BG, pb), (PT_dot, BG_dot, pdot))
    jax.block_until_ready((cl, cl_dot))
    cold = time.perf_counter() - t0
    print(f"[SPECTRUM jvp] lmax={LMAX} lensing={LENSING}: cold(compile+run) {cold:.1f}s "
          f"peak {peak_gb():.2f} GB", flush=True)
    # warm
    t0 = time.perf_counter()
    _, cl_dot = jvp_stage(SS.get_Cl_batched, (PT, BG, pb), (PT_dot, BG_dot, pdot))
    jax.block_until_ready(cl_dot)
    warm = time.perf_counter() - t0
    print(f"[SPECTRUM jvp] warm {warm:.2f}s => compile~{cold-warm:.1f}s", flush=True)
    print("[spec_compile_probe] done", flush=True)


if __name__ == "__main__":
    main()
