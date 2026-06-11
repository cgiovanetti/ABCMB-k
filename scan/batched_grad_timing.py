"""batched_grad_timing.py — throughput of the staged batched AD gradient vs the
single-cosmology loop.

The batched gradient (scan/batched_grad.py, validated to 1.75e-5 vs single-path)
processes ALL B cosmologies per call, so its per-cosmology cost AMORTIZES with B;
the driver's `loop` method does B separate single-cosmology jacfwds (no
amortization, ~85 s each at l_max=2508). This measures the warm (post-compile)
per-cosmology-gradient time for both at a common l_max, to quantify the win.

Run via srun on a GPU node, PYTHONPATH=$(pwd). Env: BGT_LMAX(512), BGT_B(16),
BGT_WRT(h,omega_b,omega_cdm,n_s,ln10As).
"""
import os, time
_SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/c/carag")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_SCRATCH, ".jax_cache_abcmb"))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from abcmb.main import Model
from scan.batched_grad import raw_dict, derived_and_tangents, staged_cl_and_grad

LMAX = int(os.environ.get("BGT_LMAX", 512))
B = int(os.environ.get("BGT_B", 16))
WRT = os.environ.get("BGT_WRT", "h,omega_b,omega_cdm,n_s,ln10As").split(",")
P = len(WRT)


def peak_gb():
    try:
        return max(d.memory_stats()["peak_bytes_in_use"] for d in jax.devices('gpu')) / 1e9
    except Exception:
        return float('nan')


def main():
    print(f"devices: {jax.devices()}  lmax={LMAX}  B={B}  P={P}  wrt={WRT}", flush=True)
    model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=False,
                  output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
                  rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
                  rtol_small_k_PE=1e-5, max_steps_PE=16384)
    rng = np.random.default_rng(0)
    raw_ps = [raw_dict(h=0.6736 + 0.01 * rng.normal(),
                       omega_cdm=0.12 + 0.002 * rng.normal(),
                       n_s=0.965 + 0.005 * rng.normal()) for _ in range(B)]
    full_ps, params_dots = derived_and_tangents(model, raw_ps, WRT)

    # ---- staged batched gradient: compile, then warm timing ----
    t0 = time.perf_counter()
    out = staged_cl_and_grad(model, full_ps, params_dots); jax.block_until_ready(out)
    t_comp = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = staged_cl_and_grad(model, full_ps, params_dots); jax.block_until_ready(out)
    t_warm = time.perf_counter() - t0
    per_cosmo_batched = t_warm / B
    print(f"\n[staged batched] B={B}: compile {t_comp:.0f}s, WARM {t_warm:.2f}s "
          f"=> {per_cosmo_batched:.2f}s / cosmo-gradient (all P={P}); peak {peak_gb():.2f} GB",
          flush=True)

    # ---- single-path jacfwd: 1 cosmology, all P (the driver 'loop' unit) ----
    base = {k: jnp.asarray(float(v)) for k, v in raw_ps[0].items()}

    def cl0(scalar, key):
        d = dict(base); d[key] = scalar
        o = model.run_cosmology_abbr(model.add_derived_parameters(d))
        return o.ClTT, o.ClTE, o.ClEE

    def single_grad():
        return [jax.jacfwd(lambda s: cl0(s, key))(base[key]) for key in WRT]
    t0 = time.perf_counter()
    g = single_grad(); jax.block_until_ready(g)
    t_sc = time.perf_counter() - t0
    t0 = time.perf_counter()
    g = single_grad(); jax.block_until_ready(g)
    t_sw = time.perf_counter() - t0
    print(f"[single-path]    1 cosmo: compile {t_sc:.0f}s, WARM {t_sw:.2f}s "
          f"=> {t_sw:.2f}s / cosmo-gradient (all P={P})", flush=True)

    print(f"\n  >>> SPEEDUP per cosmo-gradient = {t_sw / per_cosmo_batched:.1f}x "
          f"at B={B}, l_max={LMAX} (grows with B; +sharding multiplies by n_dev)", flush=True)
    print("[batched_grad_timing] done", flush=True)


if __name__ == "__main__":
    main()
