"""
Test if the 'chunking bug' propagates to the PerturbationTable (and presumably
downstream to Cls). Tighter rtol/atol should make both methods converge.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from jax import vmap

from abcmb.main import Model
from abcmb.perturbations import strip_bg_kappa

FIDUCIAL = {
    'h': 0.6736, 'omega_cdm': 0.120, 'omega_b': 0.02237,
    'A_s': 2.1e-9, 'n_s': 0.9649, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5,
    'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}


def _to_float(v):
    arr = jnp.asarray(v)
    if arr.dtype.kind in 'iub':
        return arr.astype(jnp.float64)
    return arr


def build_one_bg(model, params):
    full_p = model.add_derived_parameters(params)
    pre_bg = model.get_BG_pre_recomb(full_p)
    cpu_dev = jax.devices('cpu')[0]
    recomb_in_cpu = jax.device_put(pre_bg.recomb_inputs, cpu_dev)
    p_cpu = jax.device_put(full_p, cpu_dev)
    recomb_output = eqx.filter_jit(model.RecModel, backend='cpu')(
        (recomb_in_cpu, p_cpu))
    try:
        recomb_output = jax.device_put(recomb_output, jax.devices('gpu')[0])
    except Exception:
        pass
    recomb_output = jax.tree_util.tree_map(_to_float, recomb_output)
    full_p = jax.tree_util.tree_map(_to_float, full_p)
    bg = model.get_BG(full_p, pre_bg, recomb_output)
    return full_p, bg


def stack_pytrees(pytrees):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *pytrees)


def main():
    model = Model(
        user_species=None, output_Cl=True, l_max=800, lensing=False,
        output_Pk=True, output_k_max=0.5,
        l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
    )
    PE = model.PE
    full_p, bg = build_one_bg(model, FIDUCIAL)

    bg_strip = strip_bg_kappa(bg)
    BG_b = stack_pytrees([bg_strip])
    p_b = stack_pytrees([full_p])
    lna_b = vmap(lambda lts: jnp.linspace(lts, 0., 500))(
        BG_b.lna_transfer_start)

    # full_evolution (current production reference)
    PT_single = PE.full_evolution((bg, full_p))
    jax.block_until_ready(PT_single.delta_m)

    # Chunked _evolve_chunk path (from existing buggy implementation in
    # _evolve_chunk, called via a fresh full_evolution_batched-like wrapper)
    K_CHUNK = 100
    k_axis = PE.k_axis_perturbations
    N_k = len(k_axis)
    chunks = []
    for i in range(0, N_k, K_CHUNK):
        k_chunk = k_axis[i:i + K_CHUNK]
        c = PE._evolve_chunk(k_chunk, lna_b, BG_b, p_b)
        chunks.append(c)
    modes_chunked = jnp.concatenate(chunks, axis=0)
    modes_chunked = modes_chunked.transpose(1, 3, 2, 0)  # (B, Ny, Nlna, N_k)
    PT_chunked = PE.make_output_table_batched(lna_b, modes_chunked, (BG_b, p_b))
    jax.block_until_ready(PT_chunked.delta_m)

    # Compare key fields. PT_chunked is batched (B=1).
    def rel(name, a_single, b_batched):
        b = b_batched[0]
        if a_single.shape != b.shape:
            print(f"  {name}: shape mismatch", flush=True)
            return
        diff = float(np.abs(np.asarray(b) - np.asarray(a_single)).max())
        norm = float(np.abs(np.asarray(a_single)).max())
        relerr = diff / max(norm, 1e-30)
        print(f"  {name}: max_abs={diff:.2e}  max|a|={norm:.2e}  rel={relerr:.2e}",
              flush=True)

    print("\nPerturbationTable field comparison (chunked vs single):",
          flush=True)
    rel("delta_m", PT_single.delta_m, PT_chunked.delta_m)
    rel("theta_b_prime", PT_single.theta_b_prime, PT_chunked.theta_b_prime)
    rel("metric_eta", PT_single.metric_eta, PT_chunked.metric_eta)
    rel("metric_h_prime", PT_single.metric_h_prime, PT_chunked.metric_h_prime)
    rel("metric_eta_prime", PT_single.metric_eta_prime, PT_chunked.metric_eta_prime)


if __name__ == "__main__":
    main()
