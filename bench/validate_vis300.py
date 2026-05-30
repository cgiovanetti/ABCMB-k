"""Validate the new default (visibility grid, n_lna_PE=300) on the BATCHED path
and measure the memory/throughput payoff vs the old uniform-500.

(1) Parity: call_batched()[0] vs single-call model() at the same cosmology, with
    the visibility grid -- confirms the grid works through the batched code (the
    vmap over BG_batch + BG.visibility under nested vmap) to chunking-noise level.
(2) Memory + throughput at B=64 (lensing off, ELLMAX=800, matches round-3 baseline):
    uniform-500 vs visibility-300 -> peak_gb and per_param.
"""
import os, time, json
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import abcmb
assert "ABCMB-k" in abcmb.__file__, abcmb.__file__
from abcmb.main import Model

ELLMAX = 800
FID = {
    'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225,
    'A_s': 2.12424e-9, 'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4, 'N_nu_massive': 0, 'T_nu_massive': 0.71611,
    'm_nu_massive': 0.06, 'tau_reion': 0.0544, 'Delta_z_reion': 0.5,
    'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}
BOX = {'h': (0.65, 0.70), 'omega_cdm': (0.115, 0.125), 'omega_b': (0.0220, 0.0230),
       'A_s': (1.95e-9, 2.25e-9), 'n_s': (0.950, 0.980)}


def mk(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = dict(FID)
        for k, (lo, hi) in BOX.items():
            p[k] = float(rng.uniform(lo, hi))
        out.append(p)
    return out


def model(mode, n, lensing):
    return Model(user_species=None, output_Cl=True, l_max=ELLMAX, lensing=lensing,
                 output_Pk=True, output_k_max=0.5, l_max_g=12, l_max_pol_g=10,
                 l_max_ur=17, l_max_ncdm=17, n_lna_PE=n, lna_grid_mode=mode)


def block(x): jax.block_until_ready(jax.tree_util.tree_leaves(x))
def peak():
    try: return round(jax.local_devices()[0].memory_stats()['peak_bytes_in_use']/1e9, 2)
    except Exception: return None


def main():
    print(f"# abcmb {abcmb.__file__}  default specs: mode={Model().specs['lna_grid_mode']} "
          f"n_lna_PE={Model().specs['n_lna_PE']} smooth={Model().specs['lna_vis_smooth']} "
          f"floor={Model().specs['lna_vis_floor']}", flush=True)

    # (1) PARITY: batched vs single-call (visibility grid), lensing on for a real check
    m = model("visibility", 300, True)
    pl = mk(4, seed=1)
    single = m(pl[0])
    block(single.ClTT)
    batched = m.call_batched(pl, shard=False)
    block(batched.ClTT)
    rtt = float(np.max(np.abs(np.asarray(single.ClTT) - np.asarray(batched.ClTT[0]))
                       / np.abs(np.asarray(single.ClTT))))
    ree = float(np.max(np.abs(np.asarray(single.ClEE) - np.asarray(batched.ClEE[0]))
                       / np.abs(np.asarray(single.ClEE))))
    print(f"PARITY vis-300 batched[0] vs single: TT {rtt:.2e}  EE {ree:.2e} "
          f"(chunking-noise floor ~3e-5)", flush=True)

    # (memory + throughput measured separately via mem_throughput_sweep.py --gridmode,
    #  one config per process, so peak_bytes_in_use is a clean high-water mark.)


if __name__ == "__main__":
    main()
