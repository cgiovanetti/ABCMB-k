"""Massive-neutrino safety check for the vis-300 default.

The vs-CLASS massive branch has a known setup mismatch, so instead we check that
the vis-300 grid REPRODUCES the trusted uniform-500 baseline for one massive
neutrino, per-ell-binned. Expectation (mirrors massless): ell>=5 reproduced to
well below the 0.2% accuracy floor; only the ell=2-4 quadrupole differs.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import abcmb
assert "ABCMB-k" in abcmb.__file__, abcmb.__file__
from abcmb.main import Model
from abcmb import species

ELLMAX = 2500
P = {'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225, 'A_s': 2.12424e-9,
     'n_s': 0.9709, 'Neff': 2.0308, 'YHe': 0.245, 'TCMB0': 2.34865418e-4,
     'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
     'tau_reion': 0.0544, 'Delta_z_reion': 0.5, 'z_reion_He': 3.5,
     'Delta_z_reion_He': 0.5, 'exp_reion': 1.5}


def run(mode, n):
    m = Model(user_species=(species.MassiveNeutrino,), output_Cl=True, l_max=ELLMAX,
              lensing=True, output_Pk=True, output_k_max=0.5, l_max_g=12,
              l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17, n_lna_PE=n, lna_grid_mode=mode)
    o = m(dict(P)); jax.block_until_ready(o.ClTT)
    return np.asarray(o.ClTT), np.asarray(o.ClEE)


def main():
    tt0, ee0 = run("uniform", 500)      # trusted baseline
    tt1, ee1 = run("visibility", 300)   # new default
    ell = np.arange(2, ELLMAX + 1)
    rtt = np.abs(tt1 - tt0) / np.abs(tt0)
    ree = np.abs(ee1 - ee0) / np.abs(ee0)
    print(f"# massive nu: vis-300 vs uniform-500 reproduction (rel diff)", flush=True)
    for name, lo, hi in (("lo<5", 2, 5), ("5-30", 5, 30), ("mid", 30, 1000), ("hi>1000", 1000, 2501)):
        m = (ell >= lo) & (ell < hi)
        print(f"  TT {name:8s} max {rtt[m].max()*100:.4f}%   EE max {ree[m].max()*100:.4f}%", flush=True)
    print(f"  ell=2: TT {rtt[0]*100:.4f}%  EE {ree[0]*100:.4f}%", flush=True)
    print(f"  OVERALL ell>=5: TT {rtt[ell>=5].max()*100:.4f}%  EE {ree[ell>=5].max()*100:.4f}%", flush=True)


if __name__ == "__main__":
    main()
