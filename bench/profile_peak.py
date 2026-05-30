"""Stage-by-stage GPU peak profiler for call_batched (single GPU, shard=False).

Replicates the call_batched body stage by stage, reading the device
peak_bytes_in_use / bytes_in_use after each stage so we can see WHICH stage
sets the binding high-water mark. Peak is a since-process-start high-water, so
a JUMP in peak between stages N-1 and N means stage N allocated the most.

  python bench/profile_peak.py --B 64 --massive 0
"""
import os, sys, json, argparse
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from abcmb.main import Model
from abcmb import species

ELLMAX = 800
FIDUCIAL = {
    'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225,
    'A_s': 2.12424e-9, 'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5,
    'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}
BOXES = {'h': (0.65, 0.70), 'omega_cdm': (0.115, 0.125),
         'omega_b': (0.0220, 0.0230), 'A_s': (1.95e-9, 2.25e-9),
         'n_s': (0.950, 0.980)}


def make_params(n, massive, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = dict(FIDUCIAL)
        for k, (lo, hi) in BOXES.items():
            p[k] = float(rng.uniform(lo, hi))
        if massive:
            p['N_nu_massive'] = 1
            p['Neff'] = 2.0308
        out.append(p)
    return out


def block(x):
    jax.block_until_ready(jax.tree_util.tree_leaves(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--massive", type=int, default=0)
    ap.add_argument("--kchunk", type=int, default=100)
    ap.add_argument("--stop", choices=["build", "pt", "cl", "pk"], default="pk",
                    help="run+compile ONLY up to this stage, then print final peak. "
                         "Run once per stage in fresh processes; the peak increment "
                         "between consecutive --stop values is that stage's marginal "
                         "binding contribution (no warm contamination).")
    a = ap.parse_args()
    dev = jax.local_devices()[0]

    def gb(key):
        try:
            return round(dev.memory_stats()[key] / 1e9, 3)
        except Exception:
            return None

    def report(tag):
        print(f"  [{tag:18s}] peak={gb('peak_bytes_in_use'):7} GB   "
              f"live={gb('bytes_in_use'):7} GB", flush=True)

    user_species = (species.MassiveNeutrino,) if a.massive else None
    model = Model(user_species=user_species, output_Cl=True, l_max=ELLMAX,
                  lensing=False, output_Pk=True, output_k_max=0.5,
                  l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17)
    pl = make_params(a.B, a.massive)

    # Single no-warm run UP TO --stop, in this fresh process, so peak_bytes_in_use
    # is a clean high-water for exactly {build, +pt, +cl, +pk}. Run once per --stop
    # in separate processes; the peak increment between consecutive --stop values is
    # that stage's marginal binding contribution.
    print(f"[stop={a.stop} B={a.B} massive={a.massive}]", flush=True)
    full_ps = [model.add_derived_parameters(p) for p in pl]
    pb, BGb = model._build_bgs_batched(full_ps, shardfn=None); block((pb, BGb))
    report("after build_bgs")
    if a.stop in ("pt", "cl", "pk"):
        PT = model.PE.full_evolution_batched((BGb, pb), k_chunk_size=a.kchunk); block(PT)
        report("after PT(modes)")
    if a.stop in ("cl", "pk"):
        Cl = model.SS.get_Cl_batched(PT, BGb, pb); block(Cl)
        report("after get_Cl")
    if a.stop == "pk":
        Pk = model.SS.Pk_lin_batched(model.SS.k_axis_Pk_output, 0., PT, pb); block(Pk)
        report("after Pk")
    n_k = len(model.PE.k_axis_perturbations)
    Ny = 1 + sum(int(s.num_equations) for s in model.PE.species_list)
    raw = n_k * 500 * Ny * 8 / 1e9 * a.B
    print(f"  raw modes tensor (B,Ny,Nlna,N_k) = {raw:.3f} GB ; "
          f"Ny={Ny} N_k={n_k}", flush=True)


if __name__ == "__main__":
    main()
