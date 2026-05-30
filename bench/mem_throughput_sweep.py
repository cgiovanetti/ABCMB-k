"""One-config memory + throughput probe for call_batched.

Runs a SINGLE (B, k_chunk, shard) config in its own process so the peak-memory
high-water mark is clean, then prints one JSON line. A bash driver loops configs.

  python bench/mem_throughput_sweep.py --B 256 --kchunk 48 --shard 1

Prints: {"B":..,"kchunk":..,"shard":..,"warm":..,"run":..,"per_param":..,
         "peak_gb":[per-device],"ok":true} or {... "ok":false,"err":...}
"""
import os, sys, time, json, argparse, traceback
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


def make_params(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = dict(FIDUCIAL)
        for k, (lo, hi) in BOXES.items():
            p[k] = float(rng.uniform(lo, hi))
        out.append(p)
    return out


def block(x):
    jax.block_until_ready(jax.tree_util.tree_leaves(x))


def peak_gb():
    out = []
    for d in jax.local_devices():
        try:
            out.append(round(d.memory_stats()['peak_bytes_in_use'] / 1e9, 2))
        except Exception:
            out.append(None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--kchunk", type=int, default=100)
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--massive", type=int, default=0,
                    help="if 1, add one massive neutrino (Ny ~46 -> ~250)")
    ap.add_argument("--nlna", type=int, default=500,
                    help="n_lna_PE (perturbation save grid; default 500)")
    ap.add_argument("--gridmode", type=str, default="visibility",
                    help="lna_grid_mode: uniform | visibility | recomb_dense")
    a = ap.parse_args()
    shard = bool(a.shard)
    rec = {"B": a.B, "kchunk": a.kchunk, "shard": int(shard),
           "massive": a.massive, "nlna": a.nlna, "gridmode": a.gridmode,
           "n_dev": len(jax.devices('gpu'))}
    try:
        user_species = (species.MassiveNeutrino,) if a.massive else None
        model = Model(user_species=user_species, output_Cl=True, l_max=ELLMAX,
                      lensing=False, output_Pk=True, output_k_max=0.5,
                      l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
                      n_lna_PE=a.nlna, lna_grid_mode=a.gridmode)
        pl = make_params(a.B)
        if a.massive:                       # enable 1 massive nu in each cosmology
            for p in pl:
                p['N_nu_massive'] = 1
                p['Neff'] = 2.0308          # 2 massless + 1 massive ~ Neff 3.044
        t0 = time.perf_counter()
        out = model.call_batched(pl, shard=shard, k_chunk_size=a.kchunk)
        block(out.ClTT)
        rec["warm"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        out = model.call_batched(pl, shard=shard, k_chunk_size=a.kchunk)
        block(out.ClTT)
        run = time.perf_counter() - t0
        rec.update(run=round(run, 2), per_param=round(run / a.B, 4),
                   peak_gb=peak_gb(), shape=list(out.ClTT.shape), ok=True)
    except Exception as e:
        rec.update(ok=False, err=f"{type(e).__name__}: {str(e)[:200]}",
                   peak_gb=peak_gb())
        traceback.print_exc()
    print("RESULT " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
