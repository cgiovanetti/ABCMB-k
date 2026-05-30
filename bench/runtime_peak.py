"""Separate COMPILE-time scratch from RUNTIME peak for call_batched stages.

peak_bytes_in_use is a since-start high-water and includes XLA autotuning
scratch from the first (compile) call. To get the true *runtime* high-water, we
warm (compile) once, then poll bytes_in_use from a background thread DURING a
second, already-compiled call. The max bytes_in_use over the 2nd call is the
runtime peak, independent of compile scratch.

  python bench/runtime_peak.py --B 64 --massive 0
"""
import os, sys, time, threading, argparse
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


class Poller:
    """Background thread sampling bytes_in_use; reports max over a window."""
    def __init__(self, dev):
        self.dev = dev
        self.max = 0
        self._stop = False
        self._t = None

    def _run(self):
        while not self._stop:
            try:
                b = self.dev.memory_stats()['bytes_in_use']
                if b > self.max:
                    self.max = b
            except Exception:
                pass

    def __enter__(self):
        self.max = 0
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        self._t.join()

    def gb(self):
        return round(self.max / 1e9, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--massive", type=int, default=0)
    ap.add_argument("--kchunk", type=int, default=100)
    ap.add_argument("--nlna", type=int, default=500)
    a = ap.parse_args()
    dev = jax.local_devices()[0]

    def peak():
        try: return round(dev.memory_stats()['peak_bytes_in_use'] / 1e9, 3)
        except Exception: return None

    user_species = (species.MassiveNeutrino,) if a.massive else None
    model = Model(user_species=user_species, output_Cl=True, l_max=ELLMAX,
                  lensing=False, output_Pk=True, output_k_max=0.5,
                  l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
                  n_lna_PE=a.nlna)
    pl = make_params(a.B, a.massive)
    tag = f"B={a.B} massive={a.massive} nlna={a.nlna}"

    # WARM: compile build_bgs, PT, Cl, Pk once.
    full_ps = [model.add_derived_parameters(p) for p in pl]
    pb, BGb = model._build_bgs_batched(full_ps, shardfn=None); block((pb, BGb))
    PT = model.PE.full_evolution_batched((BGb, pb), k_chunk_size=a.kchunk); block(PT)
    Cl = model.SS.get_Cl_batched(PT, BGb, pb); block(Cl)
    Pk = model.SS.Pk_lin_batched(model.SS.k_axis_Pk_output, 0., PT, pb); block(Pk)
    print(f"[{tag}] after warm: cumulative peak (incl compile scratch) = {peak()} GB", flush=True)

    # RUNTIME peak per stage on a 2nd (already-compiled) call, via polling.
    with Poller(dev) as p:
        pb, BGb = model._build_bgs_batched(full_ps, shardfn=None); block((pb, BGb))
    print(f"[{tag}] RUNTIME build_bgs   = {p.gb()} GB", flush=True)
    with Poller(dev) as p:
        PT = model.PE.full_evolution_batched((BGb, pb), k_chunk_size=a.kchunk); block(PT)
    print(f"[{tag}] RUNTIME PT(modes)   = {p.gb()} GB", flush=True)
    with Poller(dev) as p:
        Cl = model.SS.get_Cl_batched(PT, BGb, pb); block(Cl)
    print(f"[{tag}] RUNTIME get_Cl      = {p.gb()} GB", flush=True)
    with Poller(dev) as p:
        Pk = model.SS.Pk_lin_batched(model.SS.k_axis_Pk_output, 0., PT, pb); block(Pk)
    print(f"[{tag}] RUNTIME Pk          = {p.gb()} GB", flush=True)

    # full call_batched runtime peak (2nd call)
    _ = model.call_batched(pl, shard=False, k_chunk_size=a.kchunk); block(_.ClTT)
    with Poller(dev) as p:
        out = model.call_batched(pl, shard=False, k_chunk_size=a.kchunk); block(out.ClTT)
    print(f"[{tag}] RUNTIME call_batched (full) = {p.gb()} GB   "
          f"(cumulative peak now {peak()} GB)", flush=True)


if __name__ == "__main__":
    main()
