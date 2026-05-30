"""Find the big runtime transient inside _build_bgs_batched.

Polls bytes_in_use around each sub-stage (pre_recomb GPU / HyRex CPU / get_BG
GPU) on a SECOND, already-compiled call, freeing prior outputs first so each
poll reflects that sub-stage alone. Also dumps a device-memory pprof during
get_BG so we can attribute the allocation to a source line.

  python bench/profile_buildbgs.py --B 64 --massive 0
"""
import os, sys, time, threading, argparse, gc
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
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
            p['N_nu_massive'] = 1; p['Neff'] = 2.0308
        out.append(p)
    return out


def block(x):
    jax.block_until_ready(jax.tree_util.tree_leaves(x))


class Poller:
    def __init__(self, dev): self.dev = dev; self.max = 0; self._stop = False
    def _run(self):
        while not self._stop:
            try:
                b = self.dev.memory_stats()['bytes_in_use']
                if b > self.max: self.max = b
            except Exception: pass
    def __enter__(self):
        self.max = 0; self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start(); return self
    def __exit__(self, *a): self._stop = True; self._t.join()
    def gb(self): return round(self.max / 1e9, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--massive", type=int, default=0)
    a = ap.parse_args()
    dev = jax.local_devices()[0]
    def live(): return round(dev.memory_stats()['bytes_in_use']/1e9, 3)

    user_species = (species.MassiveNeutrino,) if a.massive else None
    model = Model(user_species=user_species, output_Cl=True, l_max=ELLMAX,
                  lensing=False, output_Pk=True, output_k_max=0.5,
                  l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17)
    pl = make_params(a.B, a.massive)
    tag = f"B={a.B} massive={a.massive}"

    def _to_float(v):
        arr = jnp.asarray(v)
        return arr.astype(jnp.float64) if arr.dtype.kind in 'iub' else arr

    full_ps = [model.add_derived_parameters(p) for p in pl]
    params_batch = jax.tree.map(lambda *xs: jnp.stack(xs), *full_ps)
    params_batch = jax.tree_util.tree_map(_to_float, params_batch)

    # WARM each sub-stage
    pre = model._pre_recomb_batched(params_batch); block(pre)
    cpu = jax.devices('cpu')[0]
    ri_cpu = jax.device_put(pre.recomb_inputs, cpu)
    p_cpu = jax.device_put(params_batch, cpu)
    recomb = eqx.filter_jit(jax.vmap(model.RecModel), backend='cpu')((ri_cpu, p_cpu))
    recomb = jax.tree_util.tree_map(_to_float, recomb)
    try: recomb = jax.device_put(recomb, dev)
    except Exception: pass
    BG = model._get_BG_batched(params_batch, pre, recomb); block(BG)
    print(f"[{tag}] warmed. live now {live()} GB", flush=True)

    # MEASURE each sub-stage runtime peak (2nd call)
    with Poller(dev) as p:
        pre = model._pre_recomb_batched(params_batch); block(pre)
    print(f"[{tag}] RUNTIME _pre_recomb_batched = {p.gb()} GB  (live after {live()})", flush=True)

    ri_cpu = jax.device_put(pre.recomb_inputs, cpu); p_cpu = jax.device_put(params_batch, cpu)
    with Poller(dev) as p:
        recomb = eqx.filter_jit(jax.vmap(model.RecModel), backend='cpu')((ri_cpu, p_cpu))
        recomb = jax.tree_util.tree_map(_to_float, recomb)
        try: recomb = jax.device_put(recomb, dev)
        except Exception: pass
        block(recomb)
    print(f"[{tag}] RUNTIME HyRex+transfer      = {p.gb()} GB  (live after {live()})", flush=True)

    with Poller(dev) as p:
        BG = model._get_BG_batched(params_batch, pre, recomb); block(BG)
    print(f"[{tag}] RUNTIME _get_BG_batched     = {p.gb()} GB  (live after {live()})", flush=True)

    # device memory pprof during a fresh get_BG (attribute to source line)
    try:
        del BG; gc.collect()
        BG = model._get_BG_batched(params_batch, pre, recomb); block(BG)
        path = f"bench/mem_getbg_B{a.B}_m{a.massive}.prof"
        jax.profiler.save_device_memory_profile(path)
        print(f"[{tag}] wrote device-memory pprof -> {path}", flush=True)
    except Exception as e:
        print(f"[{tag}] pprof failed: {e}", flush=True)


if __name__ == "__main__":
    main()
