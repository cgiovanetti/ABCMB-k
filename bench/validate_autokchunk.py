"""Validate adaptive k_chunk in call_batched:
 (1) at small B it equals the old fixed 100 (bit-for-bit Cl/Pk);
 (2) at large B it auto-shrinks to FIT memory where fixed-100 OOMs;
 (3) different k_chunk -> Cl/Pk agree to the diffrax-noise floor (peak-normalized).
Single GPU (shard=False)."""
import os, math, time, traceback
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from abcmb.main import Model

FID = {'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225, 'A_s': 2.12424e-9,
       'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245, 'TCMB0': 2.34865418e-4,
       'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
       'tau_reion': 0.0544, 'Delta_z_reion': 0.5, 'z_reion_He': 3.5,
       'Delta_z_reion_He': 0.5, 'exp_reion': 1.5}


def pl(B):
    return [dict(FID, h=0.66 + 0.0003 * i, omega_cdm=0.119 + 0.0001 * i)
            for i in range(B)]


def peaknorm(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def block(o):
    jax.block_until_ready(o.ClTT)


m = Model(user_species=None, output_Cl=True, l_max=800, lensing=False,
          output_Pk=True, output_k_max=0.5, l_max_g=12, l_max_pol_g=10,
          l_max_ur=17, l_max_ncdm=17)
n_k = len(m.PE.k_axis_perturbations)
print(f"N_k = {n_k}; auto formula: max(32, min(100, N_k, floor(28/3.3e-3 / B_local)))",
      flush=True)
for B in (8, 64, 128, 256, 512, 1024):
    bl = B  # shard=False -> B_local = B
    auto = max(32, min(100, n_k, int(28.0 / 3.3e-3) // max(bl, 1)))
    print(f"  B={B:>5} shard=False -> auto k_chunk={auto}", flush=True)

# (1) small-B parity: auto (=100) vs explicit 100
o_auto = m.call_batched(pl(8), shard=False); block(o_auto)
o_100 = m.call_batched(pl(8), shard=False, k_chunk_size=100); block(o_100)
print(f"[B=8] auto vs explicit-100: TT|Δ|/peak={peaknorm(o_auto.ClTT, o_100.ClTT):.2e} "
      f"Pk={peaknorm(o_auto.Pk, o_100.Pk):.2e}", flush=True)

# (3) different chunk -> Cl/Pk agree to diffrax-noise floor
o_25 = m.call_batched(pl(8), shard=False, k_chunk_size=25); block(o_25)
print(f"[B=8] kchunk=100 vs 25: TT|Δ|/peak={peaknorm(o_100.ClTT, o_25.ClTT):.2e} "
      f"TE={peaknorm(o_100.ClTE, o_25.ClTE):.2e} EE={peaknorm(o_100.ClEE, o_25.ClEE):.2e} "
      f"Pk={peaknorm(o_100.Pk, o_25.Pk):.2e}", flush=True)

# (2) OOM-avoidance demo at B=128 single GPU: fixed-100 (12800 lanes ~42GB) should
# OOM; auto (~66, ~28GB) should fit.
for tag, kc in (("auto", None), ("fixed-100", 100)):
    try:
        t0 = time.perf_counter()
        o = m.call_batched(pl(128), shard=False, k_chunk_size=kc); block(o)
        peak = jax.local_devices()[0].memory_stats()['peak_bytes_in_use'] / 1e9
        print(f"[B=128 {tag}] OK in {time.perf_counter()-t0:.1f}s peak={peak:.1f}GB",
              flush=True)
    except Exception as e:
        print(f"[B=128 {tag}] FAILED: {type(e).__name__}: {str(e)[:100]}", flush=True)
print("VALIDATE-DONE", flush=True)
