"""time_b51.py — warm per-call time of call_batched at the production batch size,
at a given rtol, to size the production walltime. Builds Model at rtol_large_k_PE
(env TB_RTOL), times one compile call + two warm calls of call_batched(B=TB_B,
shard=True). Prints the warm median. Run via srun (debug, 1 GPU). Env: TB_RTOL,
TB_B(51), TB_LMAX(2508)."""
import os, time
_SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/c/carag")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_SCRATCH, ".jax_cache_abcmb"))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from abcmb.main import Model
from scan.plik_lite import PlikLite

RTOL = float(os.environ.get("TB_RTOL", 1e-5))
B = int(os.environ.get("TB_B", 51))
LMAX = int(os.environ.get("TB_LMAX", 2508))
FIXED = {'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4, 'N_nu_massive': 1,
         'T_nu_massive': 0.71611, 'm_nu_massive': 0.06, 'Delta_z_reion': 0.5,
         'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5, 'tau_reion': 0.0544}
pl = PlikLite()
model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=True,
              output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
              rtol_large_k_PE=RTOL, atol_large_k_PE=RTOL * 1e-2,
              rtol_small_k_PE=min(1e-5, RTOL), max_steps_PE=16384)


def one(seed):
    rng = np.random.default_rng(seed)
    batch = []
    for _ in range(B):
        batch.append(dict(FIXED, h=0.6736 + 0.001 * rng.standard_normal(),
                          omega_b=0.02237, omega_cdm=0.1200, n_s=0.9649,
                          A_s=float(np.exp(3.044) / 1e10)))
    t = time.perf_counter()
    out = model.call_batched(batch, shard=True)
    jax.block_until_ready((out.ClTT, out.ClEE))
    return time.perf_counter() - t


def main():
    print(f"devices={jax.devices()} rtol={RTOL:.0e} B={B} lmax={LMAX}", flush=True)
    t0 = one(0); print(f"COLD (compile+solve): {t0:.0f}s", flush=True)
    w = [one(1), one(2)]
    print(f"WARM per-call: {np.median(w):.0f}s  (runs {w})", flush=True)


if __name__ == "__main__":
    main()
