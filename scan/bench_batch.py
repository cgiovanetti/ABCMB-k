"""Production-shape timing + per-GPU peak memory of the profile driver's batched paths.
BB_MODE=value|adgrad, BB_B=<rows>, BB_REPS=<warm repeats>. One (mode, B) per process so the
peak-memory reading is clean. Rows: h POI across +-3 sigma, nuisances uniform in +-1 sigma.
Run from the repo root inside srun with the production PA_* env (see profile_prod_plikfull_neff48.slurm)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax
import abcmb
import scan.plik_full as plkmod
assert "ABCMB-k" in abcmb.__file__ and "ABCMB-k" in plkmod.__file__, (plkmod.__file__, abcmb.__file__)
import profile_prod_ad as ppa

mode, B = os.environ["BB_MODE"], int(os.environ["BB_B"])
reps = int(os.environ.get("BB_REPS", "2"))
poi = ppa.ORDER.index("h")
rng = np.random.default_rng(0)
PV = ppa.CENTER[poi] + ppa.SIGMA[poi] * np.linspace(-3, 3, B)
X = rng.uniform(-1, 1, (B, ppa.P))
POI_IDX = np.full(B, poi)
fn = {"value": ppa.fast_values_rows, "adgrad": ppa.ad_grad_rows}[mode]

def peak_gb():
    return max(d.memory_stats().get("peak_bytes_in_use", 0) for d in jax.devices()) / 1e9

tag = f"mode={mode} B={B} plf_maxit={os.environ.get('PLF_MAXIT', '800')} chunk(value)={ppa.FD_CHUNK} bchunk(ad)={ppa.AD_BCHUNK}"
t = time.perf_counter(); out = fn(POI_IDX, X, PV); cold = time.perf_counter() - t
chi2 = np.asarray(out if mode == "value" else out[0])
print(f"RESULT {tag} cold={cold:.1f}s peakGB={peak_gb():.2f} nonfinite={int((~np.isfinite(chi2)).sum())}", flush=True)
for r in range(reps):
    t = time.perf_counter(); fn(POI_IDX, X, PV); w = time.perf_counter() - t
    print(f"RESULT {tag} warm{r}={w:.1f}s per_row={w / B:.2f}s peakGB={peak_gb():.2f}", flush=True)
print("DONE", flush=True)
