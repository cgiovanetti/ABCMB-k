"""bench_plik_runtime.py — realistic per-param runtime for a Planck-grade run.

The CHANGELOG perf numbers (0.44 s/param) are at l_max=800, lensing off. A real
plik-lite scan needs l_max>=2508 with LENSING (Planck data is lensed). The
perturbation/spectrum cost and the GPU memory both grow with l_max, so the unit
cost must be MEASURED, not extrapolated.

Sweeps B for shard=True (all visible GPUs) and shard=False (1 GPU), reporting
post-compile s/param and peak GB/device. Also folds the plik-lite chi^2 in so the
reported time is the true end-to-end frequentist unit cost.

Env:
  BENCH_LMAX     (2508)         l_max
  BENCH_LENS     (1)            lensing on/off
  BENCH_MASSIVE  (0)            1 massive nu (0.06 eV) instead of massless
  BENCH_BLIST    "16,32,64,128" comma list of B for shard=True
  BENCH_BLIST1   "4,8,16"       comma list of B for shard=False (1 GPU)
  BENCH_REPS     (3)            timed reps after warmup
"""
import os, time, gc
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from abcmb.main import Model
from scan.plik_lite import PlikLite

As = float(np.exp(3.044) / 1e10)
BASE = {
    'h': 0.6736, 'omega_cdm': 0.1200, 'omega_b': 0.02237, 'A_s': As,
    'n_s': 0.9649, 'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5, 'z_reion_He': 3.5,
    'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}


def peak_gb():
    try:
        return max(d.memory_stats()["peak_bytes_in_use"]
                   for d in jax.devices('gpu')) / 1e9
    except Exception:
        return float('nan')


def reset_peak():
    # no public reset; rely on monotonic peak and report per-config best-effort
    pass


def make_batch(B, seed):
    rng = np.random.default_rng(seed)
    ds = rng.uniform(-0.003, 0.003, size=B)
    out = []
    for i in range(B):
        p = dict(BASE); p['omega_cdm'] = BASE['omega_cdm'] + float(ds[i])
        out.append(p)
    return out


def main():
    LMAX = int(os.environ.get("BENCH_LMAX", 2508))
    LENS = os.environ.get("BENCH_LENS", "1") == "1"
    MASSIVE = os.environ.get("BENCH_MASSIVE", "0") == "1"
    BLIST = [int(x) for x in os.environ.get("BENCH_BLIST", "16,32,64,128").split(",") if x]
    BLIST1 = [int(x) for x in os.environ.get("BENCH_BLIST1", "4,8,16").split(",") if x]
    REPS = int(os.environ.get("BENCH_REPS", 3))

    if MASSIVE:
        BASE['N_nu_massive'] = 1
        BASE['Neff'] = 3.044

    gpus = jax.devices('gpu')
    print(f"=== bench: lmax={LMAX} lensing={LENS} massive={MASSIVE} "
          f"nGPU={len(gpus)} ===", flush=True)

    pl = PlikLite()
    model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=LENS,
                  output_Pk=False, l_max_g=12, l_max_pol_g=10,
                  l_max_ur=17, l_max_ncdm=17)

    def chi2_batched(outb):
        res = pl.chi2_from_abcmb(outb.ClTT, outb.ClTE, outb.ClEE, outb.l,
                                 profile=True, with_prior=True)
        return res['chi2']

    def run_sweep(blist, shard, tag):
        print(f"\n--- {tag} (shard={shard}) ---", flush=True)
        print(f"{'B':>5} {'compile_s':>10} {'s/param':>9} {'cosmo/s':>8} "
              f"{'peakGB':>8}", flush=True)
        for B in blist:
            try:
                batch = make_batch(B, seed=B)
                # warmup / compile
                t0 = time.perf_counter()
                outb = model.call_batched(batch, shard=shard)
                c2 = chi2_batched(outb)
                jax.block_until_ready((outb.ClTT, c2))
                tc = time.perf_counter() - t0
                # timed reps
                ts = []
                for _ in range(REPS):
                    t1 = time.perf_counter()
                    outb = model.call_batched(batch, shard=shard)
                    c2 = chi2_batched(outb)
                    jax.block_until_ready((outb.ClTT, c2))
                    ts.append(time.perf_counter() - t1)
                dt = float(np.median(ts))
                print(f"{B:>5} {tc:>10.1f} {dt/B:>9.3f} {B/dt:>8.2f} "
                      f"{peak_gb():>8.2f}", flush=True)
                del outb, c2; gc.collect()
            except Exception as e:
                msg = str(e).splitlines()[0][:80]
                print(f"{B:>5} {'OOM/ERR':>10}  {msg}", flush=True)
                break

    if len(gpus) > 1:
        run_sweep(BLIST, shard=True, tag=f"{len(gpus)}-GPU sharded")
    run_sweep(BLIST1, shard=False, tag="single-GPU")


if __name__ == "__main__":
    main()
