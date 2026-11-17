"""Compare kencarp_validate.py outputs: kencarp_fixed against kvaerno5 per setup (times, peak memory, Cl and tangent
differences, NaN rows).

    python scan/diag/kencarp_validate_compare.py <dir with {setup}_{solver}.npz> [solver, default kencarp_fixed]
"""
import glob
import os
import sys
import numpy as np

D = sys.argv[1]
NEW = sys.argv[2] if len(sys.argv) > 2 else "kencarp_fixed"
for setup in ("massless", "edges", "mnu", "act", "memory"):
    fk, fc = (os.path.join(D, f"{setup}_{s}.npz") for s in ("kvaerno5", NEW))
    if not (os.path.exists(fk) and os.path.exists(fc)):
        continue
    k, c = np.load(fk), np.load(fc)
    line = f"{setup:9s} B={int(k['B'])} dirs={int(k['ndir'])}:"
    for w in ("values", "grad"):
        if f"{w}_t" in k.files and f"{w}_t" in c.files:
            line += f"  {w} {k[f'{w}_t'][-1]:.0f} -> {c[f'{w}_t'][-1]:.0f} s"
    line += f"  peak {float(k['peak_gb']):.1f} -> {float(c['peak_gb']):.1f} GiB/GPU"
    print(line)
    for w in ("values", "grad"):
        if f"{w}_cls" not in k.files or f"{w}_cls" not in c.files:
            continue
        r, x = k[f"{w}_cls"], c[f"{w}_cls"]
        bad = ~np.isfinite(x).all(axis=(0, 2)) | ~np.isfinite(r).all(axis=(0, 2))
        ok = ~bad
        d = {"TT": np.abs(x[0] - r[0]) / np.abs(r[0]), "EE": np.abs(x[2] - r[2]) / np.abs(r[2]),
             "TE": np.abs(x[1] - r[1]) / np.sqrt(np.abs(r[0] * r[2]))}
        if len(r) > 3:
            d["pp"] = (np.abs(x[3] - r[3]) / np.abs(r[3]))[:, 30:]
        print(f"    {w} Cl, max over rows and ell: " + "  ".join(f"{s} {v[ok].max():.1e}" for s, v in d.items())
              + (f"  ({int(bad.sum())} rows non-finite)" if bad.any() else ""))
    if "grad_dcl" in k.files and "grad_dcl" in c.files:
        r, x = k["grad_dcl"], c["grad_dcl"]                  # (3, B, ndir, n_l)
        e = np.abs(x - r).max(axis=(1, 3)) / np.abs(r).max(axis=(1, 3))    # (3, ndir)
        print("    tangent diff / peak, worst direction: " + "  ".join(f"{s} {e[i].max():.1e}" for i, s in enumerate(("TT", "TE", "EE"))))
