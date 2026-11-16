"""Compare kencarp_bench.py outputs against the Kvaerno5 run: warm/cold times, Cl differences (TT, EE, pp relative;
TE over sqrt(TT EE)), and the forward-mode tangents per LCDM direction (difference over the tangent's peak).

    python scan/diag/kencarp_compare.py logs/kencarp_bench/kvaerno5.npz logs/kencarp_bench/kencarp_10.npz ...
"""
import sys
import numpy as np

NAMES = ["omega_b", "omega_cdm", "h", "ln10As", "n_s", "tau"]
ref = np.load(sys.argv[1])
L0 = 2                                                        # ell index offset: C_ell arrays start at ell = 2


def cl_diffs(c, r):
    """max over B and ell of the relative differences; c, r: (n_spec, B, n_l) TT, TE, EE[, pp]."""
    tt, te, ee = (np.abs(c[i] - r[i]) for i in range(3))
    out = {"TT": (tt / np.abs(r[0])).max(), "TE": (te / np.sqrt(np.abs(r[0] * r[2]))).max(),
           "EE": (ee / np.abs(r[2])).max()}
    if len(c) > 3:
        out["pp"] = (np.abs(c[3] - r[3]) / np.abs(r[3]))[:, 30:].max()
    return out


def row(tag, d):
    t = [f"{k} {d[f'{k}_t_cold']:.0f}/{d[f'{k}_t_warm']:.0f} s" for k in ("values", "grad") if f"{k}_t_warm" in d.files]
    return f"{tag:18s} " + "   ".join(t)


print("cold/warm wall time (B = %d, %d GPUs)" % (int(ref["B"]), int(ref["ndev"])))
print(row("kvaerno5", ref))
for f in sys.argv[2:]:
    d = np.load(f)
    tag = f"{d['solver']} N={int(d['n_iter'])}"
    print(row(tag, d))
    for k in ("values", "grad"):
        if f"{k}_t_warm" in d.files and f"{k}_t_warm" in ref.files:
            print(f"    {k} warm speed-up vs Kvaerno5: {ref[f'{k}_t_warm'] / d[f'{k}_t_warm']:.2f}x")
    if "values_cls" in d.files:
        print("    Cl vs Kvaerno5 (values): " + ", ".join(f"{s} {v:.1e}" for s, v in cl_diffs(d["values_cls"], ref["values_cls"]).items()))
    if "grad_dcl" in d.files:
        dc, rc = d["grad_dcl"], ref["grad_dcl"]                  # (3 spectra, B, 6 dirs, n_l)
        for j, nm in enumerate(NAMES):
            e = [np.abs(dc[s, :, j] - rc[s, :, j]).max() / np.abs(rc[s, :, j]).max() for s in range(3)]
            print(f"    d/d{nm:9s} tangent diff / peak: TT {e[0]:.1e}  TE {e[1]:.1e}  EE {e[2]:.1e}")
