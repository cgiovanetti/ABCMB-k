"""Smoothness of the low-ell EE lnP tables (SimAll, SRoll2) around the Planck best-fit D_ell^EE:
local curvature d2 lnP / dD2 from the table's second differences vs the Gaussian scale, and the
second difference's scatter (table noise) over windows of 1..50 steps."""
import numpy as np
D = "/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/data"
tabs = {"simall": f"{D}/planck_2018_lowE_native/prob_table.txt",
        "sroll2": f"{D}/planck_sroll2_lowE_native/sroll2_prob_table.txt"}
step = 1e-4
for name, path in tabs.items():
    P = np.loadtxt(path)                                  # (3000, 28), ell 2..29
    print(f"== {name} {P.shape}")
    for il in (0, 2, 4, 8, 15, 27):
        c = P[:, il]
        j = int(np.argmax(c))                             # the peak (most probable D)
        w = slice(max(j - 300, 2), min(j + 300, len(c) - 2))
        d2 = (c[2:] - 2 * c[1:-1] + c[:-2]) / step ** 2   # curvature per (muK^2)^2
        d2w = d2[w.start - 1:w.stop - 1]
        # smooth curvature from a quadratic fit over +-300 steps
        x = np.arange(w.start, w.stop) * step
        q = np.polyfit(x - j * step, c[w], 2)
        sig_D = np.sqrt(-1 / (2 * q[0])) if q[0] < 0 else np.nan
        resid = c[w] - np.polyval(q, x - j * step)
        print(f"  ell {il + 2:2d}: peak D {j * step:.4f}, quad sigma_D {sig_D:.4g} muK^2 = {sig_D / step:.0f} steps; "
              f"fit curvature {2 * q[0]:.3g}; raw 2nd-diff median {np.median(d2w):.3g} sd {np.std(d2w):.3g}; "
              f"rms resid of quad fit {np.std(resid):.3g}")
