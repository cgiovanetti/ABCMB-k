"""relative difference of each act_lmax_check config's lensed Cls to ref9500, in ell bins (cosmology 0)."""
import sys, os, numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "act_lmax")
r = np.load(os.path.join(D, "ref9500.npz")); e0, c0 = r["ells"], r["cls"]
edges = [2000, 3000, 4000, 4500, 5000, 5250, 5500, 5750, 6000, 6200, 6325]
for cfg in sys.argv[1:]:
    d = np.load(os.path.join(D, f"{cfg}.npz")); e, c = d["ells"], d["cls"]
    print(f"{cfg}: max over 8 cosmologies of |C/C_ref - 1| (TE: / sqrt(TT EE))")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (e >= lo) & (e < hi); m0 = (e0 >= lo) & (e0 < hi)
        tt = np.max(np.abs(c[:, 0, m] / c0[:, 0, m0] - 1)); ee = np.max(np.abs(c[:, 2, m] / c0[:, 2, m0] - 1))
        te = np.max(np.abs(c[:, 1, m] - c0[:, 1, m0]) / np.sqrt(c0[:, 0, m0] * c0[:, 2, m0]))
        sgn = np.sign(np.mean(c[:, 0, m] / c0[:, 0, m0] - 1))
        print(f"  l {lo:5d}-{hi:5d}: TT {tt:.2e} (sign {sgn:+.0f})  TE {te:.2e}  EE {ee:.2e}")
