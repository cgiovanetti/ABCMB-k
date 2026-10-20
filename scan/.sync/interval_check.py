"""Coverage of profile_prod_ad.poi_interval on exact Gaussian profiles (sigma = 1): a measurement
x ~ N(mu_true, 1), profile chi2(mu) = (mu - x)^2 on a 21-point grid. With the bound mu >= 0 the
boundary-aware intervals must cover mu_true at the nominal rate for every mu_true (they are the
Feldman-Cousins unified intervals here), switching to one-sided limits as x falls below the bound.
Without a bound, the usual Delta-chi2 crossings must cover at the nominal rate."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/c/carag/ABCMB-k/scan")
import profile_prod_ad as ppa

rng = np.random.default_rng(0)
NT = 4000
for cl in (ppa.CL1, ppa.CL95):
    print(f"--- CL {cl:.4f}  t_threshold at d = 0, 0.5, 1, 2, 5: "
          f"{np.round(ppa.t_threshold(cl, np.array([0, .5, 1, 2, 5.])), 3)}")
    for mu_true in (0.0, 0.3, 1.0, 2.0, 4.0):
        cov_b = one = 0; cov_u = 0
        for _ in range(NT):
            x = rng.normal(mu_true)
            # bounded: grid from the bound up to a few sigma past the data
            g = np.linspace(0.0, max(x, 0) + 5.0, 21)
            lo, best, hi = ppa.poi_interval(g, (g - x) ** 2, cl, (0.0, None))
            cov_b += (lo <= mu_true <= hi); one += (lo == 0.0)
            # unbounded: grid centred on the data
            g2 = np.linspace(x - 5, x + 5, 21)
            lo2, _, hi2 = ppa.poi_interval(g2, (g2 - x) ** 2, cl)
            cov_u += (lo2 <= mu_true <= hi2)
        print(f"mu_true {mu_true:3.1f}: bounded coverage {cov_b / NT:.3f} (one-sided in {one / NT:.2f} of trials); "
              f"unbounded coverage {cov_u / NT:.3f}")
# one worked example: data below the bound
g = np.linspace(0, 5, 21)
for x in (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0):
    lo, _, hi = ppa.poi_interval(np.linspace(0, max(x, 0) + 5, 21), (np.linspace(0, max(x, 0) + 5, 21) - x) ** 2, 0.90, (0.0, None))
    print(f"x={x:+.1f}: 90% interval [{lo:.3f}, {hi:.3f}]")
