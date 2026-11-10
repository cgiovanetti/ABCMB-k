"""ACT DR6 CMB-only (candl) data layout: ell range of the theory input, bins, where the kept bins' windows end."""
import numpy as np
import candl, candl_data
like = candl.Like(candl_data.ACT_DR6_TTTEEE)
print("ells", like.ells[0], like.ells[-1], len(like.ells))
print("spec_order", like.spec_order, "bins", like.N_bins, "total", like.N_bins_total)
print("required params", like.required_nuisance_parameters, "data_model", [type(t).__name__ for t in like.data_model])
print("priors", [vars(p) for p in like.priors])
W = np.asarray(like.window_functions)
for i, s in enumerate(like.spec_order):
    w = np.asarray(like.window_functions[i])            # (n_ell, n_bins)
    nz = np.where(np.any(np.abs(w) > 0, axis=1))[0]
    eff = like.effective_ells[like.bins_start_ix[i]:like.bins_stop_ix[i]]
    cum = np.cumsum(np.abs(w).sum(axis=1)); cum /= cum[-1]
    lmax_999 = like.ells[np.searchsorted(cum, 1 - 1e-6)]
    print(f"{s}: windows shape {w.shape}, nonzero ell {like.ells[nz[0]]}..{like.ells[nz[-1]]}, "
          f"bin centres {eff[0]:.0f}..{eff[-1]:.0f}; |W| mass beyond ell: 1e-6 at {lmax_999}")
    for L in (5000, 5500, 6000, 6200, 6325, 6400, 6500, 7000):
        print(f"   |W| fraction above {L}: {1 - cum[np.searchsorted(like.ells, L)]:.3e}")
