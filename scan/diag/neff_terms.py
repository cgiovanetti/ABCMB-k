"""Split the _v6bbn Delta_Neff_init profile chi2 into its pieces at every grid point: Y_p and D/H data terms,
the BBN nuisance priors, and the CMB rest (total minus those), on the LINX Neff axis. xstar is in the driver's
scaled coordinates: theta = cen + sig * x (sig: config for cosmology, tau_n_sig / 1 for the BBN nuisances)."""
import importlib.util
import numpy as np
R = "/pscratch/sd/c/carag/ABCMB-k/scan/results"
z = np.load(f"{R}/profile_prod_ad_Delta_Neff_init_v6bbn.npz", allow_pickle=True)
spec = importlib.util.spec_from_file_location("cfg", str(z["config"])); m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m); C = m.CONFIG
NU = [str(s) for s in z["nuis"]]
sig = {**C["sig"], "tau_n_fac": C["bbn_like"]["tau_n_sig"], **{f"q{k}": 1.0 for k in range(12)}}
th = np.array([[C["cen"][n] + sig[n] * x for n, x in zip(NU, row)] for row in z["xstar"]])
P = {n: th[:, i] for i, n in enumerate(NU)}
chi2, d = z["chi2"], z["derived_bbn"]
Yp, DH, Neff = d[:, 3], d[:, 4], d[:, 1]
cY = ((Yp - 0.2458) / 0.0013) ** 2; cD = ((DH - 2.527) / 0.030) ** 2
cT = ((P["tau_n_fac"] - 1.0) / 0.000682) ** 2; cQ = sum(P[f"q{k}"] ** 2 for k in range(12))
cC = chi2 - cY - cD - cT - cQ
j = int(np.argmin(chi2))
print(f"{'Neff':>6} {'dtot':>6} {'dCMB':>6} {'dYp':>6} {'dD/H':>6} {'dq':>6} | {'Yp':>7} {'D/H':>6} {'omega_b':>8} {'q1':>5} {'q2':>5} {'q3':>5} | Yp, D/H, q pulls (abs chi2)")
for i in range(len(chi2)):
    print(f"{Neff[i]:6.3f} {chi2[i]-chi2[j]:6.2f} {cC[i]-cC[j]:6.2f} {cY[i]-cY[j]:6.2f} {cD[i]-cD[j]:6.2f} {cQ[i]+cT[i]-cQ[j]-cT[j]:6.2f} | "
          f"{Yp[i]:.5f} {DH[i]:6.3f} {P['omega_b'][i]:.6f} {P['q1'][i]:+.2f} {P['q2'][i]:+.2f} {P['q3'][i]:+.2f} | {cY[i]:.2f} {cD[i]:.2f} {cQ[i]+cT[i]:.2f}")
for nm, c in (("CMB", cC), ("Yp", cY), ("D/H", cD), ("rate+tau_n priors", cQ + cT), ("D/H + priors", cD + cQ + cT)):
    a, b, _ = np.polyfit(Neff, c, 2)
    print(f"{nm:18s}: along the profile path min at Neff {-b / (2 * a):.3f}, sigma {1 / np.sqrt(a):.3f}" if a > 0
          else f"{nm:18s}: not convex along the path; slope {np.polyfit(Neff, c, 1)[0]:+.2f} per unit Neff")
