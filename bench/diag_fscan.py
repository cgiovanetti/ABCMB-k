"""diag_fscan.py -- is the BFGS stall a diffrax adaptive-stepping ROUGHNESS floor in
chi2, and does a TIGHTER solver tolerance smooth it?

Value==gradient (diag_consistency proved f_cb==f_AD exactly), and the worst row (25)
descends smoothly along -g, so most of the stall is the contaminated Hinv (fresh start
fixes). But row 121 (high ln10As) INCREASES f even for tiny -g steps -> suspected
solver roughness. Test: fine 1-D scan of f along -g at a few rows, at the production
rtol=1e-5 AND a tighter rtol=1e-6. If the curve is smooth-decreasing -> gradient fine,
just needs the right step (Hinv). If jittery at ~0.3 -> roughness; if 1e-6 is much
smoother -> tightening rtol is the principled fix.

NEVER login node. One node, fits debug (~15 min): a few call_batched chunks.
"""
import os
os.environ.setdefault("PA_CONFIG", "scan/configs/lcdm.py")
import numpy as np
import jax.numpy as jnp
import scan.profile_prod_ad as P
from abcmb.main import Model

st = np.load("scan/results/profile_prod_ad_STATE.npz", allow_pickle=True)
POI_IDX = st["POI_IDX"]; PV = st["PV"]; x = np.array(st["x"], float)
g = np.array(st["g"], float); gn = np.abs(g).max(1)
rows = [25, 91, 121]                                          # worst, a mid, the anomalous one

# second model at tighter rtol (same config as P.model otherwise)
RT2 = 1e-6
model2 = Model(user_species=P.USER_SPECIES, output_Cl=True, l_max=P.LMAX, lensing=True,
               output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
               rtol_large_k_PE=RT2, atol_large_k_PE=RT2 * 1e-2,
               rtol_small_k_PE=min(1e-5, RT2), max_steps_PE=16384)

def f_of(model, POI_b, X, PV_b):
    batch = [P.build_dict(P.assemble_phys(int(POI_b[i]), X[i], PV_b[i])) for i in range(len(PV_b))]
    out = model.call_batched(batch, shard=P.DO_SHARD)
    return P._chi2_from_out(out)

alphas = np.linspace(0.0, 0.12, 19)
for b in rows:
    d = -g[b] / max(np.linalg.norm(g[b]), 1e-30)
    X = np.clip(x[b][None, :] + alphas[:, None] * d[None, :], -P.XBOX, P.XBOX)
    POI_b = np.full(len(alphas), POI_IDX[b]); PV_b = np.full(len(alphas), PV[b])
    f1 = f_of(P.model, POI_b, X, PV_b)
    f2 = f_of(model2, POI_b, X, PV_b)
    print(f"\n=== row {b} ({P.ORDER[int(POI_IDX[b])]}, ||g||={gn[b]:.3f}) f along -g ===", flush=True)
    print(" alpha    df(rtol1e-5)   df(rtol1e-6)")
    for i, a in enumerate(alphas):
        print(f" {a:5.3f}   {f1[i]-f1[0]:+10.4f}    {f2[i]-f2[0]:+10.4f}")
    # roughness = std of 2nd differences (curvature jitter) over the small-step region
    rough1 = np.std(np.diff(f1, 2)); rough2 = np.std(np.diff(f2, 2))
    print(f"  roughness(2nd-diff std): rtol1e-5={rough1:.4f}  rtol1e-6={rough2:.4f}  "
          f"min df: 1e-5={f1.min()-f1[0]:+.4f} 1e-6={f2.min()-f2[0]:+.4f}")
