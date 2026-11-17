"""Which LINX Hessian-row lanes go non-finite, and where (TODO item 20). Rows: LINX inputs in a text file (omega_b,
Delta_Neff_init, log10_m_chi, tau_n_fac, q0..q11 per line). Single-row program per row, then the batched stage
program (hessian_rows) on all rows. CPU only.
    PYTHONPATH=$(pwd) JAX_PLATFORM_NAME=cpu python scan/diag/linx_curv_nan.py logs/linx_curv_x_it2.txt"""
import os, sys, time
import numpy as np
os.environ.setdefault("EQX_ON_ERROR", "nan")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from scan.bbn_stage import LinxStage, OUT_NAMES

T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

S = LinxStage("key_recommended", g_chi=4.0, curv=("log10_m_chi",))
X = np.loadtxt(sys.argv[1], ndmin=2)
f1 = jax.jit(S._one_vjk)
for r, x in enumerate(X):
    y, J, K = (np.asarray(a) for a in f1(jnp.asarray(x)))
    bad = ~np.isfinite(K[:, 0, :])
    log(f"row {r} m {10 ** x[2]:.2f} MeV DN {x[1]:+.3f}: y finite {np.isfinite(y).all()}, J finite {np.isfinite(J).all()}, "
        f"K non-finite {int(bad.sum())}/{bad.size}" + (f" at (output, input) {[(OUT_NAMES[i], S.in_names[j]) for i, j in np.argwhere(bad)][:8]}" if bad.any() else
                                                       f"; d2Neff/dm2 {K[1, 0, 2]:+.4e}"))
K = S.hessian_rows(X)
log(f"batched hessian_rows: non-finite lanes {np.where(~np.isfinite(K).reshape(len(X), -1).all(1))[0].tolist()}")
