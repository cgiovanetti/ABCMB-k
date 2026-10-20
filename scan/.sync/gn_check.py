"""Checks of the Gauss-Newton machinery in scan/profile_prod_ad.py (PA_CONFIG=cfg_ab.py, PYTHONPATH=repo):
 (1) plik residual: r^T siginv r + prior == clipy's penalized chi2 at the same nuisances;
 (2) chi2 from the gradient path == the values path;
 (3) gradient == the previous driver's (jvp of the profiled chi2 along the C_ell tangents);
 (4) curvature: chi2 along test directions u vs the quadratic model F + g.u + u.H.u/2
     (u = the Newton step, capped at 1 sigma, and random 0.5-sigma directions)."""
import os, sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/c/carag/ABCMB-k/scan")
import jax, jax.numpy as jnp
import profile_prod_ad as ppa
from scan.batched_grad import staged_chi2_and_grad

poi = ppa.ORDER.index("h")
grid = ppa._grid_for("h")
pv = np.array([grid[0], grid[len(grid) // 2], grid[-1]])
N = len(pv)
FREE = np.tile(ppa.nuis_idx_of(poi), (N, 1))
BASE = np.tile(ppa.CENTER, (N, 1)); BASE[:, poi] = pv
X = np.zeros((N, ppa.P)); X[2] = np.random.default_rng(1).uniform(-0.5, 0.5, ppa.P)

F = ppa.values_rows(BASE, FREE, X)
c, G, H = ppa.grad_gn_rows(BASE, FREE, X)
print("(2) chi2 values path", np.round(F, 4), " gradient path", np.round(c, 4), " max diff", np.abs(F - c).max(), flush=True)

# (1) residual consistency at row 1
out = ppa.model.call_batched([ppa.build_dict(ppa.theta_of(BASE[1], FREE[1], X[1]))], shard=False)
cls2d = ppa.plf.abcmb_cls_to_clik(out.ClTT, out.ClTE, out.ClEE, out.l)
nu = ppa._plf_prof_B(cls2d)[1][0]
r = ppa.plf.residual(cls2d[0], nu)
a = float(r @ ppa.plf.siginv @ r + ppa.plf.prior_penalty(nu)); b = float(ppa.plf.penalized_chi2(cls2d[0], nu))
print(f"(1) r^T W r + prior = {a:.8f}  clipy penalized chi2 = {b:.8f}  diff {a - b:.2e}", flush=True)

# (3) previous driver's gradient: jvp of the envelope-profiled chi2 along the same C_ell tangents
def chi2_of_cls_old(ClTT, ClTE, ClEE):
    lvec = ppa.model.SS.ells
    Dtt = ppa.pl.abcmb_cl_to_Dl(ClTT, lvec); Dee = ppa.pl.abcmb_cl_to_Dl(ClEE, lvec)
    c2d = ppa.plf.abcmb_cls_to_clik(ClTT, ClTE, ClEE, lvec)
    nu_ = jax.lax.stop_gradient(ppa._plf_prof_B(jax.lax.stop_gradient(c2d))[1])
    c2 = jax.vmap(ppa.plf.penalized_chi2)(c2d, nu_)
    return c2 + ppa.lowee.chi2(Dee) + ppa.lowtt.chi2(Dtt)

import equinox as eqx
thetas = [jnp.asarray(ppa.theta_of(BASE[b], FREE[b], X[b])) for b in range(N)]
full_ps = [ppa._phys_to_derived(t) for t in thetas]
per = [[eqx.filter(jax.jvp(ppa._phys_to_derived, (thetas[b],), (jnp.zeros(ppa.D).at[int(FREE[b, j])].set(ppa.SIG[int(FREE[b, j])]),))[1],
                   eqx.is_inexact_array) for j in range(ppa.P)] for b in range(N)]
dots = [jax.tree.map(lambda *xs: jnp.stack(xs), *[per[b][j] for b in range(N)]) for j in range(ppa.P)]
_, g_old = staged_chi2_and_grad(ppa.model, full_ps, dots, chi2_of_cls_old, shard=ppa.DO_SHARD, spec_b_local=4)
g_old = np.asarray(g_old)
print("(3) max |g_new - g_old| / max|g_old| per row:", np.abs(G - g_old).max(1) / np.abs(g_old).max(1), flush=True)

# (4) curvature along test directions
rng = np.random.default_rng(2)
U, rows = [], []
for b in range(N):
    d = -np.linalg.solve(H[b], G[b]); d *= min(1.0, 1.0 / np.abs(d).max())
    U.append(d); rows.append(b)
    for _ in range(2):
        u = rng.normal(size=ppa.P); u *= 0.5 / np.abs(u).max(); U.append(u); rows.append(b)
U = np.array(U); rows = np.array(rows)
Fu = ppa.values_rows(BASE[rows], FREE[rows], X[rows] + U)
for k, b in enumerate(rows):
    lin = G[b] @ U[k]; quad = 0.5 * U[k] @ H[b] @ U[k]
    act = Fu[k] - F[b]
    print(f"(4) row{b} {'newton' if k % 3 == 0 else 'random'}: actual dchi2 {act:+9.4f}  model {lin + quad:+9.4f} "
          f"(lin {lin:+9.4f} quad {quad:+8.4f})  curvature ratio actual-lin/quad = {(act - lin) / quad:.3f}", flush=True)
print("eig(H) per row:", [np.round(np.linalg.eigvalsh(h), 3).tolist() for h in H], flush=True)
