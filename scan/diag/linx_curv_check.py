"""LINX second derivatives for TODO item 20: the Hessian rows d J / d x_a of the LINX stage (x = omega_b, Delta_Neff_init,
log10_m_chi, tau_n_fac, q0..q11; outputs YHe, Neff, T_nu, Y_p, D/H) by forward-over-forward AD, against central
differences of the AD Jacobian (a one-off check of the derivative, not used by the driver). Rows: _v1emwimp
profile points (logs/linx_curv_x.npy). CPU only.

    PYTHONPATH=$(pwd) JAX_PLATFORM_NAME=cpu python scan/diag/linx_curv_check.py [a ...]   (a = 1 DN, 2 log10_m)
"""
import os, sys, time
import numpy as np
os.environ.setdefault("EQX_ON_ERROR", "nan")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(os.environ.get("SCRATCH", "/pscratch/sd/c/carag"),
                                                                ".jax_cache_abcmb"))
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from scan.bbn_stage import LinxStage, OUT_NAMES

T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

S = LinxStage("key_recommended", g_chi=4.0)
X = np.load("logs/linx_curv_x.npy")
AS = [int(a) for a in sys.argv[1:]] or [2]


def vj_nosg(x):
    """(y, J) as _one_vj, without the stop_gradient on the background in the network block (the outer jvp along a
    background input must reach the network's u-derivatives through the background)."""
    u0 = jnp.concatenate([x[:1], x[S.i_tau:]])
    net = lambda b, u: S._rest(b, u[0] * S._ETA, u[1], u[2:])
    lm = S._lm(x)
    bg_out, bg_dn = jax.jvp(lambda d: S._background(d, lm), (x[1],), (jnp.ones_like(x[1]),))
    y, y_dn = jax.jvp(lambda b: net(b, u0), (bg_out,), (bg_dn,))
    cols = [y_dn[:, None]]
    _, bg_dm = jax.jvp(lambda m: S._background(x[1], m), (x[2],), (jnp.ones_like(x[2]),))
    cols.append(jax.jvp(lambda b: net(b, u0), (bg_out,), (bg_dm,))[1][:, None])
    Ju = jax.jacfwd(lambda u: net(bg_out, u))(u0)
    return y, jnp.concatenate([Ju[:, :1]] + cols + [Ju[:, 1:]], axis=1)


def vjk(x, a):
    e = jnp.zeros_like(x).at[a].set(1.0)
    (y, J), (_, K) = jax.jvp(vj_nosg, (x,), (e,))
    return y, J, K


f_vj = jax.jit(S._one_vj)
f_vjk = {a: jax.jit(lambda x, a=a: vjk(x, a)) for a in AS}
names = S.in_names
for r, x0 in enumerate(X):
    x0 = jnp.asarray(x0)
    t = time.time(); y, J = f_vj(x0); jax.block_until_ready(J)
    log(f"row {r}: m {10 ** float(x0[2]):.1f} MeV DN {float(x0[1]):+.3f}; y {np.round(np.asarray(y), 6).tolist()} "
        f"({time.time() - t:.0f}s)")
    for a in AS:
        t = time.time(); y2, J2, K = f_vjk[a](x0); jax.block_until_ready(K); tk = time.time() - t
        K = np.asarray(K); J2 = np.asarray(J2)
        log(f"  K row {names[a]}: finite {bool(np.isfinite(K).all())} ({tk:.0f}s); J (no stop_gradient) vs J "
            f"max rel {np.max(np.abs(J2 - np.asarray(J)) / (np.abs(np.asarray(J)) + 1e-30)):.1e}")
        for h in (1e-2, 3e-3):
            Jp = np.asarray(f_vj(x0.at[a].add(h))[1]); Jm = np.asarray(f_vj(x0.at[a].add(-h))[1])
            Kd = (Jp - Jm) / (2 * h)
            for k, on in enumerate(OUT_NAMES):
                sc = np.abs(Kd[k]).max() + 1e-300
                big = np.argsort(-np.abs(Kd[k]))[:4]
                log(f"    h {h:g} {on:14s}: max|K_AD - K_cd| / max|K_cd| {np.abs(K[k] - Kd[k]).max() / sc:.2e}; "
                    + ", ".join(f"{names[b]} AD {K[k, b]:+.4e} cd {Kd[k, b]:+.4e}" for b in big))
