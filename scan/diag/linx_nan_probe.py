"""Where does the second-order forward tangent of LINX's background go NaN? (TODO item 20 / 21; CPU)
Second directional derivatives of the background vector field dY in the state and in (rho_extra_init, m_chi), at
states along a solved background, with jax_debug_nans to name the primitive."""
import sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from abcmb.linx.background import BackgroundModel
from abcmb.linx import thermo, const

from abcmb.linx.special_funcs import Li, Riemann_zeta
for n in (3, 4):                                       # thermo._Li_BE's constant = the series' value at z = 1
    print(f"Li({n}, 1) - zeta({n}) = {float(Li(n, 1.0) - Riemann_zeta(n)):.3e}", flush=True)
bg = BackgroundModel(g_chi=4.0)
m_chi = 10 ** 1.373                                    # 23.6 MeV
rho_x = (7 / 8) * (4 / 11) ** (4 / 3) * thermo.rho_massless_BE(const.T_start, 0., 2) * (-0.623)
args0 = (0.0, rho_x, m_chi)
# states: T_g from T_start down to T_end, T_nu ~ T_g (a few % apart), lna ~ -log(T)
Tg = jnp.geomspace(const.T_start, const.T_end * 1.01, 40)
Y = (jnp.log(const.T_start / Tg), Tg, Tg * 0.99)


def f(lna, Tg, Tnu, rx, m):
    return jnp.stack(bg.dY(0.0, (lna, Tg, Tnu), (0.0, rx, m)))


def d2(fun, x, v):
    return jax.jvp(lambda z: jax.jvp(fun, (z,), (v,))[1], (x,), (v,))[1]


def probe(name, fun):
    out = jax.jit(jax.vmap(fun))(*Y)
    bad = ~np.isfinite(np.asarray(out))
    print(f"{name:28s}: non-finite at {int(bad.any(-1).sum()) if bad.ndim > 1 else int(bad.sum())} of {len(Tg)} states"
          + (f"; first T_g {float(Tg[np.where(bad.reshape(len(Tg), -1).any(1))[0][0]]):.4g} MeV" if bad.any() else ""),
          flush=True)
    return bad.any()


# first and second derivatives along each input
for k, nm in enumerate(["lna", "T_g", "T_nu"]):
    e = jnp.zeros(3).at[k].set(1.0)
    probe(f"d dY / d{nm}", lambda a, b, c, e=e: jax.jvp(lambda z: f(z[0], z[1], z[2], rho_x, m_chi),
                                                        (jnp.stack([a, b, c]),), (e,))[1])
    probe(f"d2 dY / d{nm}2", lambda a, b, c, e=e: d2(lambda z: f(z[0], z[1], z[2], rho_x, m_chi), jnp.stack([a, b, c]), e))
probe("d dY / dm", lambda a, b, c: jax.jvp(lambda m: f(a, b, c, rho_x, m), (m_chi,), (1.0,))[1])
probe("d2 dY / dm2", lambda a, b, c: d2(lambda m: f(a, b, c, rho_x, m), m_chi, 1.0))
probe("d2 dY / drho_x2", lambda a, b, c: d2(lambda r: f(a, b, c, r, m_chi), rho_x, 1.0))
probe("d2 dY / dT_g dm (mixed)", lambda a, b, c: jax.jvp(lambda m: jax.jvp(lambda t: f(a, t, c, rho_x, m), (b,), (1.0,))[1],
                                                        (m_chi,), (1.0,))[1])
# the output processing of the background (rho_chi_v, p_chi_v, N_eff, T_g) along T_g and m
probe("d2 rho_chi / dT_g2", lambda a, b, c: d2(lambda t: thermo.rho_chi(t, m_chi, 4.0, False), b, 1.0))
probe("d2 rho_chi / dm2", lambda a, b, c: d2(lambda m: thermo.rho_chi(b, m, 4.0, False), m_chi, 1.0))
probe("d2 p_chi / dm2", lambda a, b, c: d2(lambda m: thermo.p_chi(b, m, 4.0, False), m_chi, 1.0))

if len(sys.argv) > 1:                                  # name the primitive of the first NaN of one probe
    jax.config.update("jax_debug_nans", True)
    k = int(sys.argv[1]); e = jnp.zeros(3).at[k].set(1.0)
    for i in range(len(Tg)):
        d2(lambda z: f(z[0], z[1], z[2], rho_x, m_chi), jnp.stack([Y[0][i], Y[1][i], Y[2][i]]), e)
