"""Accuracy of LINX's polylog series (values) against mpmath on the real axis. CPU."""
import numpy as np, mpmath as mp
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from abcmb.linx.special_funcs import _Li_series
mp.mp.dps = 30
for n in (2, 3, 4):
    zs = np.r_[-2.7, -1.5, -1.0, -0.8, -0.6, -0.52, -0.3, 0.2, 0.52, 0.7, 0.95, 1.0, 1.05, 1.6, 2.6]
    errs = []
    for z in zs:
        ref = float(mp.re(mp.polylog(n, mp.mpf(z))))
        errs.append(abs(float(_Li_series(n, jnp.asarray(z))) - ref) / abs(ref))
    print(f"n={n}: " + ", ".join(f"{z:+.2f}:{e:.0e}" for z, e in zip(zs, errs)), flush=True)
