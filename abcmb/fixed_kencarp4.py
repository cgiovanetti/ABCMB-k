"""FixedKenCarp4: diffrax's KenCarp4 with a fixed number of chord iterations per implicit stage.

Same tableau, stage predictor, Jacobian timing (once per step, at the second stage) and dense output as
diffrax.KenCarp4 (diffrax/_solver/kencarp4.py, imported from there). Differences: one (implicit) term, so the
explicit half of the IMEX pair is unused; and each implicit stage runs exactly n_iter chord iterations
    f <- f - M^-1 (f - vf(t_i, ypart_i + gamma dt f)),   M = I - gamma dt J,
with M^-1 formed once per step, unrolled, with no convergence test (no while loop). As in diffrax, the iteration
matrix carries no gradient: forward-mode AD differentiates the iterations themselves, not J.
"""
from typing import ClassVar

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax._solver.kencarp4 import _implicit_tableau, _KenCarp4Interpolation

jax.config.update("jax_enable_x64", True)


class FixedKenCarp4(diffrax.AbstractSolver):
    """KenCarp4 (ESDIRK half) with a static number n_iter of chord iterations per implicit stage. y: 1-D array."""
    n_iter: int = eqx.field(static=True, default=10)

    term_structure: ClassVar = diffrax.AbstractTerm
    interpolation_cls: ClassVar = _KenCarp4Interpolation

    def order(self, terms):
        return 4

    def error_order(self, terms):
        return 4

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        tab = _implicit_tableau
        dt = t1 - t0
        gdt = tab.a_diagonal[1] * dt
        ts = [t0] + [t0 + c * dt for c in tab.c]
        vf = lambda t, y, a: terms.vf(t, y, a)
        f = [vf(t0, y0, args)]                                   # explicit first stage
        minv = None
        for i in range(1, 6):
            ypart = y0 + dt * sum(a * fj for a, fj in zip(tab.a_lower[i - 1], f))
            fi = sum(p * fj for p, fj in zip(tab.a_predictor[i - 1], f))
            if minv is None:                                     # second stage: J once per step, no gradient
                y_jac, a_jac = jax.lax.stop_gradient((ypart + gdt * fi, args))
                J = jax.jacfwd(lambda y: vf(ts[i], y, a_jac))(y_jac)
                minv = jnp.linalg.inv(jnp.eye(y0.shape[0]) - gdt * J)
            for _ in range(self.n_iter):
                fi = fi - minv @ (fi - vf(ts[i], ypart + gdt * fi, args))
            f.append(fi)
        y1 = ypart + gdt * fi                                    # stiffly accurate: the last stage is y1
        y_error = dt * sum(b * fj for b, fj in zip(tab.b_error, f))
        k = jnp.stack([dt * fj for fj in f])
        dense_info = dict(y0=y0, y1=y1, k=(jnp.zeros_like(k), k))
        return y1, y_error, dense_info, None, diffrax.RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
