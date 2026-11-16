"""FixedKenCarp4 (abcmb/fixed_kencarp4.py) against diffrax's Kvaerno5 on stiff toy problems: values and
forward-mode tangents (ForwardMode adjoint, as ABCMB uses). CPU."""
import diffrax
import jax
import jax.numpy as jnp
import pytest
from abcmb.fixed_kencarp4 import FixedKenCarp4

jax.config.update("jax_enable_x64", True)


def robertson(t, y, a):                          # stiff, nonlinear (rates span 1e-2 .. 3e7)
    k1, k2, k3 = a
    return jnp.array([-k1 * y[0] + k3 * y[1] * y[2],
                      k1 * y[0] - k3 * y[1] * y[2] - k2 * y[1] ** 2,
                      k2 * y[1] ** 2])


def vdp(t, y, a):                                # stiff Van der Pol, mu = a[0]
    return jnp.array([y[1], a[0] * ((1 - y[0] ** 2) * y[1] - y[0])])


def solve(solver, f, y0, a, t1, rtol):
    sol = diffrax.diffeqsolve(diffrax.ODETerm(f), solver, t0=0.0, t1=t1, dt0=1e-6, y0=y0, args=a,
                              stepsize_controller=diffrax.PIDController(rtol=rtol, atol=rtol * 1e-3),
                              saveat=diffrax.SaveAt(ts=jnp.linspace(0.1 * t1, t1, 5)), max_steps=100000,
                              adjoint=diffrax.ForwardMode())
    return sol.ys


CASES = [(robertson, jnp.array([1.0, 0.0, 0.0]), jnp.array([0.04, 3e7, 1e4]), 40.0),
         (vdp, jnp.array([2.0, 0.0]), jnp.array([100.0]), 50.0)]


@pytest.mark.parametrize("f, y0, a, t1", CASES)
@pytest.mark.parametrize("n_iter", [8, 10])
def test_as_accurate_as_kvaerno5(f, y0, a, t1, n_iter):
    """at the same tolerance, FixedKenCarp4's value and tangent errors (against Kvaerno5 at 1000x tighter) are
    within a few times Kvaerno5's own."""
    da = a * 1e-1
    run = lambda s, rtol: jax.jvp(lambda a: solve(s, f, y0, a, t1, rtol), (a,), (da,))
    y_ref, t_ref = run(diffrax.Kvaerno5(), 1e-10)
    err = lambda y, r: float(jnp.abs((y - r) / (jnp.abs(r).max(0) + 1e-12)).max())
    for k, (y, t) in ((n, run(s, 1e-7)) for n, s in (("kv", diffrax.Kvaerno5()), ("fix", FixedKenCarp4(n_iter=n_iter)))):
        if k == "kv":
            ey_kv, et_kv = err(y, y_ref), err(t, t_ref)
        else:
            ey, et = err(y, y_ref), err(t, t_ref)
    print(f"{f.__name__} N={n_iter}: value err {ey:.2e} (Kvaerno5 {ey_kv:.2e}), tangent err {et:.2e} (Kvaerno5 {et_kv:.2e})")
    assert ey < 5 * ey_kv + 1e-6 and et < 5 * et_kv + 1e-5
