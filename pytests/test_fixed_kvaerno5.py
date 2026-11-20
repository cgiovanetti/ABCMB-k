"""FixedKvaerno5 (abcmb/fixed_kvaerno5.py): the structured inverse against a dense one, the guard on couplings the
structure drops, and the solver against diffrax's Kvaerno5 on stiff toy problems (values and forward-mode tangents,
ForwardMode adjoint as ABCMB uses). CPU."""
import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from abcmb.fixed_kvaerno5 import (FixedKvaerno5, check_structure, dropped_mask, jacobian_structure,
                                  structured_inverse)
from test_fixed_kencarp4 import CASES, solve

jax.config.update("jax_enable_x64", True)


def full(n):
    """the structure of a dense pattern: every variable a hub (the toys are 2-3 variables)."""
    return jacobian_structure(np.ones((n, n), dtype=bool))


def chain_matrix(n=60, hubs=(0, 1, 2, 30), seed=0):
    """tridiagonal chains tied together by a few dense hub rows / columns, like the Einstein-Boltzmann Jacobian."""
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n))
    i = np.arange(n - 1)
    J[i, i + 1] = rng.normal(size=n - 1) * 30
    J[i + 1, i] = -rng.normal(size=n - 1) * 30
    J[np.arange(n), np.arange(n)] = -rng.uniform(0, 50, n)
    for h in hubs:
        cols = rng.choice(n, 12, replace=False)
        J[h, cols] = rng.normal(size=12)
        J[rng.choice(n, 12, replace=False), h] = rng.normal(size=12)
    return J


@pytest.mark.parametrize("block", [3, 8, 100])
def test_structured_inverse(block):
    J = chain_matrix()
    M = jnp.eye(len(J)) - 0.3 * jnp.asarray(J)
    s = jacobian_structure(J != 0, block)
    hub = set(s[0])
    assert all(v in hub for v in (0, 1, 2, 30)) and max(len(b) for b in s[1]) <= block
    err = jnp.abs(structured_inverse(M, s) @ M - jnp.eye(len(J))).max()
    assert err < 1e-10, err


def test_structure_from_denser_pattern():
    """a pattern that contains the true one (here: everything a hub) still gives the inverse."""
    J = chain_matrix()
    M = jnp.eye(len(J)) - 0.3 * jnp.asarray(J)
    s = jacobian_structure(np.ones_like(J, dtype=bool), 8)
    assert len(s[1]) == 0
    assert jnp.abs(structured_inverse(M, s) @ M - jnp.eye(len(J))).max() < 1e-10


@pytest.mark.parametrize("f, y0, a, t1", CASES)
@pytest.mark.parametrize("n_iter", [4, 6])
def test_as_accurate_as_kvaerno5(f, y0, a, t1, n_iter):
    """at the same tolerance, the value and tangent errors (against Kvaerno5 at 1000x tighter) are within a few
    times Kvaerno5's own, also on nonlinear problems where few iterations rely on the convergence term."""
    da = a * 1e-1
    run = lambda s, rtol: jax.jvp(lambda a: solve(s, f, y0, a, t1, rtol), (a,), (da,))
    y_ref, t_ref = run(diffrax.Kvaerno5(), 1e-10)
    err = lambda y, r: float(jnp.abs((y - r) / (jnp.abs(r).max(0) + 1e-12)).max())
    y, t = run(diffrax.Kvaerno5(), 1e-7)
    ey_kv, et_kv = err(y, y_ref), err(t, t_ref)
    y, t = run(FixedKvaerno5(structure=full(len(y0)), n_iter=n_iter), 1e-7)
    ey, et = err(y, y_ref), err(t, t_ref)
    print(f"{f.__name__} N={n_iter}: value err {ey:.2e} (Kvaerno5 {ey_kv:.2e}), tangent err {et:.2e} (Kvaerno5 {et_kv:.2e})")
    assert ey < 5 * ey_kv + 1e-6 and et < 5 * et_kv + 1e-5


@pytest.mark.parametrize("block", [3, 8, 16])
def test_factored_application(block):
    """structured_factors + apply_factors give M^-1 r without forming M^-1."""
    from abcmb.fixed_kvaerno5 import apply_factors, structured_factors
    J = chain_matrix()
    M = jnp.eye(len(J)) - 0.3 * jnp.asarray(J)
    s = jacobian_structure(J != 0, block)
    r = jnp.asarray(np.random.default_rng(1).normal(size=len(J)))
    x = apply_factors(structured_factors(M, s), s, r)
    assert jnp.abs(M @ x - r).max() < 1e-10


@pytest.mark.parametrize("kw", [dict(unroll=2), dict(unroll=1)])
def test_rolled_loop_same_solution(kw):
    """the fori_loop form of the iterations gives the unrolled result."""
    f, y0, a, t1 = CASES[0]
    ya = solve(FixedKvaerno5(structure=full(len(y0)), n_iter=6), f, y0, a, t1, 1e-7)
    yb = solve(FixedKvaerno5(structure=full(len(y0)), n_iter=6, **kw), f, y0, a, t1, 1e-7)
    assert jnp.abs(ya - yb).max() < 1e-12


@pytest.mark.parametrize("block", [3, 8])
def test_dropped_mask_is_what_the_inverse_ignores(block):
    """structured_inverse(M) is the exact inverse of M with the dropped_mask entries set to zero."""
    J = chain_matrix()
    s = jacobian_structure(J != 0, block)
    M = jnp.asarray(np.eye(len(J)) - 0.3 * J + 0.01 * np.random.default_rng(2).normal(size=J.shape))
    kept = jnp.where(dropped_mask(s, len(J)), 0.0, M)
    assert jnp.abs(structured_inverse(M, s) @ kept - jnp.eye(len(J))).max() < 1e-10
    assert not dropped_mask(s, len(J))[J != 0].any()        # nothing of the true pattern is dropped


def chain_ode(far):
    """y' = J y: a stiff tridiagonal chain of 40 plus the long-range coupling J[3, 30] = far."""
    n = 40
    J = -np.diag(np.linspace(1.0, 1e4, n)) + np.diag(np.full(n - 1, 5.0), 1) + np.diag(np.full(n - 1, -5.0), -1)
    pattern = J != 0                                         # the chain only: (3, 30) not in it
    J[3, 30] = far
    return jnp.asarray(J), pattern


def solve_checked(J, structure):
    sol = diffrax.diffeqsolve(diffrax.ODETerm(lambda t, y, J: J @ y), FixedKvaerno5(structure=structure), t0=0.0,
                              t1=1.0, dt0=1e-4, y0=jnp.ones(len(J)), args=J,
                              stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-10),
                              saveat=diffrax.SaveAt(ts=jnp.linspace(0.1, 1.0, 4), solver_state=True),
                              max_steps=100000, adjoint=diffrax.ForwardMode())
    return check_structure(sol.ys, sol.solver_state)


def test_guard_stops_a_missed_coupling():
    """a structure built without a coupling the problem has: the solve raises instead of returning."""
    J, pattern = chain_ode(far=50.0)
    s = jacobian_structure(pattern, 8)
    assert dropped_mask(s, len(J))[3, 30]
    with pytest.raises(Exception, match="kvaerno5_fixed"):
        jax.block_until_ready(solve_checked(J, s))


def test_guard_quiet_when_the_structure_is_complete():
    """the same problem with the coupling in the pattern (or the coupling absent): no error, Kvaerno5's solution."""
    for far in (50.0, 0.0):
        J, pattern = chain_ode(far)
        s = jacobian_structure(pattern | (J != 0), 8)
        y = solve_checked(J, s)
        ref = diffrax.diffeqsolve(diffrax.ODETerm(lambda t, y, J: J @ y), diffrax.Kvaerno5(), t0=0.0, t1=1.0,
                                  dt0=1e-4, y0=jnp.ones(len(J)), args=J,
                                  stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-13),
                                  saveat=diffrax.SaveAt(ts=jnp.linspace(0.1, 1.0, 4)), max_steps=100000).ys
        assert jnp.abs(y - ref).max() < 1e-5 * jnp.abs(ref).max()
