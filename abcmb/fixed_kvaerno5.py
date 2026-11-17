"""FixedKvaerno5: diffrax's Kvaerno5 with a fixed number of chord iterations per implicit stage and, optionally, a
structured inverse of the iteration matrix.

Same tableau, stage predictor, Jacobian timing (once per step, at the second stage), error estimate and dense output
as diffrax.Kvaerno5 (diffrax/_solver/kvaerno5.py, imported from there), so it takes the same steps. As in
fixed_kencarp4.py each implicit stage runs exactly n_iter chord iterations
    f <- f - M^-1 (f - vf(t_i, ypart_i + gamma dt f)),   M = I - gamma dt J,
unrolled, with M^-1 formed once per step and carrying no gradient. The last update of each stage is added to the
error estimate (conv_weight), so a step whose iterations have not converged is rejected by the step-size controller
instead of accepted.

Structured inverse (structure=jacobian_structure(pattern)): the Einstein-Boltzmann Jacobian is ~2% dense, multipole
chains (tridiagonal) tied together by a few "hub" variables (metric, the lowest moments). With the hubs and a
separator every `block` variables along the chains ordered first, M = [[H, U], [V, T]] with T block diagonal, and
    M^-1 = [[S^-1, -S^-1 U T^-1], [-T^-1 V S^-1, T^-1 + T^-1 V S^-1 U T^-1]],   S = H - U T^-1 V,
which needs one inverse of the hub block and one of each small block in place of an n x n inverse. The pattern only
has to contain the true one: a structure that misses a coupling gives an approximate M^-1, slower iterations and
(through conv_weight) rejected steps, not a wrong solution.
"""
from typing import ClassVar, Optional

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from diffrax._local_interpolation import ThirdOrderHermitePolynomialInterpolation
from diffrax._solver.kvaerno5 import _kvaerno5_tableau

jax.config.update("jax_enable_x64", True)


def jacobian_structure(pattern, block=8):
    """(hubs, blocks) for structured_inverse from a boolean (n, n) Jacobian pattern, as nested tuples (static,
    hashable). Hubs: every variable with an entry off the tridiagonal in its row or column, plus a separator after
    each `block` consecutive other variables; blocks: the runs in between, padded with n."""
    nz = np.asarray(pattern, dtype=bool)
    n = len(nz)
    off = nz & ~np.eye(n, dtype=bool)
    far = off & (np.abs(np.subtract.outer(np.arange(n), np.arange(n))) != 1)
    hub = set(np.nonzero(far.any(0) | far.any(1))[0].tolist())
    blocks, cur = [], []
    for i in range(n):
        if i in hub or len(cur) == block:
            hub.add(i)
            if cur:
                blocks.append(cur)
            cur = []
        else:
            cur.append(i)
    if cur:
        blocks.append(cur)
    which = -np.ones(n, dtype=int)
    for b, v in enumerate(blocks):
        which[v] = b
    i, j = np.nonzero(off)
    assert not ((which[i] >= 0) & (which[j] >= 0) & (which[i] != which[j])).any(), "blocks couple to each other"
    return tuple(sorted(hub)), tuple(tuple(v) + (n,) * (block - len(v)) for v in blocks)


def structured_inverse(M, structure):
    """M^-1 for M with the sparsity of `structure` (see the module docstring)."""
    hub, blk = (np.asarray(s, dtype=int) for s in structure)
    n = M.shape[0]
    if blk.size == 0:
        return jnp.linalg.inv(M)
    nb, b = blk.shape
    h = len(hub)
    pad = blk == n
    order = np.concatenate([hub, blk.reshape(-1)])
    pos = np.zeros(n, dtype=int)
    pos[order[order < n]] = np.nonzero(order < n)[0]
    Mx = jnp.zeros((n + 1, n + 1), M.dtype).at[:n, :n].set(M)
    T = jnp.where(pad[:, :, None] | pad[:, None, :], jnp.eye(b, dtype=M.dtype), Mx[blk[:, :, None], blk[:, None, :]])
    Ti = jnp.linalg.inv(T)                                            # (nb, b, b)
    U = Mx[hub[:, None, None], blk[None]]                             # (h, nb, b)
    V = Mx[blk[:, :, None], hub[None, None, :]]                       # (nb, b, h)
    W = jnp.einsum("nij,njg->nig", Ti, V).reshape(nb * b, h)          # T^-1 V
    Z = jnp.einsum("hni,nij->hnj", U, Ti).reshape(h, nb * b)          # U T^-1
    A = jnp.linalg.inv(M[hub[:, None], hub[None, :]] - U.reshape(h, nb * b) @ W)
    AZ = A @ Z
    D = (W @ AZ).reshape(nb, b, nb, b).at[np.arange(nb), :, np.arange(nb), :].add(Ti).reshape(nb * b, nb * b)
    return jnp.block([[A, -AZ], [-W @ A, D]])[pos[:, None], pos[None, :]]


class FixedKvaerno5(diffrax.AbstractSolver):
    """Kvaerno5 with a static number n_iter of chord iterations per implicit stage. y: 1-D array."""
    n_iter: int = eqx.field(static=True, default=6)
    structure: Optional[tuple] = eqx.field(static=True, default=None)
    conv_weight: float = eqx.field(static=True, default=100.0)

    term_structure: ClassVar = diffrax.AbstractTerm
    interpolation_cls: ClassVar = ThirdOrderHermitePolynomialInterpolation.from_k

    def order(self, terms):
        return 5

    def error_order(self, terms):
        return 5

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        tab = _kvaerno5_tableau
        dt = t1 - t0
        gdt = tab.a_diagonal[1] * dt
        ts = [t0] + [t0 + c * dt for c in tab.c]
        f = [terms.vf(t0, y0, args)]
        minv = None
        conv = 0.0
        for i in range(1, len(tab.c) + 1):
            ypart = y0 + dt * sum(a * fj for a, fj in zip(tab.a_lower[i - 1], f))
            fi = sum(p * fj for p, fj in zip(tab.a_predictor[i - 1], f))
            if minv is None:
                y_jac, a_jac = jax.lax.stop_gradient((ypart + gdt * fi, args))
                M = jnp.eye(y0.shape[0]) - gdt * jax.jacfwd(lambda y: terms.vf(ts[i], y, a_jac))(y_jac)
                minv = jax.lax.stop_gradient(jnp.linalg.inv(M) if self.structure is None
                                             else structured_inverse(M, self.structure))
            for _ in range(self.n_iter):
                dfi = minv @ (fi - terms.vf(ts[i], ypart + gdt * fi, args))
                fi = fi - dfi
            conv = conv + jnp.abs(gdt * dfi)
            f.append(fi)
        y1 = ypart + gdt * fi
        y_error = jnp.abs(dt * sum(b * fj for b, fj in zip(tab.b_error, f))) + self.conv_weight * conv
        dense_info = dict(y0=y0, y1=y1, k=jnp.stack([dt * fj for fj in f]))
        return y1, y_error, dense_info, None, diffrax.RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
