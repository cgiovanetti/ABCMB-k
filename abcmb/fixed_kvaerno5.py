"""FixedKvaerno5: diffrax's Kvaerno5 with a fixed number of chord iterations per implicit stage and a structured
inverse of the iteration matrix.

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
which needs one inverse of the hub block and one of each small block in place of an n x n inverse. It is exact when
the pattern contains every nonzero of J. The solver state carries the largest entry of M that the structure drops
(couples two different blocks); check_structure turns one above GUARD_TOL into an error, so a
pattern that misses a coupling stops the solve instead of slowing or biasing it. PerturbationEvolver finds the pattern
from the cosmologies it solves (perturbations.py, with_structure_for).
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

# A dropped entry of M (gamma dt J_ij, against M's identity diagonal) below GUARD_TOL changes M^-1 by about as little,
# far below the chord iterations' own convergence test; above it the solve stops (check_structure).
GUARD_TOL = 1e-10
GUARD_MSG = ("kvaerno5_fixed: the Jacobian couples variables that the structured inverse treats as independent (its "
             "sparsity probes missed a coupling). Use PE_solver='kvaerno5'.")


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


def dropped_mask(structure, n):
    """Boolean (n, n): the entries of M that structured_inverse never reads (row and column in different blocks)."""
    which = np.full(n + 1, -1)
    for b, v in enumerate(structure[1]):
        which[list(v)] = b                                            # the padding index n is sliced off below
    w = which[:n]
    return (w[:, None] >= 0) & (w[None, :] >= 0) & (w[:, None] != w[None, :])


def check_structure(x, violation):
    """x, or an error if the solve met a coupling its structure drops (violation: the FixedKvaerno5 solver state)."""
    return eqx.error_if(x, violation > GUARD_TOL, GUARD_MSG)


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


def structured_factors(M, structure):
    """(R, L, Ti) with M^-1 r = L (R r) + blocks(Ti, r): R = S^-1 [I_H, -U T^-1] (h, n), L = [I_H; -T^-1 V] (n, h), Ti the
    block inverses (nb, b, b). The same M^-1 as structured_inverse without forming it: an application costs two thin
    products and the small blocks (about 2 h n + nb b^2 multiplications instead of n^2)."""
    hub, blk = (np.asarray(s, dtype=int) for s in structure)
    n = M.shape[0]
    nb, b = blk.shape
    h = len(hub)
    pad = blk == n
    Mx = jnp.zeros((n + 1, n + 1), M.dtype).at[:n, :n].set(M)
    T = jnp.where(pad[:, :, None] | pad[:, None, :], jnp.eye(b, dtype=M.dtype), Mx[blk[:, :, None], blk[:, None, :]])
    Ti = jnp.linalg.inv(T)
    U = Mx[hub[:, None, None], blk[None]]                             # (h, nb, b)
    V = Mx[blk[:, :, None], hub[None, None, :]]                       # (nb, b, h)
    W = jnp.einsum("nij,njg->nig", Ti, V)                             # T^-1 V  (nb, b, h)
    Z = jnp.einsum("hni,nij->hnj", U, Ti)                             # U T^-1  (h, nb, b)
    A = jnp.linalg.inv(M[hub[:, None], hub[None, :]] - jnp.einsum("hnj,njg->hg", U, W))
    R = jnp.zeros((h, n + 1), M.dtype).at[:, hub].set(A).at[:, blk].set(-jnp.einsum("hg,gnj->hnj", A, Z))[:, :n]
    L = jnp.zeros((n + 1, h), M.dtype).at[hub, np.arange(h)].set(1.0).at[blk].set(-W)[:n]
    return R, L, Ti


def apply_factors(factors, structure, r):
    """M^-1 r from structured_factors."""
    R, L, Ti = factors
    blk = np.asarray(structure[1], dtype=int)
    n = r.shape[0]
    nb, b = blk.shape
    where = np.full(n, nb * b, dtype=int)                             # position of each variable in the blocks
    flat = blk.reshape(-1)
    where[flat[flat < n]] = np.nonzero(flat < n)[0]
    t = jnp.einsum("nij,nj->ni", Ti, jnp.concatenate([r, jnp.zeros(1, r.dtype)])[blk]).reshape(-1)
    return L @ (R @ r) + jnp.concatenate([t, jnp.zeros(1, r.dtype)])[where]


class FixedKvaerno5(diffrax.AbstractSolver):
    """Kvaerno5 with a static number n_iter of chord iterations per implicit stage. y: 1-D array. The solver state is
    the largest dropped entry of M so far, for check_structure."""
    structure: tuple = eqx.field(static=True)                   # jacobian_structure(pattern)
    n_iter: int = eqx.field(static=True, default=6)
    conv_weight: float = eqx.field(static=True, default=100.0)
    factored: bool = eqx.field(static=True, default=False)      # apply M^-1 as structured_factors
    unroll: Optional[int] = eqx.field(static=True, default=None)  # None: iterations unrolled; k: fori_loop(unroll=k)
    jit_vf: bool = eqx.field(static=True, default=True)         # trace the vector field once (nested jit) per step

    term_structure: ClassVar = diffrax.AbstractTerm
    interpolation_cls: ClassVar = ThirdOrderHermitePolynomialInterpolation.from_k

    def order(self, terms):
        return 5

    def error_order(self, terms):
        return 5

    def init(self, terms, t0, t1, y0, args):
        return jnp.zeros((), jnp.result_type(y0))

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        tab = _kvaerno5_tableau
        dt = t1 - t0
        vf = eqx.filter_jit(terms.vf) if self.jit_vf else terms.vf
        gdt = tab.a_diagonal[1] * dt
        ts = [t0] + [t0 + c * dt for c in tab.c]
        f = [vf(t0, y0, args)]
        minv = None
        conv = 0.0
        for i in range(1, len(tab.c) + 1):
            ypart = y0 + dt * sum(a * fj for a, fj in zip(tab.a_lower[i - 1], f))
            fi = sum(p * fj for p, fj in zip(tab.a_predictor[i - 1], f))
            if minv is None:
                y_jac, a_jac = jax.lax.stop_gradient((ypart + gdt * fi, args))
                J = jax.jacfwd(lambda y: vf(ts[i], y, a_jac))(y_jac)
                M = jnp.eye(y0.shape[0]) - gdt * J
                idx = np.flatnonzero(dropped_mask(self.structure, y0.shape[0]))   # gathered: ~3x cheaper than a mask
                if idx.size:
                    dropped = jnp.abs(gdt) * jnp.max(jnp.abs(J.reshape(-1)[idx]))
                    solver_state = jax.lax.stop_gradient(jnp.maximum(solver_state, dropped))
                if self.factored and len(self.structure[1]):
                    fac = jax.lax.stop_gradient(structured_factors(M, self.structure))
                    minv = lambda r: apply_factors(fac, self.structure, r)
                else:
                    mat = jax.lax.stop_gradient(structured_inverse(M, self.structure))
                    minv = lambda r: mat @ r

            def chord(_, c, ti=ts[i], ypart=ypart):
                d = minv(c[0] - vf(ti, ypart + gdt * c[0], args))
                return c[0] - d, d

            c = (fi, jnp.zeros_like(fi))
            if self.unroll is None or self.unroll >= self.n_iter:
                for _ in range(self.n_iter):
                    c = chord(0, c)
            else:
                c = jax.lax.fori_loop(0, self.n_iter, chord, c, unroll=self.unroll)
            fi, dfi = c
            conv = conv + jnp.abs(gdt * dfi)
            f.append(fi)
        y1 = ypart + gdt * fi
        y_error = jnp.abs(dt * sum(b * fj for b, fj in zip(tab.b_error, f))) + self.conv_weight * conv
        dense_info = dict(y0=y0, y1=y1, k=jnp.stack([dt * fj for fj in f]))
        return y1, y_error, dense_info, solver_state, diffrax.RESULTS.successful

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)
