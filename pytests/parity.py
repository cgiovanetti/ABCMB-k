"""
Batch-vs-loop parity helper for ABCMB refactor.

Used in Phase D+ to assert that batched (vmap'd over params) execution
matches a Python loop over the same params. Lives separately from
snapshots.py because snapshot tests check single-call reproducibility while
parity tests check batched vs serial equivalence.
"""

import numpy as np
import jax


def stack_pytrees(pytrees):
    """jax.tree.map(jnp.stack, *pytrees) for batched-param construction."""
    import jax.numpy as jnp
    return jax.tree.map(lambda *xs: jnp.stack(xs), *pytrees)


def assert_batch_matches_loop(fn, params_list, rtol=1e-9, atol=0.0,
                               extract=None):
    """Run `fn` once on a B-stacked params batch and once in a Python loop
    over the same B param dicts, then assert elementwise closeness.

    Parameters
    ----------
    fn : callable
        Takes one params (or batched-params) input. Must produce a pytree
        whose array leaves can be sliced along a leading B axis. If the
        signature differs, wrap the call into `extract`.
    params_list : list[dict]
        B param dicts.
    rtol, atol : float
        Tolerances for np.testing.assert_allclose. Default rtol=1e-9 allows
        a bit of XLA-fusion slack vs the strict 1e-12 snapshot tolerance.
    extract : callable or None
        If provided, applied to fn(...) output before comparison. Useful when
        fn returns a complex pytree and we want only specific fields. The
        return value of extract must support .__getitem__(i) along the B axis.
    """
    import jax.numpy as jnp

    B = len(params_list)
    # batched
    params_batch = stack_pytrees([jax.tree.map(lambda v: jnp.asarray(v), p)
                                   for p in params_list])
    batched = fn(params_batch)
    if extract is not None:
        batched = extract(batched)

    # loop
    looped = []
    for p in params_list:
        out = fn(p)
        if extract is not None:
            out = extract(out)
        looped.append(out)
    # stack the loop outputs
    stacked = stack_pytrees(looped)

    # compare leaf-by-leaf
    leaves_batched = jax.tree_util.tree_leaves(batched)
    leaves_loop = jax.tree_util.tree_leaves(stacked)
    if len(leaves_batched) != len(leaves_loop):
        raise AssertionError(
            f"Pytree structure mismatch: batched has {len(leaves_batched)} "
            f"leaves, looped has {len(leaves_loop)}")

    failures = []
    for i, (lb, ll) in enumerate(zip(leaves_batched, leaves_loop)):
        lb_arr = np.asarray(lb)
        ll_arr = np.asarray(ll)
        if lb_arr.shape != ll_arr.shape:
            failures.append(
                f"  leaf {i}: shape mismatch batched {lb_arr.shape} "
                f"vs loop {ll_arr.shape}")
            continue
        try:
            np.testing.assert_allclose(lb_arr, ll_arr, rtol=rtol, atol=atol)
        except AssertionError as e:
            diff = np.abs(lb_arr - ll_arr)
            ref = np.maximum(np.abs(ll_arr), 1e-300)
            rel = (diff / ref).max()
            failures.append(
                f"  leaf {i}: max_rel={float(rel):.2e} "
                f"max_abs={float(diff.max()):.2e}\n"
                f"    {str(e).splitlines()[0]}")
    if failures:
        raise AssertionError(
            f"Batch-vs-loop parity failed for B={B}:\n" + "\n".join(failures))
