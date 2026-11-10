"""batched_grad.py — staged forward-mode AD on the per-k batched pipeline.

Instead of jacfwd-ing the whole cross-device pipeline (which fuses into a ~20-min
monolith), apply jax.jvp to each already-filter_jit'd batched stage separately,
in eager Python, threading the array tangent between stages -- the same way
``Model.call_batched`` orchestrates the primal. Differentiating the batched
stages means the gradient inherits the params-axis batching, k-chunking and
sharding.

This lives in scan/ and only calls the existing batched-stage methods
(``_pre_recomb_batched``, HyRex ``_recmodel_cpu``, ``_get_BG_batched``,
``PE.full_evolution_batched``, ``SS.get_Cl_batched``), with no core-code edits.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from abcmb.main import _recmodel_cpu


# ---------- HyRex on a CPU mesh (the staged jvp's stage 2) ----------
_HYREX_MESH_CACHE = {}


def hyrex_cpu_mesh(model, B):
    """HyRex wrapper + placement functions for the staged jvp's stage 2.

    The HyRex primal is cheap (~0.9 s for 16 cosmologies) but its P-direction jvp is a
    serial CPU loop over every (cosmology, direction) lane (~22 s at B=16, P=7) and does
    not shrink when more GPUs share the batch. With several host devices
    (``XLA_FLAGS=--xla_force_host_platform_device_count=N`` before JAX starts) the B axis
    is sharded over ``n_use`` = the largest divisor of B <= N of them and each device
    runs its own lanes in parallel: 22 -> 4.3 s at B=16 on 16 devices. The wrapper must
    NOT be ``_recmodel_cpu`` (``filter_jit(..., backend='cpu')`` gathers sharded inputs
    onto device 0); a plain ``filter_jit`` runs on the mesh the inputs are placed on.
    Returns ``(hy, put_b, put_pb, n_use)``: ``put_b`` places a B-leading tree, ``put_pb``
    a (P, B, ...) direction-stacked tree. Single device (``_recmodel_cpu``) when n_use is 1."""
    cpus = jax.devices('cpu')
    n_cpu = len(cpus)
    n_use = max(d for d in range(1, n_cpu + 1) if B % d == 0)
    if n_use == 1:
        cpu = cpus[0]
        put = lambda t: jax.device_put(t, cpu)
        return _recmodel_cpu(model.RecModel, True), put, put, 1
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    key = (id(model.RecModel), n_use)
    hit = _HYREX_MESH_CACHE.get(key)
    if hit is not None:
        return hit[0], hit[1], hit[2], n_use
    # Built once per (RecModel, n_use): a fresh Mesh/NamedSharding per call is a fresh jit
    # cache key, i.e. a recompile of the sharded HyRex program on every gradient call.
    hy = eqx.filter_jit(jax.vmap(model.RecModel))
    mesh = Mesh(np.asarray(cpus[:n_use]), axis_names=('batch',))
    sh_b, sh_pb, repl = (NamedSharding(mesh, P('batch')), NamedSharding(mesh, P(None, 'batch')),
                         NamedSharding(mesh, P()))

    def put_b(tree):
        return jax.tree.map(lambda x: jax.device_put(x, sh_b if x.ndim >= 1 else repl)
                            if eqx.is_array(x) else x, tree)

    def put_pb(tree):
        return jax.tree.map(lambda x: jax.device_put(x, sh_pb if x.ndim >= 2 else repl)
                            if eqx.is_array(x) else x, tree)
    _HYREX_MESH_CACHE[key] = (hy, put_b, put_pb, model.RecModel, mesh)
    return hy, put_b, put_pb, n_use


# ---------- per-stage jvp helper (partition static, thread array tangents) ----------
@eqx.filter_jit
def _cls_tt_te_ee(SS, PT, BG, params):
    """TT/TE/EE of SS.get_Cl_batched. It also returns Clpp and lensed BB; slicing them off inside
    one jit lets XLA drop their primal and tangent work."""
    return SS.get_Cl_batched(PT, BG, params)[:3]


def jvp_stage(fn, primals_full, tangents_diff, n_dirs=None):
    """fn(*primals_full) with forward-mode. primals_full: tuple of full pytrees
    (eqx.Modules / dicts). tangents_diff: tuple of DIFF pytrees (arrays where the
    primal is an inexact array, None at static), one per primal. Returns
    (primal_out_full, out_tangent_diff) where out_tangent_diff is filtered to the
    inexact-array leaves (ready to thread into the next stage).

    n_dirs : int or None. If given, every tangent leaf carries a leading axis of
        length n_dirs (a stack of directions) and ALL directions go through ONE
        jvp, vmapped over that axis (what upstream's jacfwd does): the primal is
        evaluated once and the n_dirs tangent lanes ride along, instead of one
        full primal re-solve per direction. The primal output is unbatched
        (out_axes=None); the tangent leaves come back as (n_dirs, ...)."""
    diffs, statics = [], []
    for p in primals_full:
        d, s = eqx.partition(p, eqx.is_inexact_array)
        diffs.append(d); statics.append(s)

    def f_arr(*ds):
        return fn(*[eqx.combine(d, s) for d, s in zip(ds, statics)])

    if n_dirs is None:
        out, out_dot = jax.jvp(f_arr, tuple(diffs), tuple(tangents_diff))
    else:
        out, out_dot = jax.vmap(lambda *ts: jax.jvp(f_arr, tuple(diffs), ts),
                                in_axes=0, out_axes=(None, 0))(*tangents_diff)
    return out, eqx.filter(out_dot, eqx.is_inexact_array)


def _to_float(tree):
    """cast int/bool leaves to float64 (as run_cosmology_abbr / _build_bgs_batched
    do) so primal and tangent pytrees have matching inexact-array structure."""
    def f(v):
        arr = jnp.asarray(v)
        return arr.astype(jnp.float64) if arr.dtype.kind in 'iub' else arr
    return jax.tree_util.tree_map(f, tree)


def _dir_shardfn_from(sharded_leaf):
    """shardfn for DIRECTION-STACKED tangents (axis 0 = directions, axis 1 = B):
    the same mesh / 'batch' axis as ``Model._make_shardfn`` (read off a leaf it
    sharded), applied to axis 1. Leaves with ndim < 2 (per-direction scalars) are
    replicated."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    mesh = sharded_leaf.sharding.mesh
    dir_sh = NamedSharding(mesh, P(None, 'batch'))
    repl_sh = NamedSharding(mesh, P())

    def f(tree):
        def per_leaf(x):
            if eqx.is_array(x):
                return jax.device_put(x, dir_sh if x.ndim >= 2 else repl_sh)
            return x
        return jax.tree.map(per_leaf, tree)
    return f


# ---------- the staged forward-mode push: params_batch -> (Cl, dCl) ----------
def _stage_log(log, name, tree, t0):
    """stage_log hook: block on the stage output, report its wall time and the GPUs' memory."""
    import time
    jax.block_until_ready(tree)
    st = [d.memory_stats() or {} for d in jax.devices()]
    log(f"[stage] {name}: {time.time() - t0:.1f}s, in use {max(m.get('bytes_in_use', 0) for m in st) / 2**30:.1f} GB, "
        f"peak {max(m.get('peak_bytes_in_use', 0) for m in st) / 2**30:.1f} GB")
    return time.time()


def staged_cl_and_grad(model, full_ps, params_dots, k_chunk_size=100, shard=None,
                       spec_b_local=None, stage_log=None):
    """full_ps: list[B] of derived param dicts (primal).
    params_dots: list[P] of derived-param-tangent dicts, each stacked over B
        (i.e., params_dots[j] is the dB-stacked tangent of params_batch in
        direction j). Returns (ClTT,ClTE,ClEE) primal (B,n_l) and a list[P] of
        (dClTT,dClTE,dClEE) tangents (B,n_l).

    The P directions are stacked on a leading axis and pushed through each
    stage's jvp vmapped over that axis (``jvp_stage(..., n_dirs=)``): one primal
    evaluation per stage carries all the tangent lanes, and the primal Cls are
    read off that same jvp, so the pipeline's primal is solved ONCE per call.

    spec_b_local : int or None. Cosmologies PER DEVICE per spectrum-stage jvp
        call. The spectrum jvp is the gradient's memory peak (its lensing and
        line-of-sight intermediates are (P, B, ...) tensors of ~1.2 GiB per
        cosmology at P=7, far above the ~0.2 GB/cosmology the tables need), and
        it is embarrassingly parallel over B, so stage 5 is run on sub-batches of
        spec_b_local x n_dev cosmologies and the Cls / tangents concatenated: no
        extra work, no change to the spectrum code, peak ~ spec_b_local/B_local
        of the full-batch stage. None = the whole batch in one call.

    k_chunk_size threads through full_evolution_batched on both the primal and the
    jvp. Under jvp the augmented ForwardMode-Kvaerno5 loop is fused over
    vmap(k_chunk, B), so a smaller k_chunk shrinks that fusion and the compile.

    shard : bool or None. If True (or None with >1 visible GPU), shard the B
        axis across all visible GPUs via the SAME GSPMD shardfn call_batched
        uses (``Model._make_shardfn``). B (and the B axis of every
        params_dots tangent) is padded to a multiple of the device count by
        REPLICATING the last cosmology; the padding is sliced off ALL outputs
        (primal Cls + every per-direction Cl tangent). The shardfn is applied
        BEFORE the stages -- to the stacked primal params, to the
        direction-stacked tangents (axis 1 = B, via ``_dir_shardfn_from``), and
        to recomb/recomb_dot after the CPU->GPU device_put (mirroring
        _build_bgs_batched). Sharding AFTER the setup would build everything on
        device 0 and OOM at large B.

    stage_log : callable(str) or None. Diagnostics: after each stage, block and log its
        wall time and the GPUs' current / peak memory (serializes the stages)."""
    PE = model.PE; SS = model.SS
    try:
        gpu = jax.devices('gpu')[0]
    except Exception:
        gpu = None

    shardfn, n_dev = model._make_shardfn(shard)
    do_shard = shardfn is not None

    # --- pad B to a multiple of n_dev (replicate last cosmology) so the B axis
    # divides evenly across the mesh; the padding is sliced off below. Both the
    # primal list and EACH stacked tangent dict are padded on axis 0. ---
    B = len(full_ps)
    Pn = len(params_dots)
    full_ps_run = list(full_ps)
    params_dots_run = list(params_dots)
    if do_shard and B % n_dev != 0:
        pad = n_dev - (B % n_dev)
        full_ps_run = full_ps_run + [full_ps_run[-1]] * pad
        params_dots_run = [
            jax.tree.map(lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)],
                                                   axis=0), pdot)
            for pdot in params_dots_run]

    # stack primal params over B, cast int/bool->float (match tangent structure)
    params_batch = _to_float(jax.tree.map(lambda *xs: jnp.stack(xs), *full_ps_run))
    if do_shard:
        params_batch = shardfn(params_batch)
        dir_shardfn = _dir_shardfn_from(params_batch['h'])
    else:
        dir_shardfn = None

    # stack the directions on a leading axis: leaves (P, B, ...)
    pdot = jax.tree.map(lambda *xs: jnp.stack(xs), *params_dots_run)
    if do_shard:
        pdot = dir_shardfn(pdot)

    import time
    t_st = time.time()
    _lg = (lambda name, tree, t: _stage_log(stage_log, name, tree, t)) if stage_log else (lambda name, tree, t: t)
    # ---- stage 1: pre-recomb (GPU) ----
    pre_BG, pre_BG_dot = jvp_stage(model._pre_recomb_batched, (params_batch,), (pdot,),
                                   n_dirs=Pn)
    t_st = _lg("pre-recomb", pre_BG_dot, t_st)
    # ---- stage 2: HyRex (CPU mesh): inputs (recomb_inputs, params), tangents
    # (pre_BG_dot.recomb_inputs, pdot). Primal + tangents to the host devices (B axis
    # sharded over them when there are several, see hyrex_cpu_mesh); result to GPU.
    hy, put_b, put_pb, _ = hyrex_cpu_mesh(model, len(full_ps_run))
    ri_cpu = put_b(pre_BG.recomb_inputs)
    p_cpu = put_b(params_batch)
    ri_dot_cpu = put_pb(pre_BG_dot.recomb_inputs)
    p_dot_cpu = put_pb(pdot)
    recomb, recomb_dot = jvp_stage(lambda ri, p: hy((ri, p)),
                                   (ri_cpu, p_cpu), (ri_dot_cpu, p_dot_cpu), n_dirs=Pn)
    if do_shard:
        recomb = shardfn(recomb)
        recomb_dot = dir_shardfn(recomb_dot)
    elif gpu is not None:
        recomb = jax.device_put(recomb, gpu)
        recomb_dot = jax.device_put(recomb_dot, gpu)
    t_st = _lg("hyrex", recomb_dot, t_st)
    # ---- stage 3: get_BG (GPU) ----
    BG, BG_dot = jvp_stage(model._get_BG_batched,
                           (params_batch, pre_BG, recomb), (pdot, pre_BG_dot, recomb_dot),
                           n_dirs=Pn)
    t_st = _lg("get_BG", BG_dot, t_st)
    # ---- stage 4: perturbations (GPU, k-chunked) ----
    PT, PT_dot = jvp_stage(
        lambda bg, p: PE.full_evolution_batched((bg, p), k_chunk_size=k_chunk_size),
        (BG, params_batch), (BG_dot, pdot), n_dirs=Pn)
    t_st = _lg("perturbations", PT_dot, t_st)
    # ---- stage 5: spectrum (GPU), in sub-batches of spec_b_local x n_dev ----
    get_cls3 = lambda PT_, BG_, p_: _cls_tt_te_ee(SS, PT_, BG_, p_)
    Brun = len(full_ps_run)
    spec_chunk = None if spec_b_local is None else spec_b_local * (n_dev if do_shard else 1)
    if spec_chunk is None or spec_chunk >= Brun:
        cls, cl_dot = jvp_stage(get_cls3, (PT, BG, params_batch),
                                (PT_dot, BG_dot, pdot), n_dirs=Pn)
    else:
        def _sl(tree, s, e, axis):
            return jax.tree.map(
                lambda x: jax.lax.slice_in_dim(x, s, e, axis=axis)
                if eqx.is_array(x) and x.ndim > axis else x, tree)
        cls_parts, dot_parts = [], []
        for s0 in range(0, Brun, spec_chunk):
            e0 = min(s0 + spec_chunk, Brun)
            prim = _sl((PT, BG, params_batch), s0, e0, 0)       # leaves (b, ...)
            tang = _sl((PT_dot, BG_dot, pdot), s0, e0, 1)       # leaves (P, b, ...)
            if do_shard:                                        # keep the B sharding
                prim = shardfn(prim); tang = dir_shardfn(tang)
            c, cd = jvp_stage(get_cls3, prim, tang, n_dirs=Pn)
            cls_parts.append(c); dot_parts.append(cd)
            del prim, tang
        cls = tuple(jnp.concatenate([c[i] for c in cls_parts], axis=0) for i in range(3))
        cl_dot = tuple(jnp.concatenate([d[i] for d in dot_parts], axis=1) for i in range(3))
        del cls_parts, dot_parts
    t_st = _lg("spectrum", cl_dot, t_st)
    del pre_BG_dot, recomb_dot, BG_dot, PT, PT_dot

    # cl_dot is the filtered tangent of (ClTT,ClTE,ClEE): a 3-tuple of (P, B, n_l)
    ClTT, ClTE, ClEE = cls
    grads = [tuple(g[j] for g in cl_dot) for j in range(Pn)]   # list[P] of (dTT, dTE, dEE)

    # --- slice off the B-padding (if any) on the primal Cls + every tangent ---
    if len(full_ps_run) != B:
        ClTT = ClTT[:B]; ClTE = ClTE[:B]; ClEE = ClEE[:B]
        grads = [tuple(g[:B] for g in cl_dot) for cl_dot in grads]
    return (ClTT, ClTE, ClEE), grads
