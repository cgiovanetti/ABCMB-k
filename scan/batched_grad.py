"""batched_grad.py — staged forward-mode AD on the per-k batched pipeline.

Instead of jacfwd-ing the whole cross-device pipeline (which fuses into a ~20-min
monolith), apply jax.jvp to each already-filter_jit'd batched stage separately,
in eager Python, threading the array tangent between stages -- the same way
``Model.call_batched`` orchestrates the primal. Differentiating the batched
stages means the gradient inherits the params-axis batching, k-chunking, and
(once wired) sharding.

This lives in scan/ and only calls the existing batched-stage methods
(``_pre_recomb_batched``, HyRex ``_recmodel_cpu``, ``_get_BG_batched``,
``PE.full_evolution_batched``, ``SS.get_Cl_batched``), with no core-code edits.

Run via srun on a GPU node, PYTHONPATH=$(pwd). Env: BG_LMAX(128), BG_WRT(omega_cdm,n_s).
"""
import os, time
_SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/c/carag")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_SCRATCH, ".jax_cache_abcmb"))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from abcmb.main import Model
from abcmb.main import _recmodel_cpu

LMAX = int(os.environ.get("BG_LMAX", 128))
WRT = os.environ.get("BG_WRT", "omega_cdm,n_s").split(",")

FIXED = {'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
         'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
         'Delta_z_reion': 0.5, 'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5,
         'exp_reion': 1.5}
ORDER = ['h', 'omega_b', 'omega_cdm', 'n_s', 'ln10As', 'tau_reion']
A_S = float(np.exp(3.044) / 1e10)


def raw_dict(h=0.6736, omega_b=0.02237, omega_cdm=0.1200, n_s=0.9649,
             ln10As=3.044, tau_reion=0.0544):
    return dict(FIXED, h=h, omega_b=omega_b, omega_cdm=omega_cdm, n_s=n_s,
                A_s=float(np.exp(ln10As) / 1e10), tau_reion=tau_reion)


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
    a (P, B, ...) direction-stacked tree. Single device (the old path) when only one CPU
    device exists, B is prime, or ``BG_HYREX_SHARD=0``."""
    cpus = jax.devices('cpu')
    n_cpu = len(cpus)
    n_use = max(d for d in range(1, n_cpu + 1) if B % d == 0)
    if n_use == 1 or os.environ.get("BG_HYREX_SHARD", "1") == "0":
        cpu = cpus[0]
        put = lambda t: jax.device_put(t, cpu)
        return _recmodel_cpu(model.RecModel, True), put, put, 1
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    key = (id(model.RecModel), n_use)
    hit = _HYREX_MESH_CACHE.get(key)
    if hit is not None:
        return hit[0], hit[1], hit[2], n_use
    # Build the wrapper, the mesh and the shardings ONCE per (RecModel, n_use) and reuse them:
    # a fresh Mesh/NamedSharding per call is a fresh jit cache key, i.e. a recompile of the
    # sharded HyRex program on every gradient call (the in-pipeline stage-2 time was stuck at
    # ~22 s while the same jvp ran in 4.3 s standalone, where the mesh was built once).
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


def diff_of(tree):
    return eqx.filter(tree, eqx.is_inexact_array)


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
def staged_cl_and_grad(model, full_ps, params_dots, k_chunk_size=100, shard=None,
                       dir_chunk=None, spec_b_local=None):
    """full_ps: list[B] of derived param dicts (primal).
    params_dots: list[P] of derived-param-tangent dicts, each stacked over B
        (i.e., params_dots[j] is the dB-stacked tangent of params_batch in
        direction j). Returns (ClTT,ClTE,ClEE) primal (B,n_l) and a list[P] of
        (dClTT,dClTE,dClEE) tangents (B,n_l).

    The P directions are stacked on a leading axis and pushed through each
    stage's jvp vmapped over that axis (``jvp_stage(..., n_dirs=)``): one primal
    evaluation per stage carries all the tangent lanes, and the primal Cls are
    read off that same jvp, so the pipeline's primal is solved ONCE per call.
    (Before 2026-10-14 the directions were looped one at a time and every
    jax.jvp re-solved the full primal: 1 + P primal solves per gradient.)

    dir_chunk : int or None. Number of directions per vmapped jvp; None = all P
        at once (fastest; the tangent of the saved modes tensor is P times the
        primal's). Smaller chunks cap memory at the cost of ceil(P/dir_chunk)
        primal solves; dir_chunk=1 is the old one-direction-at-a-time memory.

    spec_b_local : int or None. Cosmologies PER DEVICE per spectrum-stage jvp
        call. The spectrum jvp is the gradient's memory peak (its lensing and
        line-of-sight intermediates are (P, B, ...) tensors of ~1.2 GiB per
        cosmology at P=7, far above the ~0.2 GB/cosmology the tables need), and
        it is embarrassingly parallel over B, so stage 5 is run on sub-batches of
        spec_b_local x n_dev cosmologies and the Cls / tangents concatenated: no
        extra work, no change to the spectrum code, peak ~ spec_b_local/B_local
        of the full-batch stage. None = the whole batch in one call.

    k_chunk_size threads through full_evolution_batched on both the primal and the
    jvp. It is a gradient-path knob, independent of the primal call_batched (which
    keeps k_chunk=100): under jvp the augmented ForwardMode-Kvaerno5 loop is fused
    over vmap(k_chunk, B), so a smaller k_chunk shrinks that fusion and the compile,
    at the cost of more cached chunks at runtime.

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
        device 0 and OOM at large B."""
    PE = model.PE; SS = model.SS
    cpu = jax.devices('cpu')[0]
    try:
        gpu = jax.devices('gpu')[0]
    except Exception:
        gpu = None

    shardfn, n_dev = model._make_shardfn(shard)
    do_shard = shardfn is not None

    _t_mem = [time.perf_counter()]

    def _memlog(tag, *xs):
        """BG_MEMLOG=1: block on xs and print GPU bytes in use / peak + seconds since the
        previous stage (first call per stage includes its compile)."""
        if os.environ.get("BG_MEMLOG", "0") != "1" or gpu is None:
            return
        jax.block_until_ready(xs)
        st = gpu.memory_stats()
        now = time.perf_counter()
        print(f"[memlog] {tag:28s} in_use {st['bytes_in_use']/1e9:7.2f} GB  "
              f"peak {st['peak_bytes_in_use']/1e9:7.2f} GB  {now - _t_mem[0]:7.1f} s", flush=True)
        _t_mem[0] = now

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

    # stack the directions on a leading axis: leaves (P, B, ...); chunk over it
    if dir_chunk is None or dir_chunk >= Pn:
        dir_chunk = Pn
    cls_primal = None
    cl_dots = []                           # per chunk: (dTT, dTE, dEE) each (n_d, B, n_l)
    for d0 in range(0, Pn, dir_chunk):
        pdots = params_dots_run[d0:d0 + dir_chunk]
        n_d = len(pdots)
        pdot = jax.tree.map(lambda *xs: jnp.stack(xs), *pdots)
        if do_shard:
            pdot = dir_shardfn(pdot)

        # ---- stage 1: pre-recomb (GPU) ----
        pre_BG, pre_BG_dot = jvp_stage(model._pre_recomb_batched, (params_batch,), (pdot,),
                                       n_dirs=n_d)
        _memlog("stage1 pre_recomb jvp", pre_BG, pre_BG_dot)
        # ---- stage 2: HyRex (CPU mesh): inputs (recomb_inputs, params), tangents
        # (pre_BG_dot.recomb_inputs, pdot). Primal + tangents to the host devices (B axis
        # sharded over them when there are several, see hyrex_cpu_mesh); result to GPU.
        hy, put_b, put_pb, n_cpu_use = hyrex_cpu_mesh(model, len(full_ps_run))
        ri_cpu = put_b(pre_BG.recomb_inputs)
        p_cpu = put_b(params_batch)
        ri_dot_cpu = put_pb(pre_BG_dot.recomb_inputs)
        p_dot_cpu = put_pb(pdot)
        _memlog("stage2a inputs to cpu mesh", ri_cpu, p_cpu, ri_dot_cpu, p_dot_cpu)
        recomb, recomb_dot = jvp_stage(lambda ri, p: hy((ri, p)),
                                       (ri_cpu, p_cpu), (ri_dot_cpu, p_dot_cpu), n_dirs=n_d)
        if os.environ.get("BG_MEMLOG", "0") == "1":
            _xs = [x for x in jax.tree_util.tree_leaves(recomb_dot) if eqx.is_array(x)]
            print(f"[memlog] stage2 recomb_dot sharding: {str(_xs[0].sharding)[:100] if _xs else 'none'}", flush=True)
        _memlog("stage2b hyrex jvp on cpu", recomb, recomb_dot)
        if do_shard:
            recomb = shardfn(recomb)
            recomb_dot = dir_shardfn(recomb_dot)
        elif gpu is not None:
            recomb = jax.device_put(recomb, gpu)
            recomb_dot = jax.device_put(recomb_dot, gpu)
        # ---- stage 3: get_BG (GPU) ----
        _memlog(f"stage2 hyrex jvp ({n_cpu_use} cpu dev)", recomb, recomb_dot)
        BG, BG_dot = jvp_stage(model._get_BG_batched,
                               (params_batch, pre_BG, recomb), (pdot, pre_BG_dot, recomb_dot),
                               n_dirs=n_d)
        _memlog("stage3 get_BG jvp", BG, BG_dot)
        # ---- stage 4: perturbations (GPU, k-chunked) ----
        PT, PT_dot = jvp_stage(
            lambda bg, p: PE.full_evolution_batched((bg, p), k_chunk_size=k_chunk_size),
            (BG, params_batch), (BG_dot, pdot), n_dirs=n_d)
        _memlog("stage4 perturbations jvp", PT, PT_dot)
        # ---- stage 5: spectrum (GPU), in sub-batches of spec_b_local x n_dev ----
        Brun = len(full_ps_run)
        spec_chunk = None if spec_b_local is None else spec_b_local * (n_dev if do_shard else 1)
        if spec_chunk is None or spec_chunk >= Brun:
            cls, cl_dot = jvp_stage(SS.get_Cl_batched, (PT, BG, params_batch),
                                    (PT_dot, BG_dot, pdot), n_dirs=n_d)
        else:
            def _sl(tree, s, e, axis):
                return jax.tree.map(
                    lambda x: jax.lax.slice_in_dim(x, s, e, axis=axis)
                    if eqx.is_array(x) and x.ndim > axis else x, tree)
            cls_parts, dot_parts = [], []
            for s0 in range(0, Brun, spec_chunk):
                e0 = min(s0 + spec_chunk, Brun)
                prim = _sl((PT, BG, params_batch), s0, e0, 0)       # leaves (b, ...)
                tang = _sl((PT_dot, BG_dot, pdot), s0, e0, 1)       # leaves (n_d, b, ...)
                if do_shard:                                        # keep the B sharding
                    prim = shardfn(prim); tang = dir_shardfn(tang)
                c, cd = jvp_stage(SS.get_Cl_batched, prim, tang, n_dirs=n_d)
                cls_parts.append(c); dot_parts.append(cd)
                del prim, tang
            cls = tuple(jnp.concatenate([c[i] for c in cls_parts], axis=0) for i in range(3))
            cl_dot = tuple(jnp.concatenate([d[i] for d in dot_parts], axis=1) for i in range(3))
            del cls_parts, dot_parts
        _memlog("stage5 spectrum jvp", cls, cl_dot)
        # cl_dot is the filtered tangent of (ClTT,ClTE,ClEE): a 3-tuple of (n_d, B, n_l)
        if cls_primal is None:
            cls_primal = cls
        cl_dots.append(cl_dot)
        del pre_BG_dot, recomb_dot, BG_dot, PT, PT_dot

    ClTT, ClTE, ClEE = cls_primal
    # (P, B, n_l) per spectrum -> list[P] of (dTT, dTE, dEE), the historical layout
    stacked = tuple(jnp.concatenate([c[s] for c in cl_dots], axis=0) for s in range(3))
    grads = [tuple(g[j] for g in stacked) for j in range(Pn)]

    # --- slice off the B-padding (if any) on the primal Cls + every tangent ---
    if len(full_ps_run) != B:
        ClTT = ClTT[:B]; ClTE = ClTE[:B]; ClEE = ClEE[:B]
        grads = [tuple(g[:B] for g in cl_dot) for cl_dot in grads]
    return (ClTT, ClTE, ClEE), grads


# ---------- chi2-aware variant: contract the Cl tangents through a likelihood ----------
def staged_chi2_and_grad(model, full_ps, params_dots, chi2_of_cls,
                         k_chunk_size=100, shard=None, dir_chunk=None, spec_b_local=None):
    """Like staged_cl_and_grad but returns (chi2 (B,), grad (B,P)) by contracting
    each Cl tangent through a pure-jnp likelihood closure.

    chi2_of_cls : (ClTT, ClTE, ClEE) [each (B, n_l)] -> chi2 (B,). Must be
        differentiable in the Cls (the driver's envelope-profiled plik-lite +
        low-ell objective is). All P directions go through ONE vmapped jvp of the
        closure (NOT the pipeline), so its primal runs once: for full plik that is
        the inner nuisance profile, ~47 s for 42 rows (it ran 1 + P times before).
    params_dots[j] is the B-stacked derived-param tangent in direction j; the
        returned grad column j is d chi2 / d (that direction).
    shard : forwarded to staged_cl_and_grad (B-axis GSPMD sharding over all
        visible GPUs; padding is sliced off the Cls before the chi2 contraction,
        so chi2/grad already have the unpadded length B).
    dir_chunk : forwarded to staged_cl_and_grad (directions per vmapped jvp).
    spec_b_local : forwarded to staged_cl_and_grad (cosmologies per device per
        spectrum-stage call).
    """
    (ClTT, ClTE, ClEE), cl_grads = staged_cl_and_grad(
        model, full_ps, params_dots, k_chunk_size=k_chunk_size, shard=shard,
        dir_chunk=dir_chunk, spec_b_local=spec_b_local)
    stk = tuple(jnp.stack([cl_dot[s] for cl_dot in cl_grads]) for s in range(3))  # (P, B, n_l)
    chi2, c2_dots = jax.vmap(lambda *t: jax.jvp(chi2_of_cls, (ClTT, ClTE, ClEE), t),
                             out_axes=(None, 0))(*stk)
    grad = jnp.moveaxis(c2_dots, 0, -1)           # (B, P)
    return chi2, grad


# ---------- derived-param tangents w.r.t. raw nuisances (eager, per cosmo) ----------
def derived_and_tangents(model, raw_ps, wrt):
    """For each raw param dict, derive params and the tangent of the derived dict
    w.r.t. each wrt key (forward-mode through add_derived_parameters only).
    Returns (full_ps list[B], params_dots list[P] of B-stacked tangent dicts)."""
    full_ps = [model.add_derived_parameters(p) for p in raw_ps]
    P = len(wrt)
    per_cosmo_dots = []   # [B] of [P] tangent dicts
    for p in raw_ps:
        rd = {k: jnp.asarray(float(v)) for k, v in p.items()}
        dots_j = []
        for key in wrt:
            tangent = {k: (jnp.ones(()) if k == key else jnp.zeros(())) for k in rd}
            _, fd = jax.jvp(lambda d: model.add_derived_parameters(d), (rd,), (tangent,))
            dots_j.append(eqx.filter(fd, eqx.is_inexact_array))
        per_cosmo_dots.append(dots_j)
    # restack: params_dots[j] = stack over B of per_cosmo_dots[b][j]
    params_dots = []
    for j in range(P):
        params_dots.append(jax.tree.map(lambda *xs: jnp.stack(xs),
                                        *[per_cosmo_dots[b][j] for b in range(len(raw_ps))]))
    return full_ps, params_dots


# ---------- single-path reference: jacfwd of Cl per cosmology ----------
def single_path_cl_grad(model, raw_p, wrt):
    """ClTT/TE/EE and their gradients w.r.t. `wrt` for ONE cosmology, via the
    PROVEN single-cosmology jacfwd path. Returns dict key->(dTT,dTE,dEE)."""
    base = {k: jnp.asarray(float(v)) for k, v in raw_p.items()}

    def cl_of(scalar, key):
        d = dict(base); d[key] = scalar
        out = model.run_cosmology_abbr(model.add_derived_parameters(d))
        return out.ClTT, out.ClTE, out.ClEE
    res = {}
    for key in wrt:
        prim = jax.jacfwd(lambda s: cl_of(s, key))(base[key])
        res[key] = prim                      # tuple (dTT,dTE,dEE), each (n_l,)
    return res


def main():
    print(f"devices: {jax.devices()}  lmax={LMAX}  wrt={WRT}", flush=True)
    model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=False,
                  output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
                  rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
                  rtol_small_k_PE=1e-5, max_steps_PE=16384)
    raw_ps = [raw_dict(), raw_dict(h=0.68, omega_cdm=0.119, n_s=0.96)]

    full_ps, params_dots = derived_and_tangents(model, raw_ps, WRT)
    print(f"derived {len(full_ps)} cosmologies; P={len(params_dots)} directions", flush=True)

    t0 = time.perf_counter()
    (ClTT, ClTE, ClEE), grads = staged_cl_and_grad(model, full_ps, params_dots)
    jax.block_until_ready(grads)
    print(f"[staged] value+grad compile+run {time.perf_counter()-t0:.1f}s "
          f"(B={len(raw_ps)}, P={len(WRT)})", flush=True)

    # ---- validate batched grad vs single-path jacfwd, cosmology 0 ----
    print("\n[validate] batched staged grad vs single-path jacfwd (cosmo 0):", flush=True)
    ref = single_path_cl_grad(model, raw_ps[0], WRT)
    specs = ['TT', 'TE', 'EE']
    worst = 0.0
    for j, key in enumerate(WRT):
        for s in range(3):
            bat = np.asarray(grads[j][s][0])       # batched dCl for cosmo 0, spectrum s
            rfd = np.asarray(ref[key][s])
            denom = np.maximum(np.abs(rfd), np.percentile(np.abs(rfd), 90) + 1e-30)
            rel = np.abs(bat - rfd) / denom
            m = float(np.nanmedian(rel)); worst = max(worst, m)
            print(f"  d Cl{specs[s]} / d {key:10s}: median rel = {m:.2e}", flush=True)
    print(f"\n  >>> worst MEDIAN rel(batched, single-path jacfwd) = {worst:.2e} "
          f"(<~1e-3 => batched AD gradient is CORRECT)", flush=True)
    print("[batched_grad] done", flush=True)


if __name__ == "__main__":
    main()
