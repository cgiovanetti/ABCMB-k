"""Where the perturbation solve spends its time (TODO.md item 31): per-k step counts of a solver and the cost of the
pieces of one implicit step (RHS, Jacobian, LU, inverse, solves) at the batch the engine runs (k chunk x B).

    python scan/diag/pe_probe.py --engine act|plik [--solver kvaerno5,kencarp_fixed] [--nk 64] [--parts 1]
"""
import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx
from abcmb.main import Model
from abcmb import species
import abcmb.perturbations as _P

jax.config.update("jax_enable_x64", True)
ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="act")
ap.add_argument("--solver", default="kvaerno5,kencarp_fixed")
ap.add_argument("--nk", type=int, default=64)
ap.add_argument("--parts", type=int, default=1)
ap.add_argument("--batch", type=int, default=800)
ap.add_argument("--chunk", default="", help="lo:hi k indices timed as one engine chunk, tiled --tile times")
ap.add_argument("--tile", type=int, default=4)
ap.add_argument("--out", default="", help="npz of the final states (accuracy between solvers)")
ap.add_argument("--spec", action="append", default=[], help="extra Model spec key=value (int, float or bool)")
a = ap.parse_args()
SAVE_TS = jnp.asarray([-7.3, -7.1, -6.9, -6.5, -5.0, -3.0, -1.0, 0.0])
import os
TIGHT = float(os.environ.get("PE_TIGHT", "1"))      # tolerances divided by this (reference runs)
import ast
SPEC = {k: ast.literal_eval(v) for k, v in (s.split("=") for s in a.spec)}

KW = dict(output_Cl=True, lensing=True, output_Pk=False, rtol_large_k_PE=1e-5 / TIGHT, atol_large_k_PE=1e-7 / TIGHT,
          rtol_limber_k_PE=1e-3 / TIGHT, atol_limber_k_PE=1e-5 / TIGHT,
          rtol_small_k_PE=1e-5 / TIGHT, max_steps_PE=16384 * (8 if TIGHT > 1 else 1), lna_grid_mode="recomb", bbn_type="table",
          user_species=(species.MassiveNeutrino,))
if a.engine == "act":
    KW.update(l_max=6325, l_max_g=40, l_max_pol_g=60, delta_l_max=5000, nonlinear="halofit",
              use_bessel_tables=False, lensing_streamed=True)
else:
    KW.update(l_max=2508, l_max_g=12, l_max_pol_g=10)
P = {"omega_b": 0.022581, "omega_cdm": 0.12321, "h": 0.6624, "A_s": np.exp(3.04991) / 1e10, "n_s": 0.96732,
     "tau_reion": 0.05526, "Neff": 3.044, "N_nu_massive": 1, "T_nu_massive": 0.71611, "m_nu_massive": 0.06,
     "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5, "exp_reion": 1.5, "TCMB0": 2.34865418e-4}


def build(solver):
    m = Model(**dict(KW, PE_solver=solver, **SPEC))
    params, BG = m._build_one_bg(m.add_derived_parameters(P))
    if solver == "kvaerno5_fixed":
        m = eqx.tree_at(lambda m: m.PE, m, m.PE.with_structure_for((BG, params), batched=False))
    return m, params, BG


def stats_one_k(PE, k, args):
    """evolution_one_k's solve without the save grid: (steps, accepted, rejected)."""
    lna_start = jnp.minimum(PE.get_starting_time(k, args), -10.)
    y_ini = PE.initial_conditions_one_k(k, lna_start, args)
    sp = PE.specs
    if sp["PE_solver"] == "kvaerno5_fixed":
        solver = _P.FixedKvaerno5(n_iter=sp["kvaerno5_n_iter"], structure=sp["pe_structure"],
                                  conv_weight=sp["kvaerno5_conv_weight"], factored=sp["kvaerno5_factored"],
                                  unroll=sp["kvaerno5_unroll"], jit_vf=sp["kvaerno5_jit_vf"])
    else:
        solver = _P.FixedKenCarp4(n_iter=sp["kencarp_n_iter"]) if sp["PE_solver"] == "kencarp_fixed" else diffrax.Kvaerno5()
    lim = k > sp["k_limber_start"]
    big = k > sp["k_split_PE"]
    rtol = jnp.where(lim, sp["rtol_limber_k_PE"], jnp.where(big, sp["rtol_large_k_PE"], sp["rtol_small_k_PE"]))
    atol = jnp.where(lim, sp["atol_limber_k_PE"], jnp.where(big, sp["atol_large_k_PE"], sp["atol_small_k_PE"]))
    ctrl = diffrax.PIDController(pcoeff=sp["pcoeff_PE"], icoeff=sp["icoeff_PE"], dcoeff=sp["dcoeff_PE"], rtol=rtol,
                                 atol=atol, dtmax=jnp.where(k < sp["k_max_dtmax_PE"], sp["dtmax_PE"], jnp.inf))
    sol = diffrax.diffeqsolve(diffrax.ODETerm(PE.get_derivatives), solver, t0=lna_start, t1=0.0, dt0=1e-2, y0=y_ini,
                              stepsize_controller=ctrl, max_steps=sp["max_steps_PE"],
                              saveat=diffrax.SaveAt(ts=SAVE_TS), args=(k, *args), adjoint=diffrax.RecursiveCheckpointAdjoint())
    st = sol.stats
    return st["num_steps"], st["num_accepted_steps"], st["num_rejected_steps"], lna_start, sol.ys


for solver in a.solver.split(","):
    m, params, BG = build(solver)
    PE = m.PE
    kax = np.asarray(PE.k_axis_perturbations)
    n = 1 + sum(int(s.num_equations) for s in PE.species_list)
    print(f"[probe] {a.engine} {solver}: n_y {n}, n_k {len(kax)}, k {kax[0]:.2e}..{kax[-1]:.3f}, k_max_cmb "
          f"{float(PE.specs.get('k_max_cmb', np.nan)):.3f}, k_limber_start {float(PE.specs['k_limber_start']):.3f}", flush=True)
    idx = np.unique(np.linspace(0, len(kax) - 1, a.nk).round().astype(int))
    if a.chunk:
        lo, hi = (int(v) for v in a.chunk.split(":"))
        idx = np.tile(np.arange(lo, hi), a.tile)
    f = jax.jit(jax.vmap(lambda k: stats_one_k(PE, k, (BG, params))))
    t0 = time.time(); out = [np.asarray(o) for o in f(jnp.asarray(kax[idx]))]; t1 = time.time()
    out = [np.asarray(o) for o in f(jnp.asarray(kax[idx]))]; t2 = time.time()
    print(f"[probe] {solver} stats call: cold {t1 - t0:.1f} s, warm {t2 - t1:.1f} s (vmap over {len(idx)} k)")
    for j in range(0, len(idx), max(1, len(idx) // 16)):
        print(f"    k {kax[idx[j]]:.4e}  steps {out[0][j]:6d}  acc {out[1][j]:6d}  rej {out[2][j]:5d}  lna0 {out[3][j]:.2f}")
    # chunks of 100 run to their slowest member: steps the engine pays for
    allsteps = np.asarray(jax.jit(jax.vmap(lambda k: stats_one_k(PE, k, (BG, params))[0]))(jnp.asarray(kax))) \
        if len(idx) < len(kax) and a.nk >= 10 ** 6 else None
    if a.out:
        np.savez(a.out, k=kax[idx], y=out[4], steps=out[0])
    print(f"[probe] {solver}: sum steps over sampled k {out[0].sum()}, max {out[0].max()}", flush=True)

if a.parts:
    k = jnp.asarray(np.geomspace(1e-3, 0.5, a.batch))
    lna = jnp.full(a.batch, -7.2)
    y = jax.random.normal(jax.random.PRNGKey(0), (a.batch, n)) * 1e-2
    args1 = lambda kk: (kk, BG, params)
    vf = jax.jit(jax.vmap(lambda kk, t, yy: PE.get_derivatives(t, yy, args1(kk))))
    jac = jax.jit(jax.vmap(lambda kk, t, yy: jax.jacfwd(lambda z: PE.get_derivatives(t, z, args1(kk)))(yy)))
    J = jac(k, lna, y)
    M = jnp.eye(n) - 0.01 * J
    inv = jax.jit(jax.vmap(jnp.linalg.inv))
    lu = jax.jit(jax.vmap(jax.scipy.linalg.lu_factor))
    Mi = inv(M); LU = lu(M)
    mv = jax.jit(lambda A, v: jnp.einsum("bij,bj->bi", A, v))
    lus = jax.jit(jax.vmap(lambda f_, v: jax.scipy.linalg.lu_solve(f_, v)))
    sol = jax.jit(jax.vmap(jnp.linalg.solve))

    def timeit(name, fn, *xs, reps=20):
        jax.block_until_ready(fn(*xs))
        t0 = time.time()
        for _ in range(reps):
            r = fn(*xs)
        jax.block_until_ready(r)
        print(f"    {name:28s} {(time.time() - t0) / reps * 1e3:8.2f} ms  (batch {a.batch}, n {n})", flush=True)

    print(f"[probe] pieces of one step on one GPU:")
    timeit("RHS", vf, k, lna, y)
    timeit("Jacobian (jacfwd)", jac, k, lna, y)
    timeit("inverse", inv, M)
    timeit("LU factor", lu, M)
    timeit("solve (LU + 1 rhs)", sol, M, y)
    timeit("M^-1 @ v", mv, Mi, y)
    timeit("lu_solve", lus, LU, y)
    nz = np.asarray(jnp.abs(J).max(0) > 0)
    print(f"[probe] Jacobian pattern: {nz.sum()} of {n * n} entries nonzero ({nz.mean():.1%}); "
          f"bandwidth lower {max(i - j for i, j in zip(*np.nonzero(nz)))}, upper {max(j - i for i, j in zip(*np.nonzero(nz)))}")
    np.save(f"logs/kencarp_bench/jac_pattern_{a.engine}.npy", nz)
