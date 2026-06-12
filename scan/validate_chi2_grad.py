"""validate_chi2_grad.py — correctness gate for staged_chi2_and_grad.

Validates the BATCHED chi2 gradient (scan/batched_grad.staged_chi2_and_grad,
contracting the staged per-k Cl tangents through the real plik-lite + low-ell
objective) against the SINGLE-PATH AD gradient of the SAME chi2 objective
(jax.jacfwd through run_cosmology_abbr). This is the gate before wiring
PA_GRADMETHOD=batched into the driver.

Small l_max for speed (the contraction logic is l_max-independent). Run via srun,
PYTHONPATH=$(pwd).  VCG_LMAX(128) VCG_B(3) VCG_LENSING(0)
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                      os.path.join(os.environ.get("SCRATCH", "/pscratch/sd/c/carag"),
                                   ".jax_cache_abcmb"))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from abcmb.main import Model
from scan.plik_lite import PlikLite
from scan.lowl_like import LowLEE, LowLTT
from scan.batched_grad import staged_chi2_and_grad, _to_float

LMAX = int(os.environ.get("VCG_LMAX", 128))
B = int(os.environ.get("VCG_B", 3))
LENSING = os.environ.get("VCG_LENSING", "0") != "0"
# VCG_SHARD: if "1", ALSO compare the sharded staged grad vs the unsharded one
# (GSPMD partitions the SAME program -> expect agreement ~1e-12; >1e-6 = bug).
SHARD = os.environ.get("VCG_SHARD", "0") != "0"

FIXED = {'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
         'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
         'Delta_z_reion': 0.5, 'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5,
         'exp_reion': 1.5}
ORDER = ['h', 'omega_b', 'omega_cdm', 'n_s', 'ln10As', 'tau_reion']
SIG = np.array([0.0054, 0.00015, 0.0012, 0.0042, 0.014, 0.0073])
CEN = np.array([0.6736, 0.02237, 0.1200, 0.9649, 3.044, 0.0544])

pl = PlikLite()
lowee = LowLEE()
lowtt = LowLTT()
model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=LENSING,
              output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
              rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
              rtol_small_k_PE=1e-5, max_steps_PE=16384)


def chi2_of_cls(ClTT, ClTE, ClEE):
    """Pure-jnp envelope-profiled plik-lite + low-ell chi2 from (batched) Cls."""
    l = model.SS.ells
    Dtt = pl.abcmb_cl_to_Dl(ClTT, l); Dte = pl.abcmb_cl_to_Dl(ClTE, l)
    Dee = pl.abcmb_cl_to_Dl(ClEE, l)
    m0 = pl.bin_model(Dtt, Dte, Dee)
    A_star = jax.lax.stop_gradient(pl.profile_A(m0, with_prior=True)[1])
    diff = pl.X_data - m0 / (A_star[..., None] ** 2)
    c2 = jnp.einsum("...i,ij,...j->...", diff, pl.invcov, diff) \
        + ((A_star - 1.0) / 0.0025) ** 2
    c2 = c2 + lowee.chi2(Dee) + lowtt.chi2(Dtt)
    return c2


def phys_to_derived(th6):
    p = dict(FIXED)
    p['h'] = th6[0]; p['omega_b'] = th6[1]; p['omega_cdm'] = th6[2]
    p['n_s'] = th6[3]; p['A_s'] = jnp.exp(th6[4]) / 1e10; p['tau_reion'] = th6[5]
    # _to_float the DERIVED dict so the jvp tangent has the SAME inexact-array
    # tree structure as staged_cl_and_grad's _to_float'd primal params_batch
    # (else int-valued derived keys like N_nu_massive -> None tangent mismatch the
    # float-array primal partition). This is the documented batched-AD gotcha.
    return _to_float(model.add_derived_parameters(p))


def main():
    print(f"devices={jax.devices()} lmax={LMAX} B={B} lensing={LENSING} "
          f"shard_cmp={SHARD}", flush=True)
    rng = np.random.default_rng(1)
    # B cosmologies dispersed around LCDM center (in sigma units, modest)
    thetas = [jnp.asarray(CEN + 0.3 * SIG * rng.normal(size=6)) for _ in range(B)]
    poi_idx = 3  # n_s as the (fixed) POI; the 5 nuisances are the other params
    nuis = [i for i in range(6) if i != poi_idx]

    # ---- batched gradient (dchi2/dx5, scaled coords) ----
    full_ps = [phys_to_derived(t) for t in thetas]
    per = []
    for t in thetas:
        dots = []
        for i in nuis:
            tan = jnp.zeros(6).at[i].set(SIG[i])
            _, fd = jax.jvp(phys_to_derived, (t,), (tan,))
            dots.append(eqx.filter(fd, eqx.is_inexact_array))
        per.append(dots)
    params_dots = [jax.tree.map(lambda *xs: jnp.stack(xs),
                                *[per[b][j] for b in range(B)]) for j in range(5)]
    # reference path is UNSHARDED (single device); the optional shard comparison
    # below runs the IDENTICAL program partitioned across all GPUs.
    chi2_b, grad_b = staged_chi2_and_grad(model, full_ps, params_dots, chi2_of_cls,
                                          k_chunk_size=100, shard=False)
    chi2_b = np.asarray(chi2_b); grad_b = np.asarray(grad_b)   # (B,), (B,5)
    print(f"batched chi2 (unsharded) = {chi2_b}", flush=True)

    # ---- SHARDED vs UNSHARDED gate (GSPMD partitions the same program) ----
    if SHARD:
        print("\n[validate] SHARDED vs UNSHARDED staged chi2 grad "
              f"(ndev={len(jax.devices('gpu'))}):", flush=True)
        chi2_s, grad_s = staged_chi2_and_grad(model, full_ps, params_dots,
                                              chi2_of_cls, k_chunk_size=100,
                                              shard=True)
        chi2_s = np.asarray(chi2_s); grad_s = np.asarray(grad_s)
        d_chi2 = float(np.max(np.abs(chi2_s - chi2_b)
                              / np.maximum(np.abs(chi2_b), 1.0)))
        d_grad = float(np.max(np.abs(grad_s - grad_b)
                              / np.maximum(np.abs(grad_b), 1.0)))
        print(f"  chi2: {chi2_s}", flush=True)
        print(f"  max rel(sharded, unsharded) chi2 = {d_chi2:.2e}", flush=True)
        print(f"  max rel(sharded, unsharded) grad = {d_grad:.2e}", flush=True)
        verdict = "PASS (~1e-12, same program)" if max(d_chi2, d_grad) < 1e-6 \
            else "FAIL (>1e-6 => sharding bug, do NOT proceed to timing)"
        print(f"  >>> SHARD GATE: max rel = {max(d_chi2, d_grad):.2e}  {verdict}",
              flush=True)

    # ---- single-path reference: jacfwd of chi2(run_cosmology_abbr) per cosmo ----
    def chi2_single(th6):
        out = model.run_cosmology_abbr(phys_to_derived(th6))
        # add leading axis so chi2_of_cls (batched) works, then squeeze
        c = chi2_of_cls(out.ClTT[None], out.ClTE[None], out.ClEE[None])
        return c[0]

    print("\n[validate] batched dchi2/dx5 vs single-path jacfwd:", flush=True)
    worst = 0.0
    for b in range(B):
        g_full = np.asarray(jax.jacfwd(chi2_single)(thetas[b]))   # (6,) dchi2/dtheta
        # scaled nuisance gradient: dchi2/dx5_k = dchi2/dtheta_{nuis[k]} * SIG[nuis[k]]
        g_ref = np.array([g_full[i] * SIG[i] for i in nuis])      # (5,)
        rel = np.abs(grad_b[b] - g_ref) / np.maximum(np.abs(g_ref), 1.0)
        m = float(rel.max()); worst = max(worst, m)
        print(f"  cosmo {b}: max rel = {m:.2e}   |g_ref|max={np.abs(g_ref).max():.3e}",
              flush=True)
        if b == 0:
            print(f"    g_batched = {grad_b[b]}", flush=True)
            print(f"    g_ref     = {g_ref}", flush=True)
    print(f"\n  >>> worst max-rel(batched chi2 grad, single-path) = {worst:.2e} "
          f"(<~1e-3 => CORRECT)", flush=True)
    print("[validate_chi2_grad] done", flush=True)


if __name__ == "__main__":
    main()
