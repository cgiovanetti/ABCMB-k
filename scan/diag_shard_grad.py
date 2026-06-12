"""diag_shard_grad.py — isolate where sharded vs unsharded staged grad differs.

The l128 gate found chi2 grad max-rel(sharded,unsharded)=3.34e-5 while the
chi2 VALUE matched at 1.5e-9. This script localizes the discrepancy:
  (1) Cl-level grad agreement (staged_cl_and_grad, pre-chi2) -- peak-normalized
      AND absolute per spectrum/direction. If the Cl tangents agree at ~fp64
      reduction-reorder level (~1e-9..1e-7), the pipeline sharding is CORRECT
      and the chi2 number's larger spread is the einsum/LoS reduction order
      under GSPMD (benign, the documented permille floor).
  (2) per-component chi2-grad breakdown (sharded vs unsharded vs the magnitude
      of each component) so a 3e-5 *relative* figure can be read as the tiny
      ABSOLUTE diff it is.

Run via srun, PYTHONPATH=$(pwd).  DSG_LMAX(128) DSG_B(4)
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
from scan.batched_grad import staged_cl_and_grad, staged_chi2_and_grad, _to_float

LMAX = int(os.environ.get("DSG_LMAX", 128))
B = int(os.environ.get("DSG_B", 4))
FIXED = {'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
         'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
         'Delta_z_reion': 0.5, 'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5,
         'exp_reion': 1.5}
SIG = np.array([0.0054, 0.00015, 0.0012, 0.0042, 0.014, 0.0073])
CEN = np.array([0.6736, 0.02237, 0.1200, 0.9649, 3.044, 0.0544])

pl = PlikLite(); lowee = LowLEE(); lowtt = LowLTT()
model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=False,
              output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
              rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
              rtol_small_k_PE=1e-5, max_steps_PE=16384)


def chi2_of_cls(ClTT, ClTE, ClEE):
    l = model.SS.ells
    Dtt = pl.abcmb_cl_to_Dl(ClTT, l); Dte = pl.abcmb_cl_to_Dl(ClTE, l)
    Dee = pl.abcmb_cl_to_Dl(ClEE, l)
    m0 = pl.bin_model(Dtt, Dte, Dee)
    A_star = jax.lax.stop_gradient(pl.profile_A(m0, with_prior=True)[1])
    diff = pl.X_data - m0 / (A_star[..., None] ** 2)
    c2 = jnp.einsum("...i,ij,...j->...", diff, pl.invcov, diff) \
        + ((A_star - 1.0) / 0.0025) ** 2
    return c2 + lowee.chi2(Dee) + lowtt.chi2(Dtt)


def phys_to_derived(th6):
    p = dict(FIXED)
    p['h'] = th6[0]; p['omega_b'] = th6[1]; p['omega_cdm'] = th6[2]
    p['n_s'] = th6[3]; p['A_s'] = jnp.exp(th6[4]) / 1e10; p['tau_reion'] = th6[5]
    return _to_float(model.add_derived_parameters(p))


def build_inputs():
    rng = np.random.default_rng(1)
    thetas = [jnp.asarray(CEN + 0.3 * SIG * rng.normal(size=6)) for _ in range(B)]
    poi_idx = 3
    nuis = [i for i in range(6) if i != poi_idx]
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
    return full_ps, params_dots


def main():
    print(f"devices={jax.devices()} lmax={LMAX} B={B}", flush=True)
    full_ps, params_dots = build_inputs()

    # ---- (1) Cl-level grad: unsharded vs sharded ----
    (TTu, TEu, EEu), gu = staged_cl_and_grad(model, full_ps, params_dots,
                                             k_chunk_size=100, shard=False)
    (TTs, TEs, EEs), gs = staged_cl_and_grad(model, full_ps, params_dots,
                                             k_chunk_size=100, shard=True)
    print("\n[Cl-level] sharded vs unsharded staged_cl_and_grad:", flush=True)
    specs = ['TT', 'TE', 'EE']
    for j in range(len(gu)):
        for s in range(3):
            a = np.asarray(gu[j][s]); b = np.asarray(gs[j][s])  # (B,n_l)
            peak = np.abs(a).max() + 1e-300
            pn = np.abs(a - b).max() / peak           # peak-normalized
            ab = np.abs(a - b).max()                  # absolute
            print(f"  dCl{specs[s]} dir{j}: peak-norm {pn:.2e}  abs {ab:.2e}",
                  flush=True)
    # primal Cl agreement
    for nm, a, b in [('TT', TTu, TTs), ('TE', TEu, TEs), ('EE', EEu, EEs)]:
        a = np.asarray(a); b = np.asarray(b)
        print(f"  primal Cl{nm}: peak-norm "
              f"{np.abs(a-b).max()/(np.abs(a).max()+1e-300):.2e}", flush=True)

    # ---- (2) chi2-grad component breakdown ----
    c2u, gu2 = staged_chi2_and_grad(model, full_ps, params_dots, chi2_of_cls,
                                    k_chunk_size=100, shard=False)
    c2s, gs2 = staged_chi2_and_grad(model, full_ps, params_dots, chi2_of_cls,
                                    k_chunk_size=100, shard=True)
    gu2 = np.asarray(gu2); gs2 = np.asarray(gs2)
    print("\n[chi2-grad] per-component (cosmo 0): value / abs-diff / rel-to-comp",
          flush=True)
    for k in range(gu2.shape[1]):
        u = gu2[0, k]; v = gs2[0, k]
        rel = abs(u - v) / max(abs(u), 1e-30)
        print(f"  comp {k}: unshard {u:+.6e}  abs-diff {abs(u-v):.2e}  "
              f"rel-to-comp {rel:.2e}", flush=True)
    # the metric the gate used (floor 1.0) vs a true relative metric
    floor_metric = np.max(np.abs(gs2 - gu2) / np.maximum(np.abs(gu2), 1.0))
    true_rel = np.max(np.abs(gs2 - gu2) / np.maximum(np.abs(gu2), 1e-12))
    abs_metric = np.max(np.abs(gs2 - gu2))
    print(f"\n  gate metric (floor 1.0)      = {floor_metric:.2e}", flush=True)
    print(f"  true relative (floor 1e-12)  = {true_rel:.2e}", flush=True)
    print(f"  max ABSOLUTE diff            = {abs_metric:.2e}", flush=True)
    print(f"  |grad| range = [{np.abs(gu2).min():.2e}, {np.abs(gu2).max():.2e}]",
          flush=True)
    print("[diag_shard_grad] done", flush=True)


if __name__ == "__main__":
    main()
