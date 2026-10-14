"""modax feasibility benchmark on a synthetic 46-variable Boltzmann-like stiff hierarchy
(same size as ABCMB-k's production perturbation state): metric eta, CDM, baryons (delta, theta),
photons F_0..F_12 + G_0..G_10 (Thomson-coupled, stiff early), massless neutrinos N_0..N_17, in
synchronous gauge with lna as the time variable, a toy analytic background aH(lna) and an x_e(lna)
table in CUDA constant memory (linear lookup, as modax's Mukhanov-Sasaki example does).
Per trajectory p = (k, c0..c6): 7 cosmology-like parameters that scale the background terms, so
the forward sensitivities w.r.t. them (jax.jvp through modax.solve, sens_param_columns=1..7)
mimic ABCMB-k's 7-direction gradient.
Times: modax rodas5P forward at N=572 and 572*B, and with 7 sensitivities; optionally diffrax
Kvaerno5 vmapped in chunks of 100 on the same system (DIFFRAX=1) for the lockstep comparison.
Usage: [B=16] [DIFFRAX=1] python synth_bench.py"""
import os, sys, time, json, math
import numpy as np
B = int(os.environ.get("B", "16")); RTOL = float(os.environ.get("RTOL", "1e-5")); ATOL = float(os.environ.get("ATOL", "1e-7"))
LMAX_G, LMAX_PG, LMAX_NU = 12, 10, 17
N_K = 572
# ---------------- state layout ----------------
names = ["eta", "delta_c", "delta_b", "theta_b"] + [f"F{l}" for l in range(LMAX_G + 1)] + \
        [f"G{l}" for l in range(LMAX_PG + 1)] + [f"N{l}" for l in range(LMAX_NU + 1)]
idx = {n: i for i, n in enumerate(names)}
D = len(names)
assert D == 46, D
# ---------------- toy background ----------------
LNA0, LNA1, NTAB = -15.0, 0.0, 300
lna_tab = np.linspace(LNA0, LNA1, NTAB)
def xe_of(lna):  # fully ionised early, recombination step near lna=-7 (z~1100), freeze-out 2e-4
    return 2e-4 + (1.0 - 2e-4) * 0.5 * (1.0 - np.tanh((lna + 7.0) / 0.12))
XE_TAB = np.ascontiguousarray(xe_of(lna_tab), dtype=np.float64)
DLNA = float(lna_tab[1] - lna_tab[0])
OM_R, OM_M, H0 = 9e-5, 0.3, 2.2e-4     # aH in Mpc^-1 units: H0 = 100 h km/s/Mpc / c ~ 2.2e-4 (h=0.67)
# ---------------- generate the modax RHS source (unrolled, constant indices) ----------------
def gen_rhs_source():
    L = []
    L.append("def ode_fn(y, t, p):")
    L.append("    k = p[0]; c0 = p[1]; c1 = p[2]; c2 = p[3]; c3 = p[4]; c4 = p[5]; c5 = p[6]; c6 = p[7]; c7 = p[8]")
    L.append("    a = math.exp(t)")
    L.append(f"    aH = {H0} * c0 * math.sqrt({OM_R} * c1 / (a * a) + {OM_M} * c2 / a)")
    # analytic recombination history (no tables: the real port carries the background as scalar
    # coefficients in p too); c7 moves the recombination epoch, so it is a differentiable parameter
    L.append("    xe = 2e-4 + (1.0 - 2e-4) / (1.0 + math.exp((t + 7.0 * c7) / 0.06))")
    # Thomson rate in lna units: kappa' / aH, huge early (stiff); c3 scales omega_b-like
    L.append("    kappa = 7.0e-2 * c3 * xe / (a * a) / aH")
    L.append("    R = 0.6 * c4 * a")                            # 3 rho_b / 4 rho_g ~ a
    L.append("    cs2 = 1.0e-5 * c5 / a")                      # baryon sound speed^2 (toy)
    L.append(f"    k_aH = k / aH")
    for n in names:
        L.append(f"    {n} = y[{idx[n]}]")
    # metric: h' from the (toy) Poisson-like constraint; eta' from momentum
    L.append(f"    rho_g = {OM_R} * c1 / (a * a * a * a); rho_nu = 0.69 * rho_g; rho_c = {OM_M} * c2 / (a * a * a) * 0.85; rho_b = {OM_M} * c2 / (a * a * a) * 0.15 * c3")
    L.append("    sum_rho_delta = rho_g * F0 + rho_nu * N0 + rho_c * delta_c + rho_b * delta_b")
    L.append("    sum_rhoP_theta = (4.0 / 3.0) * rho_g * (0.75 * k * F1) + (4.0 / 3.0) * rho_nu * (0.75 * k * N1) + rho_b * theta_b")
    L.append("    H2 = aH * aH")
    L.append(f"    h_prime = 2.0 / H2 * (k * k * eta + 1.5 * {H0 * H0} * c6 * a * a * sum_rho_delta)")
    L.append(f"    eta_prime = 1.5 * {H0 * H0} * c6 * a * a / aH / (k * k) * sum_rhoP_theta")
    L.append("    theta_g = 0.75 * k * F1; sigma_g = 0.5 * F2; Pi = F2 + G0 + G2")
    out = ["eta_prime", "-0.5 * h_prime", "-theta_b / aH - 0.5 * h_prime",
           "-theta_b + cs2 * k_aH * k * delta_b + kappa * (theta_g - theta_b) / R"]
    # photons: F_l hierarchy (dF/dlna = (1/aH) dF/dtau)
    out.append("k_aH * (-F1) - (2.0 / 3.0) * h_prime")                                   # F0
    out.append("k_aH * (F0 - 2.0 * F2) / 3.0 * 1.0 + kappa * (4.0 / 3.0) * (theta_b - theta_g) / k * 1.0")  # F1 ~ (4/3k) theta_g'
    for l in range(2, LMAX_G + 1):
        term = f"k_aH * ({l} * F{l - 1} - {l + 1} * F{l + 1}) / {2 * l + 1}" if l < LMAX_G else f"k_aH * F{l - 1} - {l + 1} / (k_aH * 1.0 + 1e-30) * 0.0 * F{l}"
        src = " + (4.0 / 15.0) * h_prime * 0.0 - kappa * (F2 - 0.1 * Pi)" if l == 2 else f" - kappa * F{l}"
        if l == 2:
            term = f"k_aH * (2.0 * F1 - 3.0 * F3) / 5.0 + (4.0 / 15.0) * (h_prime + 6.0 * eta_prime) - kappa * (F2 - 0.1 * Pi)"
            out.append(term)
        else:
            out.append(term + src)
    for l in range(0, LMAX_PG + 1):
        if l == 0:
            out.append("k_aH * (-G1) - kappa * (G0 - 0.5 * Pi)")
        elif l == 2:
            out.append("k_aH * (2.0 * G1 - 3.0 * G3) / 5.0 - kappa * (G2 - 0.1 * Pi)")
        elif l < LMAX_PG:
            out.append(f"k_aH * ({l} * G{l - 1} - {l + 1} * G{l + 1}) / {2 * l + 1} - kappa * G{l}")
        else:
            out.append(f"k_aH * G{l - 1} - kappa * G{l}")
    for l in range(0, LMAX_NU + 1):
        if l == 0:
            out.append("k_aH * (-N1) - (2.0 / 3.0) * h_prime")
        elif l == 1:
            out.append("k_aH * (N0 - 2.0 * N2) / 3.0")
        elif l == 2:
            out.append("k_aH * (2.0 * N1 - 3.0 * N3) / 5.0 + (4.0 / 15.0) * (h_prime + 6.0 * eta_prime)")
        elif l < LMAX_NU:
            out.append(f"k_aH * ({l} * N{l - 1} - {l + 1} * N{l + 1}) / {2 * l + 1}")
        else:
            out.append(f"k_aH * N{l - 1} - {l + 1} * N{l} / (k_aH * 1.0 + 1e-30) * 0.0")
    assert len(out) == D, len(out)
    L.append("    return (" + ",\n            ".join(out) + ")")
    return "\n".join(L)
SRC = gen_rhs_source()
ns = {"math": math}
exec(SRC, ns)
ode_fn = ns["ode_fn"]
# ---------------- ensemble ----------------
k_axis = np.logspace(-4, np.log10(0.4), N_K)
def y0_of(k):  # adiabatic-ish toy ICs
    y = np.zeros(D); y[idx["eta"]] = 1.0; y[idx["delta_c"]] = -1e-3; y[idx["delta_b"]] = -1e-3; y[idx["F0"]] = -4e-3 / 3; y[idx["N0"]] = -4e-3 / 3
    y[idx["F1"]] = 1e-6 * k; y[idx["N1"]] = 1e-6 * k; y[idx["theta_b"]] = 1e-6 * k * k
    return y
Y0 = np.stack([y0_of(k) for k in k_axis])
P1 = np.concatenate([k_axis[:, None], np.ones((N_K, 8))], axis=1)         # (N_K, 9)
def ensemble(Bc):
    rng = np.random.default_rng(0)
    scales = 1.0 + 0.05 * rng.standard_normal((Bc, 8)); scales[0] = 1.0
    P = np.concatenate([np.concatenate([k_axis[:, None], np.tile(s, (N_K, 1))], axis=1) for s in scales], axis=0)
    return np.tile(Y0, (Bc, 1)), P
T_SPAN = np.linspace(LNA0, LNA1, 300)
# Jacobian sparsity superset: tridiagonal within each hierarchy + dense coupling through h', eta', kappa terms
S = np.zeros((D, D), dtype=bool)
for i in range(D):
    for j in range(max(0, i - 1), min(D, i + 2)):
        S[i, j] = True
dense_rows = [idx[n] for n in ("eta", "delta_c", "delta_b", "theta_b", "F0", "F1", "F2", "G0", "G2", "N0", "N2")]
dense_cols = [idx[n] for n in ("eta", "delta_c", "delta_b", "theta_b", "F0", "F1", "F2", "G0", "G2", "N0", "N1")]
S[dense_rows, :] = True; S[:, dense_cols] = True
print(f"D={D} N_K={N_K} B={B} rtol={RTOL} atol={ATOL} sparsity nnz={S.sum()}/{D*D}", flush=True)
import jax, jax.numpy as jnp
import modax
from modax.rodas5P import solve
def timed(fn, *a, reps=2, **kw):
    out = fn(*a, **kw); jax.block_until_ready(out); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(*a, **kw); jax.block_until_ready(out); ts.append(time.perf_counter() - t0)
    return out, min(ts)
res = {}
for Bc in sorted({1, B}):
    y0, P = ensemble(Bc)
    t0c = time.perf_counter()
    y, stats = solve(ode_fn, jnp.asarray(y0), jnp.asarray(T_SPAN), jnp.asarray(P), rtol=RTOL, atol=ATOL,
                     max_steps=200_000, return_stats=True, sparsity=S, pcoeff=0.25, icoeff=0.8)
    jax.block_until_ready(y); t_cold = time.perf_counter() - t0c
    f = lambda P_: solve(ode_fn, jnp.asarray(y0), jnp.asarray(T_SPAN), P_, rtol=RTOL, atol=ATOL, max_steps=200_000, sparsity=S, pcoeff=0.25, icoeff=0.8)
    _, t_fwd = timed(f, jnp.asarray(P))
    steps = {kk: (int(np.median(np.asarray(v))), int(np.max(np.asarray(v)))) for kk, v in stats.items()} if isinstance(stats, dict) else str(stats)[:200]
    print(f"[modax fwd] N={N_K*Bc}: cold {t_cold:.1f} s, warm {t_fwd:.2f} s = {t_fwd/Bc:.2f} s/cosmo; finite={bool(np.isfinite(np.asarray(y)).all())}; steps(median,max)={steps}", flush=True)
    res[f"fwd_N{N_K*Bc}"] = {"warm_s": t_fwd, "s_per_cosmo": t_fwd / Bc, "cold_s": t_cold, "steps": steps}
    # 7-direction forward sensitivities: jvp w.r.t. the 7 cosmology-like columns (not k)
    tang = jnp.asarray(np.concatenate([np.zeros((N_K * Bc, 1)), 0.01 * np.ones((N_K * Bc, 7)), np.zeros((N_K * Bc, 1))], axis=1))
    g = lambda P_: solve(ode_fn, jnp.asarray(y0), jnp.asarray(T_SPAN), P_, rtol=RTOL, atol=ATOL, max_steps=200_000,
                         sparsity=S, pcoeff=0.25, icoeff=0.8, sens_param_columns=list(range(1, 8)))
    jv = lambda P_: jax.jvp(g, (P_,), (tang,))
    try:
        t0c = time.perf_counter(); yv, yd = jv(jnp.asarray(P)); jax.block_until_ready(yd); t_cold = time.perf_counter() - t0c
        _, t_jvp = timed(jv, jnp.asarray(P))
        print(f"[modax jvp 7 dirs] N={N_K*Bc}: cold {t_cold:.1f} s, warm {t_jvp:.2f} s = {t_jvp/Bc:.2f} s/cosmo; tangent finite={bool(np.isfinite(np.asarray(yd)).all())}", flush=True)
        res[f"jvp7_N{N_K*Bc}"] = {"warm_s": t_jvp, "s_per_cosmo": t_jvp / Bc, "cold_s": t_cold}
    except Exception as ex:
        import traceback; traceback.print_exc()
        res[f"jvp7_N{N_K*Bc}"] = {"error": f"{type(ex).__name__}: {str(ex)[:300]}"}
if os.environ.get("DIFFRAX", "0") == "1":
    import diffrax, equinox as eqx
    def f_jax(t, y, p):
        return jnp.stack(ode_fn_jax(y, t, p))
    # jax version of the same RHS: swap math -> jnp and the table lookup -> jnp.interp
    src_jax = SRC.replace("math.exp", "jnp.exp").replace("math.sqrt", "jnp.sqrt").replace("def ode_fn", "def ode_fn_jax")
    pass
    nsj = {"jnp": jnp}
    exec(src_jax, nsj); ode_fn_jax = nsj["ode_fn_jax"]
    @eqx.filter_jit
    def chunk(y0c, Pc):
        def one(y0i, pi):
            ctrl = diffrax.PIDController(pcoeff=0.25, icoeff=0.8, dcoeff=0.0, rtol=RTOL, atol=ATOL, step_ts=jnp.asarray(T_SPAN))
            sol = diffrax.diffeqsolve(diffrax.ODETerm(f_jax), diffrax.Kvaerno5(), t0=LNA0, t1=LNA1, dt0=1e-2, y0=y0i,
                                      stepsize_controller=ctrl, max_steps=200_000, saveat=diffrax.SaveAt(ts=jnp.asarray(T_SPAN)),
                                      args=pi, adjoint=diffrax.ForwardMode())
            return sol.ys
        return jax.vmap(one)(y0c, Pc)
    for Bc in sorted({1, B}):
        y0, P = ensemble(Bc)
        def run():
            outs = [chunk(jnp.asarray(y0[i:i + 100]), jnp.asarray(P[i:i + 100])) for i in range(0, len(P), 100)]
            return jnp.concatenate(outs, 0)
        t0c = time.perf_counter(); yk = run(); jax.block_until_ready(yk); t_cold = time.perf_counter() - t0c
        t0w = time.perf_counter(); yk = run(); jax.block_until_ready(yk); t_warm = time.perf_counter() - t0w
        print(f"[diffrax Kvaerno5 vmap/100] N={N_K*Bc}: cold {t_cold:.1f} s, warm {t_warm:.2f} s = {t_warm/Bc:.2f} s/cosmo", flush=True)
        res[f"diffrax_N{N_K*Bc}"] = {"warm_s": t_warm, "s_per_cosmo": t_warm / Bc}
        if Bc == 1:
            ym = np.asarray(solve(ode_fn, jnp.asarray(y0), jnp.asarray(T_SPAN), jnp.asarray(P), rtol=RTOL, atol=ATOL, max_steps=200_000, sparsity=S, pcoeff=0.25, icoeff=0.8))
            yk = np.asarray(yk)
            scale = np.max(np.abs(yk), axis=(0, 1)) + 1e-300
            print(f"[modax vs diffrax] max peak-normalized |diff| = {np.max(np.abs(ym - yk) / scale):.3e}", flush=True)
print("RESULT " + json.dumps(res, default=str), flush=True)
