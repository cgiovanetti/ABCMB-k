"""bbn_stage.py — batched LINX BBN for the profile driver.

The engine's own bbn_type="linx" path runs LINX (background thermodynamics + the nuclear network) inside
add_derived_parameters, one cosmology at a time on the CPU, and the driver's gradient block would then push
every (row x direction) tangent through it. Here the driver does the BBN step itself for all rows of a call at
once: (omega_b, Delta_Neff_init) -> (YHe, Neff, T_nu_massless), the three quantities the engine's LINX branch
sets, vmapped + jitted on one host CPU device in chunks of B_CHUNK cosmologies, plus their 3x2 Jacobian by
forward-mode AD. The engine then runs with bbn_type="" (user-supplied YHe / Neff / T_nu_massless) and the
driver's _phys_to_derived injects y + J (theta - theta_row), whose jvp is the exact tangent by the chain rule.

One device: a vmap over 64 cosmologies runs in ~23 s warm on a Perlmutter CPU node (XLA threads the batched
linear algebra); splitting the batch over 16 host devices is slower (every device's program competes for the
same cores, 31 s, plus one compile per device), a sharded Mesh turns the ODE loops' batch-wide stop test into
a cross-device all-reduce per step that times out, and pmap trips lineax's linear_solve primitive
(ravel_vector "pytree does not match out_structure", lineax 0.0.x / jax 0.8.1). Solver settings: the
constants below (the network 100x tighter than the engine's; see SMOOTHNESS).

The Jacobian is one batched two-jvp program (values_and_jac) since the vendored LINX hands the network a
dense fixed-size grid (below): on the old step grid the vmapped jvp through the implicit (Kvaerno) network
solve raised for any batch of more than one lane (scan/.mm/linxdev/batched_jvp_matrix.log) and the
Jacobians had to run one cosmology at a time, nearly serially (~6 s per row). Note EQX_ON_ERROR=nan (the
driver sets it before importing equinox, so a failed solve is a NaN rather than an abort): equinox reduces
the error predicate over the batch, so one failing lane nan-s its whole chunk; values_and_jac recomputes
such lanes alone. The per-cosmology jvp is never jax.jacfwd: its vmap over the two tangent directions pushes
an identically zero tangent through the background solve on the omega_b lane and ends in a NaN on some
Delta_Neff_init values (scan/.mm/linx_nan_scan.log); _one_vj does two single jvps instead.

SMOOTHNESS (2026-10-22). With LINX as published, YHe is rough in Delta_Neff_init at 1.5e-5 over +-0.3 sigma
(1e-3..1e-2 in chi2 with dchi2/dYHe ~ 100-1000) and biased low by 8.5e-5; in the first _v4linx run that
rejected LM steps once the predicted drop reached ~1e-2 (7-16 stalled rows per rank at ||g|| 0.15-0.4, none
in _v3 with fixed YHe) and the 6 h job timed out. Not a tolerance: the network's rtol does not touch it and
the background only matches at rtol 1e-10 at 6x the time. It is the background's output GRID: its ~300
adaptive steps, on which the network interpolated a(t) and T(t) linearly in t, and which move with
Delta_Neff_init. Fixed inside the vendored LINX (abcmb/linx/background.py: SaveAt(dense=True) and n_out
log-spaced times read from the dense interpolant; abundances.py: log-log interpolation): roughness 5e-8,
YHe 0.244238 = the converged value, no change in cost, fewer network steps. A stage-side cubic re-grid of
the raw steps did the same from 2026-10-22 morning until this. What is left is the background's own global
error in T_nu / T_g (7e-6 in Neff, ~1e-4 in chi2). scan/.mm/linx_roughness.log, linx_bg_roughness.log,
linxdev/val12_pinned.log.

Delta_Neff_init is LINX's Delta Neff at the START of BBN (inert radiation at T_start): the cosmology's Neff is
~ 3.044 + 0.262 * Delta_Neff_init and is read back from the background solution, never computed from the input.
"""
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
from abcmb.linx.background import BackgroundModel
from abcmb.linx.abundances import AbundanceModel
from abcmb.linx.nuclear import NuclearRates
from abcmb.linx import const as linxconst
from abcmb.linx import thermo as linxThermo
from abcmb import constants as cnst

jax.config.update("jax_enable_x64", True)

# optimistix's implicit-function-theorem jvp of a root find (the network's implicit Kvaerno stages) runs
# lineax.linear_solve with throw=True even though diffrax calls the root find with throw=False: a Newton
# stage that diverged, which the primal handles by rejecting the step and retrying smaller, then raises
# inside the TANGENT ("A linear solver received non-finite input"), and in EQX_ON_ERROR=nan mode nan-s
# the whole batch. It is lane- and compilation-dependent (any batch of ~64 cosmologies hits it, so did
# jax.jacfwd and, occasionally, single lanes). Non-throwing solves process-wide: the garbage tangent of a
# rejected stage is masked by the rejection exactly as its primal is (scan/.mm/linxdev/nothrow_test.log:
# batched = per-lane Jacobians to 1e-4). ABCMB's own implicit solves are covered by the same rule.
_lx_linear_solve = lx.linear_solve


def _linear_solve_nothrow(*args, **kwargs):
    kwargs.setdefault("throw", False)
    return _lx_linear_solve(*args, **kwargs)


lx.linear_solve = _linear_solve_nothrow

B_CHUNK = 64            # cosmologies per vmapped values call (bounds memory and the union-of-steps of a vmapped
B_PAD = 16              # solve); chunks are padded to a multiple of B_PAD so only 4 batch shapes ever compile
JAC_DEVICES = 4         # host devices the per-cosmology Jacobian programs are spread over. Each device compiles
                        # its own copy (~2 min, and the persistent cache cannot store them: host callbacks), so
                        # more devices buy concurrency at a fixed cost: for ~10 gradient calls of 64 rows at ~4 s
                        # per lane, 4 devices minimise compile + run time (~20 min total; 16 devices ~35 min)
BG_RTOL, BG_ATOL = 1e-8, 1e-10   # the background: the engine's defaults (tighter buys nothing now: SMOOTHNESS)
BG_MAX_STEPS = 512
BG_N_OUT = 4096                  # the background's dense-interpolant output grid (abcmb/linx/background.py n_out)
AB_RTOL, AB_ATOL = 1e-8, 1e-11   # the network: 100x tighter than the engine's 1e-6 / 1e-9 for +12% time; YHe
AB_MAX_STEPS = 16384             # roughness along omega_b 7.8e-7 -> 3.4e-8; ~5000 steps at 1e-8 (65536 is the
                                 # budget the step counts were measured with; scan/.mm/linxdev/steps_dev.log)


class LinxStage:
    """(omega_b, Delta_Neff_init) -> (YHe (CMB mass fraction), Neff, T_nu_massless / T_CMB), batched on one
    host CPU device; values() and values_and_jac() take and return numpy."""

    def __init__(self, reaction_net="key_PRIMAT_2023"):
        self.bg = BackgroundModel()
        self.ab = AbundanceModel(NuclearRates(nuclear_net=reaction_net))
        self.nq = len(self.ab.nuclear_net.reactions)
        self.cpus = jax.devices("cpu")
        self.dev = self.cpus[0]
        self._v = eqx.filter_jit(jax.vmap(self._one_x))   # values()
        self._v1 = eqx.filter_jit(self._one_x)             # per cosmology, unbatched: the check scripts
        self._vj = eqx.filter_jit(jax.vmap(self._one_vj))  # values_and_jac(): the batched two-jvp Jacobian
        self._vj1 = eqx.filter_jit(self._one_vj)           # per cosmology: the fallback for lanes that fail batched
        self.n_fallback = 0                                # lanes recomputed one at a time so far (trace line)

    # ---- one cosmology: exactly the engine's bbn_type="linx" branch (abcmb/main.py) ----
    def _background(self, dneff):
        return self.bg(dneff, rtol=BG_RTOL, atol=BG_ATOL, max_steps=BG_MAX_STEPS, n_out=BG_N_OUT)

    def _rest(self, bg_out, eta_fac):
        """(YHe, Neff, T_nu) from the background outputs and eta_fac = omega_b * Omegabh2_to_eta0 / eta0."""
        t_vec, a_vec, rho_g, rho_nu, rho_NP, P_NP, Neff_vec = bg_out
        Y = self.ab(rho_g, rho_nu, rho_NP, P_NP, t_vec=t_vec, a_vec=a_vec, eta_fac=eta_fac,
                    tau_n_fac=jnp.asarray(1.0), nuclear_rates_q=jnp.zeros(self.nq),
                    rtol=AB_RTOL, atol=AB_ATOL, max_steps=AB_MAX_STEPS)
        YHe_BBN = 4 * Y[5]
        Yp_CMB = 1.0 / (4 * cnst.mH / cnst.mHe * (1 / YHe_BBN - 1) + 1)
        T_nu = linxThermo.T_nu(rho_nu[-1]) / linxThermo.T_g(rho_g[-1])
        return jnp.stack([Yp_CMB, Neff_vec[-1], T_nu])

    _ETA = linxconst.Omegabh2_to_eta0 / linxconst.eta0

    def _one(self, omega_b, dneff):
        return self._rest(self._background(dneff), omega_b * self._ETA)

    def _one_x(self, x):
        return self._one(x[0], x[1])

    def _one_vj(self, x):
        """value and the 3x2 Jacobian from two single-direction jvps. Not jax.jacfwd: its vmap over the two
        tangent directions pushes an identically zero tangent through the background solve on the omega_b
        lane, and on some Delta_Neff_init values that ends in a NaN reaching the stiff solver's linear solve
        (scan/.mm/linx_nan_scan.log, linx_jac_localize.log); each single jvp is finite there. The background
        depends on Delta_Neff_init only, the network on (background, eta_fac), so the omega_b direction never
        enters the background at all."""
        omega_b, dneff = x[0], x[1]
        eta = omega_b * self._ETA
        bg_out, bg_dot = jax.jvp(self._background, (dneff,), (jnp.ones_like(dneff),))
        y, y_dn = jax.jvp(lambda b: self._rest(b, eta), (bg_out,), (bg_dot,))
        bg_const = jax.lax.stop_gradient(bg_out)
        _, y_ob = jax.jvp(lambda e: self._rest(bg_const, e), (eta,), (jnp.asarray(self._ETA, dtype=eta.dtype),))
        return y, jnp.stack([y_ob, y_dn], axis=1)

    # ---- batching: chunks of <= B_CHUNK, each padded to a multiple of B_PAD (shape reuse) ----
    def _chunks(self, omega_b, dneff):
        x = np.stack([np.asarray(omega_b, float), np.asarray(dneff, float)], 1)
        for s in range(0, len(x), B_CHUNK):
            xc = x[s:s + B_CHUNK]; n = len(xc); m = -(-n // B_PAD) * B_PAD
            if m > n:
                xc = np.concatenate([xc, np.repeat(xc[-1:], m - n, 0)])
            yield n, jax.device_put(jnp.asarray(xc), self.dev)

    def _per_lane(self, fn, x, n_dev):
        """fn on each row of x (B, 2), one jitted program per row dispatched asynchronously round-robin over
        n_dev host devices; identical rows (the driver pads its batches by repeating a row) run once."""
        n = min(n_dev, len(self.cpus))
        uniq, inv = np.unique(np.round(x, 15), axis=0, return_inverse=True)
        outs = [fn(jax.device_put(jnp.asarray(uniq[k]), self.cpus[k % n])) for k in range(len(uniq))]
        return outs, np.asarray(inv).reshape(-1)

    def values(self, omega_b, dneff):
        """(B, 3) = [YHe, Neff, T_nu_massless] per cosmology, the vmapped program on one host device in chunks
        (~0.5 s per cosmology warm)."""
        return np.concatenate([np.array(self._v(xc), dtype=float)[:n] for n, xc in self._chunks(omega_b, dneff)])

    def values_and_jac(self, omega_b, dneff):
        """((B, 3) values, (B, 3, 2) d[YHe, Neff, T_nu] / d[omega_b, Delta_Neff_init]) from the batched two-jvp
        program in chunks (the primal of the same program; it agrees with values() to ~5e-8 in YHe). A lane
        that comes back non-finite (in EQX_ON_ERROR=nan mode a solver failure in ANY lane of a chunk nan-s the
        whole chunk: equinox reduces the error predicate over the batch) is recomputed one cosmology at a time
        with the un-vmapped program over JAC_DEVICES host devices; a lane still non-finite alone stays NaN
        (the driver treats the row as failed)."""
        ys, Js = [], []
        for n, xc in self._chunks(omega_b, dneff):
            y, J = self._vj(xc)
            y = np.array(y, dtype=float)[:n]; J = np.array(J, dtype=float)[:n]
            bad = ~(np.isfinite(y).all(1) & np.isfinite(J).all((1, 2)))
            if bad.any():
                xb = np.asarray(xc)[:n][bad]
                outs, inv = self._per_lane(self._vj1, xb, JAC_DEVICES)
                y[bad] = np.array([np.asarray(o[0], dtype=float) for o in outs]).reshape(-1, 3)[inv]
                J[bad] = np.array([np.asarray(o[1], dtype=float) for o in outs]).reshape(-1, 3, 2)[inv]
                self.n_fallback += int(bad.sum())
                print(f"[bbn] {int(bad.sum())} of {n} Jacobian lanes recomputed alone "
                      f"({int((~np.isfinite(J[bad]).all((1, 2))).sum())} still non-finite)", flush=True)
            ys.append(y); Js.append(J)
        return np.concatenate(ys), np.concatenate(Js)
