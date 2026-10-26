"""bbn_stage.py — batched LINX BBN for the profile driver.

The engine's own bbn_type="linx" path runs LINX (background thermodynamics + the nuclear network) inside
add_derived_parameters, one cosmology at a time on the CPU, and the driver's gradient block would then push
every (row x direction) tangent through it. Here the driver does the BBN step itself for all rows of a call at
once: (omega_b, Delta_Neff_init, tau_n_fac, q_1..q_nq) -> (YHe, Neff, T_nu_massless, Y_p(BBN), 1e5 D/H), the
three quantities the engine's LINX branch sets plus the two BBN observables, vmapped + jitted on one host CPU
device in chunks of B_CHUNK cosmologies, plus their 5 x n_in Jacobian by forward-mode AD. The engine then runs
with bbn_type="" (user-supplied YHe / Neff / T_nu_massless) and the driver's _phys_to_derived injects
y + J (theta - theta_row), whose jvp is the exact tangent by the chain rule; with the BBN likelihood on
(config bbn_like) the driver adds the Y_p and D/H terms and profiles tau_n_fac and the rate shifts q (one per
reaction, in sigmas of LINX's log-normal rate uncertainties) as parameters of every row.

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


OUT_NAMES = ("YHe", "Neff", "T_nu_massless", "YHe_BBN", "DH")   # the stage's outputs: the engine's three
                                                                 # (YHe = the CMB mass fraction) + the BBN
                                                                 # observables (4 Y_He4, 1e5 D/H)


class LinxStage:
    """x = (omega_b, Delta_Neff_init, tau_n_fac, q_1..q_nq) -> (YHe (CMB mass fraction), Neff,
    T_nu_massless / T_CMB, YHe_BBN, D/H x 1e5), batched on one host CPU device; values() and values_and_jac()
    take and return numpy. The inputs after the first two are LINX's BBN nuisances: the neutron-lifetime
    factor and one log-normal rate shift (in sigmas) per reaction of the network; a caller without them
    passes 1 and 0."""

    def __init__(self, reaction_net="key_PRIMAT_2023"):
        self.reaction_net = reaction_net
        self.bg = BackgroundModel()
        self.ab = AbundanceModel(NuclearRates(nuclear_net=reaction_net))
        self.nq = len(self.ab.nuclear_net.reactions)
        self.reaction_names = tuple(self.ab.nuclear_net.reactions_names)
        self.in_names = ("omega_b", "Delta_Neff_init", "tau_n_fac") + tuple(f"q{i}" for i in range(self.nq))
        self.n_in = len(self.in_names)
        self.cpus = jax.devices("cpu")
        self.dev = self.cpus[0]
        self._v = eqx.filter_jit(jax.vmap(self._one_x))   # values()
        self._v1 = eqx.filter_jit(self._one_x)             # per cosmology, unbatched: the check scripts
        self._vj = eqx.filter_jit(jax.vmap(self._one_vj))  # values_and_jac(): the batched Jacobian
        self._vj1 = eqx.filter_jit(self._one_vj)           # per cosmology: the fallback for lanes that fail batched
        self.n_fallback = 0                                # lanes recomputed one at a time so far (trace line)

    def x_default(self, omega_b, dneff):
        """(N, n_in) inputs with the nuisances at their reference values (tau_n_fac 1, q 0)."""
        ob = np.asarray(omega_b, float).reshape(-1); dn = np.asarray(dneff, float).reshape(-1)
        x = np.zeros((len(ob), self.n_in)); x[:, 0] = ob; x[:, 1] = dn; x[:, 2] = 1.0
        return x

    # ---- one cosmology: exactly the engine's bbn_type="linx" branch (abcmb/main.py) ----
    def _background(self, dneff):
        return self.bg(dneff, rtol=BG_RTOL, atol=BG_ATOL, max_steps=BG_MAX_STEPS, n_out=BG_N_OUT)

    def _rest(self, bg_out, eta_fac, tau_n_fac, q):
        """(YHe, Neff, T_nu, YHe_BBN, DH) from the background outputs, eta_fac = omega_b * Omegabh2_to_eta0
        / eta0, the neutron-lifetime factor and the rate shifts q (nq,)."""
        t_vec, a_vec, rho_g, rho_nu, rho_NP, P_NP, Neff_vec = bg_out
        Y = self.ab(rho_g, rho_nu, rho_NP, P_NP, t_vec=t_vec, a_vec=a_vec, eta_fac=eta_fac,
                    tau_n_fac=tau_n_fac, nuclear_rates_q=q,
                    rtol=AB_RTOL, atol=AB_ATOL, max_steps=AB_MAX_STEPS)
        YHe_BBN = 4 * Y[5]
        Yp_CMB = 1.0 / (4 * cnst.mH / cnst.mHe * (1 / YHe_BBN - 1) + 1)
        T_nu = linxThermo.T_nu(rho_nu[-1]) / linxThermo.T_g(rho_g[-1])
        DH = 1e5 * Y[2] / Y[1]
        return jnp.stack([Yp_CMB, Neff_vec[-1], T_nu, YHe_BBN, DH])

    _ETA = linxconst.Omegabh2_to_eta0 / linxconst.eta0

    def _one_x(self, x):
        return self._rest(self._background(x[1]), x[0] * self._ETA, x[2], x[3:])

    def _one_vj(self, x):
        """value (5,) and Jacobian (5, n_in). The Delta_Neff_init column goes through the background's jvp
        and the network; every other input enters the network only, so those columns come from forward-mode
        over the network alone with the background held (jax.jacfwd there is fine: the zero-tangent background
        solve that made jacfwd over ALL inputs go NaN, scan/.mm/linx_nan_scan.log, never happens)."""
        eta = x[0] * self._ETA
        bg_out, bg_dot = jax.jvp(self._background, (x[1],), (jnp.ones_like(x[1]),))
        y, y_dn = jax.jvp(lambda b: self._rest(b, eta, x[2], x[3:]), (bg_out,), (bg_dot,))
        bg_c = jax.lax.stop_gradient(bg_out)
        u0 = jnp.concatenate([x[:1], x[2:]])                      # (omega_b, tau_n_fac, q)
        Ju = jax.jacfwd(lambda u: self._rest(bg_c, u[0] * self._ETA, u[1], u[2:]))(u0)   # (5, n_in - 1)
        return y, jnp.concatenate([Ju[:, :1], y_dn[:, None], Ju[:, 1:]], axis=1)

    # ---- batching: chunks of <= B_CHUNK, each padded to a multiple of B_PAD (shape reuse) ----
    def _chunks(self, x):
        x = np.asarray(x, float).reshape(-1, self.n_in)
        for s in range(0, len(x), B_CHUNK):
            xc = x[s:s + B_CHUNK]; n = len(xc); m = -(-n // B_PAD) * B_PAD
            if m > n:
                xc = np.concatenate([xc, np.repeat(xc[-1:], m - n, 0)])
            yield n, jax.device_put(jnp.asarray(xc), self.dev)

    def _per_lane(self, fn, x, n_dev):
        """fn on each row of x (B, n_in), one jitted program per row dispatched asynchronously round-robin
        over n_dev host devices; identical rows (the driver pads its batches by repeating a row) run once."""
        n = min(n_dev, len(self.cpus))
        uniq, inv = np.unique(np.round(x, 15), axis=0, return_inverse=True)
        outs = [fn(jax.device_put(jnp.asarray(uniq[k]), self.cpus[k % n])) for k in range(len(uniq))]
        return outs, np.asarray(inv).reshape(-1)

    @staticmethod
    def n_reactions(reaction_net="key_PRIMAT_2023"):
        """number of reactions (= rate-shift nuisances q) of a network, without building the models."""
        return len(NuclearRates(nuclear_net=reaction_net).reactions)

    def values(self, x):
        """(B, 5) = [YHe, Neff, T_nu_massless, YHe_BBN, DH] for the rows of x (B, n_in), the vmapped program on
        one host device in chunks (~0.5 s per cosmology warm)."""
        return np.concatenate([np.array(self._v(xc), dtype=float)[:n] for n, xc in self._chunks(x)])

    def values_and_jac(self, x):
        """((B, 5) values, (B, 5, n_in) Jacobian d[outputs] / d[inputs]) for the rows of x (B, n_in) from the
        batched Jacobian program in chunks (the primal of the same program; it agrees with values() to ~5e-8
        in YHe). A lane that comes back non-finite (in EQX_ON_ERROR=nan mode a solver failure in ANY lane of a
        chunk nan-s the whole chunk: equinox reduces the error predicate over the batch) is recomputed one
        cosmology at a time with the un-vmapped program over JAC_DEVICES host devices; a lane still non-finite
        alone stays NaN (the driver treats the row as failed)."""
        ys, Js = [], []
        for n, xc in self._chunks(x):
            y, J = self._vj(xc)
            y = np.array(y, dtype=float)[:n]; J = np.array(J, dtype=float)[:n]
            bad = ~(np.isfinite(y).all(1) & np.isfinite(J).all((1, 2)))
            if bad.any():
                xb = np.asarray(xc)[:n][bad]
                outs, inv = self._per_lane(self._vj1, xb, JAC_DEVICES)
                y[bad] = np.array([np.asarray(o[0], dtype=float) for o in outs]).reshape(-1, len(OUT_NAMES))[inv]
                J[bad] = np.array([np.asarray(o[1], dtype=float) for o in outs]).reshape(-1, len(OUT_NAMES), self.n_in)[inv]
                self.n_fallback += int(bad.sum())
                print(f"[bbn] {int(bad.sum())} of {n} Jacobian lanes recomputed alone "
                      f"({int((~np.isfinite(J[bad]).all((1, 2))).sum())} still non-finite)", flush=True)
            ys.append(y); Js.append(J)
        return np.concatenate(ys), np.concatenate(Js)
