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

The Jacobian is NOT vmapped: the jvp through the implicit (Kvaerno) network solve goes wrong under vmap on
lane-specific inputs (about one in five, with a batch of one as well, never without vmap;
scan/.mm/linx_jac_bisect.log), and in a batch one such lane poisons every lane through the loop's
batch-wide reductions (all 64 NaN at B = 64, linx_jac_bisect3.log); with errors left as exceptions
lineax's error_if aborts the call ("A linear solver received non-finite input"). The un-vmapped jvp is
exact on every lane (~4 s each), so values_and_jac runs one jitted jvp program per cosmology, dispatched
asynchronously round-robin over JAC_DEVICES host CPU devices so they run concurrently. Each device
compiles its own copy on first use in every process (~2 min; the persistent compilation cache refuses
these executables, they carry host callbacks), hence the small default. EQX_ON_ERROR=nan (the driver
sets it before importing equinox) keeps a failed solve a NaN rather than an abort.

SMOOTHNESS (2026-10-22). With LINX as the engine runs it, YHe is rough in Delta_Neff_init at 1.5e-5 over
+-0.3 sigma (1e-3..1e-2 in chi2 with dchi2/dYHe ~ 100-1000) and every distinct executable disagrees with
every other at the same ~1e-5 level; in the first _v4linx run that rejected LM steps once the predicted
drop reached ~1e-2: 7-16 stalled rows per rank at ||g|| 0.15-0.4 (none in _v3 with fixed YHe), spurious
flags, deflation walks and branches, and the 6 h job timed out. The roughness is not a tolerance: the
network's rtol does not touch it (1e-6, 1e-8, 1e-9 all give 1.5e-5) and the background only reaches it at
rtol 1e-10 (16384-step budget, ~1900 steps, 8e-8 at 6x the time per cosmology; 1e-9: 3.8e-7 at 3.6x;
the 2048 / 4096 budgets fail). It is the background's output GRID: the ~300 steps the network interpolates
LINEARLY in t (abundances.py dY: a(t), T(t); mid-step error ~1e-4 in a) and which move with
Delta_Neff_init. _regrid re-samples the background onto N_FINE log-spaced times with a cubic before the
network sees it: roughness 8e-8 along Delta_Neff_init, 2e-8 along omega_b (network at 1e-8 / 1e-11),
executables within 3e-8..8e-8 of each other, +25% time; the re-gridded YHe sits 6-8e-5 above the raw-grid
one (the interpolation bias). What is left is the background's own global error in T_nu / T_g (6e-7 in
T_nu, 7e-6 in Neff, ~1e-4 in chi2), untouched by the re-grid. scan/.mm/linx_roughness.log,
linx_bg_roughness.log, linx_regrid_check.log.

Delta_Neff_init is LINX's Delta Neff at the START of BBN (inert radiation at T_start): the cosmology's Neff is
~ 3.045 + 0.272 * Delta_Neff_init and is read back from the background solution, never computed from the input.
"""
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from interpax import interp1d
from abcmb.linx.background import BackgroundModel
from abcmb.linx.abundances import AbundanceModel
from abcmb.linx.nuclear import NuclearRates
from abcmb.linx import const as linxconst
from abcmb.linx import thermo as linxThermo
from abcmb import constants as cnst

jax.config.update("jax_enable_x64", True)

B_CHUNK = 64            # cosmologies per vmapped values call (bounds memory and the union-of-steps of a vmapped
B_PAD = 16              # solve); chunks are padded to a multiple of B_PAD so only 4 batch shapes ever compile
JAC_DEVICES = 4         # host devices the per-cosmology Jacobian programs are spread over. Each device compiles
                        # its own copy (~2 min, and the persistent cache cannot store them: host callbacks), so
                        # more devices buy concurrency at a fixed cost: for ~10 gradient calls of 64 rows at ~4 s
                        # per lane, 4 devices minimise compile + run time (~20 min total; 16 devices ~35 min)
BG_RTOL, BG_ATOL = 1e-8, 1e-10   # the background: the engine's defaults. Tighter needs a bigger step budget (rtol
BG_MAX_STEPS = 512               # 1e-10: ~1900 steps, 6x the time; scan/.mm/linx_regrid_check.log) -- see N_FINE
AB_RTOL, AB_ATOL = 1e-8, 1e-11   # the network: 100x tighter than the engine's 1e-6 / 1e-9 for +12% time; YHe
AB_MAX_STEPS = 8192              # roughness along omega_b 7.8e-7 -> 3.4e-8 (scan/.mm/linx_roughness.log)
N_FINE = 8192           # the background's ~300 adaptive steps are re-gridded (cubic) to this many log-spaced
                        # times before the network sees them (_regrid): the network interpolates LINEARLY in t
                        # on whatever grid it is given, and on the raw step grid that error (~1e-4 in a(t)
                        # mid-step) moves with the step sequence, i.e. with Delta_Neff_init: 1.5e-5 roughness
                        # in YHe (1e-3..1e-2 in chi2) that no tolerance touches (scan/.mm/linx_bg_roughness.log)


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
        self._vj1 = eqx.filter_jit(self._one_vj)           # per cosmology: the Jacobian (module docstring)
        self.n_fallback = 0                                # kept for the driver's trace line (always 0)

    # ---- one cosmology: exactly the engine's bbn_type="linx" branch (abcmb/main.py) ----
    def _background(self, dneff):
        return self._regrid(self.bg(dneff, rtol=BG_RTOL, atol=BG_ATOL, max_steps=BG_MAX_STEPS))

    @staticmethod
    def _regrid(bg_out):
        """The background's (t, a, rho_g, rho_nu, rho_NP, P_NP, Neff) on its raw step grid (t[0] = 0, then the
        adaptive steps, then the event step repeated up to max_steps) -> the same on t[0] plus N_FINE - 1
        log-spaced times up to the event step: log a, log rho_g, log rho_nu cubic in log t (interpax), the
        repeated tail first continued as power laws so the cubic sees distinct smooth nodes; rho_NP, P_NP from
        their exact a^-4 law; Neff untouched (only its last entry is used). The last fine node is the event step,
        so [-1] values are unchanged."""
        t, a, rg, rn, rNP, PNP, Neff = bg_out
        n = t.shape[0]; i = jnp.arange(n)
        last = jnp.argmax(t)                                   # first index of the repeated tail
        lt, la, lg, ln = jnp.log(t), jnp.log(a), jnp.log(rg), jnp.log(rn)
        k = jnp.maximum(i - last, 0)
        ext = lambda f: jnp.where(i > last, f[last] + k * (f[last] - f[last - 1]), f)
        ltx = ext(lt)
        ltq = jnp.linspace(lt[1], lt[last], N_FINE - 1)
        fq = interp1d(ltq, ltx[1:], jnp.stack([ext(la), ext(lg), ext(ln)], 1)[1:], method="cubic")
        af = jnp.concatenate([a[:1], jnp.exp(fq[:, 0])])
        rgf = jnp.concatenate([rg[:1], jnp.exp(fq[:, 1])])
        rnf = jnp.concatenate([rn[:1], jnp.exp(fq[:, 2])])
        tf = jnp.concatenate([t[:1], jnp.exp(ltq)])
        rNPf = rNP[0] * (a[0] / af) ** 4
        return tf, af, rgf, rnf, rNPf, rNPf / 3, Neff

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
        """((B, 3) values, (B, 3, 2) d[YHe, Neff, T_nu] / d[omega_b, Delta_Neff_init]): the values from
        values(), the Jacobian from one un-vmapped two-jvp program per cosmology over JAC_DEVICES devices
        (its primal agrees with values() to ~5e-8 in YHe, scan/.mm/linx_regrid_check.log)."""
        x = np.stack([np.asarray(omega_b, float), np.asarray(dneff, float)], 1)
        y = self.values(omega_b, dneff)
        outs, inv = self._per_lane(self._vj1, x, JAC_DEVICES)
        J = np.array([np.asarray(o[1], dtype=float) for o in outs]).reshape(-1, 3, 2)
        return y, J[inv]
