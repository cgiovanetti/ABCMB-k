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
(ravel_vector "pytree does not match out_structure", lineax 0.0.x / jax 0.8.1). The LINX solver settings
are the engine's (max_steps 512 / 4096): raising them changes nothing (scan/.mm/linx_nan_scan.log).

The Jacobian is NOT vmapped: the jvp through the implicit (Kvaerno) network solve goes wrong under vmap on
lane-specific inputs (about one in five, with a batch of one as well, never without vmap;
scan/.mm/linx_jac_bisect.log), and in a batch one such lane poisons every lane through the loop's
batch-wide reductions (all 64 NaN at B = 64, linx_jac_bisect3.log); with errors left as exceptions
lineax's error_if aborts the call ("A linear solver received non-finite input"). The un-vmapped jvp is
exact on every lane (~4 s each), so values_and_jac runs one jitted jvp program per cosmology, dispatched
asynchronously round-robin over JAC_DEVICES host CPU devices so they run concurrently. Each device
compiles its own copy on first use in every process (~2 min; the persistent compilation cache refuses
these executables, they carry host callbacks), hence the small default. The vmapped values are fine
(they equal the per-lane ones on every lane). EQX_ON_ERROR=nan (the driver sets it before importing
equinox) keeps a failed solve a NaN rather than an abort.

Delta_Neff_init is LINX's Delta Neff at the START of BBN (inert radiation at T_start): the cosmology's Neff is
~ 3.045 + 0.272 * Delta_Neff_init and is read back from the background solution, never computed from the input.
"""
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
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
BG_MAX_STEPS = 512      # the engine's defaults
AB_MAX_STEPS = 4096


class LinxStage:
    """(omega_b, Delta_Neff_init) -> (YHe (CMB mass fraction), Neff, T_nu_massless / T_CMB), batched on one
    host CPU device; values() and values_and_jac() take and return numpy."""

    def __init__(self, reaction_net="key_PRIMAT_2023"):
        self.bg = BackgroundModel()
        self.ab = AbundanceModel(NuclearRates(nuclear_net=reaction_net))
        self.nq = len(self.ab.nuclear_net.reactions)
        self.cpus = jax.devices("cpu")
        self.dev = self.cpus[0]
        self._v = eqx.filter_jit(jax.vmap(self._one_x))
        self._vj1 = eqx.filter_jit(self._one_vj)       # per cosmology, never vmapped (module docstring)
        self.n_fallback = 0                            # kept for the driver's trace line (always 0 now)

    # ---- one cosmology: exactly the engine's bbn_type="linx" branch (abcmb/main.py) ----
    def _background(self, dneff):
        return self.bg(dneff, max_steps=BG_MAX_STEPS)

    def _rest(self, bg_out, eta_fac):
        """(YHe, Neff, T_nu) from the background outputs and eta_fac = omega_b * Omegabh2_to_eta0 / eta0."""
        t_vec, a_vec, rho_g, rho_nu, rho_NP, P_NP, Neff_vec = bg_out
        Y = self.ab(rho_g, rho_nu, rho_NP, P_NP, t_vec=t_vec, a_vec=a_vec, eta_fac=eta_fac,
                    tau_n_fac=jnp.asarray(1.0), nuclear_rates_q=jnp.zeros(self.nq), max_steps=AB_MAX_STEPS)
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

    def values(self, omega_b, dneff):
        """(B, 3) = [YHe, Neff, T_nu_massless] per cosmology."""
        return np.concatenate([np.array(self._v(xc), dtype=float)[:n] for n, xc in self._chunks(omega_b, dneff)])

    def values_and_jac(self, omega_b, dneff):
        """((B, 3) values, (B, 3, 2) d[YHe, Neff, T_nu] / d[omega_b, Delta_Neff_init]): one un-vmapped jvp
        program per cosmology, dispatched asynchronously round-robin over the host devices."""
        x = np.stack([np.asarray(omega_b, float), np.asarray(dneff, float)], 1)
        n = min(JAC_DEVICES, len(self.cpus))
        outs = [self._vj1(jax.device_put(jnp.asarray(x[k]), self.cpus[k % n])) for k in range(len(x))]
        y = np.array([np.asarray(o[0], dtype=float) for o in outs]).reshape(-1, 3)
        J = np.array([np.asarray(o[1], dtype=float) for o in outs]).reshape(-1, 3, 2)
        return y, J
