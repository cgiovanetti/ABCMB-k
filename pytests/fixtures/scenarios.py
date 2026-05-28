"""
Scenario definitions for Phase C snapshot fixtures.

Each scenario is a (specs, user_species_callable, params) triple that produces
one Model run. The set is designed to cover every branch in
abcmb.main.add_derived_parameters and abcmb.main.Model construction that the
upcoming refactor must preserve under batching:

  - lcdm_tau          : default reion parameterization (input_tau_reion=True)
  - lcdm_z            : alternate reion parameterization (input_tau_reion=False)
  - lcdm_massive_nu   : one species.MassiveNeutrino in user_species
  - bbn_table         : bbn_type="table" branch (PArthENoPE interp)
  - bbn_linx          : bbn_type="linx" branch (slowest; can be skipped)

`user_species` is a callable returning a tuple, because importing
abcmb.species at scenario definition time would force JAX startup. The
generator imports + calls it.
"""

# kept small for fast generation; we just need branch coverage.
ELLMAX = 800


def _lcdm_tau_specs():
    return dict(
        output_Cl=True, l_max=ELLMAX, lensing=False,
        output_Pk=True, output_k_max=0.5,
        l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
        input_tau_reion=True,
    )


def _lcdm_z_specs():
    s = _lcdm_tau_specs()
    s["input_tau_reion"] = False
    return s


def _lcdm_massive_nu_specs():
    s = _lcdm_tau_specs()
    return s  # no spec change; user_species carries the change


def _bbn_table_specs():
    s = _lcdm_tau_specs()
    s["bbn_type"] = "table"
    return s


def _bbn_linx_specs():
    s = _lcdm_tau_specs()
    s["bbn_type"] = "linx"
    return s


def _user_species_none():
    return None


def _user_species_massive_nu():
    from abcmb import species
    return (species.MassiveNeutrino,)


def _params_lcdm_tau():
    return {
        'h': 0.6736, 'omega_cdm': 0.120, 'omega_b': 0.02237,
        'A_s': 2.1e-9, 'n_s': 0.9649, 'Neff': 3.044, 'YHe': 0.245,
        'TCMB0': 2.34865418e-4,
        'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
        'tau_reion': 0.0544, 'Delta_z_reion': 0.5,
        'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
    }


def _params_lcdm_z():
    p = _params_lcdm_tau()
    del p['tau_reion']
    p['z_reion'] = 7.67
    return p


def _params_lcdm_massive_nu():
    p = _params_lcdm_tau()
    p['N_nu_massive'] = 1
    return p


def _params_bbn_table():
    # bbn_type=table interpolates YHe from omega_b + Neff. Do NOT supply YHe.
    p = _params_lcdm_tau()
    del p['YHe']
    return p


def _params_bbn_linx():
    # LINX computes Neff and YHe. Do NOT supply N_nu_massless, Neff,
    # T_nu_massless, or YHe.
    p = _params_lcdm_tau()
    del p['Neff']
    del p['YHe']
    p['Delta_Neff_init'] = 0.0
    return p


SCENARIOS = {
    "lcdm_tau": dict(
        specs=_lcdm_tau_specs, user_species=_user_species_none,
        params=_params_lcdm_tau,
    ),
    "lcdm_z": dict(
        specs=_lcdm_z_specs, user_species=_user_species_none,
        params=_params_lcdm_z,
    ),
    "lcdm_massive_nu": dict(
        specs=_lcdm_massive_nu_specs, user_species=_user_species_massive_nu,
        params=_params_lcdm_massive_nu,
    ),
    "bbn_table": dict(
        specs=_bbn_table_specs, user_species=_user_species_none,
        params=_params_bbn_table,
    ),
    "bbn_linx": dict(
        specs=_bbn_linx_specs, user_species=_user_species_none,
        params=_params_bbn_linx,
    ),
}

# Order matters for stable snapshot keying.
SCENARIO_NAMES = list(SCENARIOS.keys())
