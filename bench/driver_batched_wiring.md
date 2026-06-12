# PA_GRADMETHOD=batched wiring plan (scan/profile_prod_ad.py)

The batched gradient module is `scan/batched_grad.py`:
- `staged_cl_and_grad(model, full_ps, params_dots, k_chunk_size)` -> (Cl, [dCl per dir])
- `staged_chi2_and_grad(model, full_ps, params_dots, chi2_of_cls, k_chunk_size)`
  -> (chi2 (B,), grad (B,P))   [NEW this session]

## What the driver needs
`iterate_fg(poi_idx, X, PV, vg, method)` returns (F (B,), G (B,5)) where G is
dchi2/dx5 in SCALED nuisance coords (x5 = (theta - CEN)/SIG on the 5 non-POI params).
The "loop"/"vmap" methods do single-cosmology jacfwd per grid point. The "batched"
method rides the per-k pipeline.

## New code (driver side)

```python
# chi2 as a pure function of the (batched) Cls -- envelope-profiled A_planck
# (stop_gradient) + low-ell. Mirrors chi2_scaled_single's body but on Cls.
def chi2_of_cls(ClTT, ClTE, ClEE):
    l = model.SS.ells
    Dtt = pl.abcmb_cl_to_Dl(ClTT, l); Dte = pl.abcmb_cl_to_Dl(ClTE, l)
    Dee = pl.abcmb_cl_to_Dl(ClEE, l)
    m0 = pl.bin_model(Dtt, Dte, Dee)                       # (B, ndata)
    A_star = jax.lax.stop_gradient(pl.profile_A(m0, with_prior=True)[1])   # (B,)
    diff = pl.X_data - m0 / (A_star[..., None] ** 2)
    c2 = jnp.einsum("...i,ij,...j->...", diff, pl.invcov, diff) \
         + ((A_star - 1.0) / 0.0025) ** 2
    if lowee is not None: c2 = c2 + lowee.chi2(Dee)
    if lowtt is not None: c2 = c2 + lowtt.chi2(Dtt)
    return c2                                              # (B,)

# physical->derived + the 5 SCALED nuisance tangents, per grid point
from scan.batched_grad import _to_float as _bg_to_float
def _phys_to_derived(th6):                                 # th6: (6,) physical
    p = dict(FIXED)
    p['h'] = th6[0]; p['omega_b'] = th6[1]; p['omega_cdm'] = th6[2]
    p['n_s'] = th6[3]; p['A_s'] = jnp.exp(th6[4]) / 1e10; p['tau_reion'] = th6[5]
    # MUST _to_float the DERIVED dict: staged_cl_and_grad _to_float's the primal,
    # so int derived keys (N_nu_massive = jnp.array(1), main.py:653) must be float
    # here too or the jvp tangent (None after inexact-filter) mismatches the
    # float-array primal partition. (FIXED carries N_nu_massive as a python int.)
    return _bg_to_float(model.add_derived_parameters(p))

def batched_grad_fg(poi_idx, X, PV):
    import equinox as eqx
    from scan.batched_grad import staged_chi2_and_grad
    nuis = [i for i in range(6) if i != poi_idx]
    B = len(PV)
    thetas = [jnp.asarray(assemble_phys(poi_idx, X[b], PV[b])) for b in range(B)]
    full_ps = [_phys_to_derived(t) for t in thetas]
    # per-cosmo tangents in the 5 scaled directions (tangent = SIG[i] e_i)
    per = []
    for t in thetas:
        dots = []
        for i in nuis:
            tan = jnp.zeros(6).at[i].set(SIG[i])
            _, fd = jax.jvp(_phys_to_derived, (t,), (tan,))
            dots.append(eqx.filter(fd, eqx.is_inexact_array))
        per.append(dots)
    params_dots = [jax.tree.map(lambda *xs: jnp.stack(xs),
                                *[per[b][j] for b in range(B)]) for j in range(5)]
    chi2, grad = staged_chi2_and_grad(model, full_ps, params_dots, chi2_of_cls,
                                      k_chunk_size=GRAD_KCHUNK)
    return np.asarray(chi2), np.asarray(grad)              # (B,), (B,5)
```

Then in `iterate_fg`, add `if method == "batched": return batched_grad_fg(poi_idx, X, PV)`.
NOTE the driver keeps VALUES on the fast call_batched path (fast_values) for the
Armijo line search; the batched grad supplies only G (and a chi2 we can ignore or
cross-check). Consistency rule (commit 76127ca) preserved.

## Open: SHARDING the gradient (multi-GPU)
staged_cl_and_grad currently builds on gpu[0]. To shard like call_batched, wrap the
stacked primal+tangent pytrees with call_batched's shardfn BEFORE the stages and pad
B to a multiple of n_dev. Defer until single-GPU path validated + tractable.

## Gate before wiring: production-shape compile must be bounded (grad_prod_shape.py).
GRAD_KCHUNK chosen from the k_chunk sweep (smaller cuts compile; balance vs warm).
