import jax.numpy as jnp
import numpy as np
import equinox as eqx

print("Testing which Reaction field causes the warning...")

# Simulate Reaction.__init__ step by step
name = 'npdg'
in_states = (0, 1)
out_states = (2,)
alpha = 4.7161402e9
beta = 1.5
gamma = -25.81502

# Load spline data
import os
file_dir = os.path.dirname(os.path.abspath(__file__))
T9_vec, mu_median_vec, expsigma_vec = np.loadtxt(
    file_dir+'/ABCMB/linx/data/nuclear_rates/key_PRIMAT_2023/npdg.txt',
    unpack=True 
)

print(f"T9_vec type: {type(T9_vec)}, is JAX: {isinstance(T9_vec, jax.Array)}")
print(f"mu_median_vec type: {type(mu_median_vec)}, is JAX: {isinstance(mu_median_vec, jax.Array)}")
print(f"expsigma_vec type: {type(expsigma_vec)}, is JAX: {isinstance(expsigma_vec, jax.Array)}")

# Compute symmetry factors
multiplicity_in = jnp.array([in_states.count(i) for i in set(in_states)])
frwrd_symmetry_fac = float(jnp.prod(1. / multiplicity_in))

multiplicity_out = jnp.array([out_states.count(i) for i in set(out_states)])
bkwrd_symmetry_fac = float(jnp.prod(1. / multiplicity_out))

print(f"\nfrwrd_symmetry_fac: {frwrd_symmetry_fac}, type: {type(frwrd_symmetry_fac)}")
print(f"bkwrd_symmetry_fac: {bkwrd_symmetry_fac}, type: {type(bkwrd_symmetry_fac)}")

# Now try creating a Module with all these fields
class TestReaction(eqx.Module):
    name: str = eqx.field(static=True)
    in_states: tuple = eqx.field(static=True)
    out_states: tuple = eqx.field(static=True)
    frwrd_symmetry_fac: float = eqx.field(static=True)
    bkwrd_symmetry_fac: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    T9_vec: list = eqx.field(static=True)
    mu_median_vec: list = eqx.field(static=True)
    expsigma_vec: list = eqx.field(static=True)
    interp_type: str = eqx.field(static=True)
    frwrd_rate_param_func: callable = eqx.field(static=True)

print("\nCreating TestReaction...")
test_rxn = TestReaction(
    name=name,
    in_states=in_states,
    out_states=out_states,
    frwrd_symmetry_fac=frwrd_symmetry_fac,
    bkwrd_symmetry_fac=bkwrd_symmetry_fac,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    T9_vec=T9_vec,
    mu_median_vec=mu_median_vec,
    expsigma_vec=expsigma_vec,
    interp_type='linear',
    frwrd_rate_param_func=None
)
print("TestReaction created!")