"""HyRex CPU stage, sharded over host devices. Times the batched HyRex primal and its
P=7-direction vmapped jvp at B cosmologies on (a) one CPU device or (b) the B axis sharded over
n_use = the largest divisor of B <= len(jax.devices('cpu')) host devices.
Run with XLA_FLAGS=--xla_force_host_platform_device_count=N to get N CPU devices.
Usage: python hyrex_bench.py B [tag]"""
import os, sys, time, json
ROOT = "/pscratch/sd/c/carag/ABCMB-k-grad"
sys.path.insert(0, ROOT)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import abcmb
assert abcmb.__file__.startswith(ROOT)
from abcmb.main import Model, _recmodel_cpu
sys.path.insert(1, os.path.join(ROOT, "scan"))
from batched_grad import jvp_stage, _to_float
B = int(sys.argv[1]); TAG = sys.argv[2] if len(sys.argv) > 2 else "x"; Pd = 7
ORDER = ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion", "Neff"]
CEN = np.array([0.6662, 0.02229, 0.11654, 0.95986, 3.0446, 0.05933, 2.8268])
SIG = np.array([0.0054, 0.00015, 0.0013, 0.0042, 0.014, 0.0073, 0.1])
FIXED = {"YHe": 0.2454, "TCMB0": 2.34865418e-4, "N_nu_massive": 1, "T_nu_massive": 0.71611,
         "m_nu_massive": 0.06, "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5,
         "exp_reion": 1.5}
SPECS = dict(output_Cl=True, l_max=2508, lensing=True, output_Pk=False, l_max_g=12, l_max_pol_g=10,
             l_max_ur=17, l_max_ncdm=17, rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
             rtol_small_k_PE=1e-5, max_steps_PE=16384)
model = Model(**SPECS)
cpus = jax.devices('cpu'); n_cpu = len(cpus)
n_use = max(d for d in range(1, n_cpu + 1) if B % d == 0)
print(f"B={B} cpu devices={n_cpu} n_use={n_use} affinity={len(os.sched_getaffinity(0))} XLA_FLAGS={os.environ.get('XLA_FLAGS')}", flush=True)
z = np.clip(np.random.default_rng(0).standard_normal((max(B, 1), 7)), -2, 2); z[0] = 0.
TH = CEN + SIG * z[:B]
def raw(th):
    p = dict(FIXED)
    for name, v in zip(ORDER, th):
        if name == "ln10As": p["A_s"] = jnp.exp(v) / 1e10
        else: p[name] = v
    return p
def derived(th): return _to_float(model.add_derived_parameters(raw(th)))
full_ps = [derived(jnp.asarray(th)) for th in TH]
pdots = [jax.tree.map(lambda *xs: jnp.stack(xs), *[
    eqx.filter(jax.jvp(derived, (jnp.asarray(th),), (jnp.zeros(7).at[j].set(SIG[j]),))[1],
               eqx.is_inexact_array) for th in TH]) for j in range(Pd)]
pdot = jax.tree.map(lambda *xs: jnp.stack(xs), *pdots)
params_batch = jax.tree.map(lambda *xs: jnp.stack(xs), *full_ps)
pre_BG, pre_BG_dot = jvp_stage(model._pre_recomb_batched, (params_batch,), (pdot,), n_dirs=Pd)
if n_use > 1:
    mesh = Mesh(np.asarray(cpus[:n_use]), axis_names=('batch',))
    sh_b, sh_pb, repl = NamedSharding(mesh, P('batch')), NamedSharding(mesh, P(None, 'batch')), NamedSharding(mesh, P())
    put_b = lambda t: jax.tree.map(lambda x: jax.device_put(x, sh_b if x.ndim >= 1 else repl) if eqx.is_array(x) else x, t)
    put_pb = lambda t: jax.tree.map(lambda x: jax.device_put(x, sh_pb if x.ndim >= 2 else repl) if eqx.is_array(x) else x, t)
else:
    put_b = put_pb = lambda t: jax.device_put(t, cpus[0])
ri_cpu, p_cpu = put_b(pre_BG.recomb_inputs), put_b(params_batch)
ri_dot, p_dot = put_pb(pre_BG_dot.recomb_inputs), put_pb(pdot)
def shard_of(tree):
    xs = [x for x in jax.tree_util.tree_leaves(tree) if eqx.is_array(x)]
    return str(xs[0].sharding)[:90] if xs else "no arrays"
print(f"inputs: ri {shard_of(ri_cpu)} | ri_dot {shard_of(ri_dot)}", flush=True)
WRAP = os.environ.get("HYREX_WRAP", "backend")   # backend: _recmodel_cpu (filter_jit backend='cpu'); plain: filter_jit no backend
hy = _recmodel_cpu(model.RecModel, True) if WRAP == "backend" else eqx.filter_jit(jax.vmap(model.RecModel))
res = {"B": B, "n_cpu": n_cpu, "n_use": n_use, "tag": TAG, "wrap": WRAP}
ts = []
for rep in range(3):
    t0 = time.perf_counter(); out = jax.block_until_ready(hy((ri_cpu, p_cpu))); ts.append(time.perf_counter() - t0)
res["primal_s"] = [round(t, 2) for t in ts]
print(f"[primal {WRAP}] {res['primal_s']}  out sharding={shard_of(out)}", flush=True)
ts = []
for rep in range(3):
    t0 = time.perf_counter()
    o, od = jvp_stage(lambda ri, p: hy((ri, p)), (ri_cpu, p_cpu), (ri_dot, p_dot), n_dirs=Pd)
    jax.block_until_ready((o, od)); ts.append(time.perf_counter() - t0)
res["jvp_s"] = [round(t, 2) for t in ts]
print(f"[jvp P={Pd} {WRAP}] {res['jvp_s']}  out_dot sharding={shard_of(od)}", flush=True)
leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(out) if eqx.is_array(x)]
dleaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(od) if eqx.is_array(x)]
np.savez(os.path.join(os.path.dirname(__file__), f"hyrex_bench_B{B}_{TAG}.npz"), *leaves, **{f"d{i}": d for i, d in enumerate(dleaves)})
print("RESULT " + json.dumps(res), flush=True)
