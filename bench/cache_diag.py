"""Minimal persistent-compile-cache diagnostic. Does jax write cache entries at
all in this env? Run twice (same cache dir) — 2nd jit should be instant."""
import os, sys, time, logging
cd = sys.argv[1]
os.makedirs(cd, exist_ok=True)
logging.basicConfig(level=logging.INFO)
import jax, jax.numpy as jnp
print("jax", jax.__version__)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", cd)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

@jax.jit
def f(x):
    for _ in range(50):
        x = jnp.tanh(x @ x) * 1.0001
    return x

t0 = time.perf_counter()
f(jnp.ones((256, 256))).block_until_ready()
print(f"first jit: {time.perf_counter()-t0:.2f}s")
print("cache entries:", os.listdir(cd))
