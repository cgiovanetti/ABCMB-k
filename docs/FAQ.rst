FAQ
===

What is this JAX thing, anyway?  Do I need to be a JAX expert to use ABCMB?
---------------------------------------------------------------------------
You should not need to know very much about JAX in order to use ABCMB.  `JAX <https://docs.jax.dev/en/latest/index.html>`_ is a Python library that lets you write fast and differentiable Python code without having to depart too far from the ordinary Python you're already familiar with.  In general, the following tips are helpful to keep in mind:

1. Always use ``jax.numpy`` as opposed to ``numpy``.  (At the top of your script, you can ``import jax.numpy as jnp`` and use ``jnp`` in place of where you might ordinarily use ``np``.)  Scipy is also not generally safe to use with JAX; use ``jax.scipy``, or if the function you need is missing from ``jax.scipy`` you can look into community packages like ``interpax``, ``diffrax``, or ``quadax``.

2. You typically can't write conditionals the way you would in Python in JAX.   Conditions that aren't based on things like floats (e.g. ``if FLAG==True``, and ``FLAG`` is set at initialization) are just fine to use with JAX.  But if a function in your custom fluid needs a conditional like ``if x > 5:``, use ``jnp.where`` instead.  The Python code::

    if x > 5:
        return x**2
    else:
        return -x**2

   Can be rewritten as the more JAX-friendly::

    return jnp.where(x>5, x**2, -x**2)

   See `JAX documentation <https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html>`_ for more common gotchas if you're finding your custom modules throw errors or are recompiling.

How do I take gradients of ABCMB output?
----------------------------------------
In general it is best to use ``jax.jacfwd``, or forward accumulation, with ABCMB.  There are many internal states to trace over, which can quickly push memory requirements out of hand, when attempting to use reverse AD like ``jax.grad`` or ``jax.jacrev`` with ABCMB.

Why am I seeing my code recompile?
----------------------------------
There are a few reasons why otherwise JAX-safe code might not call the cached JIT-compiled version.  Passing in different data types to the same JIT-compiled argument will trigger recompilation (i.e. passing in "``1``" vs "``1.``").

You may also be seeing recompilation because you wrapped ``Model.run_cosmology`` in a larger ``jit`` context.  We do not recommend enclosing ``Model.run_cosmology`` in another ``jax.jit``, for a couple reasons:

1. ``add_derived_parameters``, the first auxiliary function to be called under the hood, is intended to be called outside of ``jit``.   This in principle can be worked around by wrapping your inputs to your exterior-most ``jit`` context in ``jnp.array``.

2. LINX is CPU-optimized and has been carefully extracted from the rest of the ABCMB ``jit`` context so that it will always run on CPU, regardless of whether a GPU is present.  Wrapping ``Model.run_cosmology`` in a larger ``jit`` context will slow down your code substantially if you are running with BBN.  Future versions may also force CPU evaluation of HyRex in a similar fashion, so you will always be taking a performance hit if you choose to ``jit`` ``Model.run_cosmology``.

Finally, you may be seeing recompilation because you've encountered a bug!  After you've ruled out the causes above, feel free to open an issue on our `GitHub <https://github.com/TonyZhou729/ABCMB>`_.   If you'd like to explore the cause yourself, turn on ``jax.config.update("explain_cache_misses"=True)`` before running your recompiling code.

Can I add new methods to my custom fluids beyond what ABCMB expects?
--------------------------------------------------------------------
Yes!  ``abcmb.species.Baryon`` is a good example of a fluid that has extra methods. 
