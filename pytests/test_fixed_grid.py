"""Model.freeze_lna_grid: the frozen save / LoS grid reproduces the visibility grid's Cls at the
cosmology it was built at, and is then the same for every cosmology (lna_grid_mode="fixed")."""
import numpy as np


def test_freeze_lna_grid():
    import jax
    jax.config.update("jax_enable_x64", True)
    from abcmb.main import Model
    from pytests.fixtures.scenarios import SCENARIOS

    scen = SCENARIOS["lcdm_tau"]
    params = scen["params"]()
    model = Model(user_species=scen["user_species"](), **scen["specs"]())
    frozen = model.freeze_lna_grid(params)
    nodes = np.asarray(frozen.specs["lna_grid_nodes"])
    assert frozen.specs["lna_grid_mode"] == "fixed" and model.specs["lna_grid_mode"] == "visibility"
    assert nodes.shape == (model.specs["n_lna_PE"],) and np.all(np.diff(nodes) > 0) and nodes[-1] == 0.
    np.testing.assert_array_equal(nodes, model.lna_grid(params))

    out, out_f = model(params), frozen(params)
    for f in ("ClTT", "ClTE", "ClEE", "Pk"):
        a, b = np.asarray(getattr(out, f)), np.asarray(getattr(out_f, f))
        np.testing.assert_allclose(b, a, rtol=1e-7, atol=1e-7 * np.abs(a).max(), err_msg=f)

    p2 = dict(params, omega_b=params["omega_b"] * 1.01)
    np.testing.assert_array_equal(frozen.lna_grid(p2), nodes)
    assert not np.array_equal(model.lna_grid(p2), nodes)
