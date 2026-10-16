"""Draws from the GP belief under a product kernel with a time factor: their moments against the closed-form posterior

pixi run crosscheck belief_draws
"""

import gpjax as gpx
import jax
import jax.numpy as jnp
import numpy as np

from zermelo.methods.waypoint_bo.belief import BeliefKernel, Dataset, GPBelief
from zermelo.problems.balloon import SphereGrid, balloon_states

N_DRAWS, N_FEATURES, AMPLITUDE, NOISE = 512, 512, 2.4, 1e-2
MAX_SIGMAS = 4.0
"""Standard errors a sample mean may sit from the exact posterior mean"""
MAX_MOMENT = 0.15
"""Share of the prior variance a sample variance may miss the exact posterior variance by (the bias of finite features)"""


def main() -> None:
    grid = SphereGrid(n_lat=5, n_lon=7, lat_first=30.0, lon_first=180.0, lat_step=2.0, lon_step=2.0)
    states = balloon_states(grid, (20.2, 17.9, 15.8))
    factors = (
        BeliefKernel.of(gpx.kernels.Matern52, ["position", "altitude"], [800.0] * 3 + [3.0]),
        BeliefKernel.of(gpx.kernels.Matern32, ["hours"], [24.0]),
    )
    belief = GPBelief.empty(
        states,
        4,
        2,
        1,
        kernel_factors=factors,
        context={"hours": 1},
        amplitude=jnp.asarray(AMPLITUDE),
        noise=jnp.asarray(NOISE),
        n_features=N_FEATURES,
        refit=False,
        refit_steps=0,
        prior_mean=None,
    )
    at = jnp.asarray([9.0, 12.0, 24.0, 40.0])  # the hours each reading was taken at
    z = jnp.asarray([17, 52, 17, 80])  # cells read, one of them twice at different hours
    r = jnp.asarray([[3.0, -1.0], [-2.0, 0.5], [2.0, -1.5], [0.0, 4.0]])
    data = Dataset.empty(4, 2, 1)
    for i in range(4):
        data = data.write(i, z[i], r[i], at[i : i + 1])
    belief = belief.condition(data)
    now = jnp.asarray([30.0])
    exact_mean, exact_var = belief.predict(jnp.arange(states.size()), now)  # (n_states, 2) each

    drawn = belief.draw(jax.random.key(0), N_DRAWS, now)
    table = jnp.stack([f(states.elements()) for f in drawn])  # (N_DRAWS, n_states, 2)
    mean, var = table.mean(0), table.var(0)
    sigmas = float(jnp.max(jnp.abs(mean - exact_mean) / jnp.sqrt(exact_var / N_DRAWS)))
    moment = float(jnp.max(jnp.abs(var - exact_var)) / AMPLITUDE**2)
    unread = jnp.sqrt(var[:, 0]).max()
    print(f"posterior mean: sample vs exact, worst {sigmas:.2f} standard errors (allowed {MAX_SIGMAS})")
    print(
        f"posterior variance: sample vs exact, worst gap {moment:.3f} of the prior variance (allowed {MAX_MOMENT}); prior sd {AMPLITUDE}, largest draw sd {float(unread):.2f}"
    )
    at_read = jnp.sqrt(var[z[1], 0])
    print(f"at a cell read 18 h ago the draws spread {float(at_read):.2f} m/s; at an unread cell {float(unread):.2f} m/s")
    assert sigmas < MAX_SIGMAS, "the draws are not centred on the posterior mean"
    assert moment < MAX_MOMENT, "the draws do not have the posterior's spread"
    assert at_read < unread, "a reading did not narrow the draws where it was taken"
    print("ok")


if __name__ == "__main__":
    main()
