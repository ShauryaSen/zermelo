"""Does sampling the belief help more as the forecast gets worse?

The noise ladder of `bounds_noise`, crossed with `n_winds`: how many draws from the posterior the MPC
averages its cost over per solve. `n_winds = 0` plans on the posterior mean alone, which is the paper's own
setting; anything above that is the sampled extension.

The expectation is that the lines fan out. With a perfect forecast the posterior mean is the truth and
sampling can only add noise, so every arm should sit together at sigma = 0. As the forecast decays the mean
becomes a worse summary of what the balloon knows, and a plan that hedges across draws should hold up
better. If the lines stay parallel, sampling is not buying anything the mean does not already give.

Arms differ in `n_winds` and nothing else: same belief, same acquisition, replanning every move. They are not
bounds on each other -- their trajectories diverge, so each ends up choosing its own waypoints -- but they
are a fair comparison, which is all that is claimed.

The cost is navigation alone, as in `bounds_noise`: `resource_weight = 0`.
"""

import dataclasses

from zermelo.experiments.balloon_waypoint.setup import Implementation, arm, posterior_spread, sweep
from zermelo.experiments.balloon_waypoint.sweeps.bounds_noise import NAVIGATION_ONLY, RESOURCES, SIGMAS
from zermelo.experiments.balloon_waypoint.sweeps.era5_collect_continuous import (
    BELIEF,
    CLAIM_EVERY,
    ERA5_SH_JUL2021,
    HORIZON,
    RECORDED,
    _method,
)

WINDS = (0, 4, 16)
"""Draws from the belief averaged over per solve. Zero is the posterior mean alone, the paper's setting.

Kept to three values and spaced wide: a draw happens at every one of the 80 moves, so the cost is roughly
linear in this and the top arm sets the runtime of the whole sweep. Measure before adding a fourth:

    time pixi run sweep balloon_waypoint winds_noise_s24
"""

SEEDS = range(16)
"""Half of what the one-dimensional ladder uses, because this sweep is three arms wide at every level"""


def _arm(n_winds: int) -> object:
    """The controller averaging its cost over `n_winds` draws, named for how many"""
    navigator: Implementation = {**NAVIGATION_ONLY, "n_winds": n_winds}
    rule = dataclasses.replace(_method(posterior_spread(), improvement=False, n_fields=0), navigator=navigator)
    return arm(f"n{n_winds}", rule, BELIEF)


for _sigma in SIGMAS:
    sweep(
        f"winds_noise_s{int(round(_sigma * 10))}",
        problem=dataclasses.replace(ERA5_SH_JUL2021, forecast="noisy", error_scale=_sigma),
        horizon=HORIZON,
        claim_every=CLAIM_EVERY,
        recorded={**RECORDED, "objective_state/claim": CLAIM_EVERY},
        seeds=SEEDS,
        resources=RESOURCES,
        arms=[_arm(n) for n in WINDS],
    )
