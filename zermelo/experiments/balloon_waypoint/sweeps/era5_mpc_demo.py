"""Two MPC navigators side by side for the eye: the posterior-mean planner and the sample-average planner over 8 belief draws.

The max-variance rule chooses the waypoints in both arms (the rule that travelled best on this record); only the wind the
leg is planned under differs. Two seeds, the whole 240 h record. For stills and films, not for numbers.
"""

import dataclasses

from zermelo.experiments.balloon_waypoint.schema import Resources
from zermelo.experiments.balloon_waypoint.setup import arm, posterior_spread, sweep
from zermelo.experiments.balloon_waypoint.sweeps.era5_collect_continuous import (
    BELIEF,
    CLAIM_EVERY,
    ERA5_SH_JUL2021,
    HORIZON,
    NAVIGATOR,
    RECORDED,
    _method,
)

SAMPLED = {**NAVIGATOR, "n_winds": 8}
"""The same solver, its cost averaged over 8 winds drawn from the belief each move"""

sweep(
    "era5_mpc_demo",
    problem=ERA5_SH_JUL2021,
    horizon=HORIZON,
    claim_every=CLAIM_EVERY,
    recorded={**RECORDED, "objective_state/claim": CLAIM_EVERY},
    seeds=range(2),
    resources=Resources(cpus=4, mem_gb=16, timeout_min=240, gres=None, partition="dean", constraint=None, array_parallelism=4),
    arms=[
        arm("mean_mpc", _method(posterior_spread(), improvement=False, n_fields=0), BELIEF),
        arm("sampled_mpc", dataclasses.replace(_method(posterior_spread(), improvement=False, n_fields=0), navigator=SAMPLED), BELIEF),
    ],
)
