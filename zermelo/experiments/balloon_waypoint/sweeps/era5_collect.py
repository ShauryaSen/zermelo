"""Data collection on real ERA5: a balloon that flies to where a wind reading teaches the belief the most.

The world is our HEALPix ERA5 store cut to a lat-lon box over the southern ocean (16-21 July 2021,
3-hourly, six of the 18 balloon-band levels), with a persistence forecast 24 h stale standing in for
Stage 1's output. The arms differ only in what a waypoint is chosen for; every arm is scored on the
reconstruction error and posterior uncertainty of its claimed wind field, which is the objective of the
data-collection task. Record built by balloon-env-dev/unicorn/make_zermelo_record.py.
"""

import dataclasses
from typing import Literal

from zermelo.experiments.balloon_waypoint.registry import IRMA_JOSE, OPENING_LEGS, RECORDED, RESOURCES, matched_belief
from zermelo.experiments.balloon_waypoint.schema import BeliefConfig, BeliefKernelConfig, MethodConfig, Resources
from zermelo.experiments.balloon_waypoint.setup import (
    Implementation,
    arm,
    max_magnitude,
    point_speed,
    posterior_spread,
    predictive_confidence,
    sweep,
    uniform,
    value_iteration,
)

ERA5_SH_JUL2021 = dataclasses.replace(
    IRMA_JOSE,
    wind_path="zermelo/problems/balloon/data/era5_npac_wide.npz",
    frame=0,
    frames=81,  # the whole 240 h record at 3 h a frame: the wind drifts under the balloon
    forecast="GEFS",  # the file's own forecast column (persistence, 24 h stale)
    resource_units=40,
    step_hours=3.0,
    target=point_speed(),
    margin_lat=2,
    margin_lon=3,
)
"""Ten days over the North Pacific in summer, 20N-50N x 120-240E at 2 degrees, 14.8-20.2 km: easterlies aloft, westerlies below (the Loon regime).

Wide: at the 180-240E box of the first sweeps a balloon aloft (easterlies, 9 m/s westward) reached the western edge in four days and sat there"""

HORIZON = 80
"""Moves an episode makes: 80 x 3 h = the record's 240 h"""

CLAIM_EVERY = 5

PLANNING_BUDGET = 20

ARRIVAL_RADIUS_KM = 160.0
"""Cells here are 2 degrees (~220 km) apart, so the registry's 60 km radius could never be reached"""
"""`L`: a replan every 20 moves (60 h), one opening leg of 20 random moves, hitting times truncated at 20: at ~0.36 cells a step the planner must see ~7 cells ahead"""


def _belief() -> BeliefConfig:
    """The matched belief with a factor in the hour, since the wind moves; amplitude at the persistence error"""
    spatial = BeliefKernelConfig(kernel="gpjax.kernels.Matern52", parts=["position", "altitude"], lengthscale=[800.0] * 3 + [3.0])
    return dataclasses.replace(
        matched_belief(),
        kernel_factors=[spatial, BeliefKernelConfig(kernel="gpjax.kernels.Matern32", parts=["hours"], lengthscale=[24.0])],
        amplitude=2.4,  # m/s: the RMS of W - F in this record
        n_features=256,
    )


BELIEF = _belief()


def _method(
    utility: Implementation, *, improvement: bool, n_fields: int, combination: Literal["linear", "fraction"] = "linear"
) -> MethodConfig:
    """One rule on the value-iteration planner; travel priced off the planner's table"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(max_steps=PLANNING_BUDGET, radius=ARRIVAL_RADIUS_KM, target_chunk=None, cost_weight=0.25),
        improvement=improvement,
        n_fields=n_fields,
        n_walks=0,
        combination=combination,
        step_rate=0.5
        if combination == "linear"
        else None,  # travel is charged: at step_rate 0 the rule picked waypoints 15 cells away and no plan could reach them
        steps_from="predicted",
        n_candidates=None,
        opening_legs=OPENING_LEGS,
    )


sweep(
    "era5_collect",
    problem=ERA5_SH_JUL2021,
    horizon=HORIZON,
    claim_every=CLAIM_EVERY,
    recorded={**RECORDED, "objective_state/claim": CLAIM_EVERY},
    seeds=range(5),
    resources=Resources(cpus=2, mem_gb=16, timeout_min=150, gres=None, partition="dean", constraint=None, array_parallelism=20),
    arms=[
        arm("random_act", None, BELIEF),  # acts uniformly, claims the prior: the floor
        arm("random_search", _method(uniform(), improvement=False, n_fields=0), BELIEF),  # random waypoints, belief updated
        arm("max_variance", _method(posterior_spread(), improvement=False, n_fields=0), BELIEF),  # go where the belief is least sure
        arm(
            "entropy", _method(predictive_confidence(), improvement=True, n_fields=0, combination="fraction"), BELIEF
        ),  # go where a reading drops entropy most
        arm(
            "storm_search", _method(max_magnitude(), improvement=True, n_fields=0, combination="fraction"), BELIEF
        ),  # Vinamr's objective, for contrast
    ],
)
