import dataclasses

from zermelo.experiments.balloon_waypoint.registry import (
    ARRIVAL_RADIUS_KM,
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    IRMA_JOSE,
    N_CANDIDATES,
    N_SEEDS,
    OPENING_LEGS,
    PLANNING_BUDGET,
    RECORDED,
    RESOURCES,
    oracle_belief,
)
from zermelo.experiments.balloon_waypoint.schema import MethodConfig
from zermelo.experiments.balloon_waypoint.setup import (
    Implementation,
    arm,
    max_magnitude,
    posterior_spread,
    sweep,
    uniform,
    upper_confidence,
    value_iteration,
)

BELIEF = oracle_belief()
"""The true field on every arm, so a rule reading a spread reads zero"""


def _method(
    utility: Implementation, *, improvement: bool, n_fields: int, n_walks: int, step_rate: float, cost_weight: float
) -> MethodConfig:
    """This sweep's held values"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(max_steps=PLANNING_BUDGET, radius=ARRIVAL_RADIUS_KM, target_chunk=None, cost_weight=cost_weight),
        improvement=improvement,
        n_fields=n_fields,
        n_walks=n_walks,
        combination="linear",
        step_rate=step_rate,
        steps_from="predicted",
        n_candidates=N_CANDIDATES,
        opening_legs=OPENING_LEGS,
    )


def _sampled(utility: Implementation, *, improvement: bool) -> MethodConfig:
    """`utility` at this sweep's draw counts and charge"""
    return _method(utility, improvement=improvement, n_fields=16, n_walks=16, step_rate=0.5, cost_weight=0.25)


sweep(
    "utilities",
    problem=IRMA_JOSE,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=RESOURCES,
    # an increment differs from a level only for `max_magnitude`, the one rule reading the imagined readings
    arms=[
        arm("max_magnitude", _sampled(max_magnitude(), improvement=True), BELIEF),
        arm("posterior_spread", _sampled(posterior_spread(), improvement=False), BELIEF),
        arm("ucb2", _sampled(upper_confidence(2.0), improvement=False), BELIEF),
        # a charge would bias a uniform draw, which spans [0, 1] whatever the field
        arm("random_search", _method(uniform(), improvement=False, n_fields=0, n_walks=0, step_rate=0.0, cost_weight=0.0), BELIEF),
    ],
)
