import dataclasses

from zermelo.experiments.ambient_waypoint.registry import (
    AMBIENT1_CONTROL1,
    CLAIM_EVERY_STEPS,
    HORIZON_LENGTH,
    N_SEEDS,
    OPENING_LEGS,
    PLANNING_BUDGET,
    RECORDED,
    RESOURCES,
    matched_belief,
)
from zermelo.experiments.ambient_waypoint.schema import MethodConfig
from zermelo.experiments.ambient_waypoint.setup import (
    Implementation,
    arm,
    expected_improvement,
    max_magnitude,
    posterior_spread,
    sweep,
    uniform,
    upper_confidence,
    value_iteration,
)

BELIEF = matched_belief(AMBIENT1_CONTROL1)


def _method(utility: Implementation, *, improvement: bool, n_fields: int, n_walks: int, step_rate: float) -> MethodConfig:
    """This sweep's held values"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(max_steps=PLANNING_BUDGET, radius=0.5, target_chunk=None),
        improvement=improvement,
        n_fields=n_fields,
        n_walks=n_walks,
        combination="linear",
        step_rate=step_rate,
        steps_from="predicted",
        n_candidates=None,
        opening_legs=OPENING_LEGS,
    )


def _sampled(utility: Implementation, *, improvement: bool) -> MethodConfig:
    """`utility` at this sweep's draw counts and charge"""
    return _method(utility, improvement=improvement, n_fields=16, n_walks=16, step_rate=0.5)


sweep(
    "utilities",
    problem=AMBIENT1_CONTROL1,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=dataclasses.replace(RESOURCES, mem_gb=16, timeout_min=480, array_parallelism=40),
    # an increment differs from a level only for `max_magnitude`, the one rule reading the imagined readings
    arms=[
        arm("max_magnitude", _sampled(max_magnitude(), improvement=True), BELIEF),
        arm("posterior_spread", _sampled(posterior_spread(), improvement=False), BELIEF),
        arm("ucb2", _sampled(upper_confidence(2.0), improvement=False), BELIEF),
        arm("ei", _sampled(expected_improvement(), improvement=False), BELIEF),
        # a charge would bias a uniform draw, which spans [0, 1] whatever the field
        arm("random_search", _method(uniform(), improvement=False, n_fields=0, n_walks=0, step_rate=0.0), BELIEF),
    ],
)
