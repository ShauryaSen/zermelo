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
from zermelo.experiments.ambient_waypoint.setup import arm, max_magnitude, sweep, value_iteration

BELIEF = matched_belief(AMBIENT1_CONTROL1)


def _method(n_walks: int) -> MethodConfig:
    """This sweep's held values"""
    return MethodConfig(
        utility=max_magnitude(),
        planner=value_iteration(max_steps=PLANNING_BUDGET, radius=0.5, target_chunk=None),
        improvement=True,
        n_fields=16,
        n_walks=n_walks,
        combination="linear",
        step_rate=0.5,
        steps_from="predicted",
        n_candidates=None,
        opening_legs=OPENING_LEGS,
    )


sweep(
    "walk_count",
    problem=AMBIENT1_CONTROL1,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=RESOURCES,
    arms=[arm(f"walks{m}", _method(m), BELIEF) for m in (1, 4, 8, 16, 32)],
)
