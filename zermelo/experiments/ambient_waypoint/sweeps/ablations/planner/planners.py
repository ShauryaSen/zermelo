from typing import Any

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
from zermelo.experiments.ambient_waypoint.setup import Implementation, arm, greedy, max_magnitude, random_walk, sweep, value_iteration

BELIEF = matched_belief(AMBIENT1_CONTROL1)


def _method(planner: Implementation) -> MethodConfig:
    """This sweep's held values"""
    return MethodConfig(
        utility=max_magnitude(),
        planner=planner,
        improvement=True,
        n_fields=16,
        n_walks=16,
        combination="linear",
        step_rate=0.5,
        steps_from="predicted",
        n_candidates=None,
        opening_legs=OPENING_LEGS,
    )


PLANNERS: dict[str, Any] = {"value_iteration": value_iteration, "greedy": greedy, "random_walk": random_walk}

sweep(
    "planners",
    problem=AMBIENT1_CONTROL1,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=RESOURCES,
    arms=[arm(name, _method(build(max_steps=PLANNING_BUDGET, radius=0.5, target_chunk=None)), BELIEF) for name, build in PLANNERS.items()],
)
