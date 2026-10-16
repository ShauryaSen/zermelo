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


def _method(n_candidates: int | None) -> MethodConfig:
    """This sweep's held values"""
    return MethodConfig(
        utility=max_magnitude(),
        planner=value_iteration(max_steps=PLANNING_BUDGET, radius=0.5, target_chunk=None),
        improvement=True,
        n_fields=16,
        n_walks=16,
        combination="linear",
        step_rate=0.5,
        steps_from="predicted",
        n_candidates=n_candidates,
        opening_legs=OPENING_LEGS,
    )


sweep(
    "candidate_count",
    problem=AMBIENT1_CONTROL1,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=RESOURCES,
    arms=[
        arm("all2500", _method(None), BELIEF),
        arm("draw1250", _method(1250), BELIEF),  # the shortlist is redrawn every decision
    ],
)
