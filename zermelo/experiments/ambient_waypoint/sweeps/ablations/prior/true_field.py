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

MATCHED = matched_belief(AMBIENT1_CONTROL1)
ORACLE = dataclasses.replace(MATCHED, oracle=True)


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


RULES = {
    # a charge would bias a uniform draw, which spans [0, 1] whatever the field
    "random_search": _method(uniform(), improvement=False, n_fields=0, n_walks=0, step_rate=0.0),
    "ei": _method(expected_improvement(), improvement=False, n_fields=0, n_walks=0, step_rate=0.0),
    "thompson": _method(max_magnitude(), improvement=False, n_fields=1, n_walks=0, step_rate=0.0),
    "eui_fields16_walks16_rate0p5": _method(max_magnitude(), improvement=True, n_fields=16, n_walks=16, step_rate=0.5),
}
"""The rules of the headline sweep"""

sweep(
    "true_field",
    problem=AMBIENT1_CONTROL1,
    horizon=HORIZON_LENGTH,
    claim_every=CLAIM_EVERY_STEPS,
    recorded=RECORDED,
    seeds=range(N_SEEDS),
    resources=dataclasses.replace(RESOURCES, timeout_min=480, array_parallelism=40),
    arms=[
        arm(f"{name}_{model}", method, belief)
        for name, method in RULES.items()
        for model, belief in (("oracle", ORACLE), ("matched", MATCHED))
    ],
)
