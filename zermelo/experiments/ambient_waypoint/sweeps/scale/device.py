import dataclasses

from zermelo.experiments.ambient_waypoint.registry import (
    AMBIENT1_CONTROL1,
    CLAIM_EVERY_STEPS,
    OPENING_LEGS,
    PLANNING_BUDGET,
    RECORDED,
    RESOURCES,
    matched_belief,
)
from zermelo.experiments.ambient_waypoint.schema import MethodConfig
from zermelo.experiments.ambient_waypoint.setup import Implementation, arm, max_magnitude, sweep, uniform, value_iteration

HORIZON = 500
"""Moves an episode makes here: ten planning budgets"""


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


for name, resources in (
    ("device_cpu", dataclasses.replace(RESOURCES, gres=None, partition="dean", constraint="avx2", timeout_min=60, array_parallelism=2)),
    ("device_gpu", dataclasses.replace(RESOURCES, gres="gpu:1", partition="gpu", timeout_min=60, array_parallelism=2)),
):
    sweep(
        name,
        problem=AMBIENT1_CONTROL1,
        horizon=HORIZON,
        claim_every=CLAIM_EVERY_STEPS,
        recorded=RECORDED,
        seeds=[0],
        resources=resources,
        arms=[
            arm(
                "eui_fields16_walks16_rate0p5",  # the pathwise sampler runs on this arm
                _method(max_magnitude(), improvement=True, n_fields=16, n_walks=16, step_rate=0.5),
                matched_belief(AMBIENT1_CONTROL1),
            ),
            # no draws and no walks: this arm times the planner alone
            arm(
                "random_search",
                _method(uniform(), improvement=False, n_fields=0, n_walks=0, step_rate=0.0),
                matched_belief(AMBIENT1_CONTROL1),
            ),
        ],
    )
