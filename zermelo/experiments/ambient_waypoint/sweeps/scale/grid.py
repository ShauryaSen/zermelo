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
from zermelo.experiments.ambient_waypoint.setup import Implementation, arm, max_magnitude, sweep, uniform, value_iteration

WIDE = dataclasses.replace(AMBIENT1_CONTROL1, ambient_cells=100, controllable_cells=100)
"""n_states 120 * 104 = 12480 against 70 * 54 = 3780, and 10000 candidates against 2500"""

TARGET_CHUNK = 2500
"""Targets a plan solves at once on the wide grid: n_states * TARGET_CHUNK * n_actions floats memory per plan"""


def _method(
    utility: Implementation, *, improvement: bool, n_fields: int, n_walks: int, step_rate: float, target_chunk: int | None
) -> MethodConfig:
    """This sweep's held values"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(max_steps=PLANNING_BUDGET, radius=0.5, target_chunk=target_chunk),
        improvement=improvement,
        n_fields=n_fields,
        n_walks=n_walks,
        combination="linear",
        step_rate=step_rate,
        steps_from="predicted",
        n_candidates=None,
        opening_legs=OPENING_LEGS,
    )


for name, problem, chunk, resources in (
    ("grid50", AMBIENT1_CONTROL1, None, RESOURCES),
    (
        "grid100",
        WIDE,
        TARGET_CHUNK,
        dataclasses.replace(RESOURCES, mem_gb=32, timeout_min=480, gres="gpu:1", partition="gpu", array_parallelism=8),
    ),
):
    sweep(
        name,
        problem=problem,
        horizon=HORIZON_LENGTH,
        claim_every=CLAIM_EVERY_STEPS,
        recorded=RECORDED,
        seeds=range(N_SEEDS),
        resources=resources,
        arms=[
            arm(
                "eui_fields16_walks16_rate0p5",  # the pathwise sampler runs on this arm
                _method(max_magnitude(), improvement=True, n_fields=16, n_walks=16, step_rate=0.5, target_chunk=chunk),
                matched_belief(problem),
            ),
            # a charge would bias a uniform draw, which spans [0, 1] whatever the field
            arm(
                "random_search",
                _method(uniform(), improvement=False, n_fields=0, n_walks=0, step_rate=0.0, target_chunk=chunk),
                matched_belief(problem),
            ),
        ],
    )
