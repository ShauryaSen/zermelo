"""The world every study runs on, and the numbers every sweep of it holds equal"""

from zermelo.experiments.ambient_waypoint.schema import BeliefConfig, BeliefKernelConfig, ProblemConfig, Resources

AMBIENT1_CONTROL1 = ProblemConfig(
    ambient_axes=1,
    ambient_cell_spacing=1.0,  # one cell is one unit: every length below reads in cells
    ambient_cells=50,
    ambient_pad=10,  # the largest displacement the field can cause in a step
    controllable_axes=1,
    controllable_cell_spacing=1.0,
    controllable_cells=50,
    controllable_pad=2,  # absorbs the act noise, clipped at 2 by the control bound
    sigma_flow=0.5,
    sigma_act=0.2,
    sigma_obs=0.0,
    choices_per_axis=3,
    drift_bound_cells=10.0,
    control_step_cells=1.0,
    control_bound_cells=2.0,  # 2 rather than the 1-cell deliberate move: the act noise may overshoot as well as undershoot
    field_kernel="gpjax.kernels.Matern52",
    field_lengthscale=2.5,
    field_amplitude=3.0,
    field_features=256,
)


def matched_belief(world: ProblemConfig) -> BeliefConfig:
    """A belief holding the world's own kernel numbers, at a stated noise and fidelity"""
    return BeliefConfig(
        oracle=False,
        # one lengthscale per coordinate the two grids embed to, isotropic here
        kernel_factors=[
            BeliefKernelConfig(
                kernel=world.field_kernel,
                parts=["ambient", "controllable"],
                lengthscale=[world.field_lengthscale] * (world.ambient_axes + world.controllable_axes),
            )
        ],
        amplitude=world.field_amplitude,  # a standard deviation on both sides; the kernel squares it
        noise=1e-3,
        n_features=256,
        refit=False,
        refit_steps=100,  # unread while refitting is off, and stated anyway
    )


PLANNING_BUDGET = 50
"""`L`: a replan fires every `L` moves, costs `L` backups, and truncates a hitting time at `L` steps"""

OPENING_LEGS = 1
"""Waypoint legs of uniform random walking taken before the rule starts"""

HORIZON_LENGTH = 2000
"""`T`: moves an episode makes"""

CLAIM_EVERY_STEPS = 50
"""Moves between re-readings of the claim off a belief"""

RECORDED = {
    "state/ambient": 1,
    "state/controllable": 1,
    "decision/action": 1,
    "objective_state/oracle": 1,
    "objective_state/incumbent": 1,
    "objective_state/truth": 1,
    "objective_state/claim": CLAIM_EVERY_STEPS,
    "agent_state/steps_since_waypoint": 1,
    "agent_state/scores.predicted_step_cost": 1,
    "agent_state/scores.frac_zero_value_candidates": 1,
    "agent_state/scores.frac_reachable_candidates": 1,
    "agent_state/scores.waypoint": 1,
    "agent_state/scores.imagined_walk_to_waypoint": 1,
    "agent_state/scores.candidate_indices": 1,
    "agent_state/scores.acquisition_value": 1,
    "agent_state/policy.act": 1,
}
"""What a record stores, and the moves between the snapshots it keeps of each"""

N_SEEDS = 10
"""Instances every arm is run on, seeded `0` to `N_SEEDS - 1`"""

RESOURCES = Resources(cpus=2, mem_gb=12, timeout_min=60, gres="gpu:1", partition="gpu", constraint="avx2&gpu-high", array_parallelism=64)
"""What one cell of a sweep is given. A sweep wanting more states its own with `dataclasses.replace`.

`avx2` and `gpu-high` are node features. Dropping either lands cells on nodes where jaxlib fails to
import or a matrix multiply fails to launch. A cell wanting no device states `avx2` alone"""
