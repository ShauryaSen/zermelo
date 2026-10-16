from zermelo.experiments.ambient_waypoint.registry import RECORDED
from zermelo.experiments.ambient_waypoint.schema import BeliefConfig, BeliefKernelConfig, MethodConfig, ProblemConfig, Resources
from zermelo.experiments.ambient_waypoint.setup import Implementation, arm, expected_improvement, max_magnitude, sweep, value_iteration

WORLD = ProblemConfig(
    ambient_cells=5,
    ambient_axes=1,
    ambient_cell_spacing=1.0,
    ambient_pad=1,
    controllable_cells=4,
    controllable_axes=1,
    controllable_cell_spacing=1.0,
    controllable_pad=1,
    sigma_flow=0.03,
    sigma_act=0.02,
    sigma_obs=0.01,
    choices_per_axis=3,
    drift_bound_cells=1.0,
    control_step_cells=1.0,
    control_bound_cells=1.0,
    field_kernel="gpjax.kernels.RBF",
    field_lengthscale=1.2,  # cells; under one cell a draw is white noise the belief cannot fit
    field_amplitude=1.0,
    field_features=32,
)
"""A five by four world whose numbers are noise"""


def _belief(oracle: bool) -> BeliefConfig:
    """The model an arm holds: matched to this world, or the true field itself"""
    return BeliefConfig(
        oracle=oracle,
        kernel_factors=[BeliefKernelConfig(kernel="gpjax.kernels.RBF", parts=["ambient", "controllable"], lengthscale=[1.2, 1.2])],
        amplitude=1.0,
        noise=0.01,
        n_features=32,
        refit=False,
        refit_steps=20,
    )


def _method(utility: Implementation, *, improvement: bool, n_fields: int, n_walks: int, step_rate: float) -> MethodConfig:
    """One rule on a value-iteration planner, at the settings the arms differ in"""
    return MethodConfig(
        utility=utility,
        planner=value_iteration(max_steps=4, radius=0.05, target_chunk=None),
        improvement=improvement,
        n_fields=n_fields,
        n_walks=n_walks,
        combination="linear",
        step_rate=step_rate,
        steps_from="predicted",
        n_candidates=None,
        opening_legs=1,  # four uniform moves, leaving two the rule decides
    )


# One entry per device the cells can run on; `smoke_gpu` asks for an accelerator and a gpu-high node
for name, resources in (
    ("smoke", Resources(cpus=1, mem_gb=4, timeout_min=10, gres=None, partition="dean", constraint="avx2", array_parallelism=10)),
    (
        "smoke_gpu",
        Resources(cpus=1, mem_gb=4, timeout_min=10, gres="gpu:1", partition="dean", constraint="avx2&gpu-high", array_parallelism=10),
    ),
):
    sweep(
        name,
        problem=WORLD,
        horizon=6,
        claim_every=1,
        recorded={**RECORDED, "objective_state/claim": 1},
        seeds=[0],
        resources=resources,
        arms=[
            arm("mean", _method(max_magnitude(), improvement=True, n_fields=0, n_walks=0, step_rate=0.0), _belief(oracle=False)),
            arm("oracle", _method(max_magnitude(), improvement=True, n_fields=0, n_walks=0, step_rate=0.0), _belief(oracle=True)),
            arm(
                "ei",  # a closed form: its own increment, no draws, no walks
                _method(expected_improvement(), improvement=False, n_fields=0, n_walks=0, step_rate=0.0),
                _belief(oracle=False),
            ),
            arm("random", None, _belief(oracle=False)),  # no rule at all
        ],
    )
