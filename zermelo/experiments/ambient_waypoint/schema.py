"""The settings one episode is built from, as dataclasses hydra composes and checks"""

from dataclasses import dataclass, field
from typing import Any, Literal

from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class ProblemConfig:
    """The world one episode runs in"""

    ambient_cells: int = MISSING
    """Cells per displaced axis, before the pad"""

    ambient_axes: int = MISSING

    ambient_cell_spacing: float = MISSING
    """Coordinate distance from one cell to the next on a displaced axis"""

    ambient_pad: int = MISSING
    """Pad cells beyond the scored ones on each displaced axis"""

    controllable_cells: int = MISSING
    """Cells per steered axis, before the pad"""

    controllable_axes: int = MISSING

    controllable_cell_spacing: float = MISSING
    """Coordinate distance from one cell to the next on a steered axis"""

    controllable_pad: int = MISSING
    """Pad cells beyond the scored ones on each steered axis"""

    sigma_flow: float = MISSING
    """Standard deviation of the noise on the displacement the field causes"""

    sigma_act: float = MISSING
    """Standard deviation of the noise on the displacement the actor steers"""

    sigma_obs: float = MISSING
    """Standard deviation of one reading"""

    choices_per_axis: int = MISSING
    """Steering choices per steered axis"""

    drift_bound_cells: float = MISSING
    """Largest displacement, in cells, the field may cause in one step"""

    control_step_cells: float = MISSING
    """Cells one steering choice moves per step"""

    control_bound_cells: float = MISSING
    """Largest displacement, in cells, the actor may steer in one step"""

    field_kernel: str = MISSING
    """Dotted path to the kernel family the true field is drawn from"""

    field_lengthscale: float = MISSING
    """Lengthscale of the kernel the true field is drawn from"""

    field_amplitude: float = MISSING
    """Amplitude of the kernel the true field is drawn from, as a standard deviation"""

    field_features: int = MISSING
    """Random Fourier features the field draw is built from"""


@dataclass
class BeliefKernelConfig:
    """One factor of the belief's kernel, over the parts it names"""

    kernel: str = MISSING
    """Dotted path to the stationary kernel family this factor is"""

    parts: list[str] = MISSING
    """Which parts of a coordinate row this reads, named as the problem names them"""

    lengthscale: list[float] = MISSING
    """One per column those parts occupy, in the order `parts` names them"""


@dataclass
class BeliefConfig:
    """A method's own model of the field"""

    oracle: bool = MISSING
    """Whether the method is handed the true field instead of fitting one"""

    kernel_factors: list[BeliefKernelConfig] = MISSING
    """The factors the kernel is a product of"""

    amplitude: float = MISSING
    """Amplitude of that kernel, as a standard deviation"""

    noise: float = MISSING
    """Standard deviation the method assumes a reading has"""

    n_features: int = MISSING
    refit: bool = MISSING

    refit_steps: int = MISSING
    """Gradient steps taken per refit"""


@dataclass
class MethodConfig:
    """One fully specified waypoint rule"""

    utility: Any = MISSING
    """Which quantity of a belief is maximized, as a dotted path with its own arguments"""

    planner: Any = MISSING
    """Which policy carries the actor to a waypoint, as a dotted path with its own arguments"""

    n_fields: int = MISSING
    """Field draws averaged over (zero reads the posterior mean alone)"""

    n_walks: int = MISSING
    """Walks rolled per candidate (zero prices travel from the planner's own table)"""

    improvement: bool = MISSING
    """Whether a candidate is scored by the increment it would add rather than its level"""

    step_rate: float | None = MISSING
    """`lambda`: a full-budget trip costs this share of the spread of the candidate values"""

    steps_from: Literal["predicted", "rolled"] = MISSING
    """Whether travel is priced from the planner's table or from the walks that were rolled"""

    combination: Literal["linear", "fraction"] = MISSING
    """Whether travel subtracts from value or divides it"""

    n_candidates: int | None = MISSING
    """Cap on candidates scored per move (None scores every one)"""

    opening_legs: int = MISSING
    """Waypoint legs of uniform random walking taken before the rule starts"""


@dataclass
class Resources:
    """What one cell of a sweep is given to run in"""

    cpus: int = MISSING

    mem_gb: int = MISSING

    timeout_min: int = MISSING
    """Wall clock in minutes a cell is allowed before the scheduler evicts it"""

    gres: str | None = MISSING
    """Generic resources a cell asks the scheduler for, such as `gpu:1` (None asks for none)"""

    partition: str = MISSING
    """Scheduler partition the cells are submitted to"""

    constraint: str | None = MISSING
    """Node features a cell demands of the scheduler, joined by `&` and `|` (None demands none)"""

    array_parallelism: int = MISSING
    """Cells of one array allowed to run at once, the rest queueing behind them"""


@dataclass
class RunConfig:
    """Everything one episode needs"""

    defaults: list[Any] = field(default_factory=lambda: ["_self_"])
    """Hydra's defaults list, naming no group: a run names its arm and its seed on the command line"""

    problem: ProblemConfig = MISSING

    belief: BeliefConfig = MISSING

    method: MethodConfig | None = MISSING
    """The waypoint rule's settings (None is an arm that acts uniformly and chooses no waypoint)"""

    horizon: int = MISSING
    """Moves an episode may make"""

    claim_every: int = MISSING
    """Moves between re-readings of the claim off a belief"""

    recorded: dict[str, int] = MISSING
    """What the record stores, and the moves between the snapshots it keeps of each"""

    resources: Resources = MISSING
    """What the cell was given, carried onto the record"""

    seed: int = MISSING
    """The one number an episode replays from"""


def resolve(composed: DictConfig) -> RunConfig:
    """One composed configuration into the dataclasses it was checked against"""
    settings = OmegaConf.to_object(composed)
    if not isinstance(settings, RunConfig):
        raise TypeError(f"a composed configuration should be the schema it was registered against, and this is {type(settings).__name__}")
    return settings
