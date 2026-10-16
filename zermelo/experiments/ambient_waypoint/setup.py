"""What a sweep module is written with"""

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from zermelo.experiments.ambient_waypoint.schema import BeliefConfig, MethodConfig, ProblemConfig, Resources

cs = ConfigStore.instance()


@dataclass(frozen=True)
class Arm:
    """One fully specified rule and the model it holds, or no rule at all"""

    name: str
    method: MethodConfig | None
    belief: BeliefConfig


def _check_unspecified(block: Any) -> None:
    """Raises where any field was left unset"""
    unspecified = [f.name for f in fields(block) if getattr(block, f.name) == MISSING]
    if unspecified:
        raise ValueError(f"{type(block).__name__} was built without {', '.join(unspecified)}")


def arm(name: str, method: MethodConfig | None, belief: BeliefConfig) -> Arm:
    """One arm of a sweep, refused unless every setting of it was given"""
    if method is not None:
        _check_unspecified(method)
    _check_unspecified(belief)
    return Arm(name, method, belief)


def sweep(
    name: str,
    *,
    problem: ProblemConfig,
    horizon: int,
    claim_every: int,
    recorded: dict[str, int],
    seeds: Sequence[int],
    resources: Resources,
    arms: Sequence[Arm],
) -> None:
    """Register one sweep: its arms crossed with its seeds, the world they share, and what a cell is given"""
    _check_unspecified(problem)
    _check_unspecified(resources)
    for one in arms:
        cs.store(
            group="arm",
            name=f"{name}_{one.name}",
            package="_global_",
            node={
                "problem": problem,
                "belief": one.belief,
                "method": one.method,
                "horizon": horizon,
                "claim_every": claim_every,
                "recorded": recorded,
            },
        )
    cs.store(
        group="sweep",
        name=name,
        package="_global_",
        node={
            "resources": resources,  # read when the launcher is built, before any arm is composed
            "hydra": {
                "mode": "MULTIRUN",
                "sweeper": {
                    "params": {"+arm": ",".join(f"{name}_{one.name}" for one in arms), "seed": ",".join(str(seed) for seed in seeds)}
                },
                "sweep": {
                    "dir": f"results/ambient_waypoint/{name}/${{now:%Y-%m-%d_%H-%M-%S}}",  # sorts lexicographically, holds no colon
                    "subdir": "arm=${hydra:runtime.choices.arm},seed=${seed}",
                },
                "job": {"name": name},
            },
        },
    )


Implementation = dict[str, Any]
"""A dotted path and the arguments to build it with"""


def _built(path: str, **arguments: Any) -> Implementation:
    """One dotted path and its arguments, as the mapping hydra builds from"""
    return {"_target_": path, **arguments}


# --- utilities: which quantity of a belief a rule maximizes --------------------------------------------------------


def max_magnitude() -> Implementation:
    """The largest reading magnitude among the cells scored"""
    return _built("zermelo.methods.waypoint_bo.utility.MaxMagnitude")


def posterior_spread() -> Implementation:
    """The posterior standard deviations at the scored cells, summed"""
    return _built("zermelo.methods.waypoint_bo.utility.PosteriorSpread")


def upper_confidence(c: float) -> Implementation:
    """The largest optimistic magnitude `||mean|| + c * spread` at the scored cells"""
    return _built("zermelo.methods.waypoint_bo.utility.UpperConfidence", c=c)


def expected_improvement() -> Implementation:
    """`E[(|f| - beta)^+]` at each scored cell in closed form, `beta` the largest magnitude read so far"""
    return _built("zermelo.methods.waypoint_bo.utility.ExpectedImprovement")


def predictive_confidence() -> Implementation:
    """Negative predictive entropy over the candidate cells, under the belief re-conditioned on the scored cells"""
    return _built("zermelo.methods.waypoint_bo.utility.PredictiveConfidence")


def uniform() -> Implementation:
    """A draw ignoring the data"""
    return _built("zermelo.methods.waypoint_bo.utility.Uniform")


# --- planners: how the actor is carried to a waypoint, and how that walk is priced ----------------------------------


def value_iteration(max_steps: int, radius: float, target_chunk: int | None) -> Implementation:
    """`v_0 = 0`, `v_{k+1}(s) = 0 on arrival, else min over a of c(s, a) + sum over s' of P(s' | s, a) v_k(s')`, `max_steps` rounds"""
    return _built(
        "zermelo.methods.waypoint_bo.planner.ValueIteration", max_steps=max_steps, radius=radius, target_chunk=target_chunk, cost_weight=0.0
    )


def greedy(max_steps: int, radius: float, target_chunk: int | None) -> Implementation:
    """One backup on the distance to the target: the act whose next cell is nearest it in expectation"""
    return _built(
        "zermelo.methods.waypoint_bo.planner.Greedy", max_steps=max_steps, radius=radius, target_chunk=target_chunk, cost_weight=0.0
    )


def random_walk(max_steps: int, radius: float, target_chunk: int | None) -> Implementation:
    """One uniform act per cell, drawn when the plan is made and the same for every target"""
    return _built(
        "zermelo.methods.waypoint_bo.planner.RandomWalk", max_steps=max_steps, radius=radius, target_chunk=target_chunk, cost_weight=0.0
    )
