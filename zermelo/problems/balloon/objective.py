"""What a balloon is scored on: predicting where a chosen reading of the wind is largest

W(lat, lon, p) = (u, v)     the wind (m/s), u eastward and v northward
g(W; lat, lon)              the scalar g reads off W at one cell
claim                       g predicted at every candidate cell, as a mean and a log-variance
"""

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from zermelo.interface import BoxDomain, Decision, Domain, FunctionDomain, Objective, ProductDomain, Subset
from zermelo.problems.balloon.field import WindField
from zermelo.problems.balloon.grid import SphereGrid


class Target(ABC):
    """The scalar g an episode is scored on predicting"""

    @abstractmethod
    def __call__(self, field: WindField, candidates: dict[str, Any]) -> Float[Array, " n_candidates"]:
        """g at every candidate state"""


@dataclass(frozen=True)
class PointSpeed(Target):
    """g(W; lat, lon, p) = ||W(lat, lon, p)||"""

    def __call__(self, field: WindField, candidates: dict[str, Any]) -> Float[Array, " n_candidates"]:
        """Wind speed at each candidate's own altitude and cell"""
        return jnp.linalg.norm(field(candidates), axis=-1)


@dataclass(frozen=True)
class ColumnPeakSpeed(Target):
    """g(W; lat, lon, p) = max over p' of ||W(lat, lon, p')||, one value shared down a column"""

    n_alt: int
    """How many altitudes the column spans"""

    def __call__(self, field: WindField, candidates: dict[str, Any]) -> Float[Array, " n_candidates"]:
        """The fastest wind anywhere in each candidate's column"""
        cells = candidates["position"]  # (n_candidates, 2)
        per_altitude = jnp.stack(
            [jnp.linalg.norm(field({"position": cells, "altitude": jnp.full(cells.shape[0], p)}), axis=-1) for p in range(self.n_alt)]
        )  # (alt, n_candidates)
        return jnp.max(per_altitude, axis=0)


@dataclass(frozen=True)
class StormSearch(Objective[dict[str, Float[Array, "..."]]]):
    """Reward is the increment in the largest g the balloon has flown through, with the claim recorded beside it"""

    target: Target

    candidates: Subset[dict[str, Any]]
    """The states (u, v) is predicted at, over the state domain a belief is maintained on"""

    candidate_states: dict[str, Any] = dataclasses.field(init=False, repr=False)
    """The live candidates, gathered once"""

    candidate_indices: Int[Array, " n_candidates"] = dataclasses.field(init=False, repr=False)
    """Their indices into the state domain"""

    def __post_init__(self) -> None:
        """Gather the live candidates as concrete arrays"""
        live = jnp.flatnonzero(self.candidates.live)  # (n_candidates,) indices into the state domain
        object.__setattr__(self, "candidate_indices", live)
        object.__setattr__(self, "candidate_states", jax.tree.map(lambda a: a[live], self.candidates.elements()))

    @property
    def claim_domain(self) -> Domain:
        """A claim is a function: the mean of (u, v) at any candidate, then their log-variances"""
        return FunctionDomain(self.candidates, BoxDomain((4,)))

    def truth(self, state: dict[str, Any]) -> Float[Array, " n_candidates"]:
        """g at every candidate, at the hour the state carries"""
        return self.target(state["field"], self.candidate_states | {"hours": state["hours"]})

    def wind_speed_at(self, state: dict[str, Any]) -> Float[Array, ""]:
        """||W|| where the balloon stands"""
        return jnp.linalg.norm(state["field"](state))

    def reset(self, state: dict[str, Any]) -> dict[str, Float[Array, "..."]]:
        """The objective's memory at step zero"""
        truth = self.truth(state)  # (n_candidates,)
        return {
            "incumbent": self.wind_speed_at(state),
            "oracle": jnp.max(truth),
            "truth": truth,
            "claim": jnp.zeros((truth.shape[0], 4)),  # (n_candidates, 4): the mean of (u, v), then their log-variances
        }

    def score(
        self, objective_state: dict[str, Float[Array, "..."]], state: dict[str, Any], decision: Decision, next_state: dict[str, Any]
    ) -> tuple[dict[str, Float[Array, "..."]], Float[Array, ""]]:
        """What arriving improved on the largest wind flown through, and the claim as it stood"""
        magnitude, incumbent = self.wind_speed_at(next_state), objective_state["incumbent"]
        carried = objective_state | {
            "incumbent": jnp.maximum(incumbent, magnitude),
            "claim": decision.claim(self.candidate_states),  # (n_candidates, 4)
        }
        return carried, jnp.maximum(magnitude - incumbent, 0.0)


def balloon_interior(grid: SphereGrid, margin_lat: int, margin_lon: int) -> Bool[Array, " pos"]:
    """The grid without the band `margin_lat` rows and `margin_lon` columns deep on every side"""
    cells = grid.cell_of(grid.elements())  # (pos, 2)
    return (
        (cells[:, 0] >= margin_lat)
        & (cells[:, 0] < grid.n_lat - margin_lat)
        & (cells[:, 1] >= margin_lon)
        & (cells[:, 1] < grid.n_lon - margin_lon)
    )


def balloon_candidates(states: ProductDomain, grid: SphereGrid, margin_lat: int, margin_lon: int) -> Subset[dict[str, Any]]:
    """`states` over the grid's interior, at every altitude"""
    interior = balloon_interior(grid, margin_lat, margin_lon)  # (pos,)
    return states.narrow(interior[grid.flat_index(states.elements()["position"])])
