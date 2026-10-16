"""What a balloon sees: where it is, what it has left, the wind it measures, and the forecast it was given

s = (lat, lon, p, r, W)     where it is, which altitude, what resource is left, the true wind
W(lat, lon, p) = (u, v)     wind in m/s, u eastward and v northward
F                           the forecast, known everywhere from the start
"""

import dataclasses
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from zermelo.interface import BoxDomain, Domain, ProductDomain, Readout
from zermelo.problems.balloon.grid import SphereGrid


@dataclass(frozen=True)
class BalloonReadout(Readout):
    """Where the balloon is and what it measures of W, with F carried alongside"""

    grid: SphereGrid
    n_alt: int
    forecast: Float[Array, "alt pos uv"]
    """F at every altitude and position, handed over whole at every step"""

    states: ProductDomain
    """Where the balloon is, as the domain a belief over the wind is written on"""

    @property
    @abstractmethod
    def wind_shape(self) -> tuple[int, ...]:
        """The shape of one measurement of W"""

    @abstractmethod
    def measure(self, state: dict[str, Any]) -> Float[Array, "*wind"]:
        """What W reads where the balloon is"""

    @property
    def context(self) -> dict[str, int]:
        """The hour a reading was taken at, the wind it measures having moved by then"""
        return {"hours": 1}

    @property
    def readings(self) -> Domain:
        """Where the balloon is, what it has left, when it read, one measurement of W, and F"""
        return ProductDomain(
            {
                "position": self.states,
                "balloon_resource": BoxDomain(()),
                "context": BoxDomain((1,)),
                "wind": BoxDomain(self.wind_shape),
                "forecast": BoxDomain(self.forecast.shape),
            }
        )

    def reset(self, key: PRNGKeyArray, state: dict[str, Any]) -> dict[str, Any]:
        """The reading before acting"""
        return {
            "position": {part: state[part] for part in self.states.parts},
            "balloon_resource": state["balloon_resource"],
            "context": jnp.atleast_1d(state["hours"]),
            "wind": self.measure(state),
            "forecast": self.forecast,
        }

    def step(self, key: PRNGKeyArray, state: dict[str, Any], action: Any, next_state: dict[str, Any]) -> dict[str, Any]:
        """The reading after acting"""
        return self.reset(key, next_state)


@dataclass(frozen=True)
class PointWind(BalloonReadout):
    """W at the state itself: what a balloon carrying one anemometer reads"""

    @property
    def wind_shape(self) -> tuple[int, ...]:
        return (2,)

    def measure(self, state: dict[str, Any]) -> Float[Array, " uv"]:
        """W(lat, lon, p)"""
        return state["field"](state)


@dataclass(frozen=True)
class ColumnWind(BalloonReadout):
    """W at every altitude above and below the balloon: what a sounding of the whole column reads"""

    @property
    def wind_shape(self) -> tuple[int, ...]:
        return (self.n_alt, 2)

    def measure(self, state: dict[str, Any]) -> Float[Array, "alt uv"]:
        """W(lat, lon, p') at every p'"""
        column = {**state, "altitude": jnp.arange(self.n_alt), "position": jnp.broadcast_to(state["position"], (self.n_alt, 2))}
        return state["field"](column)
