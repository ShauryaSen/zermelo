"""A balloon problem, assembled from a wind field and the forecast of it"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from zermelo.interface import BoxDomain, Domain, Enumerable, FunctionDomain, Prior, ProductDomain, ProductPrior, Transition, Uniform, World
from zermelo.problems.balloon.field import ForecastPrior, WindError
from zermelo.problems.balloon.grid import SphereGrid, Steps
from zermelo.problems.balloon.objective import StormSearch, Target, balloon_candidates, balloon_interior
from zermelo.problems.balloon.readout import BalloonReadout
from zermelo.problems.balloon.transition import Advection, Ascent, BalloonTransition, Clock, SmoothBalloonTransition


class Highest(Prior[Any]):
    """The largest element of a finite domain, drawn with probability one"""

    def sample(self, domain: Domain[Any], key: PRNGKeyArray) -> Any:
        """The element at `size() - 1`"""
        if not isinstance(domain, Enumerable):
            raise TypeError(f"a {type(domain).__name__} has no largest element, so none can be drawn from it")
        return domain.from_index(jnp.asarray(domain.size() - 1))


class Zero(Prior[Float[Array, ""]]):
    """The scalar zero, drawn with probability one"""

    def sample(self, domain: Domain[Float[Array, ""]], key: PRNGKeyArray) -> Float[Array, ""]:
        """0.0"""
        return jnp.zeros(())


@dataclass(frozen=True)
class WindRecord:
    """W and F as (t, alt, pos, uv)

    W   True wind
    F   Forecast (issued at t=0, valid at t)
    """

    wind: Float[Array, "t alt pos uv"]
    forecast: Float[Array, "t alt pos uv"]
    hours: Float[Array, " t"]
    altitude_km: Float[Array, " alt"]
    grid: SphereGrid

    @property
    def n_alt(self) -> int:
        """How many altitudes the record holds"""
        return self.wind.shape[1]

    def at(self, frame: int) -> Float[Array, "alt pos uv"]:
        """W[frame,..,.]"""
        return self.wind[frame]

    def forecast_at(self, frame: int) -> Float[Array, "alt pos uv"]:
        """F[frame,..,.]"""
        return self.forecast[frame]

    def window(self, frames: slice) -> tuple[tuple[Float[Array, "t alt pos uv"], Float[Array, "t alt pos uv"]], Float[Array, " t"]]:
        """(W, F) over `frames`, and the hours they stand for rebased so the episode starts at zero"""
        return (self.wind[frames], self.forecast[frames]), self.hours[frames] - self.hours[frames][0]


def load_wind(path: Path) -> WindRecord:
    """The wind record stored at `path`, laid out for the grid it sits on"""
    raw = np.load(path)
    lat, lon = raw["latitude"], raw["longitude"]
    grid = SphereGrid(
        n_lat=len(lat),
        n_lon=len(lon),
        lat_first=float(lat[0]),
        lon_first=float(lon[0]),
        lat_step=float(lat[1] - lat[0]),
        lon_step=float(lon[1] - lon[0]),
    )
    shape = (raw["u"].shape[0], raw["u"].shape[1], grid.size(), 2)  # (t, alt, pos, uv)
    wind = np.stack([raw["u"], raw["v"]], axis=-1).reshape(shape)
    forecast = np.stack([raw["forecast_u"], raw["forecast_v"]], axis=-1).reshape(shape)
    return WindRecord(jnp.asarray(wind), jnp.asarray(forecast), jnp.asarray(raw["hours"]), jnp.asarray(raw["altitude_km"]), grid)


def balloon_transition(recording: WindRecord, states: ProductDomain, step_hours: float) -> BalloonTransition:
    """One step of the balloon over `states`: the wind carries it, the action moves it an altitude, and the clock updates"""
    return BalloonTransition(Advection(recording.grid, step_hours), Ascent(recording.n_alt), Clock(step_hours), states)


def smooth_transition(recording: WindRecord, states: ProductDomain, step_hours: float, climb_cap_km: float) -> SmoothBalloonTransition:
    """One step of a balloon with real coordinates, over the levels `states` holds"""
    altitudes = states.parts["altitude"]
    if not isinstance(altitudes, Steps):
        raise TypeError(f"a smooth balloon climbs between the record's levels, and the altitude part is a {type(altitudes).__name__}")
    return SmoothBalloonTransition(recording.grid, altitudes, step_hours, climb_cap_km)


def balloon_world(
    recording: WindRecord,
    states: ProductDomain,
    forecast: Float[Array, "alt pos uv"] | Float[Array, "t alt pos uv"],
    error: WindError,
    transition: Transition[Any],
    readout: type[BalloonReadout],
    resource_units: int,
    margin_lat: int,
    margin_lon: int,
    hours: Float[Array, " t"] | None,
    continuous: bool = False,
) -> World:
    """The problem a forecast poses: the balloon starts over the grid's interior, and the wind is that forecast plus an error.

    Continuous, the altitude is any real level between the record's first and last, and the wind is read between records.
    """
    grid = recording.grid
    initial_forecast = forecast if hours is None else forecast[0]  # what the agent is told at the hour it starts
    altitude = BoxDomain((), 0.0, float(recording.n_alt - 1)) if continuous else states.parts["altitude"]
    return World(
        state_domain=ProductDomain(
            {
                **states.parts,
                "position": grid.narrow(balloon_interior(grid, margin_lat, margin_lon)),
                "altitude": altitude,
                "balloon_resource": Steps((0.0,) * (resource_units + 1)),
                "hours": BoxDomain(()),
                "field": FunctionDomain(grid, BoxDomain((2,))),
            }
        ),
        prior=ProductPrior(
            {
                "position": Uniform(),
                "altitude": Uniform(),
                "balloon_resource": Highest(),
                "hours": Zero(),
                "field": ForecastPrior(forecast, error, grid, hours, smooth=continuous),
            }
        ),
        transition=transition,
        readout=readout(grid, recording.n_alt, initial_forecast, states),
    )


def balloon_objective(recording: WindRecord, states: ProductDomain, target: Target, margin_lat: int, margin_lon: int) -> StormSearch:
    """What the episode is scored on: `target` predicted over the states above the grid's interior"""
    return StormSearch(target, balloon_candidates(states, recording.grid, margin_lat, margin_lon))
