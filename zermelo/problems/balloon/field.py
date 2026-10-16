"""The wind a balloon is carried by: how it is read, and the law an episode's own is drawn by

W(lat, lon, p) = (u, v)     u: eastward, v: northward, both (m/s)
lat, lon                    degrees, on the grid's own positions
p                           which altitude, an index into the ones the record holds
t                           hours since the episode began
"""

import dataclasses
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from zermelo.interface import Domain, Function, FunctionDomain, Prior
from zermelo.problems.balloon.grid import SphereGrid


class WindField(ABC):
    """A read of W at one state"""

    @abstractmethod
    def __call__(self, state: dict[str, Any]) -> Float[Array, "*batch uv"]:
        """(u, v) where the state sits"""


@register_dataclass
@dataclass(frozen=True)
class GridWind(WindField):
    """W(lat, lon, p), held per altitude and position, read at the nearest one"""

    wind: Float[Array, "alt pos uv"]
    """(u, v) at every altitude and position, in the grid's index order"""

    grid: SphereGrid = dataclasses.field(metadata=dict(static=True))
    """What `pos` is indexed by"""

    def __call__(self, state: dict[str, Any]) -> Float[Array, "*batch uv"]:
        """wind[p, pos]"""
        return self.wind[state["altitude"], self.grid.flat_index(state["position"])]


@register_dataclass
@dataclass(frozen=True)
class DriftingWind(WindField):
    """W(lat, lon, p, t), read at the nearest position and hour"""

    wind: Float[Array, "t alt pos uv"]
    """(u, v) at every hour, altitude and position"""

    hours: Float[Array, " t"]
    """The t each record stands for"""

    grid: SphereGrid = dataclasses.field(metadata=dict(static=True))
    """What `pos` is indexed by"""

    def __call__(self, state: dict[str, Any]) -> Float[Array, "*batch uv"]:
        """wind[argmin |hours - t|, p, pos]"""
        hours = state["hours"]
        frame = jnp.argmin(jnp.abs(self.hours[:, None] - jnp.atleast_1d(hours)[None, :]), axis=0).reshape(jnp.shape(hours))
        return self.wind[frame, state["altitude"], self.grid.flat_index(state["position"])]


@register_dataclass
@dataclass(frozen=True)
class SmoothWind(WindField):
    """W(lat, lon, p, t), read linearly between the records it is held at: differentiable in every coordinate

        W(x) = sum over the 16 corners c of the cell x falls in of  prod over axes of (w_axis if c is high else 1 - w_axis) * wind[c]

    with `w_axis` the fraction of the way x sits from the low corner to the high one along (lat, lon, p, t);
    a coordinate past the last record reads the last record, so W is constant outside the grid.
    """

    wind: Float[Array, "*winds t alt pos uv"]
    """(u, v) at every hour, altitude and position; one frame for a wind that never moves. Several winds stacked along
    leading axes make a batch, read one at a time under `jax.vmap`"""

    hours: Float[Array, "*winds t"]
    """The t each frame stands for, increasing"""

    grid: SphereGrid = dataclasses.field(metadata=dict(static=True))
    """What `pos` is indexed by"""

    def __call__(self, state: dict[str, Any]) -> Float[Array, "*batch uv"]:
        """The weighted sum over the corners around (lat, lon, p, t), for one wind"""
        n_t, n_alt = self.wind.shape[0], self.wind.shape[1]
        cell = (state["position"] - self.grid.first) / self.grid.steps  # (..., 2) fractional (row, column)
        frame = jnp.interp(state.get("hours", jnp.zeros(())), self.hours, jnp.arange(n_t) * 1.0)
        axes = [cell[..., 0], cell[..., 1], jnp.asarray(state["altitude"]) * 1.0, frame]
        sizes = [self.grid.n_lat, self.grid.n_lon, n_alt, n_t]
        lows = [jnp.clip(jnp.floor(x), 0, max(n - 2, 0)).astype(jnp.int32) for x, n in zip(axes, sizes, strict=True)]
        weights = [jnp.clip(x - lo, 0.0, 1.0) for x, lo in zip(axes, lows, strict=True)]
        out = jnp.zeros((*cell.shape[:-1], 2))
        for corner in itertools.product((0, 1), repeat=4):
            at = [jnp.minimum(lo + c, n - 1) for lo, c, n in zip(lows, corner, sizes, strict=True)]
            weight = jnp.ones(())
            for w, c in zip(weights, corner, strict=True):  # multiplied rather than stacked: the axes may carry different batch shapes
                weight = weight * (w if c else 1.0 - w)
            out = out + weight[..., None] * self.wind[at[3], at[2], at[0] * self.grid.n_lon + at[1]]
        return out


def smoothed(table: Function, grid: SphereGrid, n_alt: int) -> SmoothWind:
    """A field held at the grid's cells and levels (a belief's mean, say), made readable between them"""
    cells = grid.elements()  # (pos, 2)
    probe = {"position": jnp.tile(cells, (n_alt, 1)), "altitude": jnp.repeat(jnp.arange(n_alt), cells.shape[0])}
    return SmoothWind(table(probe).reshape(1, n_alt, cells.shape[0], -1), jnp.zeros(1), grid)


class WindError(ABC):
    """Where e comes from, for a truth W = forecast + e"""

    @abstractmethod
    def sample(self, key: PRNGKeyArray, grid: SphereGrid, n_altitudes: int) -> Float[Array, "*hours alt pos uv"]:
        """One draw of e at every altitude and position"""


@dataclass(frozen=True)
class GaussianError(WindError):
    """e ~ N(0, K), independently per altitude and per component of (u, v), with

    K(a, b) = amplitude_ms^2 * exp(-||embed(a) - embed(b)||^2 / (2 * length_scale_km^2))
    """

    length_scale_km: float
    amplitude_ms: float

    jitter: float
    """Added to K's diagonal relative to amplitude_ms^2, so the factorisation succeeds"""

    def sample(self, key: PRNGKeyArray, grid: SphereGrid, n_altitudes: int) -> Float[Array, "alt pos uv"]:
        """L @ white, for L the Cholesky factor of K. Time O(pos^3), memory O(pos^2)"""
        points = grid.embed(grid.elements())  # (pos, 3)
        gap = jnp.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)  # (pos, pos)
        correlation = jnp.exp(-gap / (2.0 * self.length_scale_km**2)) + self.jitter * jnp.eye(grid.size())
        factor = self.amplitude_ms * jnp.linalg.cholesky(correlation)
        if jnp.isnan(factor).any():  # a singular K returns NaN rather than raising
            raise ValueError(f"K over {grid.size()} positions at {self.length_scale_km} km does not factorise at jitter {self.jitter}")
        white = jax.random.normal(key, (grid.size(), n_altitudes, 2))
        return jnp.moveaxis(jnp.tensordot(factor, white, axes=(1, 0)), 0, 1)  # (alt, pos, uv)


@register_dataclass
@dataclass(frozen=True)
class GEFSError(WindError):
    """e = W - F, the same field at every draw"""

    error: Float[Array, "*hours alt pos uv"]
    """W - F at every altitude and position"""

    def sample(self, key: PRNGKeyArray, grid: SphereGrid, n_altitudes: int) -> Float[Array, "*hours alt pos uv"]:
        """e, whatever the key"""
        return self.error


@dataclass(frozen=True)
class ForecastPrior(Prior[WindField]):
    """W = forecast + e, the forecast fixed and e taken once per episode"""

    forecast: Float[Array, "alt pos uv"] | Float[Array, "t alt pos uv"]
    """(u, v) predicted at every altitude and position, and what the agent is told"""

    error: WindError
    """Where e comes from"""

    grid: SphereGrid
    """What `pos` is indexed by"""

    hours: Float[Array, " t"] | None
    """The t each frame of a drifting forecast stands for, None for one frame that never moves"""

    smooth: bool = False
    """Whether W is read between records (a continuous balloon) or at the nearest one"""

    def sample(self, domain: Domain[WindField], key: PRNGKeyArray) -> WindField:
        """One episode's W"""
        if not isinstance(domain, FunctionDomain):
            raise TypeError(f"a {type(domain).__name__} holds no fields, so none can be drawn from it")
        if self.hours is None:
            wind = self.forecast + self.error.sample(key, self.grid, self.forecast.shape[0])
            return SmoothWind(wind[None], jnp.zeros(1), self.grid) if self.smooth else GridWind(wind, self.grid)
        error = self.error.sample(key, self.grid, self.forecast.shape[1])  # (alt, pos, uv) or (t, alt, pos, uv)
        wind = self.forecast + (error if error.ndim == self.forecast.ndim else error[None])
        return SmoothWind(wind, self.hours, self.grid) if self.smooth else DriftingWind(wind, self.hours, self.grid)
