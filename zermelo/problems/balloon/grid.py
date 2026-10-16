"""The sphere a balloon's position lives on, and the state domain it is one part of"""

import math
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from zermelo.interface import Domain, Embeddable, Enumerable, ProductDomain, Subset

EARTH_RADIUS_KM = 6371.0
"""The sphere embedded coordinates are measured on"""


@dataclass(frozen=True)
class SphereGrid(Domain[Float[Array, " 2"]], Enumerable[Float[Array, " 2"]], Embeddable[Float[Array, " 2"]]):
    """A latitude-longitude grid of `n_lat * n_lon` cells, an element being degrees (latitude, longitude)"""

    n_lat: int
    n_lon: int

    lat_first: float
    """The latitude of row 0, in degrees"""

    lon_first: float
    """The longitude of column 0, in degrees"""

    lat_step: float
    """Degrees from one row to the next, negative where latitude descends"""

    lon_step: float
    """Degrees from one column to the next"""

    @property
    def steps(self) -> Float[Array, " 2"]:
        """The per-axis spacing, in degrees"""
        return jnp.asarray([self.lat_step, self.lon_step])

    @property
    def first(self) -> Float[Array, " 2"]:
        """The per-axis coordinate of cell 0"""
        return jnp.asarray([self.lat_first, self.lon_first])

    @property
    def shape(self) -> Int[Array, " 2"]:
        """Cells per axis"""
        return jnp.asarray([self.n_lat, self.n_lon])

    def cell_of(self, x: Float[Array, "*batch 2"]) -> Int[Array, "*batch 2"]:
        """The per-axis cell of `x`: round((x - first) / step), clipped into range"""
        return jnp.clip(jnp.round((x - self.first) / self.steps), 0, self.shape - 1).astype(jnp.int32)

    def coords_of(self, c: Int[Array, "*batch 2"]) -> Float[Array, "*batch 2"]:
        """The degrees at per-axis cell `c`: first + c * step"""
        return self.first + c * self.steps

    def contains(self, x: Float[Array, " 2"]) -> Bool[Array, ""]:
        """True where `x` sits on a cell of this grid"""
        return jnp.allclose(self.project(x), x)

    def project(self, x: Float[Array, "*batch 2"]) -> Float[Array, "*batch 2"]:
        """`x` snapped to the nearest cell and clipped onto the grid"""
        return self.coords_of(self.cell_of(x))

    def narrow(self, witness: Bool[Array, " pos"]) -> Subset[Float[Array, " 2"]]:
        """This grid restricted to the cells `witness` marks, in this grid's index order"""
        return Subset(self, witness)

    def size(self) -> int:
        """n_lat * n_lon"""
        return self.n_lat * self.n_lon

    def index_of(self, x: Float[Array, " 2"]) -> Int[Array, ""]:
        """The flat index of `x`"""
        return self.flat_index(x)

    def flat_index(self, x: Float[Array, "*batch 2"]) -> Int[Array, "*batch"]:
        """row * n_lon + column, the longitude varying fastest. Batches, unlike `index_of`"""
        c = self.cell_of(x)
        return c[..., 0] * self.n_lon + c[..., 1]

    def from_index(self, i: Int[Array, ""]) -> Float[Array, " 2"]:
        """The element at flat index `i`"""
        return self.coords_of(jnp.stack([i // self.n_lon, i % self.n_lon], axis=-1))

    def elements(self) -> Float[Array, "pos 2"]:
        """Every cell, as (n_lat * n_lon, 2) degrees in flat index order"""
        i = jnp.arange(self.size())
        return self.coords_of(jnp.stack([i // self.n_lon, i % self.n_lon], axis=-1))

    def dim(self) -> int:
        """3: a point on the sphere"""
        return 3

    def embed(self, x: Float[Array, "*batch 2"]) -> Float[Array, "*batch 3"]:
        """radius * (cos(lat) cos(lon), cos(lat) sin(lon), sin(lat)), so a chord approximates an arc"""
        lat, lon = jnp.radians(x[..., 0]), jnp.radians(x[..., 1])
        return EARTH_RADIUS_KM * jnp.stack([jnp.cos(lat) * jnp.cos(lon), jnp.cos(lat) * jnp.sin(lon), jnp.sin(lat)], axis=-1)

    def unembed(self, v: Float[Array, "*batch 3"]) -> Float[Array, "*batch 2"]:
        """The degrees `v` points at, snapped onto the grid"""
        lat = jnp.degrees(jnp.arcsin(jnp.clip(v[..., 2] / EARTH_RADIUS_KM, -1.0, 1.0)))
        return self.project(jnp.stack([lat, jnp.degrees(jnp.arctan2(v[..., 1], v[..., 0]))], axis=-1))

    def km_per_degree(self, lat: Float[Array, "*batch"]) -> Float[Array, "*batch 2"]:
        """Kilometres one degree spans at latitude `lat`: (meridional, zonal) = R * (1, cos(lat)) * pi / 180"""
        per_degree = EARTH_RADIUS_KM * math.pi / 180.0
        return jnp.stack([jnp.full_like(lat, per_degree), per_degree * jnp.cos(jnp.radians(lat))], axis=-1)


@dataclass(frozen=True)
class Steps(Domain[Int[Array, ""]], Enumerable[Int[Array, ""]], Embeddable[Int[Array, ""]]):
    """The integers `0` to `len(coordinate) - 1`, step `i` embedding to `coordinate[i]`"""

    coordinate: tuple[float, ...]
    """Where each step sits, in the units distance between states is measured in"""

    def contains(self, x: Int[Array, ""]) -> Bool[Array, ""]:
        return (0 <= x) & (x < len(self.coordinate))

    def project(self, x: Int[Array, ""]) -> Int[Array, ""]:
        return jnp.clip(x, 0, len(self.coordinate) - 1)

    def narrow(self, witness: Bool[Array, " n"]) -> Subset[Int[Array, ""]]:
        """`witness` marks which steps are live now"""
        return Subset(self, witness)

    def size(self) -> int:
        return len(self.coordinate)

    def index_of(self, x: Int[Array, ""] | Float[Array, ""]) -> Int[Array, ""]:
        """The step itself, or the nearest one where `x` sits between steps"""
        return jnp.asarray(jnp.round(x), jnp.int32)

    def from_index(self, i: Int[Array, ""]) -> Int[Array, ""]:
        return i

    def elements(self) -> Int[Array, " n"]:
        return jnp.arange(len(self.coordinate))

    def dim(self) -> int:
        """1"""
        return 1

    def embed(self, x: Int[Array, "*batch"] | Float[Array, "*batch"]) -> Float[Array, "*batch 1"]:
        """coordinate[x], read linearly between steps where `x` is fractional"""
        return jnp.interp(x, jnp.arange(len(self.coordinate)), jnp.asarray(self.coordinate))[..., None]

    def unembed(self, v: Float[Array, "*batch 1"]) -> Int[Array, "*batch"]:
        """The step nearest `v`"""
        return jnp.argmin(jnp.abs(jnp.asarray(self.coordinate) - v[..., :1]), axis=-1)


def balloon_states(grid: SphereGrid, altitude_km: tuple[float, ...]) -> ProductDomain:
    """The state domain: where the balloon is, and which altitude it flies at."""
    return ProductDomain({"position": grid, "altitude": Steps(altitude_km)})
