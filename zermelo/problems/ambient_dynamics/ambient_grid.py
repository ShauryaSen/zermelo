"""The grid a position part lives on, in coordinates: padded, or wrapped"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from zermelo.interface import Domain, Embeddable, Enumerable, ProductDomain, Subset


@dataclass(frozen=True)
class GridDomain(Domain[Float[Array, " axes"]], Enumerable[Float[Array, " axes"]], Embeddable[Float[Array, " axes"]], ABC):
    """`n_axes` axes of `n_cells` cells `cell_spacing` apart; an element is one coordinate per axis"""

    n_axes: int
    n_cells: int

    cell_spacing: float
    """The coordinate distance from one cell to the next along an axis"""

    @property
    @abstractmethod
    def extent(self) -> float:
        """The coordinate span of one axis"""

    @property
    @abstractmethod
    def cells(self) -> int:
        """Cells per axis, any pad included"""

    @abstractmethod
    def cell_of(self, x: Float[Array, "*batch axes"]) -> Int[Array, "*batch axes"]:
        """The cell index of each axis of `x`, in [0, cells)"""

    @abstractmethod
    def coords_of(self, c: Int[Array, "*batch axes"]) -> Float[Array, "*batch axes"]:
        """The coordinate of each per-axis cell index `c`"""

    @abstractmethod
    def dim(self) -> int:
        """The width `k` of an embedded element: `n_axes`, or more where an axis needs several numbers"""

    @abstractmethod
    def embed(self, x: Float[Array, "*batch axes"]) -> Float[Array, "*batch k"]:
        """`k` real coordinates for `x`, nearby cells landing on nearby vectors"""

    @abstractmethod
    def unembed(self, v: Float[Array, "*batch k"]) -> Float[Array, "*batch axes"]:
        """The element `v` embeds, snapped back onto this grid"""

    def contains(self, x: Float[Array, " axes"]) -> Bool[Array, ""]:
        """True where `x` already sits exactly on a cell of this grid"""
        return jnp.allclose(self.project(x), x)

    def narrow(self, witness: Bool[Array, " n"]) -> Subset[Float[Array, " axes"]]:
        """This grid restricted to the cells `witness` marks, in this grid's own index order"""
        return Subset(self, witness)

    def size(self) -> int:
        """How many elements: cells ** n_axes"""
        return self.cells**self.n_axes

    def _strides(self) -> Int[Array, " axes"]:
        """The stride of each axis: cells ** (n_axes - 1 - i), so the last axis varies fastest"""
        return jnp.asarray([self.cells ** (self.n_axes - 1 - i) for i in range(self.n_axes)])

    def index_of(self, x: Float[Array, " axes"]) -> Int[Array, ""]:
        """The flat index of `x`: sum over axes of cell_of(x) * stride"""
        return jnp.sum(self.cell_of(x) * self._strides())

    def from_index(self, i: Int[Array, ""]) -> Float[Array, " axes"]:
        """The element at flat index `i`"""
        return self.coords_of(i // self._strides() % self.cells)

    def elements(self) -> Float[Array, "n axes"]:
        """Every element, as (cells ** n_axes, n_axes) coordinates in flat index order"""
        return self.coords_of(jnp.arange(self.size())[:, None] // self._strides() % self.cells)


@dataclass(frozen=True)
class PaddedGridDomain(GridDomain):
    """[0, extent] plus a ring of `pad` cells on each side of every axis, embedded as itself"""

    pad: int = 0

    @property
    def extent(self) -> float:
        """cell_spacing * (n_cells - 1), both ends of [0, extent] being cells"""
        return self.cell_spacing * (self.n_cells - 1)

    @property
    def cells(self) -> int:
        """n_cells + 2 * pad"""
        return self.n_cells + 2 * self.pad

    def project(self, x: Float[Array, "*batch axes"]) -> Float[Array, "*batch axes"]:
        """`x` rounded to the nearest cell, then clipped into cells -pad .. n_cells - 1 + pad"""
        return jnp.clip(jnp.round(x / self.cell_spacing), -self.pad, self.n_cells - 1 + self.pad) * self.cell_spacing

    def inward_direction(self, x: Float[Array, "*batch axes"]) -> Int[Array, "*batch axes"]:
        """Which way each axis must move to re-enter [0, extent]: +1 below it, -1 above it, 0 inside"""
        c = jnp.round(x / self.cell_spacing)  # the interior is cells 0 .. n_cells - 1; anything else is in the pad
        return jnp.where(c < 0, 1, jnp.where(c > self.n_cells - 1, -1, 0)).astype(jnp.int32)

    def in_pad(self, x: Float[Array, "*batch axes"]) -> Bool[Array, "*batch"]:
        """True where `x` is in the pad"""
        return jnp.any(self.inward_direction(x) != 0, axis=-1)

    def cell_of(self, x: Float[Array, "*batch axes"]) -> Int[Array, "*batch axes"]:
        """round(x / cell_spacing) + pad, making cell 0 the outermost pad cell"""
        return (jnp.round(x / self.cell_spacing) + self.pad).astype(jnp.int32)

    def coords_of(self, c: Int[Array, "*batch axes"]) -> Float[Array, "*batch axes"]:
        """(c - pad) * cell_spacing"""
        return (c - self.pad) * self.cell_spacing

    def dim(self) -> int:
        """n_axes: a padded coordinate is already its own embedding"""
        return self.n_axes

    def embed(self, x: Float[Array, "*batch axes"]) -> Float[Array, "*batch k"]:
        """`x` itself"""
        return x

    def unembed(self, v: Float[Array, "*batch k"]) -> Float[Array, "*batch axes"]:
        """`v` snapped back onto the grid"""
        return self.project(v)


@dataclass(frozen=True)
class PeriodicGridDomain(GridDomain):
    """The same grid wrapped: cell `n_cells` is cell 0, and there is no pad"""

    @property
    def extent(self) -> float:
        """cell_spacing * n_cells, a period: the last cell abuts the first"""
        return self.cell_spacing * self.n_cells

    @property
    def cells(self) -> int:
        """n_cells"""
        return self.n_cells

    def project(self, x: Float[Array, "*batch axes"]) -> Float[Array, "*batch axes"]:
        """`x` rounded to the nearest cell, wrapped into [0, extent)"""
        return jnp.round(x / self.cell_spacing) % self.n_cells * self.cell_spacing

    def cell_of(self, x: Float[Array, "*batch axes"]) -> Int[Array, "*batch axes"]:
        """round(x / cell_spacing) mod n_cells"""
        return (jnp.round(x / self.cell_spacing) % self.n_cells).astype(jnp.int32)

    def coords_of(self, c: Int[Array, "*batch axes"]) -> Float[Array, "*batch axes"]:
        """c * cell_spacing"""
        return c * self.cell_spacing

    def dim(self) -> int:
        """2 * n_axes: a cosine and a sine per axis"""
        return 2 * self.n_axes

    def embed(self, x: Float[Array, "*batch axes"]) -> Float[Array, "*batch k"]:
        """radius * (cos a, sin a) per axis, a = 2 * pi * x / extent and radius = extent / (2 * pi)"""
        angle = 2 * math.pi * x / self.extent
        return self.extent / (2 * math.pi) * jnp.concatenate([jnp.cos(angle), jnp.sin(angle)], axis=-1)

    def unembed(self, v: Float[Array, "*batch k"]) -> Float[Array, "*batch axes"]:
        """atan2(sin, cos) * extent / (2 * pi), snapped onto the grid"""
        return self.project(jnp.arctan2(v[..., self.n_axes :], v[..., : self.n_axes]) * self.extent / (2 * math.pi))


def ambient_positions(ambient: GridDomain, controllable: GridDomain) -> ProductDomain:
    """The position domain z = (y, q): the two grids as named parts"""
    return ProductDomain({"ambient": ambient, "controllable": controllable})


def ambient_candidates(ambient: PaddedGridDomain, controllable: PaddedGridDomain) -> Subset[dict[str, Any]]:
    """Where the methods get evaluated (interior of the padded grid)"""
    positions = ambient_positions(ambient, controllable)
    z = positions.elements()
    return positions.narrow(~ambient.in_pad(z["ambient"]) & ~controllable.in_pad(z["controllable"]))
