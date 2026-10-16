"""The readings a method has collected, and a model of the field fitted to them"""

import dataclasses
import functools
import operator
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

import gpjax as gpx
import jax
import jax.numpy as jnp
from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.stationary.base import StationaryKernel
from jax.scipy.linalg import solve_triangular
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from zermelo.interface import Domain, Embeddable, Enumerable, Function, ProductDomain

# float32 is the problem side's choice, not this file's; gpjax warns per dataset
warnings.filterwarnings("ignore", message=".*not of type float64.*", category=UserWarning)


@functools.lru_cache(maxsize=8)
def coordinates(positions: Domain) -> Float[Array, "n_states k"]:
    """Every element of `positions` embedded as real coordinates, cached and evaluated eagerly"""
    if not (isinstance(positions, Enumerable) and isinstance(positions, Embeddable)):
        raise TypeError(f"a belief regresses over a finite set that hands out coordinates, which a {type(positions).__name__} is not")
    with jax.ensure_compile_time_eval():
        return positions.embed(positions.elements())


@dataclass(frozen=True)
class BeliefKernel:
    """One factor of the belief's kernel, reading the columns its parts occupy:

    k_i(x, x') = family(x[columns], x'[columns]),   columns = the columns `parts` occupy
    """

    family: type[StationaryKernel]
    """Which stationary kernel `lengthscale` parameterizes"""

    parts: tuple[str, ...]
    """Which parts of a coordinate row this reads, by name"""

    lengthscale: tuple[float, ...]
    """One per column those parts occupy, in the order `parts` names them"""

    @classmethod
    def of(cls, family: type[StationaryKernel], parts: list[str], lengthscale: list[float]) -> "BeliefKernel":
        """One factor from the lists a configuration holds it as"""
        return cls(family=family, parts=tuple(parts), lengthscale=tuple(float(x) for x in lengthscale))

    def build(self, layout: dict[str, tuple[int, int]], variance: Float[Array, ""]) -> StationaryKernel:
        """`k_i`, reading the columns `layout` gives this factor's parts"""
        missing = [name for name in self.parts if name not in layout]
        if missing:
            raise KeyError(f"a kernel factor names {missing}, and a coordinate row holds {sorted(layout)}")
        columns = [column for name in self.parts for column in range(*layout[name])]
        if len(columns) != len(self.lengthscale):
            raise ValueError(f"a factor over {self.parts} spans {len(columns)} columns and gives {len(self.lengthscale)} lengthscales")
        return self.family(lengthscale=jnp.asarray(self.lengthscale), variance=variance, n_dims=len(columns), active_dims=columns)


def column_layout(positions: Domain, context: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Which columns of a coordinate row each part occupies: the position domain's parts in order, then the context's

    Example: A balloon over a sphere grid at a specific time given as context, lays out as

        columns    0     1     2       3          4
        part       <-- position -->    altitude   hours (context)
        layout     position: (0, 3)    (3, 4)     (4, 5)
    """
    if not isinstance(positions, ProductDomain):
        raise TypeError(f"a {type(positions).__name__} holds no named parts, so no kernel factor can name one")
    widths = [(name, part.dim()) for name, part in positions.parts.items() if isinstance(part, Embeddable)]
    layout, start = {}, 0
    for name, width in [*widths, *context.items()]:
        layout[name], start = (start, start + width), start + width
    return layout


def product_kernel(factors: tuple[BeliefKernel, ...], layout: dict[str, tuple[int, int]], amplitude: Float[Array, ""]) -> AbstractKernel:
    """The factors multiplied:  k(x, x') = amplitude^2 * prod over i of k_i(x, x')"""
    ones = jnp.ones(())
    built = [f.build(layout, amplitude**2 if i == 0 else ones) for i, f in enumerate(factors)]
    return functools.reduce(operator.mul, built)


@functools.lru_cache(maxsize=8)
def elements(positions: Domain) -> Any:
    """Every element of `positions`, batched along a leading axis, cached and evaluated eagerly"""
    if not isinstance(positions, Enumerable):
        raise TypeError(f"a {type(positions).__name__} cannot be enumerated")
    with jax.ensure_compile_time_eval():
        return positions.elements()


@functools.lru_cache(maxsize=8)
def _indexer(positions: Domain) -> Any:
    """`index_of` over a batch of elements, jitted and cached per domain"""
    if not isinstance(positions, Enumerable):
        raise TypeError(f"a field over a table needs an index, which a {type(positions).__name__} has none of")
    return jax.jit(jax.vmap(positions.index_of))


def lookup(positions: Domain, table: Float[Array, "n_states m"]) -> Function:
    """A field that reads `table` at whatever element it is called on"""
    index = _indexer(positions)
    return lambda x: table[index(x)]


def _fourier(x: Float[Array, "n dim"], omega: Float[Array, "n_features dim"]) -> Float[Array, "n features"]:
    """Random Fourier features of `x` at the frequencies `omega`, already divided by their lengthscales: [cos(x omega^T), sin(x omega^T)]"""
    z = x @ omega.T
    return jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)


@register_dataclass
@dataclass(frozen=True)
class Dataset:
    """`D_n`: where readings were taken, what they said, and which rows count"""

    z: Int[Array, "*batch n"]
    """The position domain's index of the cell each row sits at"""

    r: Float[Array, "*batch n m"]
    """What was read there, or what a hypothesis says is there"""

    live: Bool[Array, "*batch n"]
    """False on an unwritten buffer row, and on a walk's rows after it arrived"""

    context: Float[Array, "*batch n n_context"]
    """The coordinates each row carries beyond its cell, `n_context` wide and zero wide where there are none"""

    @classmethod
    def empty(cls, rows: int, m: int, n_context: int) -> "Dataset":
        """A buffer of `rows` rows with none of them written"""
        return cls(jnp.zeros(rows, jnp.int32), jnp.zeros((rows, m)), jnp.zeros(rows, bool), jnp.zeros((rows, n_context)))

    def write(self, row: int, z: Int[Array, ""], r: Float[Array, " m"], context: Float[Array, " n_context"]) -> "Dataset":
        """This dataset with `row` overwritten and marked live, raising past the end of the buffer"""
        if not 0 <= row < self.z.shape[0]:
            raise IndexError(f"row {row} of a {self.z.shape[0]}-row buffer: the agent's horizon is shorter than the episode's")
        return Dataset(self.z.at[row].set(z), self.r.at[row].set(r), self.live.at[row].set(True), self.context.at[row].set(context))

    def concat(self, other: "Dataset") -> "Dataset":
        """The two laid end to end along the row axis"""
        return Dataset(
            jnp.concatenate([self.z, other.z], axis=-1),
            jnp.concatenate([self.r, other.r], axis=-2),
            jnp.concatenate([self.live, other.live], axis=-1),
            jnp.concatenate([self.context, other.context], axis=-2),
        )

    def broadcast(self, batch: tuple[int, ...]) -> "Dataset":
        """This dataset repeated over the leading axes `batch`"""
        return Dataset(
            jnp.broadcast_to(self.z, (*batch, *self.z.shape)),
            jnp.broadcast_to(self.r, (*batch, *self.r.shape)),
            jnp.broadcast_to(self.live, (*batch, *self.live.shape)),
            jnp.broadcast_to(self.context, (*batch, *self.context.shape)),
        )


class Belief(ABC):
    """A model of the field over `positions`, conditioned on `data`"""

    positions: Domain
    """The cells the field is defined on: finite, and handing out coordinates"""

    data: Dataset

    @abstractmethod
    def fit(self, data: Dataset) -> Self:
        """This belief conditioned on `data`, its own parameters re-estimated from it where configured"""

    @abstractmethod
    def condition(self, data: Dataset) -> Self:
        """This belief conditioned on `data`, its parameters unchanged, over one dataset and never a batch"""

    @abstractmethod
    def predict(
        self, z: Int[Array, "*batch n"], context: Float[Array, " n_context"]
    ) -> tuple[Float[Array, "*batch n m"], Float[Array, "*batch n m"]]:
        """Posterior mean and per-component variance at those cells, at one context"""

    @abstractmethod
    def mean(self, context: Float[Array, " n_context"]) -> Function:
        """`mu_n`, the posterior mean field at one context"""

    @abstractmethod
    def draw(self, key: PRNGKeyArray, n_fields: int, context: Float[Array, " n_context"]) -> list[Function]:
        """`n_fields` fields drawn from the posterior at one context"""


@register_dataclass
@dataclass(frozen=True)
class OracleBelief(Belief):
    """The true field in place of a model: the mean is exactly it and the variance is zero"""

    positions: Domain = dataclasses.field(metadata=dict(static=True))
    data: Dataset

    field: Function = dataclasses.field(metadata=dict(static=True))
    """The field the world drew"""

    context_parts: tuple[str, ...] = dataclasses.field(metadata=dict(static=True))
    """What the context's columns are called, so the field can be read at one"""

    @classmethod
    def empty(cls, positions: Domain, rows: int, m: int, field: Function, context: dict[str, int]) -> "OracleBelief":
        """Seen nothing, over a buffer of `rows` rows, holding the field the world drew"""
        return cls(positions, Dataset.empty(rows, m, sum(context.values())), field, tuple(context))

    def fit(self, data: Dataset) -> "OracleBelief":
        return dataclasses.replace(self, data=data)

    def condition(self, data: Dataset) -> "OracleBelief":
        return dataclasses.replace(self, data=data)

    def _table(self, context: Float[Array, " n_context"]) -> Float[Array, "n_states m"]:
        """The field read at every cell, at one context"""
        cells = elements(self.positions)
        return self.field({**cells, **{name: context[j] for j, name in enumerate(self.context_parts)}})

    def predict(
        self, z: Int[Array, "*batch n"], context: Float[Array, " n_context"]
    ) -> tuple[Float[Array, "*batch n m"], Float[Array, "*batch n m"]]:
        """The field's own values, and a variance of zero"""
        mean = self._table(context)[z]  # (n_states, m): read at every cell, then gathered
        return mean, jnp.zeros_like(mean)

    def mean(self, context: Float[Array, " n_context"]) -> Function:
        return lookup(self.positions, self._table(context))

    def draw(self, key: PRNGKeyArray, n_fields: int, context: Float[Array, " n_context"]) -> list[Function]:
        """The same field every time, whatever the key"""
        return [self.mean(context)] * n_fields


@jax.jit
def _conjugate(
    gram: Float[Array, "rows rows"],
    cross: Float[Array, "rows q"],
    y: Float[Array, "rows m"],
    keep: Bool[Array, " rows"],
    noise_var: Float[Array, " rows"],
    variance: Float[Array, ""],
) -> tuple[Float[Array, "q m"], Float[Array, " q"]]:
    """Posterior mean and variance at `cross`'s columns, over the rows `keep` marks:

        K = gram + diag(noise_var) where kept, the identity elsewhere;   L L^T = K
        V = L^-1 cross,   mean = V^T L^-1 y,   var = variance - sum over rows of V^2

    Time O(rows^3 + q rows^2), memory O(q rows).
    """
    pair = keep[:, None] & keep[None, :]  # (rows, rows): both ends kept
    diagonal = jnp.where(keep, noise_var, 1.0) + 1e-6  # (rows,): dropped rows get variance 1; +1e-6 jitter for float32
    chol = jnp.linalg.cholesky(jnp.where(pair, gram, 0.0) + jnp.diag(diagonal))  # (rows, rows)
    solved = jnp.where(keep[:, None], solve_triangular(chol, cross, lower=True), 0.0)  # (rows, q), a dropped row zeroed
    mean = solved.T @ solve_triangular(chol, jnp.where(keep[:, None], y, 0.0), lower=True)  # (q, m)
    return mean, jnp.maximum(variance - jnp.sum(solved**2, axis=0), 0.0)  # (q,)


@register_dataclass
@dataclass(frozen=True)
class GPBelief(Belief):
    """`m` Gaussian processes over the field's components, independent, sharing one kernel, about a given prior mean:

    f = prior_mean + e,   e ~ GP(0, k)
    """

    positions: Domain = dataclasses.field(metadata=dict(static=True))
    data: Dataset

    kernel_factors: tuple[BeliefKernel, ...] = dataclasses.field(metadata=dict(static=True))
    """The factors the kernel is a product of, each over the parts it names"""

    context: dict[str, int] = dataclasses.field(metadata=dict(static=True))
    """The coordinates a reading carries beyond its cell, each named and how wide it is"""

    amplitude: Float[Array, ""]
    noise: Float[Array, ""]
    """The observation noise standard deviation"""

    n_features: int = dataclasses.field(metadata=dict(static=True))
    """How many random Fourier features a pathwise draw uses"""

    refit: bool = dataclasses.field(metadata=dict(static=True))
    """Whether `fit` re-estimates the three parameters by marginal likelihood"""

    refit_steps: int = dataclasses.field(metadata=dict(static=True))

    prior_mean: Function | None = dataclasses.field(metadata=dict(static=True))
    """Prior assumption of the field mean (None is zero everywhere)"""

    @classmethod
    def empty(cls, positions: Domain, rows: int, m: int, n_context: int, **config: Any) -> "GPBelief":
        """Seen nothing, over a buffer of `rows` rows"""
        return cls(positions, Dataset.empty(rows, m, n_context), **config)

    @property
    def coords(self) -> Float[Array, "n_states k"]:
        """Every cell's coordinates, in the position domain's own index order"""
        return coordinates(self.positions)

    @property
    def kernel(self) -> AbstractKernel:
        """The product of the factors, over the columns the position domain and the context lay out"""
        return product_kernel(self.kernel_factors, column_layout(self.positions, self.context), self.amplitude)

    def query(self, context: Float[Array, " n_context"]) -> Float[Array, "n_states k_context"]:
        """Every cell's coordinates at one context, that context repeated down the query"""
        return jnp.concatenate([self.coords, jnp.broadcast_to(context, (self.coords.shape[0], context.shape[-1]))], axis=-1)

    @property
    def offset(self) -> Float[Array, "n_states m"]:
        """`prior_mean` at every cell if given, or zeros when none"""
        if self.prior_mean is None:
            return jnp.zeros((self.coords.shape[0], self.data.r.shape[-1]))
        return jnp.broadcast_to(self.prior_mean(elements(self.positions)), (self.coords.shape[0], self.data.r.shape[-1]))

    def _conditioning(self) -> tuple[Float[Array, "rows dim"], Float[Array, "rows m"], Bool[Array, " rows"], Float[Array, " rows"]]:
        """Per buffer row: coordinates, the residual read there, whether the row is conditioned on, and the noise variance it carries

        Rows at one cell are repeat measurements only where no context separates them, and are then averaged
        into the earliest of them at `noise^2 / c` for `c` reads.
        """
        rows, n_states = self.data.z.shape[0], self.coords.shape[0]
        z = jnp.where(self.data.live, self.data.z, 0)  # (rows,): dead rows get cell 0, and are masked out below
        x = jnp.concatenate([self.coords[z], self.data.context], axis=-1)  # (rows, dim)
        if self.context:  # no deduplication: a context tells two readings of one cell apart
            return x, self.data.r - self.offset[z], self.data.live, jnp.full(rows, self.noise**2)
        # deduplicate: without a context a repeat visit is a repeat measurement, averaged for a better conditioned gram
        c = jnp.zeros(n_states).at[z].add(self.data.live.astype(self.data.r.dtype))  # (n_states,): readings per cell
        total = jnp.zeros((n_states, self.data.r.shape[-1])).at[z].add(jnp.where(self.data.live[:, None], self.data.r, 0.0))
        # (n_states,): earliest live row at each cell, a dead row scattering `rows` so it never wins the min
        first = jnp.full(n_states, rows).at[z].min(jnp.where(self.data.live, jnp.arange(rows), rows))
        n_reads = jnp.maximum(c[z], 1.0)  # (rows,): readings at this row's cell, floored to 1
        # rows kept: live and the first at its cell, so a cell appears once
        residual = total[z] / n_reads[:, None] - self.offset[z]  # (rows, m): r - prior_mean
        return x, residual, self.data.live & (jnp.arange(rows) == first[z]), self.noise**2 / n_reads

    def _posterior(self, n: int) -> Any:
        """The conjugate posterior over one output component of the field seen at `n` points"""
        prior = gpx.gps.Prior(kernel=self.kernel, mean_function=gpx.mean_functions.Zero())
        return prior * gpx.likelihoods.Gaussian(num_datapoints=max(n, 1), obs_stddev=self.noise)

    def fit(self, data: Dataset) -> "GPBelief":
        held = dataclasses.replace(self, data=data)
        if not self.refit:
            return held
        x, y, keep, _ = held._conditioning()
        x, y = x[keep], y[keep]  # (k, dim), (k, m): the distinct cells read
        if x.shape[0] < 2:  # a marginal likelihood on one point prefers an infinite lengthscale
            return held

        def loss(model: Any, d: Any) -> Float[Array, ""]:
            # one kernel over m independent components, so the joint log marginal likelihood is their sum
            per_component = [gpx.objectives.conjugate_mll(model, gpx.Dataset(d.X, d.y[:, j : j + 1])) for j in range(d.y.shape[-1])]
            return -jnp.sum(jnp.stack([jnp.asarray(t) for t in per_component]))

        tuned, _ = gpx.fit_scipy(
            model=held._posterior(x.shape[0]), objective=loss, train_data=gpx.Dataset(x, y), max_iters=self.refit_steps, verbose=False
        )
        fitted = tuned.prior.kernel.kernels if len(self.kernel_factors) > 1 else [tuned.prior.kernel]
        return dataclasses.replace(
            held,
            kernel_factors=tuple(
                dataclasses.replace(factor, lengthscale=tuple(jnp.asarray(k.lengthscale.value).reshape(-1).tolist()))
                for factor, k in zip(self.kernel_factors, fitted, strict=True)
            ),
            amplitude=jnp.sqrt(jnp.asarray(fitted[0].variance.value).reshape(())),
            noise=jnp.asarray(tuned.likelihood.obs_stddev.value).reshape(()),
        )

    def condition(self, data: Dataset) -> "GPBelief":
        return dataclasses.replace(self, data=data)

    def _moments(self, query: Float[Array, "q dim"]) -> tuple[Float[Array, "q m"], Float[Array, "q m"]]:
        """Posterior mean and variance per component of the field at `query`"""
        x, y, keep, noise_var = self._conditioning()
        kernel = self.kernel
        mean, var = _conjugate(kernel.gram(x).to_dense(), kernel.cross_covariance(x, query), y, keep, noise_var, self.amplitude**2)
        return mean, jnp.broadcast_to(var[:, None], mean.shape)  # (q, m) each, one kernel serving every component

    def predict(
        self, z: Int[Array, "*batch n"], context: Float[Array, " n_context"]
    ) -> tuple[Float[Array, "*batch n m"], Float[Array, "*batch n m"]]:
        mean, var = self._moments(self.query(context))  # (n_states, m) each: solved at every cell, then gathered
        return mean[z] + self.offset[z], var[z]

    def mean(self, context: Float[Array, " n_context"]) -> Function:
        return lookup(self.positions, self._moments(self.query(context))[0] + self.offset)

    def draw(self, key: PRNGKeyArray, n_fields: int, context: Float[Array, " n_context"]) -> list[Function]:
        """`n_fields` fields drawn from the posterior at one context, a shorter draw being a subset of a longer one

        Time O(n_read^3 + n_fields (n_states n_features + n_read n_states)).
        """
        x, y, keep, _ = self._conditioning()
        x, y = x[keep], y[keep]  # (n_read, dim), (n_read, m): the rows conditioned on
        width = self.data.r.shape[-1]
        drawn = [self._paths(jax.random.split(key, width)[j], x, y[:, j], n_fields, self.query(context)) for j in range(width)]
        table = jnp.stack(drawn, axis=-1) + self.offset  # (n_fields, n_states, m): the mean plus a draw of e
        return [lookup(self.positions, table[s]) for s in range(n_fields)]

    def _paths(
        self,
        key: PRNGKeyArray,
        x: Float[Array, "n_read dim"],
        y: Float[Array, " n_read"],
        n_fields: int,
        query: Float[Array, "n_states dim"],
    ) -> Float[Array, "n_fields n_states"]:
        """`n_fields` draws of one field component at every cell:

        f_s(z) = Phi(z) theta_s + k(z, x) v_s,   v_s = (K + noise^2 I)^-1 (y + eps_s - Phi(x) theta_s)
        """
        kernel = self.kernel
        omega = self._frequencies(key)  # (n_features, dim)
        scale = jnp.sqrt(self.amplitude**2 / self.n_features)
        theta = jax.random.normal(key, (n_fields, 2 * self.n_features))  # (n_fields, 2 n_features): row s is path s
        features = _fourier(query, omega) * scale  # (n_states, 2 n_features)
        prior_part = theta @ features.T  # (n_fields, n_states)
        if x.shape[0] == 0:  # nothing read: the draw is from the prior
            return prior_part
        # drawn (n_fields, n_read) then turned: filling the other way round would change path s with n_fields
        eps = self.noise * jax.random.normal(key, (n_fields, x.shape[0])).T  # (n_read, n_fields)
        gram = kernel.gram(x).to_dense() + (self.noise**2 + 1e-6) * jnp.eye(x.shape[0])  # (n_read, n_read)
        residual = y[:, None] + eps - (_fourier(x, omega) * scale) @ theta.T  # (n_read, n_fields)
        canonical = jnp.linalg.solve(gram, residual)  # (n_read, n_fields): one factorization, a column per field
        return prior_part + (kernel.cross_covariance(query, x) @ canonical).T

    def _frequencies(self, key: PRNGKeyArray) -> Float[Array, "n_features dim"]:
        """Frequencies drawn from the product kernel's spectral density, one block of columns per factor:

        a product of stationary kernels over disjoint columns is stationary, and its spectral density is the product of
        the factors' densities, so each factor's frequencies are drawn from its own density (over its own lengthscales)
        and laid into the columns it reads; the other columns get zero. gpjax's `RFF` refuses the product outright.
        """
        layout = column_layout(self.positions, self.context)
        dim = max(stop for _, stop in layout.values())
        omega = jnp.zeros((self.n_features, dim))
        for factor, k in zip(self.kernel_factors, jax.random.split(key, len(self.kernel_factors)), strict=True):
            built = factor.build(layout, jnp.ones(()))
            columns = jnp.asarray([column for name in factor.parts for column in range(*layout[name])])
            drawn = built.spectral_density.sample(key=k, sample_shape=(self.n_features, columns.shape[0]))  # (n_features, q)
            omega = omega.at[:, columns].set(drawn / jnp.asarray(factor.lengthscale))
        return omega
