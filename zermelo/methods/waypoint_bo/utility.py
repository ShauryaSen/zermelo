"""`U(D_n)`: what a set of readings is worth"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jaxtyping import Array, Float, PRNGKeyArray

from zermelo.interface import Subset
from zermelo.methods.waypoint_bo.belief import Belief, Dataset


class Utility(ABC):
    """Scores a dataset under a belief's model, considering only the live rows that sit on a candidate cell"""

    @abstractmethod
    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        """What `data` is worth, one score per leading batch axis of `data`"""


def query_context(data: Dataset) -> Float[Array, " n_context"]:
    """The context the belief is read at to score `data`: its last row's

    context[-1, ..., -1, :], one -1 per leading batch axis of a (*batch, n, n_context) array.
    """
    return data.context[(-1,) * (data.context.ndim - 1)]


@dataclass(frozen=True)
class MaxMagnitude(Utility):
    """U(D) = max over the set of ||r||, the incumbent beta_n"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        counts = data.live & candidates.live[data.z]
        # 0.0 is the infimum of a magnitude, so a set with no counted row scores the floor rather than -inf
        return jnp.max(jnp.where(counts, jnp.linalg.norm(data.r, axis=-1), 0.0), axis=-1)


@dataclass(frozen=True)
class PosteriorSpread(Utility):
    """The posterior standard deviations at the scored cells, summed"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        spread = jnp.sqrt(jnp.sum(belief.predict(data.z, query_context(data))[1], axis=-1))  # sqrt(sum_j var_j), components independent
        return jnp.sum(jnp.where(data.live & candidates.live[data.z], spread, 0.0), axis=-1)


@dataclass(frozen=True)
class ExpectedImprovement(Utility):
    """The largest `E[(|f(z)| - beta)^+]` over the scored cells, `beta` the largest magnitude read so far"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        if data.r.shape[-1] != 1:
            raise ValueError(f"a closed-form improvement of a magnitude needs a one-component field, and this one has {data.r.shape[-1]}")
        held = belief.data  # the incumbent comes from what has actually been read, not from the scored rows
        # zero where nothing has been read yet, which is the infimum of a magnitude and makes the tails below disjoint
        beta = jnp.max(jnp.where(held.live & candidates.live[held.z], jnp.linalg.norm(held.r, axis=-1), 0.0))
        mean, var = belief.predict(data.z, query_context(data))
        mu, spread = mean[..., 0], jnp.sqrt(jnp.maximum(var[..., 0], 1e-12))  # (*batch, n): spread floored to 1e-6
        # |f| clears beta above or below, disjointly for beta >= 0, so the two tails add. Each is
        # `E[(x - c)^+] = gap * Phi(gap / spread) + spread * phi(gap / spread)` at its own gap
        over, under = (mu - beta) / spread, (-mu - beta) / spread
        improvement = (mu - beta) * norm.cdf(over) + (-mu - beta) * norm.cdf(under) + spread * (norm.pdf(over) + norm.pdf(under))
        return jnp.max(jnp.where(data.live & candidates.live[data.z], improvement, 0.0), axis=-1)  # 0.0 floors an improvement


@dataclass(frozen=True)
class UpperConfidence(Utility):
    """The largest optimistic magnitude `||mu(z)|| + c * s(z)` at the scored cells"""

    c: float
    """How much of the spread to add"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        mean, var = belief.predict(data.z, query_context(data))
        bound = jnp.linalg.norm(mean, axis=-1) + self.c * jnp.sqrt(jnp.sum(var, axis=-1))
        return jnp.max(jnp.where(data.live & candidates.live[data.z], bound, 0.0), axis=-1)  # 0.0 floors a magnitude


@dataclass(frozen=True)
class PredictiveConfidence(Utility):
    """Negative predictive entropy over the candidate cells, under the belief re-conditioned on the scored data"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        query = jnp.flatnonzero(candidates.live)
        batch, rows = data.z.shape[:-1], data.z.shape[-1]
        z, r, live = data.z.reshape(-1, rows), data.r.reshape(-1, rows, data.r.shape[-1]), data.live.reshape(-1, rows)
        context = data.context.reshape(-1, rows, data.context.shape[-1])
        scored = []
        for i in range(z.shape[0]):
            rows_i = Dataset(z[i], r[i], live[i] & candidates.live[z[i]], context[i])
            var = belief.condition(rows_i).predict(query, context[i][-1])[1]
            scored.append(-jnp.sum(0.5 * jnp.log(2 * math.pi * math.e * var)))
        return jnp.stack(scored).reshape(batch)


@dataclass(frozen=True)
class Uniform(Utility):
    """A uniform draw per scored dataset, ignoring the data"""

    def __call__(self, key: PRNGKeyArray, belief: Belief, data: Dataset, candidates: Subset[Any]) -> Float[Array, " *batch"]:
        return jax.random.uniform(key, data.z.shape[:-1])
