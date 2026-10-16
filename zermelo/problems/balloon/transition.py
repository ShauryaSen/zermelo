"""One step of a balloon: the wind carries it, the action moves it between altitudes, and that spends resource

s = (lat, lon, p, r, t, W)  where it is, which altitude, what resource is left, what time, the wind it moves in
a in {0, 1, 2}              down one, hold, up one
W(lat, lon, p, t) = (u, v)  the wind there (m/s), u eastward and v northward
h                           hours in one step
(i, j)                      the grid cell of (lat, lon): i the row, j the column

Three factors write (lat, lon, p, t), and the resource follows from what they did:

    (i, j)' ~ (i, j) + d           d the wind's step in cells, each axis rounded   Advection
    p'      = p + a - 1            where r > 0 and 0 <= p' < n_altitudes           Ascent
    t'      = t + h                                                                Clock
    r'      = r - |p' - p|                                                         Resource
"""

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from zermelo.interface import BoxDomain, DiscreteDomain, Domain, Function, ProductDomain, Transition, TransitionKernel
from zermelo.problems.balloon.grid import SphereGrid, Steps

type Act = Int[Array, ""]
"""0 down one altitude, 1 hold, 2 up one"""

HOLD = 1
"""The action that changes no altitude and spends no resource"""


class Factor[Action](ABC):
    """One part of s' sampled from the whole of s"""

    @property
    @abstractmethod
    def part(self) -> str:
        """Which part of s this writes"""

    @abstractmethod
    def sample(self, key: PRNGKeyArray, state: dict[str, Any], action: Action) -> Any:
        """One draw of part'"""


@dataclass(frozen=True)
class Advection(Factor[Act]):
    """One advected step, each axis rounded independently:

        (a, b) = (i, j) + d                       d the wind's step in cells
        i' = floor(a) + Bernoulli(a - floor(a))
        j' = floor(b) + Bernoulli(b - floor(b))

    then clipped onto the grid.
    """

    grid: SphereGrid
    step_hours: float

    @property
    def part(self) -> str:
        return "position"

    def offset(self, state: dict[str, Any]) -> Float[Array, "*batch 2"]:  # states batch along a leading axis
        """d = (v, u) * h * 3.6 / km_per_degree(lat) / degrees_per_cell, the wind's step in cells"""
        wind = state["field"]({part: state[part] for part in state if part != "field"})  # (..., 2) as (u, v)
        km = jnp.stack([wind[..., 1], wind[..., 0]], axis=-1) * self.step_hours * 3.6  # (..., 2) as (lat, lon)
        return km / self.grid.km_per_degree(state["position"][..., 0]) / self.grid.steps

    def sample(self, key: PRNGKeyArray, state: dict[str, Any], action: Act) -> Float[Array, "*batch 2"]:
        """One draw of (i, j)'"""
        exact = self.grid.cell_of(state["position"]) + self.offset(state)
        rounded = jnp.floor(exact) + (jax.random.uniform(key, exact.shape) < exact - jnp.floor(exact))
        return self.grid.coords_of(jnp.clip(rounded, 0, self.grid.shape - 1).astype(jnp.int32))


@dataclass(frozen=True)
class Ascent(Factor[Act]):
    """The action moves the balloon one altitude, if it can afford it and there is one that way:

        p' = p + (a - 1)   where r > 0 and 0 <= p + (a - 1) < n_altitudes
           = p             otherwise

    Deterministic given (p, r, a)
    """

    n_altitudes: int

    @property
    def part(self) -> str:
        return "altitude"

    def moved(self, state: dict[str, Any], action: Act) -> Int[Array, "*batch"]:
        """p' - p, one of -1, 0, +1: the wanted a - 1 where r > 0 and the altitude landed on is in range, else 0"""
        wanted = action - HOLD
        landing = state["altitude"] + wanted
        return jnp.where((state["balloon_resource"] > 0) & (landing >= 0) & (landing < self.n_altitudes), wanted, 0)

    def sample(self, key: PRNGKeyArray, state: dict[str, Any], action: Act) -> Int[Array, ""]:
        """p' = p + (p' - p)"""
        return state["altitude"] + self.moved(state, action)


@dataclass(frozen=True)
class Clock(Factor[Act]):
    """The hours an episode has run, advanced whatever the action: t' = t + h"""

    step_hours: float

    @property
    def part(self) -> str:
        return "hours"

    def sample(self, key: PRNGKeyArray, state: dict[str, Any], action: Act) -> Float[Array, ""]:
        """t' = t + h"""
        return state["hours"] + self.step_hours


@dataclass(frozen=True)
class BalloonTransition(Transition[Act]):
    """s' factor by factor: the wind moves (lat, lon), the action moves p, the clock updates, and a move spends resource"""

    advection: Advection
    ascent: Ascent
    clock: Clock

    state_domain: ProductDomain
    """Every state, in the index order a kernel's arrays follow"""

    @property
    def action_domain(self) -> Domain:
        """Three actions, legal at every state: one the balloon cannot afford moves it nowhere"""
        return DiscreteDomain(3)

    @property
    def factors(self) -> tuple[Factor[Act], ...]:
        """The factors in the order their parts are written"""
        return (self.advection, self.ascent, self.clock)

    def __call__(self, key: PRNGKeyArray, state: dict[str, Any], action: Act) -> dict[str, Any]:
        """One sampled step, and the resource the altitude change spent"""
        keys = jax.random.split(key, len(self.factors))
        out = {f.part: f.sample(k, state, action) for f, k in zip(self.factors, keys, strict=True)}
        spent = jnp.abs(self.ascent.moved(state, action))
        return {**out, "balloon_resource": state["balloon_resource"] - spent, "field": state["field"]}

    def wind_step(self, hypothesis: Function) -> tuple[Int[Array, "pos_alt 4"], Float[Array, "pos_alt 4"]]:
        """The four cells one advected step may land on, and their probabilities:

            (a, b) = (i, j) + d                  d the wind's step in cells under `hypothesis`
            m = floor(a),  w = a - m
            n = floor(b),  u = b - n

            P(m,     n)     = (1 - w)(1 - u)     P(m,     n + 1) = (1 - w) u
            P(m + 1, n)     =      w (1 - u)     P(m + 1, n + 1) =      w  u

        each cell clipped onto the grid. Both (pos_alt, 4), altitude varying fastest.
        """
        grid, pos, n_alt = self.advection.grid, self.advection.grid.size(), self.ascent.n_altitudes
        cells = jnp.repeat(grid.elements(), n_alt, axis=0)  # (pos_alt, 2), altitude varying fastest
        under = {"position": cells, "altitude": jnp.tile(jnp.arange(n_alt), pos), "field": hypothesis}
        exact = grid.cell_of(cells) + self.advection.offset(under)  # (pos_alt, 2) in cells
        floor = jnp.floor(exact)  # both ends clip off this, so a landing beyond the edge puts all its mass on the edge
        low = jnp.clip(floor, 0, grid.shape - 1).astype(jnp.int32)
        high = jnp.clip(floor + 1, 0, grid.shape - 1).astype(jnp.int32)
        up = jnp.clip(exact - floor, 0.0, 1.0)  # (pos_alt, 2): u per axis, the probability of rounding up
        lat_at, lat_prob = jnp.stack([low[:, 0], high[:, 0]], -1), jnp.stack([1.0 - up[:, 0], up[:, 0]], -1)
        lon_at, lon_prob = jnp.stack([low[:, 1], high[:, 1]], -1), jnp.stack([1.0 - up[:, 1], up[:, 1]], -1)
        at = (lat_at[:, :, None] * grid.n_lon + lon_at[:, None, :]).reshape(-1, 4)  # (pos_alt, 4) flat cells
        return at, (lat_prob[:, :, None] * lon_prob[:, None, :]).reshape(-1, 4)

    def kernel(self, hypothesis: Function) -> "BalloonKernel":
        """The law at every state, with `hypothesis` in place of the wind an episode drew"""
        n_alt = self.ascent.n_altitudes
        next_pos, prob = self.wind_step(hypothesis)
        alt = jnp.arange(n_alt)[None, :]  # (1, alt), broadcasting over actions
        # the law is written with resource in hand; the world refuses the move once it runs out
        under = {"altitude": jnp.broadcast_to(alt, (3, n_alt)), "balloon_resource": jnp.ones((3, n_alt), jnp.int32)}
        moved = jnp.stack([self.ascent.moved(under, jnp.asarray(a))[a] for a in range(3)])  # (actions, alt)
        return BalloonKernel(self.state_domain, next_pos, prob, alt + moved)


@register_dataclass
@dataclass(frozen=True)
class BalloonKernel(TransitionKernel[Act]):
    """The one-step law at every state, as four reachable positions and the deterministic landing of p.

    s = (lat, lon, p)                       indexed pos * n_alt + p

    P(s' | s, a) = prob[s, k]               s' = (next_pos[s, k], altitude_at[a, p])
                 = 0                        otherwise
    """

    domain: Domain = dataclasses.field(metadata=dict(static=True))
    """What `values` is indexed by, in its own index order"""

    next_pos: Int[Array, "pos_alt 4"]
    """The four cells the wind may carry (lat, lon) to, as flat positions"""

    prob: Float[Array, "pos_alt 4"]
    """P of each, summing to one along the last axis"""

    altitude_at: Int[Array, "actions alt"]
    """p' = p + a - 1 where that is in range, else p"""

    def expectation(self, values: Float[Array, " states"], action: Act) -> Float[Array, " states"]:
        """out[pos, p] = sum over k in 0..3 of prob[s, k] * values[next_pos[s, k], altitude_at[a, p]].
        Time O(states * 4), memory O(states * 4)
        """
        n_alt = self.altitude_at.shape[1]
        table = values.reshape(-1, n_alt)  # (pos, alt)
        landed = table[:, self.altitude_at[action]]  # (pos, alt): p' applied
        at = self.next_pos.reshape(-1, n_alt, 4)  # (pos, alt, 4): the flat cells the wind may reach
        alt = jnp.arange(n_alt)[None, :, None]  # (1, alt, 1), pairing each cell with the altitude it was read at
        reached = landed[at, alt]  # (pos, alt, 4)
        return jnp.einsum("pak,pak->pa", self.prob.reshape(-1, n_alt, 4), reached).reshape(-1)

    def sample(self, key: PRNGKeyArray, index: Int[Array, ""], action: Act) -> Int[Array, ""]:
        """One draw from P(. | s, a) at `index`: one of the four cells by `prob`, then p' read off"""
        n_alt = self.altitude_at.shape[1]
        p = index % n_alt
        q = self.next_pos[index, jax.random.categorical(key, jnp.log(self.prob[index]))]
        return q * n_alt + self.altitude_at[action, p]

    def step_cost(self, action: Act) -> Float[Array, " states"]:
        """The resource one step spends: `c(s, a) = |altitude_at[a, p] - p|`"""
        n_alt = self.altitude_at.shape[1]
        per_altitude = jnp.abs(self.altitude_at[action] - jnp.arange(n_alt)).astype(jnp.float32)  # (alt,)
        return jnp.tile(per_altitude, self.next_pos.shape[0] // n_alt)  # (states,), altitude varying fastest


@dataclass(frozen=True)
class SmoothBalloonTransition(Transition[Float[Array, ""]]):
    """One step of a balloon whose every coordinate is real, written so that s' is differentiable in s and a

    s = (lat, lon, p, r, t, W)   where it is (degrees), its altitude as a fractional level, resource left, hours, the wind
    a in [-c, c]                 how far to climb this step, in km; negative descends
    z(p)                         km at level p, read linearly between the record's levels; z^-1 its inverse

        (lat, lon)' = (lat, lon) + (v, u) h 3.6 / km_per_degree(lat)       Drift: exactly with the wind
        p'          = p + d,  d = clip(z^-1(clip(z(p) + a, z_min, z_max)) - p, -r, r)   Climb: as asked, within the band and the resource
        r'          = r - |d|                                               Resource: one unit per level crossed
        t'          = t + h                                                 Clock

    Every operation is linear, a clip, or an interpolation, so `jax.grad` passes through a whole rollout.
    The wind W is read by the state's own field, which must itself be smooth (`SmoothWind`) for the gradient to mean anything.
    The world keeps the balloon on the grid (`__call__` clips the position); the pure `step` a planner differentiates does not,
    since a clip at the edge has no gradient and a plan made against it could never find the way back in.
    """

    grid: SphereGrid
    altitudes: Steps
    """The record's levels and the km each stands for"""

    step_hours: float
    climb_cap_km: float
    """`c`: the most the balloon can climb or descend in one step"""

    @property
    def action_domain(self) -> Domain:
        """A climb in km, anywhere within the cap"""
        return BoxDomain((), -self.climb_cap_km, self.climb_cap_km)

    def step(self, state: dict[str, Any], action: Float[Array, ""]) -> dict[str, Any]:
        """s' given s and a, with no randomness: the pure map a planner differentiates"""
        wind = state["field"](state)  # (2,) as (u, v)
        km = jnp.stack([wind[1], wind[0]]) * self.step_hours * 3.6  # (2,) as (lat, lon)
        position = state["position"] + km / self.grid.km_per_degree(state["position"][0])
        levels, z = jnp.arange(len(self.altitudes.coordinate)) * 1.0, jnp.asarray(self.altitudes.coordinate)
        order = jnp.argsort(z)  # interp wants its abscissae increasing; the record's levels may run top down
        wanted = jnp.interp(jnp.clip(jnp.interp(state["altitude"], levels, z) + action, z.min(), z.max()), z[order], levels[order])
        moved = jnp.clip(wanted - state["altitude"], -state["balloon_resource"], state["balloon_resource"])
        return {
            "position": position,
            "altitude": state["altitude"] + moved,
            "balloon_resource": state["balloon_resource"] - jnp.abs(moved),
            "hours": state["hours"] + self.step_hours,
        }

    def __call__(self, key: PRNGKeyArray, state: dict[str, Any], action: Float[Array, ""]) -> dict[str, Any]:
        """The same step with the position clipped onto the grid; the key is unused, the law being deterministic"""
        corners = jnp.stack([self.grid.coords_of(jnp.zeros(2, jnp.int32)), self.grid.coords_of(self.grid.shape - 1)])  # (2, 2)
        after = self.step(state, action)
        return {**after, "position": jnp.clip(after["position"], corners.min(0), corners.max(0)), "field": state["field"]}
