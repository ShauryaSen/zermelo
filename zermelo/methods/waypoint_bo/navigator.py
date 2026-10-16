"""Flying to a waypoint by first-order MPC over a smooth balloon, after Pasetsky, Lin, Guo and Dean (FOMPC, arXiv 2511.07761)

x_1..x_H                 the state along a rollout of the next H steps under a hypothesised wind
u_1..u_H in [-1, 1]      the climbs as a fraction of the cap c, the one thing optimised; the balloon flies c * u
J(u) = sum over k of ||embed(x_k) - embed(waypoint)|| + w * sum over k of |p_k - p_{k-1}|

the distance to the waypoint summed over every step (so arriving early and staying both pay), plus the resource spent.

One solve, as in the paper (their section V.6 and algorithm 1):

    1. a pool of candidates: M random ones, the previous plan shifted by one, and a "coast" plan.
       A random candidate draws a target altitude uniformly over the band the balloon can reach, applies
       near-saturated climbs for the steps a saturated rollout needs to get there, and fills the rest of the
       horizon with small uniform noise. Coast is that noise alone.
    2. every candidate is scored once; the best `descend_top` of them (the paper's 1) are refined by gradient descent.
    3. the climbs are reparameterised as u = tanh(v / 2) so the descent is unconstrained, and v moves by a
       normalised gradient step `v <- v - eta * g / ||g||`, at most `iterations` times, stopping once ||g|| < tolerance.
    4. the refined plan with the lowest J is flown one step; it is the previous plan of the next solve.

Differences from the paper, stated: the cost is a waypoint distance and a resource charge, not station keeping and
battery, and it is undiscounted; the noise on the random candidates' tails is assumed Unif(-0.2, 0.2) like coast's,
the paper giving no figure.

The wind is a batch: `fields` carries N winds along a leading axis of every array it holds, and J is the mean over
them of the single-wind cost,  J(u) = (1/N) sum over i of J(u; w_i).  With N winds drawn from the belief this is the
sample-average plan, the best fixed plan under what the balloon knows as N grows; with N = 1 and the posterior mean
it is the planner of task 1, unchanged.
"""

import functools
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from zermelo.interface import BoxDomain, Differentiable, Domain, Embeddable, Function


@dataclass(frozen=True)
class GradientMPC:
    """Plans H climbs by the FOMPC recipe, replanning from the current state every move"""

    horizon: int
    """`H`: steps rolled out and optimised over"""

    iterations: int
    """`S_max`: gradient steps per refinement, at most"""

    step_size: float
    """`eta`: the normalised gradient step, in the unconstrained parameter"""

    tolerance: float
    """`epsilon`: the gradient norm below which a refinement stops"""

    resource_weight: float
    """`w`: km of summed distance one level of resource is worth"""

    candidates: int
    """`M`: random candidates in the pool, beside the previous plan and coast"""

    descend_top: int
    """`k`: how many of the best-scoring candidates are refined by gradient descent (the paper refines one)"""

    perturbation: float
    """Half-width of the uniform noise a candidate's tail and the coast plan are made of, as a fraction of the cap"""

    n_winds: int
    """`N`: winds drawn from the belief and averaged over per solve; zero plans on the posterior mean alone"""

    def plan(
        self,
        key: PRNGKeyArray,
        dynamics: Differentiable[Float[Array, ""]],
        positions: Domain,
        fields: Function,
        state: dict[str, Any],
        target: Float[Array, " k"],
        warm: Float[Array, " horizon"],
    ) -> Float[Array, " horizon"]:
        """The H climbs (km) that minimise the mean J from `state` toward `target` over the batch `fields`, `warm` the previous plan"""
        return self._solver(cast(Hashable, dynamics), positions)(key, fields, state, target, warm)

    @functools.lru_cache(maxsize=8)  # noqa: B019 - the agent holds one navigator for one dynamics, so the cache is bounded
    def _solver(self, dynamics: Any, positions: Any) -> Callable[..., Float[Array, " horizon"]]:
        """The solve jitted once per (hashable) dynamics and positions, the field and the states being its arguments"""
        if not isinstance(dynamics, Differentiable):
            raise TypeError(f"a {type(dynamics).__name__} exposes no differentiable step")
        if not isinstance(positions, Embeddable):
            raise TypeError(f"distance to a waypoint needs coordinates, which a {type(positions).__name__} hands out none of")
        box = dynamics.action_domain
        if not isinstance(box, BoxDomain):
            raise TypeError("a gradient navigator steers a continuous balloon, whose actions are a box")
        cap, steps = box.high, jnp.arange(self.horizon)

        def rollout(
            climbs: Float[Array, " horizon"], field: Function, state: dict[str, Any]
        ) -> tuple[Float[Array, "horizon k"], Float[Array, " horizon"]]:
            """The embedded coordinates and the altitude along the H steps of `climbs` (km)"""

            def one(s: dict[str, Any], a: Float[Array, ""]) -> tuple[dict[str, Any], tuple[Float[Array, " k"], Float[Array, ""]]]:
                s2 = dynamics.step({**s, "field": field}, a)
                return s2, (positions.embed({"position": s2["position"], "altitude": s2["altitude"]}), s2["altitude"])

            return jax.lax.scan(one, state, climbs)[1]

        def one_wind(u: Float[Array, " horizon"], field: Function, state: dict[str, Any], target: Float[Array, " k"]) -> Float[Array, ""]:
            """J of the climbs `cap * u` in one wind"""
            where, altitude = rollout(cap * u, field, state)
            # the distance softened at zero, where the norm's gradient is undefined and would poison the plan with NaN
            distance = jnp.sqrt(jnp.sum((where - target) ** 2, axis=-1) + 1e-6)
            spent = jnp.abs(jnp.diff(altitude, prepend=state["altitude"]))
            return jnp.sum(distance) + self.resource_weight * jnp.sum(spent)

        def objective(u: Float[Array, " horizon"], fields: Function, state: dict[str, Any], target: Float[Array, " k"]) -> Float[Array, ""]:
            """J averaged over the winds in the batch"""
            return jnp.mean(jax.vmap(one_wind, in_axes=(None, 0, None, None))(u, fields, state, target))

        def pool(
            key: PRNGKeyArray, fields: Function, state: dict[str, Any], warm: Float[Array, " horizon"]
        ) -> Float[Array, "pool horizon"]:
            """M random candidates, the previous plan, and coast, as fractions of the cap"""
            k_target, k_noise, k_coast = jax.random.split(key, 3)
            field = jax.tree.map(lambda a: a[0], fields)  # the altitude reached does not depend on the wind, so any one wind serves
            _, up = rollout(jnp.full(self.horizon, cap), field, state)  # (H,) the altitude under a saturated climb
            _, down = rollout(jnp.full(self.horizon, -cap), field, state)  # and under a saturated descent
            now = state["altitude"]
            low, high = jnp.minimum(up[-1], down[-1]), jnp.maximum(up[-1], down[-1])  # the band reachable this horizon
            targets = jax.random.uniform(k_target, (self.candidates,), minval=low, maxval=high)
            climbing = jnp.sign(targets - now) == jnp.sign(up[-1] - now)  # (M,) whether +cap moves toward the target
            path = jnp.where(climbing[:, None], up[None], down[None])  # (M, H) the saturated altitude toward each target
            needed = jnp.sum(jnp.abs(path - now) < jnp.abs(targets - now)[:, None], axis=1)  # (M,) steps before it is reached
            saturated = jnp.where(climbing, 0.95, -0.95)[:, None] * jnp.ones((1, self.horizon))
            noise = jax.random.uniform(k_noise, (self.candidates, self.horizon), minval=-self.perturbation, maxval=self.perturbation)
            random = jnp.where(steps[None] < needed[:, None], saturated, noise)
            coast = jax.random.uniform(k_coast, (1, self.horizon), minval=-self.perturbation, maxval=self.perturbation)
            return jnp.concatenate([random, jnp.clip(warm / cap, -1.0, 1.0)[None], coast], axis=0)

        def refine(
            u: Float[Array, " horizon"], fields: Function, state: dict[str, Any], target: Float[Array, " k"]
        ) -> Float[Array, " horizon"]:
            """Normalised gradient descent on J in the unconstrained v, u = tanh(v / 2), from the candidate `u`"""
            loss = lambda v: objective(jnp.tanh(v / 2.0), fields, state, target)

            def improve(_: Any, v: Float[Array, " horizon"]) -> Float[Array, " horizon"]:
                g = jax.grad(loss)(v)
                norm = jnp.linalg.norm(g)
                return jnp.where(norm < self.tolerance, v, v - self.step_size * g / (norm + 1e-12))  # converged: hold

            v0 = 2.0 * jnp.arctanh(jnp.clip(u, -0.999, 0.999))
            return jnp.tanh(jax.lax.fori_loop(0, self.iterations, improve, v0) / 2.0)

        @jax.jit
        def solve(
            key: PRNGKeyArray, fields: Function, state: dict[str, Any], target: Float[Array, " k"], warm: Float[Array, " horizon"]
        ) -> Float[Array, " horizon"]:
            everyone = pool(key, fields, state, warm)  # (M + 2, H)
            scored = jax.vmap(objective, in_axes=(0, None, None, None))(everyone, fields, state, target)
            best = everyone[jnp.argsort(scored)[: self.descend_top]]  # (k, H)
            refined = jax.vmap(refine, in_axes=(0, None, None, None))(best, fields, state, target)
            costs = jax.vmap(objective, in_axes=(0, None, None, None))(refined, fields, state, target)
            return cap * refined[jnp.argmin(costs)]

        return solve
