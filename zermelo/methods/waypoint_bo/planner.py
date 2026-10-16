"""How the actor gets from where it stands to a waypoint, under a guessed field, and what that costs"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from zermelo.interface import Domain, TransitionKernel
from zermelo.methods.waypoint_bo.belief import coordinates


@register_dataclass
@dataclass(frozen=True)
class Policy:
    """`pi_n`: the act to take at every cell to reach each target, and what that costs"""

    act: Int[Array, "n_states n_targets"]

    hitting_cost: Float[Array, "n_states n_targets"]
    """`H^pi(s, x) = c(s, pi(s)) + sum over s' of P(s' | s, pi(s)) H^pi(s', x)`, zero on arrival and truncated at `max_steps`"""


@dataclass(frozen=True)
class Planner(ABC):
    """Turns a kernel and a set of targets into an act table, and evaluates what following it costs"""

    max_steps: int
    """`L`: backups taken, so a route is planned and costed over at most `L` steps"""

    radius: float
    """`rho`: how close to a target counts as arrived, on embedded coordinates"""

    target_chunk: int | None
    """How many targets one solve handles, `None` taking every target at once"""

    cost_weight: float
    """`w`: what one unit of a step's own cost is worth in steps. Zero charges the step alone"""

    def cost(self, kernel: TransitionKernel[Int[Array, ""]], action: Int[Array, ""]) -> Float[Array, " n_states"] | float:
        """`c(., a) = 1 + w * step_cost(., a)`"""
        charged = kernel.step_cost(action)
        return 1.0 if charged is None else 1.0 + self.cost_weight * charged

    @property
    def cost_budget(self) -> float:
        """`B = L * (1 + w)`: the most a route of `max_steps` steps costs"""
        return self.max_steps * (1.0 + self.cost_weight)

    @abstractmethod
    def act_table(
        self,
        key: PRNGKeyArray,
        kernel: TransitionKernel[Int[Array, ""]],
        targets: Int[Array, " n_targets"],
        actions: Int[Array, " n_actions"],
    ) -> Int[Array, "n_states n_targets"]:
        """The act to take at each cell when steering to each target"""

    def arrived(self, domain: Domain, targets: Int[Array, " n_targets"]) -> Bool[Array, "n_states n_targets"]:
        """Whether each cell of `domain` is within `radius` of each target"""
        return self._squared_distance(domain, targets) <= self.radius**2

    def _squared_distance(self, domain: Domain, targets: Int[Array, " n_targets"]) -> Float[Array, "n_states n_targets"]:
        """Squared embedded distance from every cell to every target"""
        coords = coordinates(domain)  # (n_states, k)
        square = jnp.sum(coords**2, axis=-1)
        return jnp.maximum(square[:, None] + square[targets][None, :] - 2 * coords @ coords[targets].T, 0.0)

    def _backup(
        self, kernel: TransitionKernel[Int[Array, ""]], values: Float[Array, "n_states n_targets"], actions: Int[Array, " n_actions"]
    ) -> Float[Array, "n_actions n_states n_targets"]:
        """One Bellman backup, all targets at once:

        out[a, s, x] = c(s, a) + sum over s' of P(s' | s, a) values[s', x]
        """
        step = jax.vmap(kernel.expectation, in_axes=(1, None), out_axes=1)
        # (n_states, 1), or a scalar left as one, onto (n_states, n_targets)
        return jnp.stack([jnp.reshape(self.cost(kernel, jnp.asarray(a)), (-1, 1)) + step(values, jnp.asarray(a)) for a in actions])

    def plan(
        self,
        key: PRNGKeyArray,
        kernel: TransitionKernel[Int[Array, ""]],
        targets: Int[Array, " n_targets"],
        actions: Int[Array, " n_actions"],
    ) -> Policy:
        """An act table and a hitting cost for every target, solved `target_chunk` targets at a time"""
        step = targets.shape[0] if self.target_chunk is None else self.target_chunk
        if targets.shape[0] <= step:
            return self._plan_block(key, kernel, targets, actions)
        # every target is its own problem, so a block of columns is the whole answer for those columns
        blocks = [self._plan_block(key, kernel, targets[at : at + step], actions) for at in range(0, targets.shape[0], step)]
        return Policy(
            jnp.concatenate([block.act for block in blocks], axis=1),  # (n_states, n_targets)
            jnp.concatenate([block.hitting_cost for block in blocks], axis=1),  # (n_states, n_targets)
        )

    @partial(jax.jit, static_argnums=0)
    def _plan_block(
        self,
        key: PRNGKeyArray,
        kernel: TransitionKernel[Int[Array, ""]],
        targets: Int[Array, " n_targets"],
        actions: Int[Array, " n_actions"],
    ) -> Policy:
        """An act table for every target, and `max_steps` rounds of

        h_0(s)     = 0
        h_{k+1}(s) = 0                                                     on arrival
                   = c(s, act(s)) + sum over s' of P(s' | s, act(s)) h_k(s')
        """
        stop = self.arrived(kernel.domain, targets)
        act = self.act_table(key, kernel, targets, actions)
        rows, columns = jnp.arange(stop.shape[0])[:, None], jnp.arange(stop.shape[1])[None, :]
        hitting = lax.fori_loop(
            0,
            self.max_steps,
            lambda _, h: jnp.where(stop, 0.0, self._backup(kernel, h, actions)[act, rows, columns]),
            jnp.zeros(stop.shape),
        )
        return Policy(act, hitting)

    @partial(jax.jit, static_argnums=0)
    def roll(
        self, walk_keys: PRNGKeyArray, kernel: TransitionKernel[Int[Array, ""]], policy: Policy, start: Int[Array, ""]
    ) -> tuple[Int[Array, "n_walks n_targets max_steps"], Bool[Array, "n_walks n_targets max_steps"]]:
        """One sampled walk per key per target from `start`: the cells stepped onto, and which of those steps moved"""
        arrived = policy.hitting_cost == 0.0  # (n_states, n_targets): H = 0 marks the cells that count as the target
        target_column = jnp.arange(policy.act.shape[1])  # (n_targets,): which column of the tables each walk reads
        step_to = jax.vmap(kernel.sample, in_axes=(None, 0, 0))  # (key, cell per target, act per target) -> next cell per target

        def one_walk(key: PRNGKeyArray) -> tuple[Int[Array, "n_targets max_steps"], Bool[Array, "n_targets max_steps"]]:
            """One walk to every target: z_0 = start, z_{t+1} ~ P(. | z_t, act(z_t)) until arrived(z_t)"""
            cell = jnp.full(target_column.shape, start)  # (n_targets,): z_0 = start
            travelling = jnp.ones(target_column.shape, bool)  # (n_targets,): not arrived(z_t)
            path, moved = [], []
            for step_key in jax.random.split(key, self.max_steps):
                # z_{t+1} = P-draw where travelling, else z_t
                cell = jnp.where(travelling, step_to(step_key, cell, policy.act[cell, target_column]), cell)
                path.append(cell)
                moved.append(travelling)
                travelling = travelling & ~arrived[cell, target_column]  # travelling and not arrived(z_{t+1})
            return jnp.stack(path, axis=-1), jnp.stack(moved, axis=-1)  # both (n_targets, max_steps)

        return jax.vmap(one_walk)(walk_keys)  # both (n_walks, n_targets, max_steps)


@dataclass(frozen=True)
class ValueIteration(Planner):
    """`max_steps` rounds of

        v_0(s)     = 0
        v_{k+1}(s) = 0                                                          on arrival
                   = min over a of [ c(s, a) + sum over s' of P(s' | s, a) v_k(s') ]

    then the act attaining that minimum
    """

    def act_table(
        self,
        key: PRNGKeyArray,
        kernel: TransitionKernel[Int[Array, ""]],
        targets: Int[Array, " n_targets"],
        actions: Int[Array, " n_actions"],
    ) -> Int[Array, "n_states n_targets"]:
        stop = self.arrived(kernel.domain, targets)
        values = lax.fori_loop(
            0, self.max_steps, lambda _, v: jnp.where(stop, 0.0, jnp.min(self._backup(kernel, v, actions), axis=0)), jnp.zeros(stop.shape)
        )
        return actions[jnp.argmin(self._backup(kernel, values, actions), axis=0)]


@dataclass(frozen=True)
class Greedy(Planner):
    """One backup on the distance to the target:

        act(s) = argmin over a of [ c(s, a) + sum over s' of P(s' | s, a) d(s', x) ]

    for `d` the embedded distance.
    """

    def act_table(
        self,
        key: PRNGKeyArray,
        kernel: TransitionKernel[Int[Array, ""]],
        targets: Int[Array, " n_targets"],
        actions: Int[Array, " n_actions"],
    ) -> Int[Array, "n_states n_targets"]:
        return actions[jnp.argmin(self._backup(kernel, jnp.sqrt(self._squared_distance(kernel.domain, targets)), actions), axis=0)]


@dataclass(frozen=True)
class RandomWalk(Planner):
    """One uniform act per cell, drawn when the plan is made and the same for every target"""

    def act_table(
        self,
        key: PRNGKeyArray,
        kernel: TransitionKernel[Int[Array, ""]],
        targets: Int[Array, " n_targets"],
        actions: Int[Array, " n_actions"],
    ) -> Int[Array, "n_states n_targets"]:
        n_states = coordinates(kernel.domain).shape[0]
        drawn = actions[jax.random.randint(key, (n_states, 1), 0, actions.shape[0])]
        return jnp.broadcast_to(drawn, (n_states, targets.shape[0]))
