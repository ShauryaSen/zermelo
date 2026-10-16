"""The method itself: pick a waypoint, commit a route to it, walk that route, pick again"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from zermelo.interface import (
    Agent,
    Analytic,
    Decision,
    Differentiable,
    Domain,
    Embeddable,
    Enumerable,
    Function,
    Observation,
    Subset,
    Transition,
    Uniform,
)
from zermelo.methods.waypoint_bo.acquisition import Acquisition, Scores
from zermelo.methods.waypoint_bo.belief import Belief, Dataset, coordinates, lookup
from zermelo.methods.waypoint_bo.navigator import GradientMPC
from zermelo.methods.waypoint_bo.planner import Policy


def posterior_moments(belief: Belief, context: Float[Array, " n_context"]) -> Function:
    """The belief's mean and log-variance per component at one context, readable at any cell"""
    mean, var = belief.predict(jnp.arange(coordinates(belief.positions).shape[0]), context)
    return lookup(belief.positions, jnp.concatenate([mean, jnp.log(var)], axis=-1))


@register_dataclass
@dataclass(frozen=True)
class WaypointAgentState:
    """What the method carries between steps, and what a consumer plots it by"""

    belief: Belief
    """Holds the readings folded in so far, and the model's parameters as they stand"""

    scores: Scores
    """The acquisition pass that chose the standing waypoint, carried until the next one"""

    policy: Policy
    """The route committed to when the waypoint was chosen, one column wide"""

    steps_since_waypoint: Int[Array, ""]
    """Steps taken since that waypoint was chosen"""

    steps_taken: Int[Array, ""]
    """Moves made this episode"""

    claim: Function = dataclasses.field(metadata=dict(static=True))
    """What the method asserts about the field, re-read off the belief every `claim_every` moves and held unchanged in between"""


@dataclass(frozen=True, kw_only=True)
class WaypointAgent(Agent[WaypointAgentState]):
    """Bayesian optimization over waypoints, with the walk to each waypoint scored and charged"""

    candidates: Subset[Any]
    """Which cells are worth aiming at, published by the problem over the set the field lives on"""

    transition: Analytic[Int[Array, ""], Any]
    """The exact one-step law under a guessed field"""

    belief: Belief
    """An empty belief, its buffer sized to `horizon`"""

    acquisition: Acquisition

    position_key: str
    """Which part of the reading carries where the actor stands"""

    reading_key: str
    """Which part of the reading carries what it read there"""

    context_key: str
    """Which part of the reading carries the coordinates beyond the cell, absent where the readout publishes none"""

    horizon: int
    """`T`: the episode's move count, one reading per move filling one buffer row"""

    opening_legs: int
    """How many of the planner's step budgets are walked at random before the rule starts"""

    claim_every: int
    """Moves between re-readings of the claim off the belief"""

    @property
    def positions(self) -> Domain:
        """The cells the field is defined on"""
        return self.candidates.base

    def _live_candidates(self) -> Int[Array, " n_live"]:
        """Every candidate cell, as position domain indices"""
        return jnp.flatnonzero(self.candidates.live)

    def _legal_actions(self, obs: Observation) -> Int[Array, " n_actions"]:
        """The acts legal at the cell being acted from"""
        if not isinstance(obs.legal_actions, Enumerable):
            raise TypeError(f"a planner enumerates acts, and a {type(obs.legal_actions).__name__} has no index")
        return jnp.asarray(obs.legal_actions.elements())

    def _checked_kernel(self, field: Function) -> Any:
        """The law under `field`, checked to be written over the cells this method aims at"""
        kernel = self.transition.kernel(field)
        if kernel.domain != self.positions:
            raise ValueError("the transition's kernel is written over a different set from the one this agent was given")
        return kernel

    def _context(self, obs: Observation) -> Float[Array, " n_context"]:
        """The coordinates this reading carries beyond its cell, empty where the readout publishes none"""
        return obs.reading[self.context_key] if self.context_key in obs.reading else jnp.zeros(0)

    def _fold_reading(self, belief: Belief, obs: Observation) -> tuple[Int[Array, ""], Belief]:
        """Where the actor stands, and `belief` with this step's reading folded in"""
        z = self.candidates.index_of(obs.reading[self.position_key])
        row = int(jnp.sum(belief.data.live))  # rows written so far; concrete, an episode being a Python loop
        written = belief.data.write(row, z, obs.reading[self.reading_key], self._context(obs))
        return z, belief.fit(written)

    def reset(self, key: PRNGKeyArray, obs: Observation) -> WaypointAgentState:
        """Nothing read yet: an empty buffer, blank scores at the shape they keep, and the step budget already spent"""
        if self.belief.positions != self.positions:
            raise ValueError("the belief and the candidate set were built over different position domains")
        if self.acquisition.planner.max_steps * self.opening_legs >= self.horizon:
            raise ValueError("the random walk opening is longer than the episode horizon")
        n_states = coordinates(self.positions).shape[0]
        width = self.belief.data.r.shape[-1]
        n_scored = len(self._live_candidates()) if self.acquisition.n_candidates is None else self.acquisition.n_candidates
        steps = 1 if self.acquisition.n_walks == 0 else self.acquisition.planner.max_steps
        blank = Scores(
            candidate_indices=jnp.zeros(n_scored, jnp.int32),
            acquisition_value=jnp.zeros(n_scored),
            waypoint=jnp.zeros((), jnp.int32),
            imagined_walk_to_waypoint=jnp.zeros(steps, jnp.int32),
            waypoint_value_spread=jnp.zeros(()),
            predicted_step_cost=jnp.zeros(()),
            rolled_step_cost=jnp.zeros(()),
            frac_zero_value_candidates=jnp.zeros(()),
            step_charge=jnp.zeros(()),
            frac_reachable_candidates=jnp.zeros(()),
        )
        policy = Policy(jnp.zeros((n_states, 1), jnp.int32), jnp.zeros((n_states, 1)))
        context = self._context(obs)
        empty = self.belief.condition(Dataset.empty(self.horizon, width, context.shape[-1]))
        return WaypointAgentState(
            empty,
            blank,
            policy,
            jnp.asarray(self.acquisition.planner.max_steps, jnp.int32),
            jnp.zeros((), jnp.int32),
            posterior_moments(empty, context),
        )

    def decide(self, key: PRNGKeyArray, agent_state: WaypointAgentState, obs: Observation) -> tuple[WaypointAgentState, Decision]:
        """The reading folded in, and either a random opening act, a fresh waypoint, or the next act of the standing route"""
        k_draw, k_score, k_plan = jax.random.split(key, 3)
        planner = self.acquisition.planner
        z, belief = self._fold_reading(agent_state.belief, obs)
        context = self._context(obs)  # the field is read, planned and scored at the context this reading carries
        claim = posterior_moments(belief, context) if int(agent_state.steps_taken) % self.claim_every == 0 else agent_state.claim
        if int(agent_state.steps_taken) < planner.max_steps * self.opening_legs:
            # the opening: a uniform act, no waypoint and no planning
            return WaypointAgentState(
                belief, agent_state.scores, agent_state.policy, agent_state.steps_since_waypoint, agent_state.steps_taken + 1, claim
            ), Decision(jax.random.choice(k_draw, self._legal_actions(obs)), claim)
        budget_spent = agent_state.steps_since_waypoint >= planner.max_steps
        if bool(budget_spent | planner.arrived(self.positions, agent_state.scores.waypoint[None])[z, 0]):
            actions = self._legal_actions(obs)
            scores = self.acquisition.choose(k_score, belief, self.transition, z, actions, self.candidates, context)
            # the walk actually taken is planned under the posterior mean, whatever field the winner was scored under
            policy = planner.plan(k_plan, self._checked_kernel(belief.mean(context)), scores.waypoint[None], actions)
            steps = jnp.zeros((), jnp.int32)
        else:
            scores, policy, steps = agent_state.scores, agent_state.policy, agent_state.steps_since_waypoint + 1
        return WaypointAgentState(belief, scores, policy, steps, agent_state.steps_taken + 1, claim), Decision(policy.act[z, 0], claim)


@register_dataclass
@dataclass(frozen=True)
class GradientAgentState:
    """What the gradient variant carries: the same as `WaypointAgentState`, with the standing climbs in place of a table"""

    belief: Belief
    scores: Scores

    plan: Float[Array, " horizon"]
    """The climbs last solved for, the first flown and the rest warm-starting the next solve"""

    steps_since_waypoint: Int[Array, ""]
    steps_taken: Int[Array, ""]
    claim: Function = dataclasses.field(metadata=dict(static=True))


@dataclass(frozen=True, kw_only=True)
class GradientAgent(WaypointAgent):
    """The same waypoints, flown by first-order MPC over a continuous balloon instead of a table over cells.

    Waypoints are still chosen and priced on the grid (`transition`, `acquisition`); the leg to one is flown by
    `navigator` on `dynamics` under the belief's mean wind, replanned every move.
    """

    dynamics: Differentiable[Float[Array, ""]]
    """The continuous law the leg is planned on, the world's own"""

    navigator: GradientMPC

    interpolate: Callable[[Function], Function]
    """Turns a field held at the cells (the belief's mean) into one readable between them, as the problem defines it"""

    def _here(self, obs: Observation) -> dict[str, Any]:
        """The continuous state the reading describes, as the dynamics read it"""
        where, context = obs.reading[self.position_key], self._context(obs)
        return {
            "position": jnp.asarray(where["position"]) * 1.0,  # real, whatever the readout handed over
            "altitude": jnp.asarray(where["altitude"]) * 1.0,
            "balloon_resource": jnp.asarray(obs.reading["balloon_resource"]) * 1.0,
            "hours": context[0] if context.shape[0] else jnp.zeros(()),
        }

    @property
    def _space(self) -> Any:
        """The positions, checked to be indexed and to hand out coordinates"""
        if not (isinstance(self.positions, Enumerable) and isinstance(self.positions, Embeddable)):
            raise TypeError(f"a waypoint is an index with coordinates, which a {type(self.positions).__name__} has neither of")
        return self.positions

    def _target(self, waypoint: Int[Array, ""]) -> Float[Array, " k"]:
        """The waypoint's coordinates"""
        return self._space.embed(self._space.from_index(waypoint))

    def _pricing_actions(self) -> Int[Array, " n_actions"]:
        """The grid's acts, which the acquisition prices a trip with"""
        pricing = self.transition
        if not (isinstance(pricing, Transition) and isinstance(pricing.action_domain, Enumerable)):
            raise TypeError("trips are priced on a transition with finitely many acts")
        return jnp.asarray(pricing.action_domain.elements())

    def _winds(self, key: PRNGKeyArray, belief: Belief, context: Float[Array, " n_context"]) -> Function:
        """The winds a leg is planned over, as one batch: `n_winds` draws from the posterior, or its mean alone"""
        n = self.navigator.n_winds
        drawn = belief.draw(key, n, context) if n > 0 else [belief.mean(context)]
        return jax.tree.map(lambda *a: jnp.stack(a), *[self.interpolate(f) for f in drawn])

    def reset(self, key: PRNGKeyArray, obs: Observation) -> GradientAgentState:  # type: ignore[override]
        """As the table agent, with an all-hold plan in place of the table"""
        base = super().reset(key, obs)
        plan = jnp.zeros(self.navigator.horizon)
        return GradientAgentState(base.belief, base.scores, plan, base.steps_since_waypoint, base.steps_taken, base.claim)

    def decide(  # type: ignore[override]
        self, key: PRNGKeyArray, agent_state: GradientAgentState, obs: Observation
    ) -> tuple[GradientAgentState, Decision]:
        """The reading folded in, and either a random opening climb, a fresh waypoint, or the next climb toward the standing one"""
        k_draw, k_score, k_winds, k_plan = jax.random.split(key, 4)
        planner = self.acquisition.planner
        z, belief = self._fold_reading(agent_state.belief, obs)
        context = self._context(obs)
        claim = posterior_moments(belief, context) if int(agent_state.steps_taken) % self.claim_every == 0 else agent_state.claim
        taken = agent_state.steps_taken + 1
        if int(agent_state.steps_taken) < planner.max_steps * self.opening_legs:
            # the opening: a uniform climb, no waypoint and no planning
            state = GradientAgentState(belief, agent_state.scores, agent_state.plan, agent_state.steps_since_waypoint, taken, claim)
            return state, Decision(Uniform().sample(obs.legal_actions, k_draw), claim)
        here = self._here(obs)
        where = self._space.embed({"position": here["position"], "altitude": here["altitude"]})
        budget_spent = agent_state.steps_since_waypoint >= planner.max_steps
        arrived = jnp.linalg.norm(where - self._target(agent_state.scores.waypoint)) <= planner.radius
        if bool(budget_spent | arrived):
            scores = self.acquisition.choose(k_score, belief, self.transition, z, self._pricing_actions(), self.candidates, context)
            plan, steps = jnp.zeros_like(agent_state.plan), jnp.zeros((), jnp.int32)
        else:
            scores, plan, steps = agent_state.scores, agent_state.plan, agent_state.steps_since_waypoint + 1
        # the leg is flown under N draws from the posterior, or under its mean alone, whatever field the waypoint was scored under
        fields = self._winds(k_winds, belief, context)
        plan = self.navigator.plan(k_plan, self.dynamics, self.positions, fields, here, self._target(scores.waypoint), plan)
        carried = jnp.concatenate([plan[1:], plan[-1:]])  # what is left of the plan once its first climb is flown
        return GradientAgentState(belief, scores, carried, steps, taken, claim), Decision(plan[0], claim)


@dataclass(frozen=True, kw_only=True)
class OpenLoopAgent(GradientAgent):
    """`GradientAgent` that solves once a leg and then flies what it solved, never replanning

    The value of the best fixed plan, which is what a controller earns when it cannot react. Everything else
    is the gradient agent's: the same waypoints, the same solver, the same winds. Only the moment of solving
    differs -- here it is the first move of a leg, there it is every move.

    Flown under `OracleBelief` this is the hindsight bound instead: the plan is solved on the wind the balloon
    will actually meet, and with the dynamics deterministic given that wind no policy can beat the best plan
    for it.
    """

    def decide(  # type: ignore[override]
        self, key: PRNGKeyArray, agent_state: GradientAgentState, obs: Observation
    ) -> tuple[GradientAgentState, Decision]:
        """As `GradientAgent`, except that a standing plan is flown out rather than solved again"""
        k_draw, k_score, k_winds, k_plan = jax.random.split(key, 4)
        planner = self.acquisition.planner
        z, belief = self._fold_reading(agent_state.belief, obs)
        context = self._context(obs)
        claim = posterior_moments(belief, context) if int(agent_state.steps_taken) % self.claim_every == 0 else agent_state.claim
        taken = agent_state.steps_taken + 1
        if int(agent_state.steps_taken) < planner.max_steps * self.opening_legs:
            state = GradientAgentState(belief, agent_state.scores, agent_state.plan, agent_state.steps_since_waypoint, taken, claim)
            return state, Decision(Uniform().sample(obs.legal_actions, k_draw), claim)
        here = self._here(obs)
        where = self._space.embed({"position": here["position"], "altitude": here["altitude"]})
        budget_spent = agent_state.steps_since_waypoint >= planner.max_steps
        arrived = jnp.linalg.norm(where - self._target(agent_state.scores.waypoint)) <= planner.radius
        opening = bool(budget_spent | arrived)
        if opening:
            scores = self.acquisition.choose(k_score, belief, self.transition, z, self._pricing_actions(), self.candidates, context)
            plan, steps = jnp.zeros_like(agent_state.plan), jnp.zeros((), jnp.int32)
        else:
            scores, plan, steps = agent_state.scores, agent_state.plan, agent_state.steps_since_waypoint + 1
        if opening:  # the one solve of this leg, on the winds the leg's waypoint was scored under
            fields = self._winds(k_winds, belief, context)
            plan = self.navigator.plan(k_plan, self.dynamics, self.positions, fields, here, self._target(scores.waypoint), plan)
        carried = jnp.concatenate([plan[1:], plan[-1:]])  # the plan less the climb being flown
        return GradientAgentState(belief, scores, carried, steps, taken, claim), Decision(plan[0], claim)
