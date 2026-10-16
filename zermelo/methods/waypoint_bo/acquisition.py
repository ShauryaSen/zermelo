"""alpha(x) = w(x) - c cost(tau) or w(x) / cost(tau), where w(x) = E over fields and walks of [ U(D + obs(tau)) - U(D) ]"""

from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from zermelo.interface import Analytic, Subset, TransitionKernel
from zermelo.methods.waypoint_bo.belief import Belief, Dataset, elements
from zermelo.methods.waypoint_bo.planner import Planner, Policy
from zermelo.methods.waypoint_bo.utility import Utility


@register_dataclass
@dataclass(frozen=True)
class Scores:
    """One acquisition pass: what each candidate was worth, which won, and how the pass went"""

    candidate_indices: Int[Array, " n_candidates"]
    """The cells scored, as position domain indices"""

    acquisition_value: Float[Array, " n_candidates"]
    """`alpha` at each of them"""

    waypoint: Int[Array, ""]
    """`x_{n+1}`: the candidate that won"""

    imagined_walk_to_waypoint: Int[Array, " steps"]
    """The winner's route under the first field and walk"""

    waypoint_value_spread: Float[Array, ""]
    """Standard deviation of the winner's value over fields and walks"""

    predicted_step_cost: Float[Array, ""]
    """`H(z, x)`: what the winner's route is expected to spend, off the planner's own table"""

    rolled_step_cost: Float[Array, ""]
    """`mean cost(tau)`: what the winner's rolled walks actually spent"""

    frac_zero_value_candidates: Float[Array, ""]
    """Share of the candidates scored that are worth exactly zero before travel is charged"""

    step_charge: Float[Array, ""]
    """What one unit of imagined spend was charged on this pass"""

    frac_reachable_candidates: Float[Array, ""]
    """Share of the candidates scored whose expected spend fell under the planner's cost budget"""


@dataclass(frozen=True)
class Acquisition:
    """Scores candidate waypoints and takes the best"""

    utility: Utility
    planner: Planner

    n_fields: int
    """`S`: posterior fields drawn per decision, or the posterior mean at zero"""

    n_walks: int
    """`M`: walks imagined per field, or the destination alone at zero"""

    n_candidates: int | None
    """How many candidates to score: `None` every live one, an integer that many drawn uniformly each decision"""

    improvement: bool
    """Whether to score the held readings and the imagined ones together, less the held ones alone"""

    step_rate: float | None
    """`lambda`: a full-budget trip costs this share of the spread of the candidate values"""

    steps_from: Literal["predicted", "rolled"]
    """Whether travel is counted off the hitting-time table or off the rolled walks"""

    combination: Literal["linear", "fraction"]
    """Whether travel is subtracted from the worth or divides it"""

    def __post_init__(self) -> None:
        if self.n_walks == 0 and self.steps_from == "rolled":
            raise ValueError("steps_from='rolled' needs walks to average; set n_walks above zero or price off the planner")
        if (self.step_rate is None) != (self.combination == "fraction"):
            raise ValueError("a linear combination is priced by step_rate and a fraction takes none; state exactly one of the two")

    def walk_cost(
        self,
        kernel: TransitionKernel[Int[Array, ""]],
        walk: Int[Array, "n_walks n_candidates steps"],
        moved: Bool[Array, "n_walks n_candidates steps"],
        policy: Policy,
        start: Int[Array, ""],
        actions: Int[Array, " n_actions"],
    ) -> Float[Array, "n_walks n_candidates"]:
        """What each walk costs, over the steps that moved:

        cost(tau) = sum over t of c(z_t, act(z_t, x)),   z_0 = start
        """
        if kernel.step_cost(actions[0]) is None:
            return jnp.sum(moved, axis=-1).astype(jnp.float32)
        table = jnp.stack([self.planner.cost(kernel, a) for a in actions], axis=-1)  # (n_states, n_actions): c(s, a)
        at = jnp.concatenate([jnp.full(walk.shape[:-1] + (1,), start), walk[..., :-1]], axis=-1)  # z_t, the state each step left
        column = jnp.arange(walk.shape[1])[None, :, None]  # (1, n_candidates, 1): the target each walk steers to
        cost = table[at, policy.act[at, column]]  # (n_walks, n_candidates, steps): c(z_t, act(z_t, x))
        return jnp.sum(jnp.where(moved, cost, 0.0), axis=-1)

    def choose(
        self,
        key: PRNGKeyArray,
        belief: Belief,
        transition: Analytic[Int[Array, ""], Any],
        position: Int[Array, ""],
        actions: Int[Array, " n_actions"],
        candidates: Subset[Any],
        context: Float[Array, " n_context"],
    ) -> Scores:
        """Every candidate scored under every field and walk, and the argmax over them"""
        k_indices, k_field, k_plan, k_walk, k_utility = jax.random.split(key, 5)
        candidate_indices = jnp.flatnonzero(candidates.live)
        if self.n_candidates is not None:
            # a prefix of one permutation, so a shorter shortlist is a subset of every longer one
            candidate_indices = jax.random.permutation(k_indices, candidate_indices)[: self.n_candidates]
        # keyed by index, so the draws of a smaller n_walks are a subset of a larger one
        walk_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(k_walk, jnp.arange(max(self.n_walks, 1)))
        held, n_scored = belief.data, candidate_indices.shape[0]
        # pi[mu_n], solved once and reused
        mean_policy = self.planner.plan(k_plan, transition.kernel(belief.mean(context)), candidate_indices, actions)
        utility_samples, predicted_samples, rolled_samples, first_draw_walks = [], [], [], None

        # f-hat^(1..S)
        fields = [belief.mean(context)] if self.n_fields == 0 else belief.draw(k_field, self.n_fields, context)
        score_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(k_utility, jnp.arange(len(fields)))  # keyed by index
        for k_score, field in zip(score_keys, fields, strict=True):
            kernel = transition.kernel(field)  # p(.|f-hat)
            values = field(elements(kernel.domain))  # f-hat(z), one row per cell of the position domain
            held_under_field = Dataset(held.z, values[held.z], held.live, held.context)  # D_n, its readings re-taken from f-hat
            base_utility = self.utility(k_score, belief, held_under_field, candidates) if self.improvement else jnp.zeros(())  # U(D_n)

            if self.n_walks == 0:  # (1, n_candidates, 1): tau = (x), the destination and no route
                walk, moved = candidate_indices[None, :, None], jnp.ones((1, n_scored, 1), bool)
            else:  # both (n_walks, n_candidates, steps): tau ~ p(. | z, x, f-hat, pi)
                walk, moved = self.planner.roll(walk_keys, kernel, mean_policy, position)
            first_draw_walks = walk[0] if first_draw_walks is None else first_draw_walks  # (n_candidates, steps) under f-hat^(1), tau^(1)

            # obs(tau) = {(z_i, f-hat(z_i), context)}: the walk's cells read off f-hat, every row at the plan's own context
            imagined = Dataset(walk, values[walk], moved, jnp.broadcast_to(context, (*walk.shape, context.shape[-1])))
            scored = held_under_field.broadcast(walk.shape[:-1]).concat(imagined) if self.improvement else imagined  # D_n + obs(tau)
            utility_term = self.utility(k_score, belief, scored, candidates) - base_utility  # (n_walks, n_candidates)
            rolled = self.walk_cost(kernel, walk, moved, mean_policy, position, actions).astype(utility_term.dtype)  # cost(tau)
            utility_samples.append(utility_term)
            rolled_samples.append(rolled)
            predicted_samples.append(jnp.broadcast_to(mean_policy.hitting_cost[position], rolled.shape))  # H(z, x)

        # every stack is (n_fields, n_walks, n_candidates); averaging and spread run over both sample axes
        utility, rolled, predicted = jnp.stack(utility_samples), jnp.stack(rolled_samples), jnp.stack(predicted_samples)
        worth = jnp.mean(utility, axis=(0, 1))  # w(x), one per candidate
        spend = predicted if self.steps_from == "predicted" else rolled
        # c = lambda * (max w - min w) / B, for B what a full-budget route may spend. A flat w charges
        # nothing, and the argmax then takes the lowest index.
        rate = 0.0 if self.step_rate is None else self.step_rate
        charge = jnp.zeros(()) if rate == 0.0 else rate * (jnp.max(worth) - jnp.min(worth)) / self.planner.cost_budget
        sampled = utility - charge * spend if self.combination == "linear" else utility / jnp.maximum(spend, 1.0)
        value = jnp.mean(sampled, axis=(0, 1))  # alpha(x)
        expected_spend = jnp.mean(predicted, axis=(0, 1))  # H(z, x), averaged over the draws that solved it
        # a candidate already within the planner's radius spends nothing and has no route to it, so it
        # would win every tie the cost term decides -- scored like the rest, but never chosen
        best = jnp.argmax(jnp.where(expected_spend == 0.0, -jnp.inf, value))
        return Scores(
            candidate_indices=candidate_indices,
            acquisition_value=value,
            waypoint=candidate_indices[best],
            imagined_walk_to_waypoint=jnp.asarray(first_draw_walks)[best],
            waypoint_value_spread=jnp.std(sampled, axis=(0, 1))[best],
            predicted_step_cost=expected_spend[best],
            rolled_step_cost=jnp.mean(rolled, axis=(0, 1))[best],
            frac_zero_value_candidates=jnp.mean((worth == 0.0).astype(value.dtype)),
            step_charge=charge,
            frac_reachable_candidates=jnp.mean((expected_spend < self.planner.cost_budget).astype(value.dtype)),
        )
