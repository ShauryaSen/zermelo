"""How a state advances: one step sampled, and the same law written down exactly"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from jaxtyping import Array, Float, Int, PRNGKeyArray

from zermelo.interface.domain import Domain


class Transition[Action](ABC):
    """A pure, immutable `(key, state, action) -> state`"""

    @property
    @abstractmethod
    def action_domain(self) -> Domain:
        """Every action this transition accepts, ever"""

    @abstractmethod
    def __call__(self, key: PRNGKeyArray, state: dict[str, Any], action: Action) -> dict[str, Any]:
        """One step: a legal state with the same parts"""

    def legal_actions(self, state: dict[str, Any]) -> Domain:
        """The actions legal at `state`, as a narrowed domain; all of them, unless overridden"""
        return self.action_domain


class TransitionKernel[Action](ABC):
    """The exact law over a finite domain the transition itself chooses, indexed by that domain's `index_of`"""

    domain: Domain
    """What `values` is indexed by. Always `Enumerable`"""

    @abstractmethod
    def expectation(self, values: Float[Array, " n"], action: Action) -> Float[Array, " n"]:
        """`E[values(next) | state, action]` at every element of `domain`"""

    @abstractmethod
    def sample(self, key: PRNGKeyArray, index: Int[Array, ""], action: Action) -> Int[Array, ""]:
        """One step drawn from the same law, as an index into `domain`"""

    def step_cost(self, action: Action) -> Float[Array, " n"] | None:
        """What one step under `action` costs beyond the step itself, at every element of `domain`"""
        return None


@runtime_checkable
class Analytic[Action, Hypothesis](Protocol):
    """A transition whose law can be written down exactly, once a hypothesis settles what it cannot see"""

    def kernel(self, hypothesis: Hypothesis) -> TransitionKernel[Action]:
        """The law under `hypothesis`, precomputed once"""
        ...


@runtime_checkable
class Differentiable[Action](Protocol):
    """A transition whose step, randomness aside, is a pure map `(state, action) -> state` that `jax.grad` passes through"""

    @property
    def action_domain(self) -> Domain:
        """Every action this transition accepts, ever"""
        ...

    def step(self, state: dict[str, Any], action: Action) -> dict[str, Any]:
        """s' given s and a, with no randomness: the parts a planner differentiates"""
        ...
