"""The control arm: acts at random and learns nothing"""

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray

from zermelo.interface import Agent, Decision, Function, Observation, Uniform


@dataclass(frozen=True)
class RandomAgent(Agent[Int[Array, ""]]):
    """Draws a legal act uniformly every step, and asserts the same thing about the field at every one of them"""

    claim: Function
    """What it asserts about the field, unchanged all episode"""

    def reset(self, key: PRNGKeyArray, obs: Observation) -> Int[Array, ""]:
        """A step count, starting at zero"""
        return jnp.zeros((), jnp.int32)

    def decide(self, key: PRNGKeyArray, agent_state: Int[Array, ""], obs: Observation) -> tuple[Int[Array, ""], Decision]:
        """One act uniform over the legal ones, finite or boxed"""
        return agent_state + 1, Decision(Uniform().sample(obs.legal_actions, key), self.claim)
