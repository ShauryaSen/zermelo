"""Running one episode and recording it as it goes"""

import resource
import time
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import keystr, tree_flatten_with_path
from jaxtyping import Array, Float, PRNGKeyArray

from zermelo.interface import Agent, Decision, FunctionDomain, Objective, Observation, Subset, World
from zermelo.run.record import Record


def _channels(instants: list[Any], under: str, recorded: dict[str, int]) -> dict[str, Any]:
    """One pytree per instant into one array per leaf, named by its dotted path, for the leaves `recorded` stores under `under`"""
    named = [{keystr(path, simple=True, separator=".") or "value": leaf for path, leaf in tree_flatten_with_path(x)[0]} for x in instants]
    stacked = {}
    for name in named[0]:
        every = recorded.get(f"{under}/{name}")
        if every is None:  # nothing downstream reads it
            continue
        at = sorted({*range(0, len(named), every), len(named) - 1})  # the last instant always
        try:
            stacked[name] = jnp.stack([named[i][name] for i in at])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{name!r} does not stack across the episode: {e}") from e
    return stacked


@dataclass(frozen=True)
class StateSnapshot:
    """Everything true at one instant of an episode"""

    state: dict[str, Any]
    """The true state, which the agent never sees"""

    observation: Observation
    """What the agent saw of it"""

    agent_state: Any
    """The agent's memory"""

    objective_state: Any
    """The objective's memory"""


@dataclass(frozen=True)
class TransitionMove:
    """What one move of an episode did and what computing it cost"""

    decision: Decision
    reward: Float[Array, ""]

    seconds: float
    """Seconds inside `decide`, measured once the result landed"""

    live_bytes: int
    """Bytes of every live jax array in the process once the decision completed"""


@dataclass
class Episode:
    """A world, an objective and an agent run for one episode, recorded as it goes"""

    world: World
    objective: Objective[Any]
    agent: Agent[Any]

    horizon: int
    """How many moves an episode may make"""

    recorded: dict[str, int]
    """What the record stores, and the moves between the snapshots it keeps of each. A name absent is not stored"""

    config: dict[str, Any]
    """The fully resolved configuration"""

    key_world: PRNGKeyArray
    """Handed to the world once, at reset"""

    key_agent: PRNGKeyArray
    """Handed to the agent once, at reset"""

    key_steps: PRNGKeyArray
    """The stream every move draws from"""

    key: PRNGKeyArray = field(init=False, repr=False)
    """What is left of that stream, split down as the episode consumes it"""

    snapshots: list[StateSnapshot] = field(init=False, repr=False, default_factory=list)
    """One more of these than there are moves"""

    moves: list[TransitionMove] = field(init=False, repr=False, default_factory=list)

    terminated: bool = field(init=False, default=False)
    started: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        """Refuse a horizon below one move"""
        if self.horizon < 1:
            raise ValueError(f"an episode makes at least one move, and this one was given a horizon of {self.horizon}")

    def run(self) -> Record:
        """One whole episode, start to record"""
        self.reset()
        while self.step():
            pass
        return self.record()

    def reset(self) -> None:
        """Draw an instance, hand it out, take the first snapshot"""
        self.terminated = False
        self.snapshots, self.moves = [], []
        self.key = self.key_steps
        state, obs = self.world.reset(self.key_world)
        self.snapshots.append(StateSnapshot(state, obs, self.agent.reset(self.key_agent, obs), self.objective.reset(state)))
        self.started = time.perf_counter()

    def step(self) -> bool:
        """One move, the snapshot it leads to, and whether the episode is still running"""
        now = self.snapshots[-1]
        k_decide, k_step, self.key = jax.random.split(self.key, 3)
        started = time.perf_counter()
        agent_state, decision = self.agent.decide(k_decide, now.agent_state, now.observation)
        jax.block_until_ready(jax.tree.leaves((agent_state, decision)))  # jax dispatches asynchronously; block so the timing is the work
        seconds = time.perf_counter() - started
        live = sum(a.nbytes for a in jax.live_arrays())  # measured before the world's step allocates more
        if not bool(now.observation.legal_actions.contains(decision.action)):
            raise ValueError(f"move {len(self.moves)}: the agent chose an action that is not legal at the state it acted from")
        if not bool(self.objective.claim_domain.contains(decision.claim)):
            raise ValueError(f"move {len(self.moves)}: the agent's claim is not an element of the objective's claim domain")
        next_state, obs = self.world.step(k_step, now.state, decision.action)
        objective_state, reward = self.objective.score(now.objective_state, now.state, decision, next_state)
        self.moves.append(TransitionMove(decision, reward, seconds, live))
        self.snapshots.append(StateSnapshot(next_state, obs, agent_state, objective_state))
        self.terminated = bool(self.objective.terminated(objective_state, next_state))
        return not self.terminated and len(self.moves) < self.horizon

    def record(self) -> Record:
        """Freeze what `recorded` names. A function-valued channel is skipped"""
        stored = [n for n, p in self.world.state_domain.parts.items() if not isinstance(p, FunctionDomain)]
        keeps_claim = not isinstance(self.objective.claim_domain, FunctionDomain)
        observation = _channels([s.observation.reading for s in self.snapshots], "observation", self.recorded)
        # only a narrowed domain carries an array; a plain one has nothing to store
        legal = (s.observation.legal_actions for s in self.snapshots)
        masks = [d.live for d in legal if isinstance(d, Subset)]
        if len(masks) == len(self.snapshots):  # a mask at every snapshot, so it stacks like any other channel
            observation |= _channels([{"legal": m} for m in masks], "observation", self.recorded)
        return Record(
            state=_channels([{name: s.state[name] for name in stored} for s in self.snapshots], "state", self.recorded),
            observation=observation,
            agent_state=_channels([s.agent_state for s in self.snapshots], "agent_state", self.recorded),
            objective_state=_channels([s.objective_state for s in self.snapshots], "objective_state", self.recorded),
            decision=_channels(
                [{"action": m.decision.action} | ({"claim": m.decision.claim} if keeps_claim else {}) for m in self.moves],
                "decision",
                self.recorded,
            ),
            reward=jnp.stack([m.reward for m in self.moves]),
            time_per_decision=jnp.asarray([m.seconds for m in self.moves]),
            memory_per_decision=jnp.asarray([m.live_bytes for m in self.moves]),
            config=self.config,
            terminated=self.terminated,
            time_per_episode=time.perf_counter() - self.started,
            peak_rss_per_process=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,  # ru_maxrss is kilobytes on linux
        )
