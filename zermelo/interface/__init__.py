"""The library API surface: what a third-party method implements or consumes.

Everything here is a set or a value in that set.

- `Domain` -- `contains`, `project`, `narrow`. Every set the framework quantifies over.
- `DiscreteDomain`, `BoxDomain` -- one of `n` options; a bounded real array.
- `ProductDomain` -- named parts, each its own domain.
- `FunctionDomain` -- elements are callables. The latent field, and a claim about it.
- `Subset` -- what `narrow` returns.
- `Function` -- an element of a `FunctionDomain`.
- `Enumerable` -- `size`, `index_of`, `from_index`, `elements`.
- `Embeddable` -- `dim`, `embed`, `unembed`.
- `Prior`, `Uniform`, `ProductPrior` -- the law an element is drawn by.
- `World` -- a state domain, a prior, a transition and a readout.
- `Transition` -- `(key, state, action) -> state`, and which actions are legal.
- `TransitionKernel`, `Analytic` -- the same law exactly, as `expectation`.
- `Readout`, `Observation` -- what an agent sees, and what it receives.
- `Objective` -- the claim domain, a reward, termination.
- `Agent` -- `reset` and `decide`.
- `Decision` -- the pair `(action, claim)`.
"""

from zermelo.interface.agent import Agent
from zermelo.interface.decision import Decision
from zermelo.interface.domain import (
    BoxDomain,
    DiscreteDomain,
    Domain,
    Embeddable,
    Enumerable,
    Function,
    FunctionDomain,
    ProductDomain,
    Subset,
)
from zermelo.interface.objective import Objective
from zermelo.interface.observation import Observation, Readout
from zermelo.interface.prior import Prior, ProductPrior, Uniform
from zermelo.interface.transition import Analytic, Differentiable, Transition, TransitionKernel
from zermelo.interface.world import World

__all__ = [
    "Agent",
    "Analytic",
    "Differentiable",
    "BoxDomain",
    "Decision",
    "DiscreteDomain",
    "Domain",
    "Embeddable",
    "Enumerable",
    "Function",
    "FunctionDomain",
    "Objective",
    "Observation",
    "Prior",
    "ProductDomain",
    "ProductPrior",
    "Readout",
    "Subset",
    "Transition",
    "TransitionKernel",
    "Uniform",
    "World",
]
