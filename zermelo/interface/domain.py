"""The sets the framework quantifies over, and the protocols some of them satisfy"""

import functools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


class Domain[X](ABC):
    """A set of legal elements"""

    @abstractmethod
    def contains(self, x: X) -> Bool[Array, ""]:
        """Whether `x` is a legal element, as a traced boolean"""

    @abstractmethod
    def project(self, x: X) -> X:
        """Snap `x` onto this set: the identity on legal elements, not necessarily the nearest one"""

    def narrow(self, witness: Any) -> "Domain[X]":
        """The sub-domain legal right now, given whatever this domain narrows by"""
        raise TypeError(f"a {type(self).__name__} cannot be narrowed")


@runtime_checkable
class Function(Protocol):
    """An element of a `FunctionDomain`: a pytree callable at an element of its source"""

    def __call__(self, x: Any) -> Float[Array, "*batch m"]: ...


@runtime_checkable
class Enumerable[X](Protocol):
    """A finite domain you can address by index"""

    def size(self) -> int:
        """How many elements there are, fixed at construction"""
        ...

    def index_of(self, x: X) -> Int[Array, ""]:
        """The index of `x`"""
        ...

    def from_index(self, i: Int[Array, ""]) -> X:
        """The element at `i`, the inverse of `index_of`"""
        ...

    def elements(self) -> X:
        """Every element, batched along a leading axis"""
        ...


@runtime_checkable
class Embeddable[X](Protocol):
    """A domain that hands out real coordinates for its elements"""

    def dim(self) -> int:
        """The coordinate width, fixed at construction"""
        ...

    def embed(self, x: X) -> Float[Array, "... k"]:
        """Coordinates for `x`; nearby elements must embed to nearby vectors"""
        ...

    def unembed(self, v: Float[Array, "... k"]) -> X:
        """The element at `v`, retracted onto the domain"""
        ...


@dataclass(frozen=True)
class Subset[X](Domain[X], Enumerable[X]):
    """An `Enumerable` domain narrowed to the elements live right now, keeping the base's `size`"""

    base: Domain[X]
    """The domain being narrowed; static and hashable"""

    live: Bool[Array, " n"]
    """One traced flag per element of `base`, in the base's own index order"""

    @property
    def _indexed(self) -> "Enumerable[X]":
        """`base` as an `Enumerable`, or a raise"""
        if not isinstance(self.base, Enumerable):
            raise TypeError(f"a {type(self.base).__name__} has no index, so it cannot be narrowed to a Subset")
        return self.base

    def contains(self, x: X) -> Bool[Array, ""]:
        return self.base.contains(x) & self.live[self._indexed.index_of(x)]

    def project(self, x: X) -> X:
        """`x` where it is live, else the first live element, else the base's first"""
        i = self._indexed.index_of(self.base.project(x))
        return self._indexed.from_index(jnp.where(self.live[i], i, jnp.argmax(self.live)))

    def size(self) -> int:
        return self._indexed.size()

    def index_of(self, x: X) -> Int[Array, ""]:
        return self._indexed.index_of(x)

    def from_index(self, i: Int[Array, ""]) -> X:
        return self._indexed.from_index(i)

    def elements(self) -> X:
        return self._indexed.elements()


@dataclass(frozen=True)
class DiscreteDomain(Domain[Int[Array, ""]], Enumerable[Int[Array, ""]]):
    """One of `n` options, as a scalar integer in `[0, n)`"""

    n: int

    def contains(self, x: Int[Array, ""]) -> Bool[Array, ""]:
        return (0 <= x) & (x < self.n)

    def project(self, x: Int[Array, ""]) -> Int[Array, ""]:
        return jnp.clip(x, 0, self.n - 1)

    def narrow(self, witness: Bool[Array, " n"]) -> Subset[Int[Array, ""]]:
        """`witness` marks which of the `n` options are live now"""
        return Subset(self, witness)

    def size(self) -> int:
        return self.n

    def index_of(self, x: Int[Array, ""]) -> Int[Array, ""]:
        return x

    def from_index(self, i: Int[Array, ""]) -> Int[Array, ""]:
        return i

    def elements(self) -> Int[Array, " n"]:
        return jnp.arange(self.n)


@dataclass(frozen=True)
class BoxDomain(Domain[Float[Array, "*shape"]], Embeddable[Float[Array, "*shape"]]):
    """A real array of `shape`, bounded elementwise by `[low, high]`"""

    shape: tuple[int, ...]
    low: float = -math.inf
    high: float = math.inf

    def contains(self, x: Float[Array, "*shape"]) -> Bool[Array, ""]:
        if x.shape != self.shape:
            raise ValueError(f"this box holds arrays of shape {self.shape}, got {x.shape}")
        return jnp.all((self.low <= x) & (x <= self.high))

    def project(self, x: Float[Array, "*shape"]) -> Float[Array, "*shape"]:
        return jnp.clip(x, self.low, self.high)

    def narrow(self, witness: tuple[float, float]) -> "BoxDomain":
        """`witness` is a tighter `(low, high)`; bounds outside the current ones are ignored"""
        return BoxDomain(self.shape, max(self.low, witness[0]), min(self.high, witness[1]))

    def dim(self) -> int:
        return math.prod(self.shape)

    def embed(self, x: Float[Array, "*shape"]) -> Float[Array, "... k"]:
        return jnp.reshape(x, (*x.shape[: x.ndim - len(self.shape)], self.dim()))  # leading axes are a batch

    def unembed(self, v: Float[Array, "... k"]) -> Float[Array, "*shape"]:
        return jnp.reshape(v, (*v.shape[:-1], *self.shape))


@dataclass(frozen=True)
class ProductDomain(Domain[dict[str, Any]]):
    """Named parts, each its own domain; an element is a dict with the same keys"""

    parts: dict[str, Domain]

    def __hash__(self) -> int:
        return hash(tuple(self.parts.items()))

    def contains(self, x: dict[str, Any]) -> Bool[Array, ""]:
        if x.keys() != self.parts.keys():
            raise ValueError(f"this domain has parts {sorted(self.parts)}, got {sorted(x)}")
        ok = jnp.bool_(True)
        for name, part in self.parts.items():
            ok &= part.contains(x[name])
        return ok

    def project(self, x: dict[str, Any]) -> dict[str, Any]:
        return {name: part.project(x[name]) for name, part in self.parts.items()}

    def narrow(self, witness: Bool[Array, " n"]) -> Subset[dict[str, Any]]:
        """`witness` marks which of the enumerated elements are live now, in this domain's index order"""
        return Subset(self, witness)

    def _indexed(self) -> list[tuple[str, Any]]:
        """The parts in declaration order, each one Enumerable, or a raise naming the part that is not"""
        for name, part in self.parts.items():
            if not isinstance(part, Enumerable):
                raise TypeError(f"part {name!r} is a {type(part).__name__}, which has no index")
        return list(self.parts.items())

    def _embeddable(self) -> list[tuple[str, Any]]:
        """The parts in declaration order, each one Embeddable, or a raise naming the part that is not"""
        for name, part in self.parts.items():
            if not isinstance(part, Embeddable):
                raise TypeError(f"part {name!r} is a {type(part).__name__}, which hands out no coordinates")
        return list(self.parts.items())

    def size(self) -> int:
        return math.prod(part.size() for _, part in self._indexed())

    def index_of(self, x: dict[str, Any]) -> Int[Array, ""]:
        i = jnp.asarray(0)
        for name, part in self._indexed():  # row-major: the last part varies fastest
            i = i * part.size() + part.index_of(x[name])
        return i

    def from_index(self, i: Int[Array, ""]) -> dict[str, Any]:
        out = {}
        for name, part in reversed(self._indexed()):
            out[name] = part.from_index(i % part.size())
            i = i // part.size()
        return {name: out[name] for name in self.parts}  # back into declaration order

    @functools.cache
    def elements(self) -> dict[str, Any]:
        """Each part's own elements, gathered into the row-major order `index_of` counts in"""
        with jax.ensure_compile_time_eval():  # forces concrete values, so a traced call cannot cache tracers
            i, stride, out = jnp.arange(self.size()), 1, {}
            for name, part in reversed(self._indexed()):
                digit = (i // stride) % part.size()  # this part's mixed-radix digit at every index
                out[name] = jax.tree.map(lambda a: a[digit], part.elements())  # a nested part batches too
                stride *= part.size()
            return {name: out[name] for name in self.parts}

    def dim(self) -> int:
        return sum(part.dim() for _, part in self._embeddable())

    def embed(self, x: dict[str, Any]) -> Float[Array, "... k"]:
        """The parts' coordinates laid end to end in declaration order"""
        return jnp.concatenate([part.embed(x[name]) for name, part in self._embeddable()], axis=-1)

    def unembed(self, v: Float[Array, "... k"]) -> dict[str, Any]:
        out, at = {}, 0
        for name, part in self._embeddable():
            out[name], at = part.unembed(v[..., at : at + part.dim()]), at + part.dim()
        return out


@dataclass(frozen=True)
class FunctionDomain(Domain[Function]):
    """The maps from `source` to `target`"""

    source: Domain
    target: Domain

    def contains(self, x: Function) -> Bool[Array, ""]:
        """Always true. Membership past the `x: Function` annotation needs `source` evaluated
        everywhere (which is impossible to do anyways)"""
        return jnp.bool_(True)

    def project(self, x: Function) -> Function:
        """The identity"""
        return x
