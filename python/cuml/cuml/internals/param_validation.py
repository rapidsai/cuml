#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Declarative validation of estimator hyperparameters.

Estimators declare the valid values of each hyperparameter they own in a
``_parameter_constraints`` class attribute. Each parameter maps to a list of
constraints, a value is valid if any of them accepts it::

    class Ridge(Base):
        _parameter_constraints = {
            "alpha": [Float(ge=0), ArrayLike],
            "fit_intercept": [Bool],
            "solver": [Options("auto", "eig", "svd", "lsmr")],
            "max_iter": [Int(ge=1), None],
        }

``None`` may be used directly in the lists, other classes become an
``isinstance`` check. Use ``Bool`` rather than ``bool``, in Cython files
``bool`` often refers to the C++ type. Constraints declared by parent classes
are inherited. Validation happens at the start of every fit-like method
(see :func:`cuml.internals.mlfunc`) and raises :class:`InvalidParameterError`.

Checks that relate several parameters to each other, or parameters to the
input data, are not expressed here and stay in the estimator's code.
"""

import functools
import math
import numbers

import cupy as cp
import numpy as np

__all__ = (
    "InvalidParameterError",
    "Constraint",
    "Int",
    "Float",
    "Options",
    "InstanceOf",
    "AnyOf",
    "Bool",
    "ArrayLike",
    "Callable",
    "RandomState",
    "as_constraint",
    "get_parameter_constraints",
    "validate_params",
)


class InvalidParameterError(ValueError, TypeError):
    """Raised when an estimator hyperparameter has an invalid value.

    Inherits from both ``ValueError`` and ``TypeError`` so that code catching
    either keeps working.
    """


class Constraint:
    """A set of valid values for a hyperparameter.

    Subclasses implement ``accepts``, ``__str__`` (used in error messages)
    and optionally ``invalid_examples`` (used by the test suite).
    """

    def accepts(self, value) -> bool:
        raise NotImplementedError

    def invalid_examples(self) -> list:
        """Values that this constraint rejects, for use in tests."""
        return []

    def __repr__(self):
        return f"<{type(self).__name__}: {self}>"


class _Number(Constraint):
    """A number, optionally bounded on either side."""

    _description: str

    def __init__(self, *, ge=None, gt=None, le=None, lt=None):
        if ge is not None and gt is not None:
            raise ValueError("Only one of `ge` and `gt` may be set")
        if le is not None and lt is not None:
            raise ValueError("Only one of `le` and `lt` may be set")
        self.ge = ge
        self.gt = gt
        self.le = le
        self.lt = lt

    def _is_type(self, value) -> bool:
        raise NotImplementedError

    def accepts(self, value) -> bool:
        return (
            self._is_type(value)
            and (self.ge is None or value >= self.ge)
            and (self.gt is None or value > self.gt)
            and (self.le is None or value <= self.le)
            and (self.lt is None or value < self.lt)
        )

    def __str__(self):
        bounds = [
            f"{op} {bound}"
            for op, bound in [
                (">=", self.ge),
                (">", self.gt),
                ("<=", self.le),
                ("<", self.lt),
            ]
            if bound is not None
        ]
        if not bounds:
            return self._description
        return f"{self._description} {' and '.join(bounds)}"


class Int(_Number):
    """An integer (including numpy integers, excluding bools)."""

    _description = "an int"

    def _is_type(self, value) -> bool:
        return isinstance(value, numbers.Integral) and not isinstance(
            value, bool
        )

    def invalid_examples(self) -> list:
        examples = [1.5, True]
        if self.ge is not None:
            examples.append(self.ge - 1)
        if self.gt is not None:
            examples.append(self.gt)
        if self.le is not None:
            examples.append(self.le + 1)
        if self.lt is not None:
            examples.append(self.lt)
        return examples


class Float(_Number):
    """A real number (including ints and numpy floats, excluding bools and NaN)."""

    _description = "a float"

    def _is_type(self, value) -> bool:
        return (
            isinstance(value, numbers.Real)
            and not isinstance(value, (bool, np.bool_))
            and not math.isnan(value)
        )

    def invalid_examples(self) -> list:
        examples = [math.nan, True]
        if self.ge is not None:
            examples.append(math.nextafter(self.ge, -math.inf))
        if self.gt is not None:
            examples.append(self.gt)
        if self.le is not None:
            examples.append(math.nextafter(self.le, math.inf))
        if self.lt is not None:
            examples.append(self.lt)
        return examples


class Options(Constraint):
    """One of a fixed set of values, typically strings."""

    def __init__(self, *values):
        self.values = values
        self._values = frozenset(values)

    def accepts(self, value) -> bool:
        try:
            return value in self._values
        except TypeError:
            # Unhashable values (e.g. arrays) are never an option
            return False

    def __str__(self):
        if len(self.values) == 1:
            return repr(self.values[0])
        return "one of {" + ", ".join(map(repr, self.values)) + "}"

    def invalid_examples(self) -> list:
        return [f"not {self.values[0]}"]


class InstanceOf(Constraint):
    """An instance of one of the given classes."""

    def __init__(self, *classes):
        self.classes = classes

    def accepts(self, value) -> bool:
        return isinstance(value, self.classes)

    def __str__(self):
        # numpy and cupy classes may share a name, only show it once
        names = " or ".join(
            dict.fromkeys(repr(cls.__name__) for cls in self.classes)
        )
        return f"an instance of {names}"


class _Bool(Constraint):
    def accepts(self, value) -> bool:
        return isinstance(value, (bool, np.bool_))

    def __str__(self):
        return "a bool"

    def invalid_examples(self) -> list:
        return [1]


class _None(Constraint):
    def accepts(self, value) -> bool:
        return value is None

    def __str__(self):
        return "None"


class _ArrayLike(Constraint):
    def accepts(self, value) -> bool:
        if isinstance(value, (list, tuple)):
            return True
        return not np.isscalar(value) and (
            hasattr(value, "__array__")
            or hasattr(value, "__array_interface__")
            or hasattr(value, "__cuda_array_interface__")
        )

    def __str__(self):
        return "an array-like"


class _Callable(Constraint):
    def accepts(self, value) -> bool:
        return callable(value)

    def __str__(self):
        return "a callable"


class AnyOf(Constraint):
    """Accepts a value if any of its member constraints accepts it."""

    def __init__(self, *constraints):
        members = []
        for c in map(as_constraint, constraints):
            members.extend(c.constraints if isinstance(c, AnyOf) else [c])
        self.constraints = members

    def accepts(self, value) -> bool:
        return any(c.accepts(value) for c in self.constraints)

    def __str__(self):
        descriptions = [str(c) for c in self.constraints]
        if len(descriptions) == 1:
            return descriptions[0]
        return f"{', '.join(descriptions[:-1])} or {descriptions[-1]}"

    def invalid_examples(self) -> list:
        # An example rejected by one member may be accepted by another
        return [
            v
            for c in self.constraints
            for v in c.invalid_examples()
            if not self.accepts(v)
        ]


def as_constraint(spec) -> Constraint:
    """Convert a declared constraint specification to a ``Constraint``."""
    if isinstance(spec, Constraint):
        return spec
    elif spec is None:
        return _None()
    elif spec is bool:
        return Bool
    elif spec is int:
        return Int()
    elif spec is float:
        return Float()
    elif isinstance(spec, type):
        return InstanceOf(spec)
    raise TypeError(f"Unsupported parameter constraint {spec!r}")


Bool = _Bool()
ArrayLike = _ArrayLike()
Callable = _Callable()


class _RandomState(Constraint):
    # Matches the values accepted by `cuml.internals.validation.check_random_seed`
    _valid = AnyOf(
        Int(ge=0, lt=2**32),
        InstanceOf(np.random.RandomState, cp.random.RandomState),
        None,
    )

    def accepts(self, value) -> bool:
        return self._valid.accepts(value)

    def __str__(self):
        return "an int in [0, 2**32 - 1], a RandomState instance or None"

    def invalid_examples(self) -> list:
        return [-1, 2**32]


RandomState = _RandomState()


@functools.cache
def get_parameter_constraints(cls) -> dict[str, Constraint]:
    """The constraints for ``cls``, merged with those of its parent classes."""
    merged = {}
    for klass in reversed(cls.__mro__):
        merged.update(vars(klass).get("_parameter_constraints", {}))
    out = {}
    for name, specs in merged.items():
        if not isinstance(specs, list):
            raise TypeError(
                f"The constraints for {cls.__name__}.{name} must be a list, "
                f"got {specs!r}"
            )
        out[name] = AnyOf(*specs)
    return out


def validate_params(estimator):
    """Validate the hyperparameters of ``estimator`` against its constraints.

    Parameters without a declared constraint are not checked.

    Raises
    ------
    InvalidParameterError
        If a hyperparameter value is not accepted by its constraint.
    """
    constraints = get_parameter_constraints(type(estimator))
    for name, value in estimator.get_params(deep=False).items():
        constraint = constraints.get(name)
        if constraint is not None and not constraint.accepts(value):
            raise InvalidParameterError(
                f"The {name!r} parameter of {type(estimator).__name__} must "
                f"be {constraint}. Got {value!r} instead."
            )
