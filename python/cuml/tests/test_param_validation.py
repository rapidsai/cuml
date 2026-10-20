#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import math

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cuml.internals.outputs
from cuml.internals.base import Base
from cuml.internals.outputs import (
    FIT_LIKE_METHODS,
    exit_internal_context,
    mlfunc,
)
from cuml.internals.param_validation import (
    AnyOf,
    ArrayLike,
    Callable,
    Float,
    InstanceOf,
    Int,
    InvalidParameterError,
    Options,
    RandomState,
    as_constraint,
    get_parameter_constraints,
    validate_params,
)
from cuml.testing.utils import get_all_base_subclasses


@pytest.mark.parametrize(
    "constraint, valid, invalid",
    [
        (Int(), [0, -3, np.int64(2), np.uint8(1)], [1.0, True, "1", None]),
        (Int(ge=1), [1, 10], [0, -1]),
        (Int(gt=0, le=3), [1, 3], [0, 4]),
        (Int(ge=0, lt=3), [0, 2], [-1, 3]),
        (
            Float(),
            [0.5, 1, np.float32(0.5), np.int32(1), math.inf],
            [math.nan, True, np.bool_(True), "0.5", None],
        ),
        (Float(ge=0), [0, 0.0, 2.5], [-1e-12, -1]),
        (Float(gt=0, lt=1), [0.5], [0, 1, 0.0, 1.0]),
        (
            Options("auto", "eig"),
            ["auto", "eig", np.str_("auto")],
            ["AUTO", "svd", None, np.array([1, 2])],
        ),
        (InstanceOf(dict), [{}], [[], None]),
        (as_constraint(bool), [True, False, np.bool_(True)], [0, 1, "True"]),
        (as_constraint(None), [None], [0, False, ""]),
        (
            ArrayLike,
            [
                [1, 2],
                (1, 2),
                np.zeros(3),
                cp.zeros(3),
                pd.Series([1.0]),
                np.array(1.0),
            ],
            [1.0, np.float64(1.0), "abc", None],
        ),
        (Callable, [len, lambda: 1], [1, "len"]),
        (
            RandomState,
            [
                None,
                0,
                2**32 - 1,
                np.random.RandomState(0),
                cp.random.RandomState(0),
            ],
            [-1, 2**32, 1.0, np.random.default_rng(0)],
        ),
    ],
)
def test_constraint_accepts(constraint, valid, invalid):
    for value in valid:
        assert constraint.accepts(value), value
    for value in invalid:
        assert not constraint.accepts(value), value


@pytest.mark.parametrize(
    "constraint",
    [
        Int(),
        Int(ge=1),
        Int(gt=0, lt=10),
        Int(ge=-1, le=5),
        Float(ge=0),
        Float(gt=0, le=1),
        Float(ge=-1, lt=1e20),
        Options("a", "b"),
        as_constraint(bool),
        AnyOf(Int(ge=1), None),
        AnyOf(Float(ge=0), ArrayLike),
        AnyOf(bool, Int()),
        RandomState,
    ],
)
def test_invalid_examples_are_rejected(constraint):
    for value in constraint.invalid_examples():
        assert not constraint.accepts(value), value


def test_anyof_invalid_examples_filters_overlap():
    # `True` is an invalid `Int`, but a valid bool
    constraint = AnyOf(bool, Int(ge=0))
    assert True not in constraint.invalid_examples()
    assert -1 in constraint.invalid_examples()


@pytest.mark.parametrize(
    "constraint, expected",
    [
        (Int(), "an int"),
        (Int(ge=1), "an int >= 1"),
        (Float(gt=0, le=1), "a float > 0 and <= 1"),
        (Options("auto", "eig"), "one of {'auto', 'eig'}"),
        (AnyOf(Options("auto"), Int(ge=1)), "'auto' or an int >= 1"),
        (InstanceOf(dict), "an instance of 'dict'"),
        (AnyOf(Int(ge=1), None), "an int >= 1 or None"),
        (
            AnyOf(Float(ge=0), ArrayLike, None),
            "a float >= 0, an array-like or None",
        ),
        (
            RandomState,
            "an int in [0, 2**32 - 1], a RandomState instance or None",
        ),
    ],
)
def test_constraint_str(constraint, expected):
    assert str(constraint) == expected


def test_builtin_types():
    # `int` and `float` map to `Int()` and `Float()`, not to isinstance checks
    constraint = AnyOf(int, float)
    assert constraint.accepts(np.int64(1))
    assert constraint.accepts(np.float32(1.5))
    assert not constraint.accepts(True)


@pytest.mark.parametrize("spec", ["random_state", [Int()], (Int(), None)])
def test_unsupported_spec(spec):
    with pytest.raises(TypeError, match="Unsupported parameter constraint"):
        as_constraint(spec)


@pytest.mark.parametrize("spec", [Int(), bool, (Int(), None)])
def test_constraints_must_be_a_list(spec):
    class _Estimator(Base):
        _parameter_constraints = {"x": spec}

    with pytest.raises(TypeError, match="_Estimator.x must be a list"):
        get_parameter_constraints(_Estimator)


@pytest.mark.parametrize("kwargs", [{"ge": 0, "gt": 0}, {"le": 1, "lt": 1}])
def test_number_bounds_exclusive(kwargs):
    with pytest.raises(ValueError, match="Only one of"):
        Int(**kwargs)


def test_invalid_parameter_error_is_value_and_type_error():
    assert issubclass(InvalidParameterError, ValueError)
    assert issubclass(InvalidParameterError, TypeError)


class _Parent(Base):
    _parameter_constraints = {"a": [Int(ge=0)], "b": [bool]}

    def __init__(self, *, a=1, b=True, verbose=False, output_type=None):
        super().__init__(verbose=verbose, output_type=output_type)
        self.a = a
        self.b = b

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "a", "b"]


class _Child(_Parent):
    # Overrides `a`, inherits `b`, adds `c`
    _parameter_constraints = {"a": [Int(ge=10)], "c": [Float()]}

    def __init__(self, *, c=0.5, a=10, **kwargs):
        super().__init__(a=a, **kwargs)
        self.c = c

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "c"]


def test_constraints_inherited():
    constraints = get_parameter_constraints(_Child)
    assert set(constraints) == {"verbose", "output_type", "a", "b", "c"}
    assert str(constraints["a"]) == "an int >= 10"


def test_validate_params():
    validate_params(_Child())

    with pytest.raises(
        InvalidParameterError,
        match=(
            r"^The 'a' parameter of _Child must be an int >= 10. "
            r"Got 1 instead.$"
        ),
    ):
        validate_params(_Child(a=1))

    with pytest.raises(
        InvalidParameterError,
        match="The 'b' parameter of _Child must be a bool. Got 'yes' instead.",
    ):
        validate_params(_Child(b="yes"))


@pytest.mark.parametrize(
    "params",
    [
        {"verbose": "loud"},
        {"verbose": 1.5},
        {"output_type": "invalid"},
        {"output_type": "NumPy"},
    ],
)
def test_validate_base_params(params):
    with pytest.raises(InvalidParameterError):
        validate_params(_Parent(**params))


@pytest.mark.parametrize(
    "params",
    [
        {"verbose": True},
        {"verbose": 5},
        {"output_type": None},
        {"output_type": "input"},
        {"output_type": "cupy"},
    ],
)
def test_validate_base_params_valid(params):
    validate_params(_Parent(**params))


def test_params_without_constraints_are_not_checked():
    class _NoConstraints(Base):
        def __init__(self, *, x=1, verbose=False, output_type=None):
            super().__init__(verbose=verbose, output_type=output_type)
            self.x = x

        @classmethod
        def _get_param_names(cls):
            return [*super()._get_param_names(), "x"]

    validate_params(_NoConstraints(x=object()))


class _Fittable(Base):
    _parameter_constraints = {"a": [Int(ge=0)]}

    def __init__(self, *, a=1, inner=None, verbose=False, output_type=None):
        super().__init__(verbose=verbose, output_type=output_type)
        self.a = a
        self.inner = inner

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "a", "inner"]

    @mlfunc(set_input_type=True)
    def fit(self, X, y=None):
        if self.inner is not None:
            # Like the multiclass meta-estimators, fit a user provided estimator
            with exit_internal_context():
                self.inner.fit(X)
        return self

    @mlfunc(set_input_type=True)
    def fit_transform(self, X, y=None):
        return self.fit(X)

    @mlfunc
    def fit_predict(self, X, y=None):
        return self.fit(X)


@pytest.fixture
def count_validations(monkeypatch):
    calls = []

    def counting_validate_params(estimator):
        calls.append(estimator)
        validate_params(estimator)

    monkeypatch.setattr(
        cuml.internals.outputs, "validate_params", counting_validate_params
    )
    return calls


def test_fit_validates_params():
    X = np.zeros((3, 2))
    _Fittable(a=0).fit(X)
    with pytest.raises(
        InvalidParameterError, match="'a' parameter of _Fittable"
    ):
        _Fittable(a=-1).fit(X)
    # Base parameters are validated too
    with pytest.raises(
        InvalidParameterError, match="'output_type' parameter of _Fittable"
    ):
        _Fittable(output_type="invalid").fit(X)


@pytest.mark.parametrize("method", ["fit_transform", "fit_predict"])
def test_nested_fit_validates_once(count_validations, method):
    # `fit_predict` isn't decorated with `set_input_type=True`, it is
    # validated because of its name
    est = _Fittable()
    getattr(est, method)(np.zeros((3, 2)))
    assert count_validations == [est]

    with pytest.raises(InvalidParameterError):
        getattr(_Fittable(a=-1), method)(np.zeros((3, 2)))


def test_exit_internal_context_validates_inner_estimator(count_validations):
    inner = _Fittable(a=-1)
    with pytest.raises(
        InvalidParameterError,
        match="'a' parameter of _Fittable must .* Got -1",
    ):
        _Fittable(inner=inner).fit(np.zeros((3, 2)))
    assert count_validations[-1] is inner


# Estimators whose parameters don't all have constraints yet. Remove an
# estimator from this list once it declares `_parameter_constraints` for all
# its parameters.
NOT_YET_CONVERTED = {
    "AgglomerativeClustering",
    "BernoulliNB",
    "CategoricalNB",
    "ColumnTransformer",
    "ComplementNB",
    "CountVectorizer",
    "DBSCAN",
    "ElasticNet",
    "EmpiricalCovariance",
    "GaussianNB",
    "GaussianRandomProjection",
    "HDBSCAN",
    "HashingVectorizer",
    "IncrementalPCA",
    "IsolationForest",
    "KBinsDiscretizer",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "KernelDensity",
    "KernelRidge",
    "LabelBinarizer",
    "LabelEncoder",
    "Lars",
    "Lasso",
    "LedoitWolf",
    "LinearRegression",
    "LinearSVC",
    "LinearSVR",
    "LogisticRegression",
    "MBSGDClassifier",
    "MBSGDRegressor",
    "MaxAbsScaler",
    "MinMaxScaler",
    "MissingIndicator",
    "MultinomialNB",
    "NearestNeighbors",
    "Normalizer",
    "OneHotEncoder",
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OrdinalEncoder",
    "PCA",
    "PolynomialFeatures",
    "PowerTransformer",
    "QuantileTransformer",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "RobustScaler",
    "SVC",
    "SVR",
    "SimpleImputer",
    "SparseRandomProjection",
    "SpectralClustering",
    "SpectralEmbedding",
    "StandardScaler",
    "TSNE",
    "TargetEncoder",
    "TfidfTransformer",
    "TfidfVectorizer",
    "TruncatedSVD",
    "UMAP",
}

ESTIMATORS = {
    name: cls
    for name, cls in get_all_base_subclasses().items()
    if "Base" not in name
    and cls.__module__.startswith("cuml.")
    and not cls.__module__.startswith("cuml.solvers")
}
CONVERTED = sorted(set(ESTIMATORS) - NOT_YET_CONVERTED)


def _default_instance(cls):
    try:
        return cls()
    except TypeError:
        pytest.skip(f"{cls.__name__}.__init__ requires arguments")


def test_not_yet_converted_is_up_to_date():
    assert NOT_YET_CONVERTED <= set(ESTIMATORS)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                name in NOT_YET_CONVERTED,
                reason="Not converted yet",
                strict=True,
            ),
        )
        for name in sorted(ESTIMATORS)
    ],
)
def test_all_params_have_constraints(name):
    cls = ESTIMATORS[name]
    params = set(_default_instance(cls).get_params(deep=False))
    missing = params - set(get_parameter_constraints(cls))
    assert not missing, f"{name} has no constraints for {sorted(missing)}"


@pytest.mark.parametrize("name", sorted(ESTIMATORS))
def test_no_constraints_for_unknown_params(name):
    cls = ESTIMATORS[name]
    own = set(vars(cls).get("_parameter_constraints", {}))
    params = set(_default_instance(cls).get_params(deep=False))
    assert own <= params, f"{name} has constraints for {sorted(own - params)}"


class _BadType:
    def __repr__(self):
        return "<bad type>"


@pytest.mark.parametrize("name", CONVERTED)
def test_invalid_params_raise_in_fit(name):
    cls = ESTIMATORS[name]
    X = np.arange(20, dtype="float32").reshape(10, 2)
    y = np.array([0, 1] * 5)
    methods = [m for m in FIT_LIKE_METHODS if hasattr(cls, m)]
    assert methods

    for param, constraint in get_parameter_constraints(cls).items():
        for value in [_BadType(), *constraint.invalid_examples()]:
            for method in methods:
                est = _default_instance(cls).set_params(**{param: value})
                with pytest.raises(
                    InvalidParameterError,
                    match=rf"The '{param}' parameter of {name} must be .* Got .* instead.",
                ):
                    getattr(est, method)(X, y)
