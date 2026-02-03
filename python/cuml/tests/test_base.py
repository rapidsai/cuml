# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import inspect

import numpy as np
import numpydoc.docscrape
import pandas as pd
import pylibraft.common.handle
import pytest
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression,
)

import cuml
from cuml._thirdparty.sklearn.utils.skl_dependencies import (
    BaseEstimator as sklBaseEstimator,
)
from cuml.internals import get_handle
from cuml.testing.datasets import small_classification_dataset
from cuml.testing.utils import get_all_base_subclasses

all_base_children = get_all_base_subclasses()


@pytest.mark.parametrize("datatype", ["float32", "float64"])
@pytest.mark.parametrize("use_integer_n_features", [True, False])
def test_base_n_features_in(datatype, use_integer_n_features):
    X_train, _, _, _ = small_classification_dataset(datatype)
    integer_n_features = 8
    clf = cuml.Base()

    if use_integer_n_features:
        clf._set_n_features_in(integer_n_features)
        assert clf.n_features_in_ == integer_n_features
    else:
        clf._set_n_features_in(X_train)
        assert clf.n_features_in_ == X_train.shape[1]


@pytest.mark.parametrize(
    "child_class",
    [name for name in all_base_children.keys() if "Base" not in name],
)
def test_base_subclass_init_matches_docs(child_class: str):
    """
    This test is comparing the docstrings for arguments in __init__ for any
    class that derives from `Base`, We ensure that 1) the base arguments exist
    in the derived class, 2) The types and default values are the same and 3)
    That the docstring matches the base class

    This is to prevent multiple different docstrings for identical arguments
    throughout the documentation

    Parameters
    ----------
    child_class : str
        Classname to test in the dict all_base_children

    """
    klass = all_base_children[child_class]

    if issubclass(klass, sklBaseEstimator):
        pytest.skip(
            "Preprocessing models do not have "
            "the base arguments in constructors."
        )

    # To quickly find and replace all instances in the documentation, the below
    # regex's may be useful
    # output_type: r"^[ ]{4}output_type :.*\n(^(?![ ]{0,4}(?![ ]{4,})).*(\n))+"
    # verbose: r"^[ ]{4}verbose :.*\n(^(?![ ]{0,4}(?![ ]{4,})).*(\n))+"
    # handle: r"^[ ]{4}handle :.*\n(^(?![ ]{0,4}(?![ ]{4,})).*(\n))+"

    def get_param_doc(param_doc_obj, name: str):
        found_doc = next((x for x in param_doc_obj if x.name == name), None)

        assert found_doc is not None, "Could not find {} in docstring".format(
            name
        )

        return found_doc

    # Load the base class signature, parse the docstring and pull out params
    base_sig = inspect.signature(cuml.Base, follow_wrapped=True)
    base_doc = numpydoc.docscrape.NumpyDocString(cuml.Base.__doc__)
    base_doc_params = base_doc["Parameters"]

    # Load the current class signature, parse the docstring and pull out params
    klass_sig = inspect.signature(klass, follow_wrapped=True)
    klass_doc = numpydoc.docscrape.NumpyDocString(klass.__doc__ or "")
    klass_doc_params = klass_doc["Parameters"]

    for name, param in base_sig.parameters.items():
        # Ensure the base param exists in the derived
        assert param.name in klass_sig.parameters

        klass_param = klass_sig.parameters[param.name]

        # Ensure the default values are the same
        assert param.default == klass_param.default

        # Make sure we aren't accidentally a *args or **kwargs
        assert (
            klass_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            or klass_param.kind == inspect.Parameter.KEYWORD_ONLY
        )

        if klass.__doc__ is not None:
            found_doc = get_param_doc(klass_doc_params, name)

            base_item_doc = get_param_doc(base_doc_params, name)

            assert found_doc.type == base_item_doc.type, (
                f"Docstring mismatch for {name}"
            )

            found = " ".join(found_doc.desc)
            expected = " ".join(base_item_doc.desc)

            assert found == expected, f"Docstring mismatch for {name}"


@pytest.mark.parametrize("child_class", list(all_base_children.keys()))
# ignore ColumnTransformer init warning
@pytest.mark.filterwarnings("ignore:Transformers are required")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_base_children__get_param_names(child_class: str):
    """
    This test ensures that the arguments in `Base.__init__` are available in
    all derived classes `_get_param_names`
    """

    klass = all_base_children[child_class]

    sig = inspect.signature(klass, follow_wrapped=True)

    try:
        bound = sig.bind()
        bound.apply_defaults()
    except TypeError:
        pytest.skip(
            "{}.__init__ requires non-default arguments to create. Skipping.".format(
                klass.__name__
            )
        )
    else:
        # Create an instance
        obj = klass(*bound.args, **bound.kwargs)

        param_names = obj._get_param_names()

        # Now ensure the base parameters are included in _get_param_names
        for name, param in sig.parameters.items():
            if (
                param.kind == inspect.Parameter.VAR_KEYWORD
                or param.kind == inspect.Parameter.VAR_POSITIONAL
            ):
                continue

            assert name in param_names


FIT_LIKE_METHODS = [
    "fit",
    "fit_transform",
    "fit_predict",
    "score",
]

PREDICT_LIKE_METHODS = [
    "transform",
    "predict",
    "predict_proba",
    "predict_log_proba",
    "decision_function",
    "decision_path",
]


# Methods that don't match the sklearn convention, these are exceptions within sklearn itself
EXCEPTIONS = {
    "KernelCenterer.fit": ["self", "K", "y"],
    "KernelCenterer.transform": ["self", "K"],
    "LabelEncoder.fit": ["self", "y"],
    "LabelEncoder.fit_transform": ["self", "y"],
    "LabelEncoder.transform": ["self", "y"],
    "LabelBinarizer.fit": ["self", "y"],
    "LabelBinarizer.fit_transform": ["self", "y"],
    "LabelBinarizer.transform": ["self", "y"],
    "LedoitWolf.score": ["self", "X_test", "y"],
}


def generate_test_common_signatures_cases():
    methods = [
        *FIT_LIKE_METHODS,
        *PREDICT_LIKE_METHODS,
        "get_params",
        "set_params",
    ]
    for cls in sorted(
        get_all_base_subclasses().values(), key=lambda cls: cls.__name__
    ):
        if cls.__module__.startswith(("cuml.tsa", "cuml.solvers")):
            # These classes aren't expected to match the sklearn interface
            continue
        elif "Base" in cls.__name__:
            continue

        for method in methods:
            if hasattr(cls, method):
                yield (cls, method)


@pytest.mark.parametrize(
    "cls, method", generate_test_common_signatures_cases()
)
def test_common_signatures(cls, method):
    sig = inspect.signature(getattr(cls, method))

    if method == "get_params":
        assert list(sig.parameters) == ["self", "deep"]
        assert (
            sig.parameters["deep"].kind
            is inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        assert sig.parameters["deep"].default is True
        return

    elif method == "set_params":
        assert len(sig.parameters) == 2
        # Most methods use `**params` here, but we don't actually care what it's called
        assert (
            list(sig.parameters.values())[-1].kind
            is inspect.Parameter.VAR_KEYWORD
        )
        return

    if method in FIT_LIKE_METHODS:
        if (first := EXCEPTIONS.get(f"{cls.__name__}.{method}")) is None:
            first = ["self", "X", "y"]
        if "sample_weight" in sig.parameters:
            first.append("sample_weight")
            assert sig.parameters["sample_weight"].default is None
        if "copy" in sig.parameters:
            first.append("copy")
            assert (
                sig.parameters["copy"].default is not inspect.Parameter.empty
            )

        assert sig.parameters["y"].default in {inspect.Parameter.empty, None}

    elif method in PREDICT_LIKE_METHODS:
        if (first := EXCEPTIONS.get(f"{cls.__name__}.{method}")) is None:
            first = ["self", "X"]
        if "copy" in sig.parameters:
            first.append("copy")

    assert list(sig.parameters)[: len(first)] == first
    rest = list(sig.parameters.values())[len(first) :]

    for name in first:
        assert (
            sig.parameters[name].kind
            is inspect.Parameter.POSITIONAL_OR_KEYWORD
        )

    for param in rest:
        assert param.kind in {
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
        }
        assert param.name not in {"X", "y", "sample_weight"}


def test_get_handle():
    # Threadlocal is cached
    assert get_handle() is get_handle()

    # n_streams doesn't use the threadlocal handle
    res = get_handle(n_streams=4)
    assert res is not get_handle()
    assert isinstance(res, pylibraft.common.handle.Handle)


def test_get_handle_device_ids():
    for device_ids in ["all", [0]]:
        res = get_handle(device_ids=device_ids)
        assert isinstance(res, pylibraft.common.handle.DeviceResourcesSNMG)

    # None uses default handle
    assert get_handle(device_ids=None) is get_handle()

    with pytest.raises(ValueError, match="n_streams"):
        # Can't mix n_streams and device_ids
        get_handle(n_streams=4, device_ids="all")


@pytest.mark.parametrize(
    "cls",
    [
        cls
        for cls in all_base_children.values()
        if getattr(cls, "_estimator_type", None) == "regressor"
        and hasattr(cls, "fit")
        and hasattr(cls, "predict")
    ],
)
def test_regressor_predict_dtype(cls):
    X, y = make_regression(n_samples=200, random_state=42)
    X32 = X.astype("float32")
    y32 = y.astype("float32")

    # Regressors always return floats. We don't specify which dtype for now.
    y_pred = cls().fit(X, y).predict(X)
    assert y_pred.dtype.kind == "f"

    # Regressors work for integral targets, but still return floats
    y_pred = cls().fit(X, y.astype("int32")).predict(X)
    assert y_pred.dtype.kind == "f"

    # If all inputs to fit AND predict are float32 we return a float32.
    # This isn't necessary for the sklearn api, but is useful
    # for GPU workloads where smaller dtypes can be beneficial.
    # It also matches the proposed (but not implemented) check discussed
    # in sklearn here: https://github.com/scikit-learn/scikit-learn/issues/22682
    y_pred = cls().fit(X32, y32).predict(X32)
    assert y_pred.dtype == np.float32


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (cuml.LogisticRegression, None),
        (cuml.RandomForestClassifier, None),
        (cuml.SVC, None),
        (cuml.SVC, {"probability": True}),
        (cuml.LinearSVC, None),
        (cuml.KNeighborsClassifier, None),
        (cuml.MBSGDClassifier, None),
    ],
)
@pytest.mark.parametrize(
    "target_kind", ["binary", "multiclass", "multitarget"]
)
@pytest.mark.parametrize("dtype_kind", ["int-monotonic", "int", "string"])
def test_classifier_label_types(cls, kwargs, target_kind, dtype_kind):
    supports_multitarget = [cuml.KNeighborsClassifier]
    binary_only = [cuml.MBSGDClassifier]
    if target_kind == "multitarget" and cls not in supports_multitarget:
        pytest.skip(f"{cls.__name__} doesn't support multitarget y")
    elif target_kind == "multiclass" and cls in binary_only:
        pytest.skip(f"{cls.__name__} doesn't support multiclass y")

    labels = {
        "int-monotonic": [0, 1, 2, 3],
        "int": [5, 10, 15, 20],
        "string": ["a", "b", "c", "d"],
    }[dtype_kind]

    if target_kind == "binary":
        X, y = make_classification(n_samples=200, random_state=42, n_classes=2)
        y = np.array(labels).take(y)
    elif target_kind == "multiclass":
        X, y = make_classification(
            n_samples=200, random_state=42, n_classes=4, n_informative=4
        )
        y = np.array(labels).take(y)
    elif target_kind == "multitarget":
        X, y = make_multilabel_classification(
            n_samples=200, random_state=42, n_classes=4
        )
        y = np.array(labels).take(y)

    model = cls(**(kwargs or {})).fit(X, y)

    # Classes are of correct dtype
    if target_kind == "multitarget":
        assert all(c.dtype == y.dtype for c in model.classes_)
    else:
        assert model.classes_.dtype == y.dtype

    # Predicted labels are of correct type, dtype, and shape
    preds = model.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.dtype == y.dtype
    assert preds.shape == y.shape
    # Just a smoketest that the classifier is better than `np.zeros`
    score = (preds == y).sum() / y.size
    assert score > 0.5

    # `predict` still supports type reflection
    with cuml.using_output_type("pandas"):
        preds2 = model.predict(X)
    assert isinstance(preds2, (pd.Series, pd.DataFrame))

    # Unsupported dtype & output type pairs raise nicely
    if dtype_kind == "string" and target_kind == "binary":
        with pytest.raises(
            TypeError, match="output_type='cupy' doesn't support"
        ):
            with cuml.using_output_type("cupy"):
                preds2 = model.predict(X)
