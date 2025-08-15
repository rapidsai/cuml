#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import inspect

import cupy as cp
import numpy as np
import pytest
from sklearn.datasets import make_classification

import cuml
import cuml.internals.mixins as cumix
from cuml.internals.base import Base
from cuml.testing.utils import ClassEnumerator

###############################################################################
#                        Helper functions and classes                         #
###############################################################################


def func_positional_arg(func):

    if hasattr(func, "__wrapped__"):
        return func_positional_arg(func.__wrapped__)

    elif hasattr(func, "__code__"):
        all_args = func.__code__.co_argcount
        if func.__defaults__ is not None:
            kwargs = len(func.__defaults__)
        else:
            kwargs = 0
        return all_args - kwargs
    return 2


@pytest.fixture(scope="session")
def dataset():
    X, y = make_classification(100, 5, random_state=42)
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    return X, y


models = ClassEnumerator(module=cuml).get_models()

# tag system based on experimental tag system from Scikit-learn >=0.21
# https://scikit-learn.org/stable/developers/develop.html#estimator-tags
tags = {
    # cuML specific tags
    "preferred_input_order": None,
    "X_types_gpu": list,
    # Scikit-learn API standard tags
    "allow_nan": bool,
    "binary_only": bool,
    "multilabel": bool,
    "multioutput": bool,
    "multioutput_only": bool,
    "no_validation": bool,
    "non_deterministic": bool,
    "pairwise": bool,
    "poor_score": bool,
    "preserves_dtype": list,
    "requires_fit": bool,
    "requires_y": bool,
    "requires_positive_X": bool,
    "requires_positive_y": bool,
    "stateless": bool,
    "X_types": list,
    "_skip_test": bool,
    "_xfail_checks": bool,
}

tags_mixins = {
    cumix.FMajorInputTagMixin: {"preferred_input_order": "F"},
    cumix.CMajorInputTagMixin: {"preferred_input_order": "C"},
    cumix.SparseInputTagMixin: {
        "X_types_gpu": ["2darray", "sparse"],
        "X_types": ["2darray", "sparse"],
    },
    cumix.StringInputTagMixin: {
        "X_types_gpu": ["2darray", "string"],
        "X_types": ["2darray", "string"],
    },
    cumix.AllowNaNTagMixin: {"allow_nan": True},
    cumix.StatelessTagMixin: {"stateless": True},
}


class dummy_regressor_estimator(Base, cumix.RegressorMixin):
    def __init__(self, *, handle=None, verbose=False, output_type=None):
        super().__init__(handle=handle)


class dummy_classifier_estimator(Base, cumix.ClassifierMixin):
    def __init__(self, *, handle=None, verbose=False, output_type=None):
        super().__init__(handle=handle)


class dummy_cluster_estimator(Base, cumix.ClusterMixin):
    def __init__(self, *, handle=None, verbose=False, output_type=None):
        super().__init__(handle=handle)


class dummy_class_with_tags(
    cumix.TagsMixin, cumix.FMajorInputTagMixin, cumix.CMajorInputTagMixin
):
    @staticmethod
    def _more_static_tags():
        return {"X_types": ["categorical"]}

    def _more_tags(self):
        return {"X_types": ["string"]}


###############################################################################
#                               Tags Tests                                    #
###############################################################################


@pytest.mark.parametrize("model", list(models.values()))
def test_get_tags(model):
    # This test ensures that our estimators return the tags defined by
    # Scikit-learn and our cuML specific tags

    assert hasattr(model, "_get_tags")

    model_tags = model._get_tags()

    if hasattr(model, "_more_static_tags"):
        import inspect

        assert isinstance(
            inspect.getattr_static(model, "_more_static_tags"), staticmethod
        )
    for tag, tag_type in tags.items():
        # preferred input order can be None or a string
        if tag == "preferred_input_order":
            if model_tags[tag] is not None:
                assert isinstance(model_tags[tag], str)
        else:
            assert isinstance(model_tags[tag], tag_type)


def test_dynamic_tags_and_composition():
    static_tags = dummy_class_with_tags._get_tags()
    dynamic_tags = dummy_class_with_tags()._get_tags()
    print(dummy_class_with_tags.__mro__)

    # In python, the MRO is so that the uppermost inherited class
    # being closest to the final class, so in our dummy_class_with_tags
    # the F Major input mixin should the C mixin
    assert static_tags["preferred_input_order"] == "F"
    assert dynamic_tags["preferred_input_order"] == "F"

    # Testing dynamic tags actually take precedence over static ones on the
    # instantiated object
    assert static_tags["X_types"] == ["categorical"]
    assert dynamic_tags["X_types"] == ["string"]


@pytest.mark.parametrize("mixin", tags_mixins.keys())
def test_tag_mixins(mixin):
    for tag, value in tags_mixins[mixin].items():
        assert mixin._more_static_tags()[tag] == value


@pytest.mark.parametrize(
    "model",
    [
        dummy_cluster_estimator,
        dummy_regressor_estimator,
        dummy_classifier_estimator,
    ],
)
def test_estimator_type_mixins(model):
    assert hasattr(model, "_estimator_type")
    if model._estimator_type in ["regressor", "classifier"]:
        assert model._get_tags()["requires_y"]
    else:
        assert not model._get_tags()["requires_y"]


@pytest.mark.parametrize("model", list(models.values()))
def test_mro(model):
    found_base = False
    for cl in reversed(inspect.getmro(model.__class__)):
        if cl == Base:
            if found_base:
                pytest.fail("Found Base class twice in the MRO")
            else:
                found_base = True


###############################################################################
#                            Fit Function Tests                               #
###############################################################################


@pytest.mark.parametrize("model_name", list(models.keys()))
# ignore random forest float64 warnings
@pytest.mark.filterwarnings("ignore:To use pickling or GPU-based")
def test_fit_function(dataset, model_name):
    # This test ensures that our estimators return self after a call to fit
    if model_name in [
        "TSNE",
        "TruncatedSVD",
        "AutoARIMA",
        "MultinomialNB",
        "LabelEncoder",
    ]:
        pytest.xfail("These models are not tested yet")

    n_pos_args_constr = func_positional_arg(models[model_name].__init__)

    if model_name in ["SparseRandomProjection", "GaussianRandomProjection"]:
        model = models[model_name](n_components=2)
    elif model_name in ["ARIMA", "AutoARIMA", "ExponentialSmoothing"]:
        model = models[model_name](np.random.normal(0.0, 1.0, (10,)))
    elif model_name in ["RandomForestClassifier", "RandomForestRegressor"]:
        model = models[model_name](n_bins=32)
    elif model_name == "KMeans":
        model = models[model_name](n_init="auto")
    else:
        if n_pos_args_constr == 1:
            model = models[model_name]()
        elif n_pos_args_constr == 2:
            model = models[model_name](5)
        else:
            model = models[model_name](5, 5)

    if hasattr(model, "fit"):
        # Unfortunately co_argcount doesn't work with decorated functions,
        # and the inspect module doesn't work with Cython. Therefore we need
        # to register the number of arguments manually if `fit` is decorated
        pos_args_spec = {
            "ARIMA": 1,
            "ElasticNet": 3,
            "Lasso": 3,
            "LinearRegression": 3,
            "LogisticRegression": 3,
            "NearestNeighbors": 2,
            "PCA": 2,
            "Ridge": 3,
            "UMAP": 2,
        }
        n_pos_args_fit = (
            pos_args_spec[model_name]
            if model_name in pos_args_spec
            else func_positional_arg(models[model_name].fit)
        )

        X, y = dataset

        if model_name == "RandomForestClassifier":
            y = y.astype(np.int32)
            assert model.fit(X, y) is model
        else:
            if n_pos_args_fit == 1:
                assert model.fit() is model
            elif n_pos_args_fit == 2:
                assert model.fit(X) is model
            else:
                assert model.fit(X, y) is model

        # test classifiers correctly set self.classes_ during fit
        if hasattr(model, "_estimator_type"):
            if model._estimator_type == "classifier":
                cp.testing.assert_array_almost_equal(
                    model.classes_, np.unique(y)
                )
