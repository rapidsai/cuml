# Copyright (c) 2025, NVIDIA CORPORATION.
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

import importlib
import inspect
import os
import pickle
import subprocess
import sys
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import sklearn
from packaging.version import Version
from sklearn.base import check_is_fitted, is_classifier, is_regressor
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import (
    ElasticNet,
    LinearRegression,
    LogisticRegression,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from cuml.accel import is_proxy

SKLEARN_VERSION = Version(sklearn.__version__)


def test_is_proxy():
    class Foo:
        pass

    assert is_proxy(LogisticRegression)
    assert is_proxy(LogisticRegression())
    assert not is_proxy(Foo)
    assert not is_proxy(Foo())


def test_class_metadata():
    cpu_cls = LogisticRegression._cpu_class

    assert LogisticRegression.__name__ == cpu_cls.__name__
    assert LogisticRegression.__qualname__ == cpu_cls.__qualname__
    assert LogisticRegression.__doc__ == cpu_cls.__doc__

    cls = importlib.import_module(
        LogisticRegression.__module__
    ).LogisticRegression
    assert cls is LogisticRegression

    cpu_sig = inspect.signature(cpu_cls)
    assert inspect.signature(LogisticRegression) == cpu_sig


def test_method_metadata():
    cpu_cls = LogisticRegression._cpu_class

    assert LogisticRegression.fit.__name__ == cpu_cls.fit.__name__
    assert LogisticRegression.fit.__doc__ == cpu_cls.fit.__doc__

    cpu_sig = inspect.signature(cpu_cls.fit)
    assert inspect.signature(LogisticRegression.fit) == cpu_sig


def test_sklearn_introspect_estimator_type():
    assert LogisticRegression._estimator_type == "classifier"
    assert LogisticRegression()._estimator_type == "classifier"
    assert is_classifier(LogisticRegression())
    assert is_regressor(LinearRegression())


@pytest.mark.skipif(
    SKLEARN_VERSION < Version("1.6"), reason="sklearn >= 1.6 only"
)
def test_sklearn_utils_get_tags():
    """sklearn.utils.get_tags was added in sklearn 1.6"""
    from sklearn.utils import get_tags

    model = LogisticRegression()
    assert get_tags(model) == get_tags(model._cpu)


@pytest.mark.skipif(
    SKLEARN_VERSION >= Version("1.6"), reason="sklearn < 1.6 only"
)
def test_BaseEstimator__get_tags():
    model = LogisticRegression()
    assert model._get_tags() == model._cpu._get_tags()


def test_BaseEstimator__validate_params():
    model = LogisticRegression()
    model._validate_params()

    model.C = "oops"
    with pytest.raises(Exception, match="C"):
        model._validate_params()


def test_BaseEstimator__get_param_names():
    names = LogisticRegression._get_param_names()
    sol = LogisticRegression._cpu_class._get_param_names()
    assert names == sol


def test_init_positional_and_keyword():
    model = PCA()
    assert model.n_components is None

    model = PCA(n_components=10)
    assert model.n_components == 10

    model = PCA(10)
    assert model.n_components == 10

    with pytest.raises(TypeError):
        # Can't pass keyword-only parameters in as positional
        PCA(10, False)


def test_repr():
    model = LogisticRegression(C=1.5)
    assert str(model) == str(model._cpu)
    assert repr(model) == repr(model._cpu)
    # smoketest _repr_mimebundle_. It changes per-call, so can't directly compare
    assert isinstance(model._repr_mimebundle_(), dict)


def test_repr_mimebundle():
    model = LogisticRegression(C=1.5)
    unfitted_html_repr = model._repr_mimebundle_()["text/html"]

    X, y = make_classification()
    model.fit(X, y)
    fitted_html_repr = model._repr_mimebundle_()["text/html"]

    assert "<span>Not fitted</span>" in unfitted_html_repr
    assert "<span>Fitted</span>" in fitted_html_repr


def test_pipeline_repr():
    """sklearn's pretty printer requires you not override __repr__
    for pipelines to repr properly"""
    model = LogisticRegression(C=1.5)
    pipe = Pipeline([("cls", model)])
    native = Pipeline([("cls", model._cpu)])
    assert str(pipe) == str(native)
    assert repr(pipe) == repr(native)
    # smoketest _repr_mimebundle_. It changes per-call, so can't directly compare
    assert isinstance(pipe._repr_mimebundle_(), dict)


def test_dir():
    params = {"C", "fit_intercept"}
    attrs = {"n_features_in_", "coef_", "intercept_"}
    methods = {"fit", "predict"}
    private = {"_cpu", "_gpu", "_synced", "_call_method"}

    model = LogisticRegression(C=1.5)
    names = set(dir(model))
    assert names.issuperset(params | methods)
    assert names.isdisjoint(attrs | private)

    X, y = make_classification()
    model.fit(X, y)
    names = set(dir(model))
    assert names.issuperset(params | attrs | methods)
    assert names.isdisjoint(private)


def test_getattr():
    model = LogisticRegression(C=1.5, fit_intercept=False)
    assert model.C == 1.5
    assert model.fit_intercept is False

    # Never proxy through private attributes
    model._cpu._xxx_private_attr = "some_value"
    with pytest.raises(AttributeError):
        model._xxx_private_attr

    # Unfit, no fit attributes
    with pytest.raises(AttributeError):
        model.coef_

    X, y = make_classification()
    model.fit(X, y)
    # Fit attributes now available
    assert model.coef_ is model._cpu.coef_


def test_getattr_supports_select_private_attributes():
    X, y = make_blobs(random_state=42)

    # Private attributes error on unfit models
    model = NearestNeighbors()
    assert not hasattr(model, "_tree")

    # Some estimators can still expose select private attrs
    model = NearestNeighbors().fit(X)
    assert model._gpu is not None
    assert model._tree is None
    assert model._fit_method == "brute"

    # But missing attributes still error appropriately
    assert not hasattr(model, "_oops_not_a_real_attr")


def test_not_implemented_attr_error():
    X, y = make_regression()
    model = ElasticNet()

    msg = (
        "The `ElasticNet.dual_gap_` attribute is not yet "
        "implemented in `cuml.accel`"
    )

    # For unfit models the original error is raised
    with pytest.raises(AttributeError) as rec:
        model.dual_gap_
    assert msg not in str(rec.value)

    model.fit(X, y)
    # Fit models raise the nicer error message
    with pytest.raises(AttributeError, match=msg):
        model.dual_gap_

    # If trained on CPU though then there's no error
    model2 = ElasticNet(positive=True).fit(X, y)
    assert hasattr(model2, "dual_gap_")


def test_setattr():
    model = LinearRegression()

    # Hyperparameters are forwarded to CPU
    model.fit_intercept = False
    assert not model._cpu.fit_intercept
    assert model._gpu is None

    # Fit uses current hyperparameters
    X, y = make_regression(n_samples=10)
    model.fit(X, y)
    assert not model._gpu.fit_intercept

    # Changing hyperparameters forwards to both
    model.fit_intercept = True
    assert model._cpu.fit_intercept
    assert model._gpu.fit_intercept

    # But changing to an unsupported value causes fallback to CPU,
    # ensuring fit attributes are forwarded to CPU
    model.positive = True
    assert model._cpu.positive
    assert model._cpu.coef_ is not None
    assert model._gpu is None

    # Changing the value of fit attributes causes fallback to CPU.
    # This should never happen in normal workflows, but we want
    # to handle even the weird things.
    model = LinearRegression().fit(X, y)
    model.coef_ = [1, 2, 3]
    assert model._gpu is None
    assert model.coef_ == [1, 2, 3]


def test_delattr():
    X, y = make_regression(n_samples=10)
    model = LinearRegression().fit(X, y)
    del model.n_features_in_
    # Deleting an attribute causes fallback to CPU. This should never happen in
    # normal workflows, but we want to handle even the weird things.
    assert model._gpu is None
    assert model._cpu.coef_ is not None
    assert not hasattr(model._cpu, "n_features_in_")
    # Smoketest that deleting internal state works. We never do this internally.
    del model._synced
    assert not hasattr(model, "_synced")


def test_get_params():
    model = LogisticRegression(
        C=1.5,
        fit_intercept=False,
        solver="newton-cholesky",
    )
    params = model.get_params()
    assert params["C"] == 1.5
    assert not params["fit_intercept"]
    assert params["solver"] == "newton-cholesky"
    assert params == model._cpu.get_params()

    params2 = model.get_params(deep=False)
    assert params2 == model._cpu.get_params(deep=False)
    assert model.get_params.__doc__ == model._cpu.get_params.__doc__


def test_set_params():
    model = LinearRegression()

    # Hyperparameters are forwarded to CPU
    assert model.set_params(fit_intercept=False) is model
    assert not model._cpu.fit_intercept
    assert model._gpu is None

    # Fit uses current hyperparameters
    X, y = make_regression(n_samples=10)
    model.fit(X, y)
    assert not model._gpu.fit_intercept

    # Changing hyperparameters forwards to both
    model.set_params(fit_intercept=True)
    assert model._cpu.fit_intercept
    assert model._gpu.fit_intercept

    # But changing to an unsupported value causes fallback to CPU,
    # ensuring fit attributes are forwarded to CPU
    model.set_params(positive=True)
    assert model._cpu.positive
    assert model._cpu.coef_ is not None
    assert model._gpu is None


@pytest.mark.parametrize("fit", [False, True])
def test_clone(fit):
    model = LogisticRegression(
        C=1.5,
        fit_intercept=False,
        solver="newton-cholesky",
    )
    params = model.get_params()

    if fit:
        X, y = make_classification(n_samples=10)
        model.fit(X, y)

    model2 = sklearn.clone(model)

    # GPU state not initialized
    assert model2._gpu is None
    assert not model2._synced
    # Parameters copied
    assert model2.get_params() == params
    # Fit attributes not copied
    assert not hasattr(model2, "n_features_in_")


def test_pickle_proxy_estimator_class():
    """Check that the *class* can be roundtripped via pickle"""
    cls = pickle.loads(pickle.dumps(LogisticRegression))
    assert cls is LogisticRegression


def test_pickle_unfit():
    model = LogisticRegression(C=1.5, solver="newton-cholesky")

    model2 = pickle.loads(pickle.dumps(model))

    assert model2.get_params() == model.get_params()
    assert model2._gpu is None
    assert not model2._synced


def test_pickle_gpu_fit():
    X, y = make_classification(n_samples=10)
    model = LogisticRegression(C=1.5, solver="newton-cholesky")
    model.fit(X, y)
    # Ensure the test case is one that is GPU accelerated
    assert model._gpu is not None

    model2 = pickle.loads(pickle.dumps(model))

    assert model2.get_params() == model.get_params()
    # GPU model exists and is fit
    assert model2._gpu is not None
    assert model2._gpu.coef_ is not None
    # CPU model has fit attributes cleared to reduce host memory
    assert not hasattr(model2._cpu, "coef_")
    assert not model2._synced


def test_pickle_cpu_fit():
    X, y = make_regression(n_samples=10)
    model = LinearRegression(positive=True)
    model.fit(X, y)
    # Ensure the test case is one that isn't GPU accelerated
    assert model._gpu is None

    model2 = pickle.loads(pickle.dumps(model))

    assert model2.get_params() == model.get_params()
    # GPU model doesn't exist
    assert model2._gpu is None
    # CPU model has fit attributes
    assert model2._cpu.coef_ is not None
    assert not model2._synced


def test_unpickle_cuml_accel_not_active():
    """Unpickling in an process without cuml.accel enabled uses the CPU model"""
    X, y = make_classification(n_samples=10)
    model = LogisticRegression(C=1.5, solver="newton-cholesky")
    model.fit(X, y)

    buf = pickle.dumps((model, X, y))
    script = dedent(
        f"""
        import pickle

        model, X, y = pickle.loads({buf!r})

        assert model.C == 1.5
        assert model.solver == "newton-cholesky"

        from cuml.accel import enabled
        from cuml.accel.estimator_proxy import ProxyBase
        from sklearn.linear_model import LogisticRegression

        # Unpickling hasn't installed the accelerator or patched sklearn
        assert not enabled()
        assert not issubclass(LogisticRegression, ProxyBase)

        # Is a scikit-learn model of expected type
        assert type(model) is LogisticRegression
        assert not hasattr(model, "_cpu")

        # Can run inference
        model.score(X, y)
        """
    )
    env = os.environ.copy()
    env.pop("CUML_ACCEL_ENABLED", None)
    res = subprocess.run(
        [sys.executable, "-c", script],
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    # Pull out attributes before assert for nicer error reporting on failure
    returncode = res.returncode
    stdout = res.stdout
    assert returncode == 0, stdout


def test_unpickle_cuml_not_installed():
    """Unpickling in an environment without cuml installed uses the CPU model"""
    X, y = make_classification(n_samples=10)
    model = LogisticRegression(C=1.5, solver="newton-cholesky")
    model.fit(X, y)

    buf = pickle.dumps((model, X, y))
    script = dedent(
        f"""
        import pickle
        import sys

        # This prevents cuml from being imported
        sys.modules["cuml"] = None

        model, X, y = pickle.loads({buf!r})

        assert model.C == 1.5
        assert model.solver == "newton-cholesky"

        # Is a scikit-learn model of expected type
        from sklearn.linear_model import LogisticRegression
        assert type(model) is LogisticRegression
        assert not hasattr(model, "_cpu")

        # Can run inference
        model.score(X, y)
        """
    )
    env = os.environ.copy()
    env.pop("CUML_ACCEL_ENABLED", None)
    res = subprocess.run(
        [sys.executable, "-c", script],
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    # Pull out attributes before assert for nicer error reporting on failure
    returncode = res.returncode
    stdout = res.stdout
    assert returncode == 0, stdout


def test_fit_gpu():
    X, y = make_regression(n_samples=10)
    model = LinearRegression()

    # Model isn't fit
    with pytest.raises(NotFittedError):
        check_is_fitted(model)

    assert model.fit(X, y) is model
    # Fit happened on GPU, attrs kept on GPU
    check_is_fitted(model)
    assert model._gpu is not None
    assert not hasattr(model._cpu, "n_features_in_")

    # Access something that requires moving to CPU
    assert model.coef_ is not None
    assert hasattr(model._cpu, "n_features_in_")

    # Refitting resets CPU estimator
    assert model.fit(X, y) is model
    assert not hasattr(model._cpu, "n_features_in_")


def test_fit_unsupported_params():
    """Hyperparameters not supported on GPU"""
    X, y = make_regression(n_samples=10)
    model = LinearRegression(positive=True)
    assert model.fit(X, y) is model
    # Fit happened on CPU
    check_is_fitted(model)
    assert model._gpu is None
    assert hasattr(model._cpu, "n_features_in_")


def test_fit_unsupported_args():
    """Hyperparameters supported on GPU, but X/y type isn't"""
    X_dense, y = make_regression(
        n_samples=100, n_features=200, random_state=42
    )
    X_dense[X_dense < 2.5] = 0.0
    X = scipy.sparse.coo_matrix(X_dense)
    model = LinearRegression(fit_intercept=True)
    assert model.fit(X, y) is model
    # Fit happened on CPU
    check_is_fitted(model)
    assert model._gpu is None
    assert hasattr(model._cpu, "n_features_in_")


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_fit_warm_start():
    """Some sklearn estimators support a `warm_start` parameter where `fit`
    updates an existing model rather than resetting it. cuml doesn't
    support this. Here we check that fitting on CPU in this case doesn't
    clear the prior CPU model but updates it"""
    X, y = make_regression(
        n_samples=50,
        n_features=200,
        n_informative=10,
        random_state=42,
    )
    # 2 fits of 5 iters each w/ warm start should be equal to 1 fit of 10 iters
    m1 = ElasticNet(random_state=42, max_iter=5, warm_start=True)
    m1.fit(X, y).fit(X, y)
    m2 = ElasticNet(random_state=42, max_iter=10)
    m2.fit(X, y)
    np.testing.assert_allclose(m1.coef_, m2.coef_)


def test_fit_gpu_predict_gpu():
    X, y = make_regression(n_samples=10)
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    assert isinstance(y_pred, np.ndarray)
    # fit attributes not transferred to host
    assert not hasattr(model._cpu, "n_features_in_")


def test_fit_cpu_predict_cpu():
    X, y = make_regression(n_samples=10)
    model = LinearRegression(positive=True).fit(X, y)
    y_pred = model.predict(X)
    assert isinstance(y_pred, np.ndarray)
    # cpu estimator used
    assert model._gpu is None


def test_common_fit_attributes():
    X, y = make_regression(n_samples=10, n_features=5)
    model = LinearRegression()
    assert not hasattr(model, "n_features_in_")
    model.fit(X, y)
    assert model.n_features_in_ == X.shape[1]


def test_fit_validates_params():
    X, y = make_classification()
    model = LogisticRegression(C="oops")

    with pytest.raises(Exception, match="C"):
        model.fit(X, y)


def test_method_that_only_exists_on_cpu_estimator():
    """For methods that cuml doesn't implement, we fallback to CPU
    before executing."""
    X, y = make_classification(n_samples=10)
    model = LogisticRegression().fit(X, y)
    # Fit on GPU, no host transfer
    assert model._gpu is not None
    assert not hasattr(model._cpu, "n_features_in_")
    # Sanity check in case this method is implemented in cuml in the future
    assert not hasattr(model._gpu, "sparsify")
    assert model.sparsify() is model
    # Unknown method caused host transfer
    assert hasattr(model._cpu, "n_features_in_")
    # Method was run on host
    assert scipy.sparse.issparse(model.coef_)


@pytest.mark.parametrize(
    "methods",
    [
        "set_output,fit,transform",
        "fit,set_output,transform",
        "set_output,fit_transform",
    ],
)
def test_set_output(methods):
    X, _ = make_classification(n_samples=100, random_state=42)

    model = PCA(n_components=5)

    # Test calling methods in different order
    for method in methods.split(","):
        if method == "set_output":
            # set_output returns self
            assert model.set_output(transform="pandas") is model
        elif method == "fit":
            model.fit(X)
        elif method == "fit_transform":
            out = model.fit_transform(X)
        else:
            assert method == "transform"
            out = model.transform(X)

    # Transform outputs a pandas dataframe with the proper columns
    assert isinstance(out, pd.DataFrame)
    assert out.columns[0].startswith("pca")
    # No host transfer required
    assert not hasattr(model._cpu, "n_features_in_")

    # If input has an index, the output has an aligned index
    X_df = pd.DataFrame(
        X,
        columns=[f"x{i}" for i in range(X.shape[1])],
        index=[f"row{i}" for i in range(X.shape[0])],
    )
    out = model.transform(X_df)
    assert (out.index == X_df.index).all()

    # Can change back to default without requiring a host transfer either
    model.set_output(transform="default")
    out = model.transform(X)
    assert isinstance(out, np.ndarray)
    # No host transfer required
    assert not hasattr(model._cpu, "n_features_in_")


def test_get_feature_names_out():
    model = PCA(n_components=5)

    # Calling on an unfit model raises appropriately
    with pytest.raises(NotFittedError):
        model.get_feature_names_out()

    X, _ = make_classification(random_state=42)
    model.fit(X)

    # Calling on fit model returns appropriate column names
    res = model.get_feature_names_out()
    assert isinstance(res, np.ndarray)
    assert res.dtype == "object"
    assert res[0].startswith("pca")

    # No host transfer required
    assert not hasattr(model._cpu, "n_features_in_")


@pytest.fixture
def metadata_routing():
    with sklearn.config_context(enable_metadata_routing=True):
        yield


@pytest.mark.parametrize("fitted", [False, True])
def test_metadata_routing(metadata_routing, fitted):
    model = LogisticRegression()

    if fitted:
        X, y = make_classification(n_samples=10)
        model.fit(X, y)

    # Check default metadata
    routing = model.get_metadata_routing()
    assert routing.fit.requests.get("sample_weight") is None
    assert routing.score.requests.get("sample_weight") is None

    # Check can set and get metadata
    assert model.set_fit_request(sample_weight=True) is model
    assert model.set_score_request(sample_weight=False) is model

    # Check all different ways sklearn accesses metadata report the same
    def check(routing):
        assert routing.fit.requests["sample_weight"] is True
        assert routing.score.requests["sample_weight"] is False

    check(model.get_metadata_routing())
    check(model._get_metadata_request())
    check(model._metadata_request)

    # Smoketest _get_default_requests
    defaults = LogisticRegression._get_default_requests()
    assert defaults.fit.requests.get("sample_weight") is None

    # No method caused host transfer
    assert not hasattr(model._cpu, "n_features_in_")


def test_metadata_routing_consumed(metadata_routing):
    """Test that a proxy estimator is a valid consumer of metadata"""
    X, y = make_classification(random_state=42)
    weights = np.random.default_rng(42).uniform(low=0.5, size=y.shape)

    lr = LogisticRegression()
    lr.set_fit_request(sample_weight=True)
    lr.set_score_request(sample_weight=False)

    search = GridSearchCV(estimator=lr, param_grid={"C": [0.9, 1]})
    search.fit(X, y, sample_weight=weights)
