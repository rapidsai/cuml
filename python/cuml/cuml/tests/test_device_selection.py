# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

import platform
from cuml.testing.test_preproc_utils import to_output_type
from cuml.testing.utils import array_equal

from cuml.cluster.hdbscan import HDBSCAN
from cuml.neighbors import NearestNeighbors
from cuml.metrics import trustworthiness
from cuml.metrics import adjusted_rand_score
from cuml.manifold import (
    UMAP,
    TSNE,
)
from cuml.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from cuml.internals.memory_utils import using_memory_type
from cuml.internals.mem_type import MemoryType
from cuml.decomposition import PCA, TruncatedSVD
from cuml.cluster import KMeans
from cuml.cluster import DBSCAN
from cuml.ensemble import RandomForestClassifier, RandomForestRegressor
from cuml.svm import SVC, SVR
from cuml.kernel_ridge import KernelRidge
from cuml.common.device_selection import DeviceType, using_device_type
from cuml.testing.utils import assert_dbscan_equal
from hdbscan import HDBSCAN as refHDBSCAN
from sklearn.neighbors import NearestNeighbors as skNearestNeighbors
from sklearn.linear_model import Ridge as skRidge
from sklearn.linear_model import ElasticNet as skElasticNet
from sklearn.linear_model import Lasso as skLasso
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.decomposition import PCA as skPCA
from sklearn.decomposition import TruncatedSVD as skTruncatedSVD
from sklearn.kernel_ridge import KernelRidge as skKernelRidge
from sklearn.cluster import KMeans as skKMeans
from sklearn.cluster import DBSCAN as skDBSCAN
from sklearn.datasets import make_regression, make_classification, make_blobs
from sklearn.ensemble import RandomForestClassifier as skRFC
from sklearn.ensemble import RandomForestRegressor as skRFR
from sklearn.svm import SVC as skSVC
from sklearn.svm import SVR as skSVR
from sklearn.manifold import TSNE as refTSNE
from sklearn.metrics import accuracy_score, r2_score
from pytest_cases import fixture_union, fixture
from importlib import import_module
import inspect
import pickle
from cuml.internals.safe_imports import gpu_only_import
import itertools as it
import pytest
import cuml
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")
pd = cpu_only_import("pandas")
cudf = gpu_only_import("cudf")


IS_ARM = platform.processor() == "aarch64"

if not IS_ARM:
    from umap import UMAP as refUMAP


def assert_membership_vectors(cu_vecs, sk_vecs):
    """
    Assert the membership vectors by taking the adjusted rand score
    of the argsorted membership vectors.
    """
    if sk_vecs.shape == cu_vecs.shape:
        cu_labels_sorted = np.argsort(cu_vecs)[::-1]
        sk_labels_sorted = np.argsort(sk_vecs)[::-1]

        k = min(sk_vecs.shape[1], 10)
        for i in range(k):
            assert (
                adjusted_rand_score(
                    cu_labels_sorted[:, i], sk_labels_sorted[:, i]
                )
                >= 0.85
            )


@pytest.mark.parametrize(
    "input", [("cpu", DeviceType.host), ("gpu", DeviceType.device)]
)
def test_device_type(input):
    initial_device_type = cuml.global_settings.device_type
    with using_device_type(input[0]):
        assert cuml.global_settings.device_type == input[1]
    assert cuml.global_settings.device_type == initial_device_type


def test_device_type_exception():
    with pytest.raises(ValueError):
        with using_device_type("wrong_option"):
            assert True


@pytest.mark.parametrize(
    "input",
    [
        ("device", MemoryType.device),
        ("host", MemoryType.host),
        ("managed", MemoryType.managed),
        ("mirror", MemoryType.mirror),
    ],
)
def test_memory_type(input):
    initial_memory_type = cuml.global_settings.memory_type
    with using_memory_type(input[0]):
        assert cuml.global_settings.memory_type == input[1]
    assert cuml.global_settings.memory_type == initial_memory_type


def test_memory_type_exception():
    with pytest.raises(ValueError):
        with using_memory_type("wrong_option"):
            assert True


def make_reg_dataset():
    X, y = make_regression(
        n_samples=2000, n_features=20, n_informative=18, random_state=0
    )
    X_train, X_test = X[:1800], X[1800:]
    y_train, y_test = y[:1800], y[1800:]
    return (
        X_train.astype(np.float32),
        y_train.astype(np.float32),
        X_test.astype(np.float32),
        y_test.astype(np.float32),
    )


def make_classification_dataset(n_classes=2):
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=18,
        n_classes=n_classes,
        random_state=0,
    )
    X_train, X_test = X[:1800], X[1800:]
    y_train, _ = y[:1800], y[1800:]
    return (
        X_train.astype(np.float32),
        y_train.astype(np.float32),
        X_test.astype(np.float32),
    )


def make_blob_dataset():
    X, y = make_blobs(
        n_samples=2000,
        n_features=20,
        centers=20,
        random_state=0,
        cluster_std=1.0,
    )
    X_train, X_test = X[:1800], X[1800:]
    y_train, y_test = y[:1800], y[1800:]
    return (
        X_train.astype(np.float32),
        y_train.astype(np.float32),
        X_test.astype(np.float32),
        y_test.astype(np.float32),
    )


X_train_class, y_train_class, X_test_class = make_classification_dataset(
    n_classes=2
)
(
    X_train_multiclass,
    y_train_multiclass,
    X_test_multiclass,
) = make_classification_dataset(n_classes=5)
X_train_reg, y_train_reg, X_test_reg, y_test_reg = make_reg_dataset()
X_train_blob, y_train_blob, X_test_blob, y_test_blob = make_blob_dataset()


def check_trustworthiness(cuml_embedding, test_data):
    X_test = to_output_type(test_data["X_test"], "numpy")
    cuml_embedding = to_output_type(cuml_embedding, "numpy")
    trust = trustworthiness(X_test, cuml_embedding, n_neighbors=10)
    ref_trust = test_data["ref_trust"]
    tol = 0.02
    assert trust >= ref_trust - tol


def check_allclose(cuml_output, test_data):
    ref_output = to_output_type(test_data["ref_y_test"], "numpy")
    cuml_output = to_output_type(cuml_output, "numpy")
    np.testing.assert_allclose(ref_output, cuml_output, rtol=0.15)


def check_allclose_without_sign(cuml_output, test_data):
    ref_output = to_output_type(test_data["ref_y_test"], "numpy")
    cuml_output = to_output_type(cuml_output, "numpy")
    assert ref_output.shape == cuml_output.shape
    ref_output, cuml_output = np.abs(ref_output), np.abs(cuml_output)
    np.testing.assert_allclose(ref_output, cuml_output, rtol=0.15)


def check_nn(cuml_output, test_data):
    ref_dists = to_output_type(test_data["ref_y_test"][0], "numpy")
    ref_indices = to_output_type(test_data["ref_y_test"][1], "numpy")
    cuml_dists = to_output_type(cuml_output[0], "numpy")
    cuml_indices = to_output_type(cuml_output[1], "numpy")
    np.testing.assert_allclose(ref_indices, cuml_indices)
    np.testing.assert_allclose(ref_dists, cuml_dists, rtol=0.15)


def fixture_generation_helper(params):
    param_names = sorted(params)
    param_combis = list(
        it.product(*(params[param_name] for param_name in param_names))
    )
    ids = ["-".join(map(str, param_combi)) for param_combi in param_combis]
    param_combis = [
        dict(zip(param_names, param_combi)) for param_combi in param_combis
    ]
    return {"scope": "session", "params": param_combis, "ids": ids}


@fixture(
    **fixture_generation_helper(
        {
            "input_type": ["numpy", "dataframe", "cupy", "cudf", "numba"],
            "fit_intercept": [False, True],
        }
    )
)
def linreg_test_data(request):
    kwargs = {
        "fit_intercept": request.param["fit_intercept"],
    }

    sk_model = skLinearRegression(**kwargs)
    sk_model.fit(X_train_reg, y_train_reg)

    input_type = request.param["input_type"]

    if input_type == "dataframe":
        modified_y_train = pd.Series(y_train_reg)
    elif input_type == "cudf":
        modified_y_train = cudf.Series(y_train_reg)
    else:
        modified_y_train = to_output_type(y_train_reg, input_type)

    return {
        "cuEstimator": LinearRegression,
        "kwargs": kwargs,
        "infer_func": "predict",
        "assert_func": check_allclose,
        "X_train": to_output_type(X_train_reg, input_type),
        "y_train": modified_y_train,
        "X_test": to_output_type(X_test_reg, input_type),
        "ref_y_test": sk_model.predict(X_test_reg),
    }


@fixture(
    **fixture_generation_helper(
        {
            "input_type": ["numpy", "dataframe", "cupy", "cudf", "numba"],
            "penalty": [None, "l2"],
            "fit_intercept": [False, True],
        }
    )
)
def logreg_test_data(request):
    kwargs = {
        "penalty": request.param["penalty"],
        "fit_intercept": request.param["fit_intercept"],
        "max_iter": 1000,
    }

    y_train_logreg = (y_train_reg > np.median(y_train_reg)).astype(np.int32)

    sk_model = skLogisticRegression(**kwargs)
    sk_model.fit(X_train_reg, y_train_logreg)

    input_type = request.param["input_type"]

    if input_type == "dataframe":
        y_train_logreg = pd.Series(y_train_logreg)
    elif input_type == "cudf":
        y_train_logreg = cudf.Series(y_train_logreg)
    else:
        y_train_logreg = to_output_type(y_train_logreg, input_type)

    return {
        "cuEstimator": LogisticRegression,
        "kwargs": kwargs,
        "infer_func": "predict",
        "assert_func": check_allclose,
        "X_train": to_output_type(X_train_reg, input_type),
        "y_train": y_train_logreg,
        "X_test": to_output_type(X_test_reg, input_type),
        "ref_y_test": sk_model.predict(X_test_reg),
    }


@fixture(
    **fixture_generation_helper(
        {
            "input_type": ["numpy", "dataframe", "cupy", "cudf", "numba"],
            "fit_intercept": [False, True],
            "selection": ["cyclic", "random"],
        }
    )
)
def lasso_test_data(request):
    kwargs = {
        "fit_intercept": request.param["fit_intercept"],
        "selection": request.param["selection"],
        "tol": 0.0001,
    }

    sk_model = skLasso(**kwargs)
    sk_model.fit(X_train_reg, y_train_reg)

    input_type = request.param["input_type"]

    if input_type == "dataframe":
        modified_y_train = pd.Series(y_train_reg)
    elif input_type == "cudf":
        modified_y_train = cudf.Series(y_train_reg)
    else:
        modified_y_train = to_output_type(y_train_reg, input_type)

    return {
        "cuEstimator": Lasso,
        "kwargs": kwargs,
        "infer_func": "predict",
        "assert_func": check_allclose,
        "X_train": to_output_type(X_train_reg, input_type),
        "y_train": modified_y_train,
        "X_test": to_output_type(X_test_reg, input_type),
        "ref_y_test": sk_model.predict(X_test_reg),
    }


@fixture(
    **fixture_generation_helper(
        {
            "input_type": ["numpy", "dataframe", "cupy", "cudf", "numba"],
            "fit_intercept": [False, True],
            "selection": ["cyclic", "random"],
        }
    )
)
def elasticnet_test_data(request):
    kwargs = {
        "fit_intercept": request.param["fit_intercept"],
        "selection": request.param["selection"],
        "tol": 0.0001,
    }

    sk_model = skElasticNet(**kwargs)
    sk_model.fit(X_train_reg, y_train_reg)

    input_type = request.param["input_type"]

    if input_type == "dataframe":
        modified_y_train = pd.Series(y_train_reg)
    elif input_type == "cudf":
        modified_y_train = cudf.Series(y_train_reg)
    else:
        modified_y_train = to_output_type(y_train_reg, input_type)

    return {
        "cuEstimator": ElasticNet,
        "kwargs": kwargs,
        "infer_func": "predict",
        "assert_func": check_allclose,
        "X_train": to_output_type(X_train_reg, input_type),
        "y_train": modified_y_train,
        "X_test": to_output_type(X_test_reg, input_type),
        "ref_y_test": sk_model.predict(X_test_reg),
    }


@fixture(
    **fixture_generation_helper(
        {
            "input_type": ["numpy", "dataframe", "cupy", "cudf", "numba"],
            "fit_intercept": [False, True],
        }
    )
)
def ridge_test_data(request):
    kwargs = {"fit_intercept": request.param["fit_intercept"], "solver": "svd"}

    sk_model = skRidge(**kwargs)
    sk_model.fit(X_train_reg, y_train_reg)

    input_type = request.param["input_type"]

    if input_type == "dataframe":
        modified_y_train = pd.Series(y_train_reg)
    elif input_type == "cudf":
        modified_y_train = cudf.Series(y_train_reg)
    else:
        modified_y_train = to_output_type(y_train_reg, input_type)

    return {
        "cuEstimator": Ridge,
        "kwargs": kwargs,
        "infer_func": "predict",
        "assert_func": check_allclose,
        "X_train": to_output_type(X_train_reg, input_type),
        "y_train": modified_y_train,
        "X_test": to_output_type(X_test_reg, input_type),
        "ref_y_test": sk_model.predict(X_test_reg),
    }


@fixture(
    **fixture_generation_helper(
        {
            "input_type": ["cupy"],
            "n_components": [2, 16],
            "init": ["spectral", "random"],
        }
    )
)
def umap_test_data(request):
    kwargs = {
        "n_neighbors": 12,
        "n_components": request.param["n_components"],
        "init": request.param["init"],
        "random_state": 42,
    }

    # todo: remove after https://github.com/rapidsai/cuml/issues/5441 is
    # fixed
    if not IS_ARM:
        ref_model = refUMAP(**kwargs)
        ref_model.fit(X_train_blob, y_train_blob)
        ref_embedding = ref_model.transform(X_test_blob)
        ref_trust = trustworthiness(X_test_blob, ref_embedding, n_neighbors=12)
    else:
        ref_trust = 0.0

    input_type = request.param["input_type"]

    if input_type == "dataframe":
        modified_y_train = pd.Series(y_train_blob)
    elif input_type == "cudf":
        modified_y_train = cudf.Series(y_train_blob)
    else:
        modified_y_train = to_output_type(y_train_blob, input_type)

    return {
        "cuEstimator": UMAP,
        "kwargs": kwargs,
        "infer_func": "transform",
        "assert_func": check_trustworthiness,
        "X_train": to_output_type(X_train_blob, input_type),
        "y_train": modified_y_train,
        "X_test": to_output_type(X_test_blob, input_type),
        "ref_trust": ref_trust,
    }


@fixture(
    **fixture_generation_helper(
        {
            "input_type": ["numpy", "dataframe", "cupy", "cudf", "numba"],
            "n_components": [2, 8],
        }
    )
)
def pca_test_data(request):
    kwargs = {
        "n_components": request.param["n_components"],
        "svd_solver": "full",
        "tol": 1e-07,
        "iterated_power": 15,
    }

    sk_model = skPCA(**kwargs)
    sk_model.fit(X_train_blob, y_train_blob)

    input_type = request.param["input_type"]

    if input_type == "dataframe":
        modified_y_train = pd.Series(y_train_blob)
    elif input_type == "cudf":
        modified_y_train = cudf.Series(y_train_blob)
    else:
        modified_y_train = to_output_type(y_train_blob, input_type)

    return {
        "cuEstimator": PCA,
        "kwargs": kwargs,
        "infer_func": "transform",
        "assert_func": check_allclose_without_sign,
        "X_train": to_output_type(X_train_blob, input_type),
        "y_train": modified_y_train,
        "X_test": to_output_type(X_test_blob, input_type),
        "ref_y_test": sk_model.transform(X_test_blob),
    }


@fixture(
    **fixture_generation_helper(
        {
            "input_type": ["numpy", "dataframe", "cupy", "cudf", "numba"],
            "n_components": [2, 8],
        }
    )
)
def tsvd_test_data(request):
    kwargs = {
        "n_components": request.param["n_components"],
        "n_iter": 15,
        "tol": 1e-07,
    }

    sk_model = skTruncatedSVD(**kwargs)
    sk_model.fit(X_train_blob, y_train_blob)

    input_type = request.param["input_type"]

    if input_type == "dataframe":
        modified_y_train = pd.Series(y_train_blob)
    elif input_type == "cudf":
        modified_y_train = cudf.Series(y_train_blob)
    else:
        modified_y_train = to_output_type(y_train_blob, input_type)

    return {
        "cuEstimator": TruncatedSVD,
        "kwargs": kwargs,
        "infer_func": "transform",
        "assert_func": check_allclose_without_sign,
        "X_train": to_output_type(X_train_blob, input_type),
        "y_train": modified_y_train,
        "X_test": to_output_type(X_test_blob, input_type),
        "ref_y_test": sk_model.transform(X_test_blob),
    }


@fixture(
    **fixture_generation_helper(
        {
            "input_type": ["numpy", "dataframe", "cupy", "cudf", "numba"],
            "metric": ["euclidean", "cosine"],
            "n_neighbors": [3, 8],
            "return_distance": [True],
        }
    )
)
def nn_test_data(request):
    kwargs = {
        "metric": request.param["metric"],
        "n_neighbors": request.param["n_neighbors"],
    }
    infer_func_kwargs = {"return_distance": request.param["return_distance"]}

    sk_model = skNearestNeighbors(**kwargs)
    sk_model.fit(X_train_blob)

    input_type = request.param["input_type"]

    return {
        "cuEstimator": NearestNeighbors,
        "kwargs": kwargs,
        "infer_func": "kneighbors",
        "infer_func_kwargs": infer_func_kwargs,
        "assert_func": check_nn,
        "X_train": to_output_type(X_train_blob, input_type),
        "X_test": to_output_type(X_test_blob, input_type),
        "ref_y_test": sk_model.kneighbors(X_test_blob),
    }


fixture_union(
    "test_data",
    [
        "linreg_test_data",
        "logreg_test_data",
        "lasso_test_data",
        "ridge_test_data",
        "umap_test_data",
        "pca_test_data",
        "tsvd_test_data",
        "nn_test_data",
    ],
)


def test_train_cpu_infer_cpu(test_data):
    cuEstimator = test_data["cuEstimator"]
    if cuEstimator is Lasso:
        pytest.skip("https://github.com/rapidsai/cuml/issues/5298")
    if cuEstimator is UMAP and IS_ARM:
        pytest.skip("https://github.com/rapidsai/cuml/issues/5441")
    model = cuEstimator(**test_data["kwargs"])
    with using_device_type("cpu"):
        if "y_train" in test_data:
            model.fit(test_data["X_train"], test_data["y_train"])
        else:
            model.fit(test_data["X_train"])
        infer_func = getattr(model, test_data["infer_func"])
        infer_func_kwargs = test_data.get("infer_func_kwargs", {})
        cuml_output = infer_func(test_data["X_test"], **infer_func_kwargs)

    assert_func = test_data["assert_func"]
    assert_func(cuml_output, test_data)


def test_train_gpu_infer_cpu(test_data):
    cuEstimator = test_data["cuEstimator"]

    model = cuEstimator(**test_data["kwargs"])
    with using_device_type("gpu"):
        if "y_train" in test_data:
            model.fit(test_data["X_train"], test_data["y_train"])
        else:
            model.fit(test_data["X_train"])
    with using_device_type("cpu"):
        infer_func = getattr(model, test_data["infer_func"])
        infer_func_kwargs = test_data.get("infer_func_kwargs", {})
        cuml_output = infer_func(test_data["X_test"], **infer_func_kwargs)

    assert_func = test_data["assert_func"]
    assert_func(cuml_output, test_data)


def test_train_cpu_infer_gpu(test_data):
    cuEstimator = test_data["cuEstimator"]
    if cuEstimator is UMAP and IS_ARM:
        pytest.skip("https://github.com/rapidsai/cuml/issues/5441")
    model = cuEstimator(**test_data["kwargs"])
    with using_device_type("cpu"):
        if "y_train" in test_data:
            model.fit(test_data["X_train"], test_data["y_train"])
        else:
            model.fit(test_data["X_train"])
    with using_device_type("gpu"):
        infer_func = getattr(model, test_data["infer_func"])
        infer_func_kwargs = test_data.get("infer_func_kwargs", {})
        cuml_output = infer_func(test_data["X_test"], **infer_func_kwargs)

    assert_func = test_data["assert_func"]
    assert_func(cuml_output, test_data)


def test_train_gpu_infer_gpu(test_data):
    cuEstimator = test_data["cuEstimator"]
    if cuEstimator is UMAP and IS_ARM:
        pytest.skip("https://github.com/rapidsai/cuml/issues/5441")
    model = cuEstimator(**test_data["kwargs"])
    with using_device_type("gpu"):
        if "y_train" in test_data:
            model.fit(test_data["X_train"], test_data["y_train"])
        else:
            model.fit(test_data["X_train"])
        infer_func = getattr(model, test_data["infer_func"])
        infer_func_kwargs = test_data.get("infer_func_kwargs", {})
        cuml_output = infer_func(test_data["X_test"], **infer_func_kwargs)

    assert_func = test_data["assert_func"]
    assert_func(cuml_output, test_data)


def test_pickle_interop(tmp_path, test_data):
    pickle_filepath = tmp_path / "model.pickle"

    cuEstimator = test_data["cuEstimator"]
    model = cuEstimator(**test_data["kwargs"])
    with using_device_type("gpu"):
        if "y_train" in test_data:
            model.fit(test_data["X_train"], test_data["y_train"])
        else:
            model.fit(test_data["X_train"])

    with open(pickle_filepath, "wb") as pf:
        pickle.dump(model, pf)

    del model

    with open(pickle_filepath, "rb") as pf:
        pickled_model = pickle.load(pf)

    with using_device_type("cpu"):
        infer_func = getattr(pickled_model, test_data["infer_func"])
        cuml_output = infer_func(test_data["X_test"])

    assert_func = test_data["assert_func"]
    assert_func(cuml_output, test_data)


@pytest.mark.skip("Hyperparameters defaults understandably different")
@pytest.mark.parametrize(
    "estimator",
    [
        LinearRegression,
        LogisticRegression,
        Lasso,
        ElasticNet,
        Ridge,
        UMAP,
        PCA,
        TruncatedSVD,
        NearestNeighbors,
    ],
)
def test_hyperparams_defaults(estimator):
    if estimator is UMAP and IS_ARM:
        pytest.skip("https://github.com/rapidsai/cuml/issues/5441")
    model = estimator()
    cu_signature = inspect.signature(model.__init__).parameters

    if hasattr(model, "_cpu_estimator_import_path"):
        model_path = model._cpu_estimator_import_path
    else:
        model_path = "sklearn" + model.__class__.__module__[4:]
    model_name = model.__class__.__name__
    cpu_model = getattr(import_module(model_path), model_name)
    cpu_signature = inspect.signature(cpu_model.__init__).parameters

    common_hyperparams = list(
        set(cu_signature.keys()) & set(cpu_signature.keys())
    )
    error_msg = "Different default values for hyperparameters:\n"
    similar = True
    for hyperparam in common_hyperparams:
        if (
            cu_signature[hyperparam].default
            != cpu_signature[hyperparam].default
        ):
            similar = False
            error_msg += (
                "\t{} with cuML default :"
                "'{}' and CPU default : '{}'"
                "\n".format(
                    hyperparam,
                    cu_signature[hyperparam].default,
                    cpu_signature[hyperparam].default,
                )
            )

    if not similar:
        raise ValueError(error_msg)


@pytest.mark.parametrize("train_device", ["cpu", "gpu"])
@pytest.mark.parametrize("infer_device", ["cpu", "gpu"])
def test_linreg_methods(train_device, infer_device):
    ref_model = skLinearRegression()
    ref_model.fit(X_train_reg, y_train_reg)
    ref_output = ref_model.score(X_train_reg, y_train_reg)

    model = LinearRegression()
    with using_device_type(train_device):
        model.fit(X_train_reg, y_train_reg)
    with using_device_type(infer_device):
        output = model.score(X_train_reg, y_train_reg)

    tol = 0.01
    assert ref_output - tol <= output <= ref_output + tol


@pytest.mark.parametrize("train_device", ["cpu", "gpu"])
@pytest.mark.parametrize("infer_device", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "infer_func_name",
    ["decision_function", "predict_proba", "predict_log_proba", "score"],
)
def test_logreg_methods(train_device, infer_device, infer_func_name):
    y_train_logreg = (y_train_reg > np.median(y_train_reg)).astype(np.int32)

    ref_model = skLogisticRegression()
    ref_model.fit(X_train_reg, y_train_logreg)
    infer_func = getattr(ref_model, infer_func_name)
    if infer_func_name == "score":
        ref_output = infer_func(X_train_reg, y_train_logreg)
    else:
        ref_output = infer_func(X_test_reg)

    model = LogisticRegression()
    with using_device_type(train_device):
        model.fit(X_train_reg, y_train_logreg)
    with using_device_type(infer_device):
        infer_func = getattr(model, infer_func_name)
        if infer_func_name == "score":
            output = infer_func(
                X_train_reg.astype(np.float64),
                y_train_logreg.astype(np.float64),
            )
        else:
            output = infer_func(X_test_reg.astype(np.float64))

    if infer_func_name == "score":
        tol = 0.01
        assert ref_output - tol <= output <= ref_output + tol
    else:
        output = to_output_type(output, "numpy")
        mask = np.isfinite(output)
        np.testing.assert_allclose(
            ref_output[mask], output[mask], atol=0.1, rtol=0.15
        )


@pytest.mark.parametrize("train_device", ["cpu", "gpu"])
@pytest.mark.parametrize("infer_device", ["cpu", "gpu"])
def test_lasso_methods(train_device, infer_device):
    ref_model = skLasso()
    ref_model.fit(X_train_reg, y_train_reg)
    ref_output = ref_model.score(X_train_reg, y_train_reg)

    model = Lasso()
    with using_device_type(train_device):
        model.fit(X_train_reg, y_train_reg)
    with using_device_type(infer_device):
        output = model.score(X_train_reg, y_train_reg)

    tol = 0.01
    assert ref_output - tol <= output <= ref_output + tol


@pytest.mark.parametrize("train_device", ["cpu", "gpu"])
@pytest.mark.parametrize("infer_device", ["cpu", "gpu"])
def test_elasticnet_methods(train_device, infer_device):
    ref_model = skElasticNet()
    ref_model.fit(X_train_reg, y_train_reg)
    ref_output = ref_model.score(X_train_reg, y_train_reg)

    model = ElasticNet()
    with using_device_type(train_device):
        model.fit(X_train_reg, y_train_reg)
    with using_device_type(infer_device):
        output = model.score(X_train_reg, y_train_reg)

    tol = 0.01
    assert ref_output - tol <= output <= ref_output + tol


@pytest.mark.parametrize("train_device", ["cpu", "gpu"])
@pytest.mark.parametrize("infer_device", ["cpu", "gpu"])
def test_kernelridge_methods(train_device, infer_device):
    ref_model = skKernelRidge()
    ref_model.fit(X_train_reg, y_train_reg)
    ref_output = ref_model.score(X_test_reg, y_test_reg)

    model = KernelRidge()
    with using_device_type(train_device):
        model.fit(X_train_reg, y_train_reg)
    with using_device_type(infer_device):
        output = model.score(X_test_reg, y_test_reg)

    tol = 0.01
    assert ref_output - tol <= output <= ref_output + tol


@pytest.mark.parametrize("train_device", ["cpu", "gpu"])
@pytest.mark.parametrize("infer_device", ["cpu", "gpu"])
def test_ridge_methods(train_device, infer_device):
    ref_model = skRidge()
    ref_model.fit(X_train_reg, y_train_reg)
    ref_output = ref_model.score(X_train_reg, y_train_reg)

    model = Ridge()
    with using_device_type(train_device):
        model.fit(X_train_reg, y_train_reg)
    with using_device_type(infer_device):
        output = model.score(X_train_reg, y_train_reg)

    tol = 0.01
    assert ref_output - tol <= output <= ref_output + tol


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.skipif(
    IS_ARM, reason="https://github.com/rapidsai/cuml/issues/5441"
)
def test_umap_methods(device):
    ref_model = refUMAP(n_neighbors=12)
    ref_embedding = ref_model.fit_transform(X_train_blob, y_train_blob)
    ref_trust = trustworthiness(X_train_blob, ref_embedding, n_neighbors=12)

    model = UMAP(n_neighbors=12)
    with using_device_type(device):
        embedding = model.fit_transform(X_train_blob, y_train_blob)
    trust = trustworthiness(X_train_blob, embedding, n_neighbors=12)

    tol = 0.02
    assert ref_trust - tol <= trust <= ref_trust + tol


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_tsne_methods(device):
    ref_model = refTSNE()
    ref_embedding = ref_model.fit_transform(X_train_blob)
    ref_trust = trustworthiness(X_train_blob, ref_embedding, n_neighbors=12)

    model = TSNE(n_neighbors=12)
    with using_device_type(device):
        embedding = model.fit_transform(X_train_blob)
    trust = trustworthiness(X_train_blob, embedding, n_neighbors=12)

    tol = 0.02
    assert trust >= ref_trust - tol


@pytest.mark.parametrize("train_device", ["cpu", "gpu"])
@pytest.mark.parametrize("infer_device", ["cpu", "gpu"])
def test_pca_methods(train_device, infer_device):
    n, p = 500, 5
    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * 0.1 + np.array([3, 4, 2, 3, 5])

    model = PCA(n_components=3)
    with using_device_type(train_device):
        transformation = model.fit_transform(X)
    with using_device_type(infer_device):
        output = model.inverse_transform(transformation)

    output = to_output_type(output, "numpy")
    np.testing.assert_allclose(X, output, rtol=0.15)


@pytest.mark.parametrize("train_device", ["cpu", "gpu"])
@pytest.mark.parametrize("infer_device", ["cpu", "gpu"])
def test_tsvd_methods(train_device, infer_device):
    n, p = 500, 5
    rng = np.random.RandomState(0)
    X = rng.randn(n, p) * 0.1 + np.array([3, 4, 2, 3, 5])

    model = TruncatedSVD(n_components=3)
    with using_device_type(train_device):
        transformation = model.fit_transform(X)
    with using_device_type(infer_device):
        output = model.inverse_transform(transformation)

    output = to_output_type(output, "numpy")
    np.testing.assert_allclose(X, output, rtol=0.15)


@pytest.mark.parametrize("train_device", ["cpu", "gpu"])
@pytest.mark.parametrize("infer_device", ["cpu", "gpu"])
def test_nn_methods(train_device, infer_device):
    ref_model = skNearestNeighbors()
    ref_model.fit(X_train_blob)
    ref_output = ref_model.kneighbors_graph(X_train_blob)

    model = NearestNeighbors()
    with using_device_type(train_device):
        model.fit(X_train_blob)
    with using_device_type(infer_device):
        output = model.kneighbors_graph(X_train_blob)

    ref_output = ref_output.todense()
    output = output.todense()
    np.testing.assert_allclose(ref_output, output, rtol=0.15)


@pytest.mark.parametrize("train_device", ["cpu", "gpu"])
@pytest.mark.parametrize("infer_device", ["cpu", "gpu"])
def test_hdbscan_methods(train_device, infer_device):

    ref_model = refHDBSCAN(
        prediction_data=True,
        approx_min_span_tree=False,
        max_cluster_size=0,
        min_cluster_size=30,
    )
    ref_trained_labels = ref_model.fit_predict(X_train_blob)

    from hdbscan.prediction import (
        all_points_membership_vectors as cpu_all_points_membership_vectors,
        approximate_predict as cpu_approximate_predict,
    )

    ref_membership = cpu_all_points_membership_vectors(ref_model)
    ref_labels, ref_probs = cpu_approximate_predict(ref_model, X_test_blob)

    gen_min_span_tree = train_device == "gpu" and infer_device == "cpu"
    model = HDBSCAN(
        prediction_data=True,
        approx_min_span_tree=False,
        max_cluster_size=0,
        min_cluster_size=30,
        gen_min_span_tree=gen_min_span_tree,
    )
    with using_device_type(train_device):
        trained_labels = model.fit_predict(X_train_blob)
    with using_device_type(infer_device):
        from cuml.cluster.hdbscan.prediction import (
            all_points_membership_vectors,
            approximate_predict,
        )

        membership = all_points_membership_vectors(model)
        labels, probs = approximate_predict(model, X_test_blob)

    assert adjusted_rand_score(trained_labels, ref_trained_labels) >= 0.95
    assert_membership_vectors(membership, ref_membership)
    assert adjusted_rand_score(labels, ref_labels) >= 0.98
    assert array_equal(probs, ref_probs, unit_tol=0.001, total_tol=0.006)


@pytest.mark.parametrize("train_device", ["cpu", "gpu"])
@pytest.mark.parametrize("infer_device", ["cpu", "gpu"])
def test_kmeans_methods(train_device, infer_device):
    n_clusters = 20
    ref_model = skKMeans(n_clusters=n_clusters, random_state=42)
    ref_model.fit(X_train_blob)
    ref_output = ref_model.predict(X_test_blob)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")

    with using_device_type(train_device):
        model.fit(X_train_blob)
    with using_device_type(infer_device):
        output = model.predict(X_test_blob)

    assert adjusted_rand_score(ref_output, output) >= 0.9


@pytest.mark.parametrize("train_device", ["cpu", "gpu"])
@pytest.mark.parametrize("infer_device", ["cpu", "gpu"])
def test_dbscan_methods(train_device, infer_device):
    eps = 8.0
    ref_model = skDBSCAN(eps=eps)
    ref_model.fit(X_train_blob)
    ref_output = ref_model.fit_predict(X_train_blob)

    model = DBSCAN(eps=eps)
    with using_device_type(train_device):
        model.fit(X_train_blob)
    with using_device_type(infer_device):
        output = model.fit_predict(X_train_blob)

    assert array_equal(
        ref_model.core_sample_indices_, ref_model.core_sample_indices_
    )
    assert adjusted_rand_score(ref_output, output) >= 0.95
    assert_dbscan_equal(
        ref_output, output, X_train_blob, model.core_sample_indices_, eps
    )


@pytest.mark.parametrize("train_device", ["cpu", "gpu"])
@pytest.mark.parametrize("infer_device", ["cpu", "gpu"])
def test_random_forest_regressor(train_device, infer_device):
    ref_model = skRFR(
        n_estimators=40,
        max_depth=16,
        min_samples_split=2,
        max_features=1.0,
        random_state=10,
    )
    model = RandomForestRegressor(
        max_features=1.0,
        max_depth=16,
        n_bins=64,
        n_estimators=40,
        n_streams=1,
        random_state=10,
    )
    ref_model.fit(X_train_reg, y_train_reg)
    ref_output = ref_model.predict(X_test_reg)

    with using_device_type(train_device):
        model.fit(X_train_reg, y_train_reg)
    with using_device_type(infer_device):
        output = model.predict(X_test_reg)

    cuml_acc = r2_score(y_test_reg, output)
    sk_acc = r2_score(y_test_reg, ref_output)

    assert np.abs(cuml_acc - sk_acc) <= 0.05


@pytest.mark.parametrize("train_device", ["cpu", "gpu"])
@pytest.mark.parametrize("infer_device", ["cpu", "gpu"])
def test_random_forest_classifier(train_device, infer_device):
    ref_model = skRFC(
        n_estimators=40,
        max_depth=16,
        min_samples_split=2,
        max_features=1.0,
        random_state=10,
    )
    model = RandomForestClassifier(
        max_features=1.0,
        max_depth=16,
        n_bins=64,
        n_estimators=40,
        n_streams=1,
        random_state=10,
    )
    ref_model.fit(X_train_blob, y_train_blob)
    ref_output = ref_model.predict(X_test_blob)

    with using_device_type(train_device):
        model.fit(X_train_blob, y_train_blob)
    with using_device_type(infer_device):
        output = model.predict(X_test_blob)

    cuml_acc = accuracy_score(y_test_blob, output)
    ref_acc = accuracy_score(y_test_blob, ref_output)

    assert cuml_acc == ref_acc


@pytest.mark.parametrize("train_device", ["cpu", "gpu"])
@pytest.mark.parametrize("infer_device", ["cpu", "gpu"])
@pytest.mark.parametrize("decision_function_shape", ["ovo", "ovr"])
@pytest.mark.parametrize("class_type", ["single_class", "multi_class"])
@pytest.mark.parametrize("probability", [True, False])
def test_svc_methods(
    train_device,
    infer_device,
    decision_function_shape,
    class_type,
    probability,
):
    if class_type == "single_class":
        X_train = X_train_class
        y_train = y_train_class
        X_test = X_test_class
    elif class_type == "multi_class":
        X_train = X_train_multiclass
        y_train = y_train_multiclass
        X_test = X_test_multiclass

    ref_model = skSVC(
        probability=probability,
        decision_function_shape=decision_function_shape,
    )
    ref_model.fit(X_train, y_train)
    if probability:
        ref_output = ref_model.predict_proba(X_test)
    else:
        ref_output = ref_model.predict(X_test)

    model = SVC(
        probability=probability,
        decision_function_shape=decision_function_shape,
    )
    with using_device_type(train_device):
        model.fit(X_train, y_train)
    with using_device_type(infer_device):
        if probability:
            output = model.predict_proba(X_test)
        else:
            output = model.predict(X_test)

    if probability:
        eps = 0.25
        mismatches = (
            (output <= ref_output - eps) | (output >= ref_output + eps)
        ).sum()
        outlier_percentage = mismatches / ref_output.size
        assert outlier_percentage < 0.03
    else:
        correct_percentage = (ref_output == output).sum() / ref_output.size
        assert correct_percentage > 0.9


@pytest.mark.parametrize("train_device", ["cpu", "gpu"])
@pytest.mark.parametrize("infer_device", ["cpu", "gpu"])
def test_svr_methods(train_device, infer_device):
    ref_model = skSVR()
    ref_model.fit(X_train_reg, y_train_reg)
    ref_output = ref_model.predict(X_test_reg)

    model = SVR()
    with using_device_type(train_device):
        model.fit(X_train_reg, y_train_reg)
    with using_device_type(infer_device):
        output = model.predict(X_test_reg)

    np.testing.assert_allclose(ref_output, output, rtol=0.15)
