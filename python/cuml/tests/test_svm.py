# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import cudf
import cupy as cp
import numpy as np
import pytest
import scipy.sparse as scipy_sparse
from cudf.pandas import LOADED as cudf_pandas_active
from numba import cuda
from sklearn import svm
from sklearn.datasets import (
    load_iris,
    make_blobs,
    make_classification,
    make_friedman1,
    make_gaussian_quantiles,
    make_regression,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import cuml
import cuml.svm as cu_svm
from cuml.common import input_to_cuml_array
from cuml.common.exceptions import NotFittedError
from cuml.testing.utils import (
    compare_probabilistic_svm,
    compare_svm,
    quality_param,
    stress_param,
    svm_array_equal,
    unit_param,
)

IS_ARM = platform.processor() == "aarch64"


def make_dataset(dataset, n_rows, n_cols, n_classes=2, n_informative=2):
    np.random.seed(137)
    if n_rows * 0.25 < 4000:
        # Use at least 4000 test samples
        n_test = 4000
        if n_rows > 1000:
            # To avoid a large increase in test time (which is between
            # O(n_rows^2) and O(n_rows^3)).
            n_rows = int(n_rows * 0.75)
        n_rows += n_test
    else:
        n_test = n_rows * 0.25
    if dataset == "classification1":
        X, y = make_classification(
            n_rows,
            n_cols,
            n_informative=n_informative,
            n_redundant=0,
            n_classes=n_classes,
            n_clusters_per_class=1,
        )
    elif dataset == "classification2":
        X, y = make_classification(
            n_rows,
            n_cols,
            n_informative=n_informative,
            n_redundant=0,
            n_classes=n_classes,
            n_clusters_per_class=2,
        )
    elif dataset == "gaussian":
        X, y = make_gaussian_quantiles(
            n_samples=n_rows, n_features=n_cols, n_classes=n_classes
        )
    elif dataset == "blobs":
        X, y = make_blobs(
            n_samples=n_rows, n_features=n_cols, centers=n_classes
        )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test)
    # correct case when not all classes made it into the training set
    if np.unique(y_train).size < n_classes:
        for i in range(n_classes):
            y_train[i] = i
    return X_train, X_test, y_train, y_test


def get_binary_iris_dataset():
    iris = load_iris()
    X = iris.data
    y = iris.target
    y = (y > 0).astype(X.dtype)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


@pytest.mark.parametrize(
    "params",
    [
        {"kernel": "linear", "C": 1},
        {"kernel": "linear", "C": 1, "tol": 1e-6},
        {"kernel": "linear", "C": 10},
        {"kernel": "rbf", "C": 1, "gamma": 1},
        {"kernel": "rbf", "C": 1, "gamma": "auto"},
        {"kernel": "rbf", "C": 0.1, "gamma": "auto"},
        {"kernel": "rbf", "C": 10, "gamma": "auto"},
        {"kernel": "rbf", "C": 1, "gamma": "scale"},
        {"kernel": "poly", "C": 1, "gamma": 1},
        {"kernel": "poly", "C": 1, "gamma": "auto"},
        {"kernel": "poly", "C": 1, "gamma": "scale"},
        {"kernel": "poly", "C": 1, "gamma": "auto", "degree": 2},
        {"kernel": "poly", "C": 1, "gamma": "auto", "coef0": 1.37},
        {"kernel": "sigmoid", "C": 1, "gamma": "auto"},
        {"kernel": "sigmoid", "C": 1, "gamma": "scale", "coef0": 0.42},
    ],
)
def test_svm_skl_cmp_kernels(params):
    # X_train, X_test, y_train, y_test = make_dataset('gaussian', 1000, 4)
    X_train, y_train = get_binary_iris_dataset()
    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X_train, y_train)

    sklSVC = svm.SVC(**params)
    sklSVC.fit(X_train, y_train)

    compare_svm(cuSVC, sklSVC, X_train, y_train, cmp_decision_func=True)


@pytest.mark.parametrize(
    "params",
    [
        {"kernel": "linear", "C": 1},
        {"kernel": "rbf", "C": 1, "gamma": 1},
        {"kernel": "poly", "C": 1, "gamma": 1},
    ],
)
@pytest.mark.parametrize("dataset", ["classification2", "gaussian", "blobs"])
@pytest.mark.parametrize(
    "n_rows", [3, unit_param(100), quality_param(1000), stress_param(5000)]
)
@pytest.mark.parametrize(
    "n_cols", [2, unit_param(100), quality_param(1000), stress_param(1000)]
)
def test_svm_skl_cmp_datasets(params, dataset, n_rows, n_cols):
    if (
        params["kernel"] == "linear"
        and dataset in ["gaussian", "classification2"]
        and n_rows > 1000
        and n_cols >= 1000
    ):
        # linear kernel will not fit the gaussian dataset, but takes very long
        return
    X_train, X_test, y_train, y_test = make_dataset(dataset, n_rows, n_cols)

    # Default to numpy for testing
    with cuml.using_output_type("numpy"):

        cuSVC = cu_svm.SVC(**params)
        cuSVC.fit(X_train, y_train)

        sklSVC = svm.SVC(**params)
        sklSVC.fit(X_train, y_train)

        compare_svm(
            cuSVC, sklSVC, X_test, y_test, coef_tol=1e-5, report_summary=True
        )


@pytest.mark.parametrize("params", [{"kernel": "rbf", "C": 1, "gamma": 1}])
@pytest.mark.parametrize("sparse", [True, False])
def test_svm_skl_cmp_multiclass(
    params, sparse, dataset="classification2", n_rows=100, n_cols=6
):
    X_train, X_test, y_train, y_test = make_dataset(
        dataset, n_rows, n_cols, n_classes=3, n_informative=6
    )

    if sparse:
        X_train = scipy_sparse.csr_matrix(X_train)
        X_test = scipy_sparse.csr_matrix(X_test)

    # Default to numpy for testing
    with cuml.using_output_type("numpy"):

        cuSVC = cu_svm.SVC(**params)
        cuSVC.fit(X_train, y_train)

        sklSVC = svm.SVC(**params)
        sklSVC.fit(X_train, y_train)

        compare_svm(
            cuSVC, sklSVC, X_test, y_test, coef_tol=1e-5, report_summary=True
        )


@pytest.mark.parametrize(
    "params",
    [
        {"kernel": "rbf", "C": 5, "gamma": 0.005, "probability": False},
        {"kernel": "rbf", "C": 5, "gamma": 0.005, "probability": True},
    ],
)
def test_svm_skl_cmp_decision_function(params, n_rows=4000, n_cols=20):

    X_train, X_test, y_train, y_test = make_dataset(
        "classification1", n_rows, n_cols
    )
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X_train, y_train)

    pred = cuSVC.predict(X_test)
    assert pred.dtype == y_train.dtype

    df1 = cuSVC.decision_function(X_test)
    assert df1.dtype == X_train.dtype

    sklSVC = svm.SVC(**params)
    sklSVC.fit(X_train, y_train)
    df2 = sklSVC.decision_function(X_test)

    assert mean_squared_error(df1, df2) < 1e-5


@pytest.mark.parametrize(
    "params",
    [
        {"kernel": "linear", "C": 1},
        {"kernel": "rbf", "C": 1, "gamma": 1},
        {"kernel": "poly", "C": 1, "gamma": 1},
        {"kernel": "sigmoid", "C": 1, "gamma": 1},
    ],
)
@pytest.mark.parametrize(
    "n_pred", [unit_param(5000), quality_param(100000), stress_param(1000000)]
)
def test_svm_predict(params, n_pred):
    n_rows = 500
    n_cols = 2
    X, y = make_blobs(
        n_samples=n_rows + n_pred,
        n_features=n_cols,
        centers=[[-5, -5], [5, 5]],
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=n_rows
    )
    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X_train, y_train)
    y_pred = cuSVC.predict(X_test)
    n_correct = np.sum(y_test == y_pred)
    accuracy = n_correct * 100 / n_pred
    assert accuracy > 99


def test_svc_predict_proba_not_available():
    X, y = make_classification()
    model = cuml.SVC().fit(X, y)

    with pytest.raises(NotFittedError, match="probability=True"):
        model.predict_proba(X)

    with pytest.raises(NotFittedError, match="probability=True"):
        model.predict_log_proba(X)


# Probabilisic SVM uses scikit-learn's CalibratedClassifierCV, and therefore
# the input array is converted to numpy under the hood. We explicitly test for
# all supported input types, to avoid errors like
# https://github.com/rapidsai/cuml/issues/3090
@pytest.mark.parametrize("in_type", ["numpy", "cudf", "cupy", "pandas"])
@pytest.mark.parametrize("n_classes", [2, 4])
def test_svc_predict_proba(in_type, n_classes):
    params = {
        "kernel": "rbf",
        "C": 1,
        "tol": 1e-3,
        "gamma": "scale",
        "probability": True,
    }
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=5,
        n_classes=n_classes,
        n_redundant=10,
        random_state=137,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=42
    )

    X_m = input_to_cuml_array(X_train).array
    y_m = input_to_cuml_array(y_train).array

    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X_m.to_output(in_type), y_m.to_output(in_type))
    sklSVC = svm.SVC(**params)
    sklSVC.fit(X_train, y_train)

    tol = 1e-2 if n_classes == 2 else 1e-1
    compare_probabilistic_svm(
        cuSVC, sklSVC, X_test, y_test, tol=tol, brier_tol=tol
    )


@pytest.mark.parametrize("class_weight", [None, {1: 10}, "balanced"])
@pytest.mark.parametrize("sample_weight", [None, True])
@pytest.mark.parametrize("probability", [False, True])
def test_svc_weights(class_weight, sample_weight, probability):
    # We are using the following example as a test case
    # https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html
    X, y = make_blobs(
        n_samples=[1000, 100],
        centers=[[0.0, 0.0], [2.0, 2.0]],
        cluster_std=[1.5, 0.5],
        random_state=137,
        shuffle=False,
    )
    if sample_weight:
        # Put large weight on class 1
        sample_weight = y * 9 + 1

    params = {
        "kernel": "linear",
        "C": 1,
        "gamma": "scale",
        "probability": probability,
    }
    params["class_weight"] = class_weight
    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X, y, sample_weight)

    if class_weight is not None or sample_weight is not None:
        # Standalone test: check if smaller blob is correctly classified in the
        # presence of class weights
        X_1 = X[y == 1, :]
        y_1 = np.ones(X_1.shape[0])
        cu_score = cuSVC.score(X_1, y_1)
        assert cu_score > 0.9

    sklSVC = svm.SVC(**params)
    sklSVC.fit(X, y, sample_weight)
    if not probability:
        # TODO: SVC estimators with probability=True don't expose all the fitted
        # attributes properly on the cuml side. This will be best resolved by
        # changing our internal representation rather than cludging on more
        # @property definitions. Skipping the attribute equivalence check here
        # for now.
        compare_svm(cuSVC, sklSVC, X, y, coef_tol=1e-5, report_summary=True)


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            {"kernel": "poly", "degree": 40, "C": 1, "gamma": "auto"},
            marks=pytest.mark.xfail(
                reason="fp overflow in kernel "
                "function due to non scaled input "
                "features"
            ),
        ),
        pytest.param(
            {
                "kernel": "poly",
                "degree": 40,
                "C": 1,
                "gamma": "scale",
                "x_arraytype": "numpy",
            }
        ),
        pytest.param(
            {
                "kernel": "poly",
                "degree": 40,
                "C": 1,
                "gamma": "scale",
                "x_arraytype": "dataframe",
            }
        ),
        pytest.param(
            {
                "kernel": "poly",
                "degree": 40,
                "C": 1,
                "gamma": "scale",
                "x_arraytype": "numba",
            }
        ),
    ],
)
def test_svm_gamma(params):
    # Note: we test different array types to make sure that the X.var() is
    # calculated correctly for gamma == 'scale' option.
    x_arraytype = params.pop("x_arraytype", "numpy")
    n_rows = 500
    n_cols = 380
    centers = [10 * np.ones(380), -10 * np.ones(380)]
    X, y = make_blobs(
        n_samples=n_rows, n_features=n_cols, random_state=137, centers=centers
    )
    X = X.astype(np.float32)
    if x_arraytype == "dataframe":
        y = cudf.Series(y)
    elif x_arraytype == "numba":
        X = cuda.to_device(X)
    # Using degree 40 polynomials and fp32 training would fail with
    # gamma = 1/(n_cols*X.std()), but it works with the correct implementation:
    # gamma = 1/(n_cols*X.var())
    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X, y)
    accuracy = cuSVC.score(X, y) * 100
    assert accuracy > 70


@pytest.mark.parametrize("x_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("y_dtype", [np.float32, np.float64, np.int32])
@pytest.mark.xfail(reason="SVC testing inflexibility (see issue #6575)")
def test_svm_numeric_arraytype(x_dtype, y_dtype):
    X, y = get_binary_iris_dataset()
    X = X.astype(x_dtype, order="F")
    y = y.astype(y_dtype)

    params = {"kernel": "rbf", "C": 1, "gamma": 0.25}
    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X, y)
    intercept_exp = 0.23468959692060373
    n_sv_exp = 15
    assert abs(cuSVC.intercept_ - intercept_exp) / intercept_exp < 1e-3
    assert cuSVC.n_support_ == n_sv_exp
    n_pred_wrong = np.sum(cuSVC.predict(X) - y)
    assert n_pred_wrong == 0


def get_memsize(svc):
    """Calculates the memory occupied by the parameters of an SVC object

    Parameters
    ----------
    svc : cuML SVC classifier object

    Return
    ------
    The GPU memory usage in bytes.
    """
    ms = 0
    for a in ["dual_coef_", "support_", "support_vectors_"]:
        x = getattr(svc, a)
        ms += np.prod(x[0].shape) * x[0].dtype.itemsize
    return ms


@pytest.mark.xfail(reason="Need rapidsai/rmm#415 to detect memleak robustly")
@pytest.mark.memleak
@pytest.mark.parametrize("params", [{"kernel": "rbf", "C": 1, "gamma": 1}])
@pytest.mark.parametrize(
    "n_rows", [unit_param(500), quality_param(1000), stress_param(1000)]
)
@pytest.mark.parametrize(
    "n_iter", [unit_param(10), quality_param(100), stress_param(1000)]
)
@pytest.mark.parametrize("n_cols", [1000])
def test_svm_memleak(params, n_rows, n_iter, n_cols, dataset="blobs"):
    """
    Test whether there is any memory leak.

    .. note:: small `n_rows`, and `n_cols` values will result in small model
        size, that will not be measured by get_memory_info.

    """
    X_train, X_test, y_train, y_test = make_dataset(dataset, n_rows, n_cols)
    handle = cuml.Handle()
    # Warmup. Some modules that are used in SVC allocate space on the device
    # and consume memory. Here we make sure that this allocation is done
    # before the first call to get_memory_info.
    tmp = cu_svm.SVC(handle=handle, **params)
    tmp.fit(X_train, y_train)
    ms = get_memsize(tmp)
    print(
        "Memory consumption of SVC object is {} MiB".format(
            ms / (1024 * 1024.0)
        )
    )

    free_mem = cuda.current_context().get_memory_info()[0]

    # Check first whether the get_memory_info gives us the correct memory
    # footprint
    cuSVC = cu_svm.SVC(handle=handle, **params)
    cuSVC.fit(X_train, y_train)
    delta_mem = free_mem - cuda.current_context().get_memory_info()[0]
    assert delta_mem >= ms

    # Main test loop
    b_sum = 0
    for i in range(n_iter):
        cuSVC = cu_svm.SVC(handle=handle, **params)
        cuSVC.fit(X_train, y_train)
        b_sum += cuSVC.intercept_
        cuSVC.predict(X_train)

    del cuSVC
    handle.sync()
    delta_mem = free_mem - cuda.current_context().get_memory_info()[0]
    print("Delta GPU mem: {} bytes".format(delta_mem))
    assert delta_mem == 0


@pytest.mark.xfail(reason="Need rapidsai/rmm#415 to detect memleak robustly")
@pytest.mark.memleak
@pytest.mark.parametrize(
    "params", [{"kernel": "poly", "degree": 30, "C": 1, "gamma": 1}]
)
def test_svm_memleak_on_exception(
    params, n_rows=1000, n_iter=10, n_cols=1000, dataset="blobs"
):
    """
    Test whether there is any mem leak when we exit training with an exception.
    The poly kernel with degree=30 will overflow, and triggers the
    'SMO error: NaN found...' exception.
    """
    X_train, y_train = make_blobs(
        n_samples=n_rows, n_features=n_cols, random_state=137, centers=2
    )
    X_train = X_train.astype(np.float32)
    handle = cuml.Handle()

    # Warmup. Some modules that are used in SVC allocate space on the device
    # and consume memory. Here we make sure that this allocation is done
    # before the first call to get_memory_info.
    tmp = cu_svm.SVC(handle=handle, **params)
    with pytest.raises(RuntimeError):
        tmp.fit(X_train, y_train)
        # SMO error: NaN found during fitting.

    free_mem = cuda.current_context().get_memory_info()[0]

    # Main test loop
    for i in range(n_iter):
        cuSVC = cu_svm.SVC(handle=handle, **params)
        with pytest.raises(RuntimeError):
            cuSVC.fit(X_train, y_train)
            # SMO error: NaN found during fitting.

    del cuSVC
    handle.sync()
    delta_mem = free_mem - cuda.current_context().get_memory_info()[0]
    print("Delta GPU mem: {} bytes".format(delta_mem))
    assert delta_mem == 0


def make_regression_dataset(dataset, n_rows, n_cols):
    np.random.seed(137)
    if dataset == "reg1":
        X, y = make_regression(
            n_rows, n_cols, n_informative=2, n_targets=1, random_state=137
        )
    elif dataset == "reg2":
        X, y = make_regression(
            n_rows,
            n_cols,
            n_informative=2,
            n_targets=1,
            random_state=137,
            noise=10,
        )
    elif dataset == "Friedman":
        X, y = make_friedman1(
            n_samples=n_rows, n_features=n_cols, noise=0.0, random_state=137
        )
    else:
        raise ValueError("Wrong option for dataste: ", dataset)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    dtype = np.float32
    X = X.astype(dtype)
    y = y.astype(dtype)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


def compare_svr(svr1, svr2, X_test, y_test, tol=1e-3):
    if X_test.shape[0] > 1:
        score1 = svr1.score(X_test, y_test)
        score2 = svr2.score(X_test, y_test)
        assert abs(score1 - score2) < tol
    else:
        y_pred1 = svr1.predict(X_test)
        y_pred2 = svr2.predict(X_test)
        mse1 = mean_squared_error(y_test, y_pred1)
        mse2 = mean_squared_error(y_test, y_pred2)
        assert (mse1 - mse2) / mse2 < tol


@pytest.mark.parametrize(
    "params",
    [
        {"kernel": "linear", "C": 1, "gamma": 1},
        {"kernel": "rbf", "C": 1, "gamma": 1},
        {"kernel": "poly", "C": 1, "gamma": 1},
    ],
)
@pytest.mark.parametrize("dataset", ["reg1", "reg2", "Friedman"])
@pytest.mark.parametrize(
    "n_rows",
    [unit_param(3), unit_param(100), quality_param(1000), stress_param(5000)],
)
@pytest.mark.parametrize(
    "n_cols",
    [unit_param(5), unit_param(100), quality_param(1000), stress_param(1000)],
)
def test_svr_skl_cmp(params, dataset, n_rows, n_cols):
    """Compare to Sklearn SVR"""
    if dataset == "Friedman" and n_cols < 5:
        # We need at least 5 feature columns for the Friedman dataset
        return
    X_train, X_test, y_train, y_test = make_regression_dataset(
        dataset, n_rows, n_cols
    )
    cuSVR = cu_svm.SVR(**params)
    cuSVR.fit(X_train, y_train)

    sklSVR = svm.SVR(**params)
    sklSVR.fit(X_train, y_train)

    compare_svr(cuSVR, sklSVR, X_test, y_test)


def test_svr_skl_cmp_weighted():
    """Compare to Sklearn SVR, use sample weights"""
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=2,
        n_targets=1,
        random_state=137,
        noise=10,
    )
    sample_weights = 10 * np.sin(np.linspace(0, 2 * np.pi, len(y))) + 10.1

    params = {"kernel": "linear", "C": 10, "gamma": 1}
    cuSVR = cu_svm.SVR(**params)
    cuSVR.fit(X, y, sample_weights)

    sklSVR = svm.SVR(**params)
    sklSVR.fit(X, y, sample_weights)

    compare_svr(cuSVR, sklSVR, X, y)


@pytest.mark.parametrize("classifier", [True, False])
@pytest.mark.parametrize("train_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("test_dtype", [np.float64, np.float32])
def test_svm_predict_convert_dtype(train_dtype, test_dtype, classifier):
    X, y = make_classification(n_samples=50, random_state=0)

    X = X.astype(train_dtype)
    y = y.astype(train_dtype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    if classifier:
        clf = cu_svm.SVC()
    else:
        clf = cu_svm.SVR()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))


@pytest.mark.skipif(
    IS_ARM,
    reason="Test fails unexpectedly on ARM. "
    "github.com/rapidsai/cuml/issues/5100",
)
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes small numeric issues in this test only ",
)
def test_svm_no_support_vectors():
    n_rows = 10
    n_cols = 3
    X = cp.random.uniform(size=(n_rows, n_cols), dtype=cp.float64)
    y = cp.ones((n_rows, 1))
    model = cuml.svm.SVR(kernel="linear", C=10)
    model.fit(X, y)
    pred = model.predict(X)

    assert svm_array_equal(pred, y, 0)

    assert model.n_support_ == 0
    assert abs(model.intercept_ - 1) <= 1e-6
    assert svm_array_equal(model.coef_, cp.zeros((1, n_cols)))
    assert model.dual_coef_.shape == (1, 0)
    assert model.support_.shape == (0,)
    assert model.support_vectors_.shape[0] == 0
    # Check disabled due to https://github.com/rapidsai/cuml/issues/4095
    # assert model.support_vectors_.shape[1] == n_cols
