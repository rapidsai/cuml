# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import pytest
import numpy as np
from numba import cuda

import cuml
import cuml.svm as cu_svm
from cuml.common import input_to_cuml_array
from cuml.common.input_utils import is_array_like
from cuml.test.utils import unit_param, quality_param, stress_param

from sklearn import svm
from sklearn.datasets import load_iris, make_blobs
from sklearn.datasets import make_regression, make_friedman1
from sklearn.datasets import make_classification, make_gaussian_quantiles
from sklearn.metrics import mean_squared_error, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import cudf


def array_equal(a, b, tol=1e-6, relative_diff=True, report_summary=False):
    diff = np.abs(a-b)
    if relative_diff:
        idx = np.nonzero(abs(b) > tol)
        diff[idx] = diff[idx] / abs(b[idx])
    equal = np.all(diff <= tol)
    if not equal and report_summary:
        idx = np.argsort(diff)
        print("Largest diffs")
        a = a.ravel()
        b = b.ravel()
        diff = diff.ravel()
        for i in idx[-5:]:
            if (diff[i] > tol):
                print(diff[i], "at", i, "values", a[i], b[i])
        print('Avgdiff:', np.mean(diff), 'stddiyy:', np.std(diff), 'avgval:',
              np.mean(b))
    return equal


def compare_svm(svm1, svm2, X, y, b_tol=None, coef_tol=None,
                report_summary=False, cmp_decision_func=False):
    """ Compares two svm classifiers
    Parameters:
    -----------
    svm1 : svm classifier to be tested
    svm2 : svm classifier, the correct model
    b_tol : float
        tolerance while comparing the constant in the decision functions
    coef_tol: float
        tolerance used while comparing coef_ attribute for linear SVM

    Support vector machines have a decision function:

    F(x) = sum_{i=1}^{n_sv} d_i K(x_i, x) + b,

    where n_sv is the number of support vectors, K is the kernel function, x_i
    are the support vectors, d_i are the dual coefficients (more precisely
    d = alpha_i * y_i, where alpha_i is the dual coef), and b is the intercept.

    For linear svms K(x_i, x) = x_i * x, and we can simplify F by introducing
    w = sum_{i=1}^{n_sv} d_i x_i, the normal vector of the separating
    hyperplane:

    F(x) = w * x + b.

    Mathematically the solution of the optimization should be unique, which
    means w and b should be unique.

    There could be multiple set of vectors that lead to the same w, therefore
    comparing parameters d_k, n_sv or the support vector indices can lead to
    false positives.

    We can only evaluate w for linear models, for nonlinear models we can only
    test model accuracy and intercept.
    """

    n = X.shape[0]
    accuracy1 = svm1.score(X, y)
    accuracy2 = svm2.score(X, y)

    # We use at least 0.1% tolerance for accuracy comparison
    accuracy_tol_min = 0.001
    if accuracy2 < 1:
        # Set tolerance to include the 95% confidence interval of svm2's
        # accuracy. In practice this gives 0.9% tolerance for a 90% accurate
        # model (assuming n_test = 4000).
        accuracy_tol = 1.96 * np.sqrt(accuracy2 * (1-accuracy2) / n)
        if accuracy_tol < accuracy_tol_min:
            accuracy_tol = accuracy_tol_min
    else:
        accuracy_tol = accuracy_tol_min

    assert accuracy1 >= accuracy2 - accuracy_tol

    if b_tol is None:
        b_tol = 100*svm1.tol  # Using the deafult tol=1e-3 leads to b_tol=0.1

    if accuracy2 < 0.5:
        # Increase error margin for classifiers that are not accurate.
        # Although analytically the classifier should always be the same,
        # we fit only until we reach a certain numerical tolerance, and
        # therefore the resulting SVM's can be different. We increase the
        # tolerance in these cases.
        #
        # A good example is the gaussian dataset with linear classifier:
        # the classes are concentric blobs, and we cannot separate that with a
        # straight line. When we have a large number of data points, then
        # any separating hyperplane that goes through the center would be good.
        b_tol *= 10
        if n >= 250:
            coef_tol = 2  # allow any direction
        else:
            coef_tol *= 10

    # Compare model parameter b (intercept). In practice some models can have
    # some differences in the model parameters while still being within
    # the accuracy tolerance.
    #
    # We skip this test for multiclass (when intercept_ is an array). Apart
    # from the larger discrepancies in multiclass case, sklearn also uses a
    # different sign convention for intercept in that case.
    if (not is_array_like(svm2.intercept_)) or svm2.intercept_.shape[0] == 1:
        if abs(svm2.intercept_) > 1e-6:
            assert abs((svm1.intercept_-svm2.intercept_)/svm2.intercept_) \
                <= b_tol
        else:
            assert abs((svm1.intercept_-svm2.intercept_)) <= b_tol

    # For linear kernels we can compare the normal vector of the separating
    # hyperplane w, which is stored in the coef_ attribute.
    if svm1.kernel == 'linear':
        if coef_tol is None:
            coef_tol = 1e-5
        cs = np.dot(svm1.coef_, svm2.coef_.T) / \
            (np.linalg.norm(svm1.coef_) * np.linalg.norm(svm2.coef_))
        assert cs > 1 - coef_tol

    if cmp_decision_func:
        if accuracy2 > 0.9 and svm1.kernel != 'sigmoid':
            df1 = svm1.decision_function(X)
            df2 = svm2.decision_function(X)
            # For classification, the class is determined by
            # sign(decision function). We should not expect tight match for
            # the actual value of the function, therfore we set large tolerance
            assert(array_equal(df1, df2, tol=1e-1, relative_diff=True,
                   report_summary=True))
        else:
            print("Skipping decision function test due to low  accuracy",
                  accuracy2)


def make_dataset(dataset, n_rows, n_cols, n_classes=2, n_informative=2):
    np.random.seed(137)
    if n_rows*0.25 < 4000:
        # Use at least 4000 test samples
        n_test = 4000
        if n_rows > 1000:
            # To avoid a large increase in test time (which is between
            # O(n_rows^2) and O(n_rows^3)).
            n_rows = int(n_rows * 0.75)
        n_rows += n_test
    else:
        n_test = n_rows * 0.25
    if dataset == 'classification1':
        X, y = make_classification(
            n_rows, n_cols, n_informative=n_informative, n_redundant=0,
            n_classes=n_classes, n_clusters_per_class=1)
    elif dataset == 'classification2':
        X, y = make_classification(
            n_rows, n_cols, n_informative=n_informative, n_redundant=0,
            n_classes=n_classes, n_clusters_per_class=2)
    elif dataset == 'gaussian':
        X, y = make_gaussian_quantiles(n_samples=n_rows, n_features=n_cols,
                                       n_classes=n_classes)
    elif dataset == 'blobs':
        X, y = make_blobs(n_samples=n_rows, n_features=n_cols,
                          centers=n_classes)
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


@pytest.mark.parametrize('params', [
    {'kernel': 'linear', 'C': 1},
    {'kernel': 'linear', 'C': 1, 'tol': 1e-6},
    {'kernel': 'linear', 'C': 10},
    {'kernel': 'rbf', 'C': 1, 'gamma': 1},
    {'kernel': 'rbf', 'C': 1, 'gamma': 'auto'},
    {'kernel': 'rbf', 'C': 0.1, 'gamma': 'auto'},
    {'kernel': 'rbf', 'C': 10, 'gamma': 'auto'},
    {'kernel': 'rbf', 'C': 1, 'gamma': 'scale'},
    {'kernel': 'poly', 'C': 1, 'gamma': 1},
    {'kernel': 'poly', 'C': 1, 'gamma': 'auto'},
    {'kernel': 'poly', 'C': 1, 'gamma': 'scale'},
    {'kernel': 'poly', 'C': 1, 'gamma': 'auto', 'degree': 2},
    {'kernel': 'poly', 'C': 1, 'gamma': 'auto', 'coef0': 1.37},
    {'kernel': 'sigmoid', 'C': 1, 'gamma': 'auto'},
    {'kernel': 'sigmoid', 'C': 1, 'gamma': 'scale', 'coef0': 0.42}
])
def test_svm_skl_cmp_kernels(params):
    # X_train, X_test, y_train, y_test = make_dataset('gaussian', 1000, 4)
    X_train, y_train = get_binary_iris_dataset()
    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X_train, y_train)

    sklSVC = svm.SVC(**params)
    sklSVC.fit(X_train, y_train)

    compare_svm(cuSVC, sklSVC, X_train, y_train, cmp_decision_func=True)


@pytest.mark.parametrize('params', [
    {'kernel': 'linear', 'C': 1},
    {'kernel': 'rbf', 'C': 1, 'gamma': 1},
    {'kernel': 'poly', 'C': 1, 'gamma': 1},
])
@pytest.mark.parametrize('dataset', ['classification2', 'gaussian', 'blobs'])
@pytest.mark.parametrize('n_rows', [3, unit_param(100), quality_param(1000),
                                    stress_param(5000)])
@pytest.mark.parametrize('n_cols', [2, unit_param(100), quality_param(1000),
                         stress_param(1000)])
def test_svm_skl_cmp_datasets(params, dataset, n_rows, n_cols):
    if (params['kernel'] == 'linear' and
            dataset in ['gaussian', 'classification2'] and
            n_rows > 1000 and n_cols >= 1000):
        # linear kernel will not fit the gaussian dataset, but takes very long
        return
    X_train, X_test, y_train, y_test = make_dataset(dataset, n_rows, n_cols)

    # Default to numpy for testing
    with cuml.using_output_type("numpy"):

        cuSVC = cu_svm.SVC(**params)
        cuSVC.fit(X_train, y_train)

        sklSVC = svm.SVC(**params)
        sklSVC.fit(X_train, y_train)

        compare_svm(cuSVC, sklSVC, X_test, y_test, coef_tol=1e-5,
                    report_summary=True)


@pytest.mark.parametrize('params', [{'kernel': 'rbf', 'C': 1, 'gamma': 1}])
def test_svm_skl_cmp_multiclass(params, dataset='classification2', n_rows=100,
                                n_cols=6):
    X_train, X_test, y_train, y_test = make_dataset(dataset, n_rows, n_cols,
                                                    n_classes=3,
                                                    n_informative=6)

    # Default to numpy for testing
    with cuml.using_output_type("numpy"):

        cuSVC = cu_svm.SVC(**params)
        cuSVC.fit(X_train, y_train)

        sklSVC = svm.SVC(**params)
        sklSVC.fit(X_train, y_train)

        compare_svm(cuSVC, sklSVC, X_test, y_test, coef_tol=1e-5,
                    report_summary=True)


@pytest.mark.parametrize('params', [
    {'kernel': 'rbf', 'C': 5, 'gamma': 0.005, "probability": False},
    {'kernel': 'rbf', 'C': 5, 'gamma': 0.005, "probability": True}])
def test_svm_skl_cmp_decision_function(params, n_rows=4000, n_cols=20):

    X_train, X_test, y_train, y_test = make_dataset('classification1', n_rows,
                                                    n_cols)
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

    if params["probability"]:
        tol = 2e-2  # See comments in SVC decision_function method
    else:
        tol = 1e-5
    assert mean_squared_error(df1, df2) < tol


@pytest.mark.parametrize('params', [
    {'kernel': 'linear', 'C': 1},
    {'kernel': 'rbf', 'C': 1, 'gamma': 1},
    {'kernel': 'poly', 'C': 1, 'gamma': 1},
    {'kernel': 'sigmoid', 'C': 1, 'gamma': 1}
])
@pytest.mark.parametrize('n_pred', [unit_param(5000), quality_param(100000),
                                    stress_param(1000000)])
def test_svm_predict(params, n_pred):
    n_rows = 500
    n_cols = 2
    X, y = make_blobs(n_samples=n_rows + n_pred, n_features=n_cols,
                      centers=[[-5, -5], [5, 5]])
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=n_rows)
    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X_train, y_train)
    y_pred = cuSVC.predict(X_test)
    n_correct = np.sum(y_test == y_pred)
    accuracy = n_correct * 100 / n_pred
    assert accuracy > 99


def compare_probabilistic_svm(svc1, svc2, X_test, y_test, tol=1e-3,
                              brier_tol=1e-3):
    """ Compare the probability output from two support vector classifiers.
    """

    prob1 = svc1.predict_proba(X_test)
    prob2 = svc2.predict_proba(X_test)
    assert mean_squared_error(prob1, prob2) <= tol

    if (svc1.n_classes_ == 2):
        brier1 = brier_score_loss(y_test, prob1[:, 1])
        brier2 = brier_score_loss(y_test, prob2[:, 1])
        # Brier score - smaller is better
        assert brier1 - brier2 <= brier_tol


# Probabilisic SVM uses scikit-learn's CalibratedClassifierCV, and therefore
# the input array is converted to numpy under the hood. We explicitly test for
# all supported input types, to avoid errors like
# https://github.com/rapidsai/cuml/issues/3090
@pytest.mark.parametrize('in_type', ['numpy', 'numba', 'cudf', 'cupy',
                                     'pandas', 'cuml'])
def test_svm_skl_cmp_predict_proba(in_type, n_rows=10000, n_cols=20):
    params = {'kernel': 'rbf', 'C': 1, 'tol': 1e-3, 'gamma': 'scale',
              'probability': True}
    X, y = make_classification(n_samples=n_rows, n_features=n_cols,
                               n_informative=2, n_redundant=10,
                               random_state=137)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8,
                                                        random_state=42)

    X_m = input_to_cuml_array(X_train).array
    y_m = input_to_cuml_array(y_train).array

    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X_m.to_output(in_type), y_m.to_output(in_type))
    sklSVC = svm.SVC(**params)
    sklSVC.fit(X_train, y_train)
    compare_probabilistic_svm(cuSVC, sklSVC, X_test, y_test, 1e-3, 1e-2)


@pytest.mark.parametrize('class_weight', [None, {1: 10}, 'balanced'])
@pytest.mark.parametrize('sample_weight', [None, True])
def test_svc_weights(class_weight, sample_weight):
    # We are using the following example as a test case
    # https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane_unbalanced.html
    X, y = make_blobs(n_samples=[1000, 100],
                      centers=[[0.0, 0.0], [2.0, 2.0]],
                      cluster_std=[1.5, 0.5],
                      random_state=137, shuffle=False)
    if sample_weight:
        # Put large weight on class 1
        sample_weight = y * 9 + 1

    params = {'kernel': 'linear', 'C': 1, 'gamma': 'scale'}
    params['class_weight'] = class_weight
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
    compare_svm(cuSVC, sklSVC, X, y, coef_tol=1e-5, report_summary=True)


@pytest.mark.parametrize('params', [
    pytest.param({'kernel': 'poly', 'degree': 40, 'C': 1, 'gamma': 'auto'},
                 marks=pytest.mark.xfail(reason="fp overflow in kernel "
                                         "function due to non scaled input "
                                         "features")),
    pytest.param({'kernel': 'poly', 'degree': 40, 'C': 1, 'gamma': 'scale',
                  'x_arraytype': 'numpy'}),
    pytest.param({'kernel': 'poly', 'degree': 40, 'C': 1, 'gamma': 'scale',
                  'x_arraytype': 'dataframe'}),
    pytest.param({'kernel': 'poly', 'degree': 40, 'C': 1, 'gamma': 'scale',
                  'x_arraytype': 'numba'}),
])
def test_svm_gamma(params):
    # Note: we test different array types to make sure that the X.var() is
    # calculated correctly for gamma == 'scale' option.
    x_arraytype = params.pop('x_arraytype', 'numpy')
    n_rows = 500
    n_cols = 380
    centers = [10*np.ones(380), -10*np.ones(380)]
    X, y = make_blobs(n_samples=n_rows, n_features=n_cols, random_state=137,
                      centers=centers)
    X = X.astype(np.float32)
    if x_arraytype == 'dataframe':
        y = cudf.Series(y)
    elif x_arraytype == 'numba':
        X = cuda.to_device(X)
    # Using degree 40 polynomials and fp32 training would fail with
    # gamma = 1/(n_cols*X.std()), but it works with the correct implementation:
    # gamma = 1/(n_cols*X.var())
    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X, y)
    accuracy = cuSVC.score(X, y) * 100
    assert accuracy > 70


@pytest.mark.parametrize('x_dtype', [np.float32, np.float64])
@pytest.mark.parametrize('y_dtype', [np.float32, np.float64, np.int32])
def test_svm_numeric_arraytype(x_dtype, y_dtype):
    X, y = get_binary_iris_dataset()
    X = X.astype(x_dtype, order="F")
    y = y.astype(y_dtype)

    params = {'kernel': 'rbf', 'C': 1, 'gamma': 0.25}
    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X, y)
    intercept_exp = 0.23468959692060373
    n_sv_exp = 15
    assert abs(cuSVC.intercept_ - intercept_exp) / intercept_exp < 1e-3
    assert cuSVC.n_support_ == n_sv_exp
    n_pred_wrong = np.sum(cuSVC.predict(X)-y)
    assert n_pred_wrong == 0


def get_memsize(svc):
    """ Calculates the memory occupied by the parameters of an SVC object

    Parameters
    ----------
    svc : cuML SVC classifier object

    Return
    ------
    The GPU memory usage in bytes.
    """
    ms = 0
    for a in ['dual_coef_', 'support_', 'support_vectors_']:
        x = getattr(svc, a)
        ms += np.prod(x[0].shape)*x[0].dtype.itemsize
    return ms


@pytest.mark.xfail(reason='Need rapidsai/rmm#415 to detect memleak robustly')
@pytest.mark.memleak
@pytest.mark.parametrize('params', [
    {'kernel': 'rbf', 'C': 1, 'gamma': 1}
])
@pytest.mark.parametrize('n_rows', [unit_param(500), quality_param(1000),
                                    stress_param(1000)])
@pytest.mark.parametrize('n_iter', [unit_param(10), quality_param(100),
                                    stress_param(1000)])
@pytest.mark.parametrize('n_cols', [1000])
@pytest.mark.parametrize('use_handle', [True, False])
def test_svm_memleak(params, n_rows, n_iter, n_cols,
                     use_handle, dataset='blobs'):
    """
    Test whether there is any memory leak.

    .. note:: small `n_rows`, and `n_cols` values will result in small model
        size, that will not be measured by get_memory_info.

    """
    X_train, X_test, y_train, y_test = make_dataset(dataset, n_rows, n_cols)
    stream = cuml.cuda.Stream()
    handle = cuml.Handle()
    handle.setStream(stream)
    # Warmup. Some modules that are used in SVC allocate space on the device
    # and consume memory. Here we make sure that this allocation is done
    # before the first call to get_memory_info.
    tmp = cu_svm.SVC(handle=handle, **params)
    tmp.fit(X_train, y_train)
    ms = get_memsize(tmp)
    print("Memory consumtion of SVC object is {} MiB".format(ms/(1024*1024.0)))

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

    del(cuSVC)
    handle.sync()
    delta_mem = free_mem - cuda.current_context().get_memory_info()[0]
    print("Delta GPU mem: {} bytes".format(delta_mem))
    assert delta_mem == 0


@pytest.mark.xfail(reason='Need rapidsai/rmm#415 to detect memleak robustly')
@pytest.mark.memleak
@pytest.mark.parametrize('params', [
    {'kernel': 'poly', 'degree': 30, 'C': 1, 'gamma': 1}
])
def test_svm_memleak_on_exception(params, n_rows=1000, n_iter=10,
                                  n_cols=1000, dataset='blobs'):
    """
    Test whether there is any mem leak when we exit training with an exception.
    The poly kernel with degree=30 will overflow, and triggers the
    'SMO error: NaN found...' exception.
    """
    X_train, y_train = make_blobs(n_samples=n_rows, n_features=n_cols,
                                  random_state=137, centers=2)
    X_train = X_train.astype(np.float32)
    stream = cuml.cuda.Stream()
    handle = cuml.Handle()
    handle.setStream(stream)

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

    del(cuSVC)
    handle.sync()
    delta_mem = free_mem - cuda.current_context().get_memory_info()[0]
    print("Delta GPU mem: {} bytes".format(delta_mem))
    assert delta_mem == 0


def make_regression_dataset(dataset, n_rows, n_cols):
    np.random.seed(137)
    if dataset == 'reg1':
        X, y = make_regression(
            n_rows, n_cols, n_informative=2, n_targets=1, random_state=137)
    elif dataset == 'reg2':
        X, y = make_regression(
            n_rows, n_cols, n_informative=2, n_targets=1, random_state=137,
            noise=10)
    elif dataset == 'Friedman':
        X, y = make_friedman1(n_samples=n_rows, n_features=n_cols, noise=0.0,
                              random_state=137)
    else:
        raise ValueError('Wrong option for dataste: ', dataset)
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
        assert abs(score1-score2) < tol
    else:
        y_pred1 = svr1.predict(X_test)
        y_pred2 = svr2.predict(X_test)
        mse1 = mean_squared_error(y_test, y_pred1)
        mse2 = mean_squared_error(y_test, y_pred2)
        assert (mse1-mse2)/mse2 < tol


@pytest.mark.parametrize('params', [
    {'kernel': 'linear', 'C': 1, 'gamma': 1},
    {'kernel': 'rbf', 'C': 1, 'gamma': 1},
    {'kernel': 'poly', 'C': 1, 'gamma': 1},
])
@pytest.mark.parametrize('dataset', ['reg1', 'reg2', 'Friedman'])
@pytest.mark.parametrize('n_rows', [unit_param(3), unit_param(100),
                                    quality_param(1000), stress_param(5000)])
@pytest.mark.parametrize('n_cols', [unit_param(5), unit_param(100),
                                    quality_param(1000), stress_param(1000)])
def test_svr_skl_cmp(params, dataset, n_rows, n_cols):
    """ Compare to Sklearn SVR """
    if (dataset == 'Friedman' and n_cols < 5):
        # We need at least 5 feature columns for the Friedman dataset
        return
    X_train, X_test, y_train, y_test = make_regression_dataset(dataset, n_rows,
                                                               n_cols)
    cuSVR = cu_svm.SVR(**params)
    cuSVR.fit(X_train, y_train)

    sklSVR = svm.SVR(**params)
    sklSVR.fit(X_train, y_train)

    compare_svr(cuSVR, sklSVR, X_test, y_test)


def test_svr_skl_cmp_weighted():
    """ Compare to Sklearn SVR, use sample weights"""
    X, y = make_regression(
        n_samples=100, n_features=5, n_informative=2, n_targets=1,
        random_state=137, noise=10)
    sample_weights = 10*np.sin(np.linspace(0, 2*np.pi, len(y))) + 10.1

    params = {'kernel': 'linear', 'C': 10, 'gamma': 1}
    cuSVR = cu_svm.SVR(**params)
    cuSVR.fit(X, y, sample_weights)

    sklSVR = svm.SVR(**params)
    sklSVR.fit(X, y, sample_weights)

    compare_svr(cuSVR, sklSVR, X, y)


@pytest.mark.parametrize('classifier', [True, False])
@pytest.mark.parametrize('train_dtype', [np.float32, np.float64])
@pytest.mark.parametrize('test_dtype', [np.float64, np.float32])
def test_svm_predict_convert_dtype(train_dtype, test_dtype, classifier):
    X, y = make_classification(n_samples=50, random_state=0)

    X = X.astype(train_dtype)
    y = y.astype(train_dtype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)

    if classifier:
        clf = cu_svm.SVC()
    else:
        clf = cu_svm.SVR()
    clf.fit(X_train, y_train)
    clf.predict(X_test.astype(test_dtype))
