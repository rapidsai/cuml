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
from cuml.test.utils import unit_param, quality_param, stress_param

from sklearn import svm
from sklearn.datasets import load_iris, make_blobs
from sklearn.datasets import make_regression, make_friedman1
from sklearn.datasets.samples_generator import make_classification, \
    make_gaussian_quantiles
from sklearn.metrics import mean_squared_error
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


def compare_svm(svm1, svm2, X, y, n_sv_tol=None, b_tol=None, coef_tol=None,
                cmp_sv=False, dcoef_tol=None, accuracy_tol=None,
                report_summary=False, cmp_decision_func=False):
    """ Compares two svm classifiers
    Parameters:
    -----------
    svm1 : svm classifier
    svm2 : svm classifier
    accuracy_tol : float, default 0.1%
        tolerance while comparing the prediction accuracy
    b_tol : float
        tolerance while comparing the constant in the decision functions
    coef_tol: float
        tolerance used while comparing coef_ attribute for linear SVM
    cmp_idx : boolean, default false
        whether to compare SVs and their indices
    dcoef_tol: float, default: do not compare dual coefficients
        tolerance used to compare dual coefs
    """

    n = X.shape[0]
    svm1_y_hat = svm1.predict(X).to_array()
    svm1_n_wrong = np.sum(np.abs(y - svm1_y_hat))
    accuracy1 = (n-svm1_n_wrong)*100/n
    svm2_y_hat = svm2.predict(X)
    if type(svm2_y_hat) != np.ndarray:
        svm2_y_hat = svm2_y_hat.to_array()
    svm2_n_wrong = np.sum(np.abs(y - svm2_y_hat))
    accuracy2 = (n-svm2_n_wrong)*100/n

    if accuracy_tol is None:
        if n >= 250 and (accuracy1 + accuracy2)/2 <= 75:
            # 1% accuracy tolerance for not so accurate SVM on "large" dataset
            accuracy_tol = 1
        else:
            accuracy_tol = 0.1

    assert abs(accuracy1 - accuracy2) <= accuracy_tol

    n_support1 = np.sum(svm1.n_support_)
    n_support2 = np.sum(svm2.n_support_)

    if n_sv_tol is None:
        n_sv_tol = max(2, n_support1*0.02)
    if b_tol is None:
        b_tol = 30*svm1.tol

    if accuracy1 < 50:
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
        n_sv_tol *= 10
        b_tol *= 10
        if n >= 250:
            coef_tol = 2  # allow any direction
        else:
            coef_tol *= 10

    assert abs(n_support1-n_support2) <= n_sv_tol

    if abs(svm2.intercept_) > 1e-6:
        assert abs((svm1.intercept_-svm2.intercept_)/svm2.intercept_) <= b_tol
    else:
        assert abs((svm1.intercept_-svm2.intercept_)) <= b_tol

    if coef_tol is None:
        coef_tol = 1e-5
    if svm1.kernel == 'linear':
        cs = np.dot(svm1.coef_, svm2.coef_.T) / \
            (np.linalg.norm(svm1.coef_) * np.linalg.norm(svm2.coef_))
        assert cs > 1 - coef_tol

    if cmp_sv or (dcoef_tol is not None):
        sidx1 = np.argsort((svm1.support_).copy_to_host())
        sidx2 = np.argsort((svm2.support_).copy_to_host())

    if cmp_sv:
        support_idx1 = ((svm1.support_).copy_to_host())[sidx1]
        support_idx2 = ((svm2.support_).copy_to_host())[sidx2]
        assert np.all(support_idx1-support_idx2) == 0
        sv1 = ((svm1.support_vectors_).copy_to_host())[sidx1, :]
        sv2 = ((svm2.support_vectors_).copy_to_host())[sidx2, :]
        assert np.all(sv1-sv2 == 0)

    if dcoef_tol is not None:
        dcoef1 = ((svm1.dual_coef_).copy_to_host())[0, sidx1]
        dcoef2 = ((svm2.dual_coef_).copy_to_host())[0, sidx2]
        assert np.all(np.abs(dcoef1-dcoef2) <= dcoef_tol)

    if cmp_decision_func:
        if accuracy2 > 90:
            df1 = svm1.decision_function(X).to_array()
            df2 = svm2.decision_function(X)
            # For classification, the class is determined by
            # sign(decision function). We should not expect tight match for
            # the actual value of the function, therfore we set large tolerance
            assert(array_equal(df1, df2, tol=1e-1, relative_diff=True,
                   report_summary=True))
        else:
            print("Skipping decision function test due to low  accuracy",
                  accuracy2)


def make_dataset(dataset, n_rows, n_cols, n_classes=2):
    np.random.seed(137)
    if dataset == 'classification1':
        X, y = make_classification(
            n_rows, n_cols, n_informative=2, n_redundant=0,
            n_classes=n_classes, n_clusters_per_class=1)
    elif dataset == 'classification2':
        X, y = make_classification(
            n_rows, n_cols, n_informative=2, n_redundant=0,
            n_classes=n_classes, n_clusters_per_class=2)
    elif dataset == 'gaussian':
        X, y = make_gaussian_quantiles(n_samples=n_rows, n_features=n_cols,
                                       n_classes=n_classes)
    elif dataset == 'blobs':
        X, y = make_blobs(n_samples=n_rows, n_features=n_cols,
                          centers=n_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
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

    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X_train, y_train)

    sklSVC = svm.SVC(**params)
    sklSVC.fit(X_train, y_train)

    compare_svm(cuSVC, sklSVC, X_test, y_test, n_sv_tol=max(2, 0.02*n_rows),
                coef_tol=1e-5, report_summary=True)


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
    y_pred = cuSVC.predict(X_test).to_array()
    n_correct = np.sum(y_test == y_pred)
    accuracy = n_correct * 100 / n_pred
    assert accuracy > 99


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
        X_df = cudf.DataFrame()
        X = X_df.from_gpu_matrix(cuda.to_device(X))
    elif x_arraytype == 'numba':
        X = cuda.to_device(X)
    # Using degree 40 polynomials and fp32 training would fail with
    # gamma = 1/(n_cols*X.std()), but it works with the correct implementation:
    # gamma = 1/(n_cols*X.var())
    cuSVC = cu_svm.SVC(**params)
    cuSVC.fit(X, y)
    y_pred = cuSVC.predict(X).to_array()
    n_correct = np.sum(y == y_pred)
    accuracy = n_correct * 100 / n_rows
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
    n_pred_wrong = np.sum(cuSVC.predict(X).to_array()-y)
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
        ms += np.prod(x.shape)*x.dtype.itemsize
    return ms


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
    Test whether there is any memory leak. Note: small n_rows, and n_cols
    values will result in small model size, that will not be measured by
    get_memory_info.
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
        y_pred1 = svr1.predict(X_test).to_array()
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
