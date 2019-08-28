# Copyright (c) 2019, NVIDIA CORPORATION.
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
import cuml
import cuml.svm
from numba import cuda
from sklearn import svm
from sklearn.datasets import load_iris, make_blobs
from sklearn.datasets.samples_generator import make_classification, \
    make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cuml.test.utils import to_nparray, np_to_cudf
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
        for i in idx[-5:]:
            if (diff[i] > tol):
                print(diff[i], "at", i, "values", a[i], b[i])
        print('Avgdiff:', np.mean(diff), 'stddiyy:', np.std(diff), 'avgval:',
              np.mean(b))
    return equal


def compare_svm(svm1, svm2, X, y, n_sv_tol=None, b_tol=None, coef_tol=None,
                cmp_sv=False, dcoef_tol=None, n_wrong_tol=0,
                report_summary=False):
    """ Compares two svm classifiers
    Parameters:
    -----------
    svm1 : svm classifier
    svm2 : svm classifier
    n_sv_tol : float, default 1%
        tolerance while comparing the number of support vectors
    b_tol : float
        tolerance while comparing the constant in the decision functions
    coef_tol: float
        tolerance used while comparing coef_ attribute for linear SVM
    cmp_idx : boolean, default false
        whether to compare SVs and their indices
    dcoef_tol: float, default: do not compare dual coefficients
        tolerance used to compare dual coefs
    """
    n_support1 = np.sum(svm1.n_support_)
    n_support2 = np.sum(svm2.n_support_)
    if n_sv_tol is None:
        n_sv_tol = max(2, n_support1*0.02)
    assert abs(n_support1-n_support2) <= n_sv_tol

    if b_tol is None:
        b_tol = 30*svm1.tol
    if abs(svm2.intercept_) > 1e-6:
        assert abs((svm1.intercept_-svm2.intercept_)/svm2.intercept_) <= b_tol
    else:
        assert abs((svm1.intercept_-svm2.intercept_)) <= b_tol

    if coef_tol is None:
        coef_tol = 20*svm1.tol
    if svm1.kernel == 'linear':
        assert array_equal(svm1.coef_, svm2.coef_, coef_tol,
                           report_summary=report_summary)

    svm1_y_hat = to_nparray(svm1.predict(X))
    svm1_n_wrong = np.sum(np.abs(y - svm1_y_hat))
    svm2_y_hat = to_nparray(svm2.predict(X))
    svm2_n_wrong = np.sum(np.abs(y - svm2_y_hat))
    assert svm1_n_wrong - svm2_n_wrong <= n_wrong_tol

    if cmp_sv or (dcoef_tol is not None):
        sidx1 = np.argsort(to_nparray(svm1.support_))
        sidx2 = np.argsort(to_nparray(svm2.support_))

    if cmp_sv:
        support_idx1 = to_nparray(svm1.support_)[sidx1]
        support_idx2 = to_nparray(svm2.support_)[sidx2]
        assert np.all(support_idx1-support_idx2) == 0
        sv1 = to_nparray(svm1.support_vectors_)[sidx1, :]
        sv2 = to_nparray(svm2.support_vectors_)[sidx2, :]
        assert np.all(sv1-sv2 == 0)

    if dcoef_tol is not None:
        dcoef1 = to_nparray(svm1.dual_coef_)[0, sidx1]
        dcoef2 = to_nparray(svm2.dual_coef_)[0, sidx2]
        assert np.all(np.abs(dcoef1-dcoef2) <= dcoef_tol)


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


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
    cuSVC = cuml.svm.SVC(**params)
    cuSVC.fit(X_train, y_train)

    sklSVC = svm.SVC(**params)
    sklSVC.fit(X_train, y_train)

    compare_svm(cuSVC, sklSVC, X_train, y_train)


@pytest.mark.parametrize('params', [
    {'kernel': 'linear', 'C': 1},
    {'kernel': 'rbf', 'C': 1, 'gamma': 1},
    {'kernel': 'poly', 'C': 1, 'gamma': 1},
])
@pytest.mark.parametrize('dataset', ['classification2', 'gaussian', 'blobs'])
@pytest.mark.parametrize('n_rows', [3, unit_param(100), quality_param(1000),
                                    stress_param(10000)])
@pytest.mark.parametrize('n_cols', [2, unit_param(100), quality_param(1000),
                         stress_param(1000)])
def test_svm_skl_cmp_datasets(params, dataset, n_rows, n_cols):
    X_train, X_test, y_train, y_test = make_dataset(dataset, n_rows, n_cols)

    cuSVC = cuml.svm.SVC(**params)
    cuSVC.fit(X_train, y_train)

    sklSVC = svm.SVC(**params)
    sklSVC.fit(X_train, y_train)

    compare_svm(cuSVC, sklSVC, X_test, y_test, n_sv_tol=max(2, 0.02*n_rows),
                coef_tol=0.01, report_summary=True)


@pytest.mark.parametrize('x_dtype', [np.float32, np.float64])
@pytest.mark.parametrize('y_dtype', [np.float32, np.float64, np.int32])
@pytest.mark.parametrize('x_arraytype', ['numpy', 'dataframe', 'numba'])
@pytest.mark.parametrize('y_arraytype', ['numpy', 'series', 'numba'])
def test_svm_numeric_arraytype(x_arraytype, y_arraytype, x_dtype, y_dtype):
    X, y = get_binary_iris_dataset()
    X = X.astype(x_dtype, order="F")
    y = y.astype(y_dtype)
    if x_arraytype == 'dataframe':
        X_in = np_to_cudf(X)
    elif x_arraytype == 'numba':
        X_in = cuda.to_device(X)
    else:
        X_in = X
    if y_arraytype == 'numba':
        y_in = cuda.to_device(y)
    elif y_arraytype == 'series':
        y_in = cudf.Series(y)
    else:
        y_in = y
    params = {'kernel': 'rbf', 'C': 1, 'gamma': 0.25}
    cuSVC = cuml.svm.SVC(**params)
    cuSVC.fit(X_in, y_in)
    intercept_exp = 0.23468959692060373
    n_sv_exp = 15
    assert abs(cuSVC.intercept_ - intercept_exp) / intercept_exp < 1e-3
    assert cuSVC.n_support_ == n_sv_exp
    n_pred_wrong = np.sum(cuSVC.predict(X).to_array()-y)
    assert n_pred_wrong == 0


@pytest.mark.parametrize('params', [
    {'kernel': 'rbf', 'C': 1, 'gamma': 1}
])
@pytest.mark.parametrize('n_rows', [100])
@pytest.mark.parametrize('n_iter', [100])
def test_svm_loop(params, n_rows, n_iter, dataset='blobs', n_cols=10):
    X_train, X_test, y_train, y_test = make_dataset(dataset, n_rows, n_cols)

    free_mem = cuda.current_context().get_memory_info()[0]
    for i in range(n_iter):
        cuSVC = cuml.svm.SVC(**params)
        cuSVC.fit(X_train, y_train)
        cuSVC.predict(X_train)

    # intercept_exp = 0.9557494777004518
    # n_sv_exp = 6656
    # assert abs(cuSVC.intercept_ - intercept_exp) / intercept_exp < 1e-3
    # assert cuSVC.n_support_ == n_sv_exp
    del(cuSVC)
    delta_mem = free_mem - cuda.current_context().get_memory_info()[0]
    delta_mem /= (1024*1024.0)  # in MiB
    # This does not work yet
    # print("Delta mem", delta_mem)
    # assert delta_mem < 1
