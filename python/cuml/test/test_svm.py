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
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from cuml.test.utils import to_nparray


def compare_svm(svm1, svm2, X, y, n_sv_tol=None, b_tol=None, coef_tol=None,
                cmp_sv=False, dcoef_tol=None):
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
        n_sv_tol = max(2, n_support1*0.01)
    assert abs(n_support1-n_support2) <= n_sv_tol

    if b_tol is None:
        b_tol = 10*svm1.tol
    assert abs(svm1.intercept_-svm2.intercept_) <= b_tol

    if coef_tol is None:
        coef_tol = svm1.tol
    if svm1.kernel == 'linear':
        assert np.all(np.abs(svm1.coef_-svm2.coef_) <= coef_tol)

    svm1_y_hat = to_nparray(svm1.predict(X))
    svm1_n_wrong = np.sum(np.abs(y - svm1_y_hat))
    svm2_y_hat = to_nparray(svm2.predict(X))
    svm2_n_wrong = np.sum(np.abs(y - svm2_y_hat))
    assert svm1_n_wrong == svm2_n_wrong

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


@pytest.mark.parametrize('params', [
    {'kernel': 'linear', 'C': 1},
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
# @pytest.mark.parametrize('name', [unit_param(None), quality_param('iris')])
def test_svm_fit_predict(params, name='iris'):
    if name == 'iris':
        iris = load_iris()
        X = iris.data
        y = iris.target
        y = (y > 0).astype(X.dtype)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        # we create 40 separable points
        # TODO: tune the sensitivity of the test for this
        print('Unit test')
        np.random.seed(0)
        X = np.r_[np.random.randn(20, 2) - [2, 2],
                  np.random.randn(20, 2) + [2, 2]]
        y = np.array([0] * 20 + [1] * 20)

    cuSVC = cuml.svm.SVC(**params)
    cuSVC.fit(X, y)

    sklSVC = svm.SVC(**params)
    sklSVC.fit(X, y)

    compare_svm(cuSVC, sklSVC, X, y)

# TODO test different input types
# @pytest.mark.parametrize('x_datatype', [np.float32, np.float64])
# @pytest.mark.parametrize('y_datatype', [np.float32, np.float64, np.int32])
