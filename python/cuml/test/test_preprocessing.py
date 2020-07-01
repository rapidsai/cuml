# Copyright (c) 2020, NVIDIA CORPORATION.
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

from cuml.preprocessing import StandardScaler, MinMaxScaler, \
                    MaxAbsScaler, scale, minmax_scale, normalize
from cuml.preprocessing import SimpleImputer
from cuml.preprocessing import PolynomialFeatures

from cuml._thirdparty.thirdparty_adapters import to_output_type
from .test_preproc_utils import small_clf_dataset  # noqa: F401
from .test_preproc_utils import small_sparse_dataset  # noqa: F401
from .test_preproc_utils import small_int_dataset  # noqa: F401

import numpy as np
from scipy import sparse as cpu_sp
from cupy import sparse as gpu_sp
from numpy.testing import assert_allclose

import operator as op
from functools import reduce


def test_minmax_scaler(small_clf_dataset):  # noqa: F811
    X_np, X = small_clf_dataset

    scaler = MinMaxScaler(copy=True)
    t_X = scaler.fit_transform(X)
    assert type(t_X) == type(X)

    data_min = np.nanmin(X_np, axis=0)
    data_range = np.nanmax(X_np, axis=0) - data_min
    data_range[data_range == 0.0] = 1.0
    scale = 1.0 / data_range
    mini = 0.0 - data_min * scale
    t_X_np = X_np * scale + mini

    r_X = scaler.inverse_transform(t_X)
    assert type(r_X) == type(t_X)

    t_X = to_output_type(t_X, 'numpy')
    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)

    r_X = to_output_type(r_X, 'numpy')
    assert_allclose(r_X, X_np, rtol=0.0001, atol=0.0001)


def test_minmax_scale(small_clf_dataset):  # noqa: F811
    X_np, X = small_clf_dataset

    t_X = minmax_scale(X)
    assert type(t_X) == type(X)

    data_min = np.nanmin(X_np, axis=0)
    data_range = np.nanmax(X_np, axis=0) - data_min
    data_range[data_range == 0.0] = 1.0
    scale = 1.0 / data_range
    mini = 0.0 - data_min * scale
    t_X_np = X_np * scale + mini

    t_X = to_output_type(t_X, 'numpy')
    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_standard_scaler(small_clf_dataset, with_mean, with_std):  # noqa: F811
    X_np, X = small_clf_dataset

    scaler = StandardScaler(copy=True, with_mean=with_mean, with_std=with_std)
    t_X = scaler.fit_transform(X)
    assert type(t_X) == type(X)

    t_X_np = np.array(X_np, copy=True)
    if with_mean:
        t_X_np -= t_X_np.mean(axis=0)
    if with_std:
        t_X_np /= t_X_np.std(axis=0)

    r_X = scaler.inverse_transform(t_X)
    assert type(r_X) == type(t_X)

    t_X = to_output_type(t_X, 'numpy')
    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)

    r_X = to_output_type(r_X, 'numpy')
    assert_allclose(r_X, X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_scale(small_clf_dataset, with_mean, with_std):  # noqa: F811
    X_np, X = small_clf_dataset

    t_X = scale(X, copy=True, with_mean=with_mean, with_std=with_std)
    assert type(t_X) == type(X)

    t_X_np = np.array(X_np, copy=True)
    if with_mean:
        t_X_np -= t_X_np.mean(axis=0)
    if with_std:
        t_X_np /= t_X_np.std(axis=0)

    t_X = to_output_type(t_X, 'numpy')
    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)


def test_maxabs_scaler(small_clf_dataset):  # noqa: F811
    X_np, X = small_clf_dataset

    scaler = MaxAbsScaler(copy=True)
    t_X = scaler.fit_transform(X)
    assert type(t_X) == type(X)

    max_abs = np.nanmax(np.abs(X_np), axis=0)
    max_abs[max_abs == 0.0] = 1.0
    t_X_np = X_np / max_abs

    r_X = scaler.inverse_transform(t_X)
    assert type(r_X) == type(t_X)

    t_X = to_output_type(t_X, 'numpy')
    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)

    r_X = to_output_type(r_X, 'numpy')
    assert_allclose(r_X, X_np, rtol=0.0001, atol=0.0001)


def test_sparse_maxabs_scaler(small_sparse_dataset):  # noqa: F811
    X_np, X = small_sparse_dataset

    scaler = MaxAbsScaler(copy=True)
    t_X = scaler.fit_transform(X)
    assert type(t_X) == type(X)

    max_abs = np.nanmax(np.abs(X_np), axis=0)
    max_abs[max_abs == 0.0] = 1.0
    t_X_np = X_np / max_abs

    r_X = scaler.inverse_transform(t_X)
    assert type(r_X) == type(t_X)

    t_X = to_output_type(t_X, 'numpy')
    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)

    r_X = to_output_type(r_X, 'numpy')
    assert_allclose(r_X, X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("norm", ['l1', 'l2', 'max'])
@pytest.mark.parametrize("return_norm", [True, False])
def test_normalize(small_clf_dataset, norm, return_norm):  # noqa: F811
    X_np, X = small_clf_dataset

    if norm == 'l1':
        norms = np.abs(X_np).sum(axis=0)
    elif norm == 'l2':
        norms = np.linalg.norm(X_np, ord=2, axis=0)
    elif norm == 'max':
        norms = np.max(abs(X_np), axis=0)

    t_X_np = np.array(X_np, copy=True)
    t_X_np /= norms

    if return_norm:
        t_X, t_norms = normalize(X, axis=0, norm=norm, return_norm=return_norm)
        t_norms = to_output_type(t_norms, 'numpy')
        assert_allclose(t_norms, norms, rtol=0.0001, atol=0.0001)
    else:
        t_X = normalize(X, axis=0, norm=norm, return_norm=return_norm)
    assert type(t_X) == type(X)

    t_X = to_output_type(t_X, 'numpy')
    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("norm", ['l1', 'l2', 'max'])
def test_sparse_normalize(small_sparse_dataset, norm):  # noqa: F811
    X_np, X = small_sparse_dataset

    def iscsc(X):
        return isinstance(X, cpu_sp.csc_matrix) or\
               isinstance(X, gpu_sp.csc_matrix)

    if iscsc(X):
        axis = 0
    else:
        axis = 1

    if norm == 'l1':
        norms = np.abs(X_np).sum(axis=axis)
    elif norm == 'l2':
        norms = np.linalg.norm(X_np, ord=2, axis=axis)
    elif norm == 'max':
        norms = np.max(abs(X_np), axis=axis)

    t_X_np = np.array(X_np, copy=True)

    if iscsc(X):
        t_X_np /= norms
    else:
        t_X_np = t_X_np.T
        t_X_np /= norms
        t_X_np = t_X_np.T

    t_X = normalize(X, axis=axis, norm=norm)
    assert type(t_X) == type(X)

    t_X = to_output_type(t_X, 'numpy')
    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("strategy", ["mean", "most_frequent", "constant"])
def test_imputer(small_int_dataset, strategy):  # noqa: F811
    X_np, X = small_int_dataset
    fill_value = np.random.randint(10, size=1)[0]

    imputer = SimpleImputer(copy=True, strategy=strategy,
                            fill_value=fill_value)
    t_X = imputer.fit_transform(X)
    assert type(t_X) == type(X)

    t_X_np = np.array(X_np, copy=True)
    n_features = t_X_np.shape[1]

    if strategy == "mean":
        mean = np.nanmean(t_X_np, axis=0)
        for i in range(n_features):
            mask = np.where(np.isnan(t_X_np[:, i]))
            t_X_np[mask, i] = mean[i]
    elif strategy == "most_frequent":
        for i in range(n_features):
            values, counts = np.unique(t_X_np[:, i], return_counts=True)
            max_idx = np.argmax(counts)
            most_frequent = values[max_idx]

            mask = np.where(np.isnan(t_X_np[:, i]))
            t_X_np[mask, i] = most_frequent
    elif strategy == "constant":
        t_X_np[np.where(np.isnan(t_X_np))] = fill_value

    assert not np.isnan(t_X_np).any()

    t_X = to_output_type(t_X, 'numpy')
    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)


@pytest.mark.parametrize("strategy", ["mean", "most_frequent", "constant"])
def test_sparse_imputer(small_sparse_dataset, strategy):  # noqa: F811
    X_np, X = small_sparse_dataset
    if isinstance(X, (cpu_sp.csr_matrix, gpu_sp.csr_matrix)):
        pytest.skip("unsupported sparse matrix")

    fill_value = np.random.randint(10, size=1)[0]

    imputer = SimpleImputer(copy=True, strategy=strategy,
                            fill_value=fill_value)
    t_X = imputer.fit_transform(X)
    assert type(t_X) == type(X)

    t_X_np = np.array(X_np, copy=True)
    n_features = t_X_np.shape[1]

    if strategy == "mean":
        mean = np.nanmean(t_X_np, axis=0)
        for i in range(n_features):
            mask = np.where(np.isnan(t_X_np[:, i]))
            t_X_np[mask, i] = mean[i]
    elif strategy == "most_frequent":
        for i in range(n_features):
            values, counts = np.unique(t_X_np[:, i], return_counts=True)
            max_idx = np.argmax(counts)
            most_frequent = values[max_idx]

            mask = np.where(np.isnan(t_X_np[:, i]))
            t_X_np[mask, i] = most_frequent
    elif strategy == "constant":
        t_X_np[np.where(np.isnan(t_X_np))] = fill_value

    assert not np.isnan(t_X_np).any()

    t_X = to_output_type(t_X, 'numpy')
    assert_allclose(t_X, t_X_np, rtol=0.0001, atol=0.0001)


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("order", ['C', 'F'])
def test_poly_features(small_clf_dataset, degree,  # noqa: F811
                       interaction_only, include_bias, order):
    X_np, X = small_clf_dataset

    polyfeatures = PolynomialFeatures(degree=degree, order=order,
                                      interaction_only=interaction_only,
                                      include_bias=include_bias)
    t_X = polyfeatures.fit_transform(X)
    assert type(X) == type(t_X)

    if isinstance(t_X, np.ndarray):
        if order == 'C':
            assert t_X.flags['C_CONTIGUOUS']
        elif order == 'F':
            assert t_X.flags['F_CONTIGUOUS']

    t_X = to_output_type(t_X, 'numpy')

    n_features = X_np.shape[1]

    start = 0 if include_bias else 1
    n_combinations = sum(ncr(n_features, i) for i in range(start, degree+1))

    n_outputs = t_X.shape[1]
    if interaction_only:
        assert n_outputs == n_combinations
    else:
        assert n_outputs > n_combinations


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("order", ['C', 'F'])
def test_sparse_poly_features(small_sparse_dataset, degree,  # noqa: F811
                              interaction_only, include_bias, order):
    X_np, X = small_sparse_dataset

    polyfeatures = PolynomialFeatures(degree=degree, order=order,
                                      interaction_only=interaction_only,
                                      include_bias=include_bias)
    t_X = polyfeatures.fit_transform(X)
    assert type(t_X) == type(X)

    t_X = to_output_type(t_X, 'numpy')

    n_features = X_np.shape[1]

    start = 0 if include_bias else 1
    n_combinations = sum(ncr(n_features, i) for i in range(start, degree+1))

    n_outputs = t_X.shape[1]
    if interaction_only:
        assert n_outputs == n_combinations
    else:
        assert n_outputs > n_combinations
