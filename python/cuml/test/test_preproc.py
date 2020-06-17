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
from cuml._thirdparty.thirdparty_adapters import to_output_type
from .test_preproc_utils import small_clf_dataset  # noqa: F401
from .test_preproc_utils import small_int_dataset  # noqa: F401
from .test_preproc_utils import assert_array_equal

from cuml.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, \
                               scale, minmax_scale, normalize
from cuml.preprocessing import SimpleImputer
from cuml.preprocessing import PolynomialFeatures
import numpy as np

import operator as op
from functools import reduce


def test_minmax_scaler(small_clf_dataset):  # noqa: F811
    (np_X, np_y), (X, y) = small_clf_dataset

    scaler = MinMaxScaler(copy=True)
    transformed_X = scaler.fit_transform(X)
    assert str(type(X)) == str(type(transformed_X))

    transformed_X = to_output_type(transformed_X, 'numpy')

    data_min = np.nanmin(np_X, axis=0)
    data_range = np.nanmax(np_X, axis=0) - data_min
    data_range[data_range == 0.0] = 1.0
    scale = 1.0 / data_range
    mini = 0.0 - data_min * scale
    transformed_np_X = np_X * scale + mini

    assert_array_equal(transformed_X, transformed_np_X,
                       mean_diff_tol=0.0001, max_diff_tol=0.0001)

    reversed_X = scaler.inverse_transform(transformed_X)
    assert_array_equal(reversed_X, np_X, mean_diff_tol=0.0001,
                       max_diff_tol=0.0001)


def test_minmax_scale(small_clf_dataset):  # noqa: F811
    (np_X, np_y), (X, y) = small_clf_dataset

    transformed_X = minmax_scale(X)
    assert str(type(X)) == str(type(transformed_X))

    transformed_X = to_output_type(transformed_X, 'numpy')

    data_min = np.nanmin(np_X, axis=0)
    data_range = np.nanmax(np_X, axis=0) - data_min
    data_range[data_range == 0.0] = 1.0
    scale = 1.0 / data_range
    mini = 0.0 - data_min * scale
    transformed_np_X = np_X * scale + mini

    assert_array_equal(transformed_X, transformed_np_X,
                       mean_diff_tol=0.0001, max_diff_tol=0.0001)


@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_standard_scaler(small_clf_dataset, with_mean, with_std):  # noqa: F811
    (np_X, np_y), (X, y) = small_clf_dataset

    scaler = StandardScaler(copy=True, with_mean=with_mean, with_std=with_std)
    transformed_X = scaler.fit_transform(X)
    assert str(type(X)) == str(type(transformed_X))

    transformed_X = to_output_type(transformed_X, 'numpy')

    t_np_X = np.array(np_X, copy=True)
    if with_mean:
        t_np_X -= t_np_X.mean(axis=0)
    if with_std:
        t_np_X /= t_np_X.std(axis=0)

    transformed_np_X = t_np_X

    assert_array_equal(transformed_X, transformed_np_X,
                       mean_diff_tol=0.0001, max_diff_tol=0.0001)

    reversed_X = scaler.inverse_transform(transformed_X)
    assert_array_equal(reversed_X, np_X, mean_diff_tol=0.0001,
                       max_diff_tol=0.0001)


@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_scale(small_clf_dataset, with_mean, with_std):  # noqa: F811
    (np_X, np_y), (X, y) = small_clf_dataset

    transformed_X = scale(X, copy=True, with_mean=with_mean, with_std=with_std)
    assert str(type(X)) == str(type(transformed_X))

    transformed_X = to_output_type(transformed_X, 'numpy')

    t_np_X = np.array(np_X, copy=True)
    if with_mean:
        t_np_X -= t_np_X.mean(axis=0)
    if with_std:
        t_np_X /= t_np_X.std(axis=0)

    assert_array_equal(transformed_X, t_np_X, mean_diff_tol=0.0001,
                       max_diff_tol=0.0001)


def test_maxabs_scaler(small_clf_dataset):  # noqa: F811
    (np_X, np_y), (X, y) = small_clf_dataset

    scaler = MaxAbsScaler(copy=True)
    transformed_X = scaler.fit_transform(X)
    assert str(type(X)) == str(type(transformed_X))

    transformed_X = to_output_type(transformed_X, 'numpy')

    max_abs = np.nanmax(np.abs(np_X), axis=0)
    max_abs[max_abs == 0.0] = 1.0
    transformed_np_X = np_X / max_abs

    assert_array_equal(transformed_X, transformed_np_X,
                       mean_diff_tol=0.0001, max_diff_tol=0.0001)

    reversed_X = scaler.inverse_transform(transformed_X)
    assert_array_equal(reversed_X, np_X, mean_diff_tol=0.0001,
                       max_diff_tol=0.0001)


@pytest.mark.parametrize("norm", ['l1', 'l2', 'max'])
@pytest.mark.parametrize("return_norm", [True, False])
def test_normalize(small_clf_dataset, norm, return_norm):  # noqa: F811
    (np_X, np_y), (X, y) = small_clf_dataset

    if norm == 'l1':
        norms = np.abs(np_X).sum(axis=0)
    elif norm == 'l2':
        norms = np.linalg.norm(np_X, ord=2, axis=0)
    elif norm == 'max':
        norms = np.max(abs(np_X), axis=0)

    t_np_X = np.array(np_X, copy=True)
    t_np_X /= norms

    if return_norm:
        t_X, t_norms = normalize(X, axis=0, norm=norm, return_norm=return_norm)
        t_norms = to_output_type(t_norms, 'numpy')
        assert_array_equal(t_norms, norms, mean_diff_tol=0.0001,
                           max_diff_tol=0.0001)
    else:
        t_X = normalize(X, axis=0, norm=norm, return_norm=return_norm)
    assert str(type(X)) == str(type(t_X))
    t_X = to_output_type(t_X, 'numpy')

    assert_array_equal(t_X, t_np_X, mean_diff_tol=0.0001, max_diff_tol=0.0001)


@pytest.mark.parametrize("strategy", ["mean", "most_frequent", "constant"])
def test_imputer(small_int_dataset, strategy):  # noqa: F811
    np_X, X = small_int_dataset
    fill_value = np.random.randint(10, size=1)[0]

    imputer = SimpleImputer(copy=True, strategy=strategy,
                            fill_value=fill_value)
    transformed_X = imputer.fit_transform(X)
    assert str(type(X)) == str(type(transformed_X))

    transformed_X = to_output_type(transformed_X, 'numpy')

    t_np_X = np.array(np_X, copy=True)
    n_features = t_np_X.shape[1]

    if strategy == "mean":
        mean = np.nanmean(t_np_X, axis=0)
        for i in range(n_features):
            mask = np.where(np.isnan(t_np_X[:, i]))
            t_np_X[mask, i] = mean[i]
    elif strategy == "most_frequent":
        for i in range(n_features):
            values, counts = np.unique(t_np_X[:, i], return_counts=True)
            max_idx = np.argmax(counts)
            most_frequent = values[max_idx]

            mask = np.where(np.isnan(t_np_X[:, i]))
            t_np_X[mask, i] = most_frequent
    elif strategy == "constant":
        t_np_X[np.where(np.isnan(t_np_X))] = fill_value

    transformed_np_X = t_np_X
    assert not np.isnan(transformed_np_X).any()

    assert_array_equal(transformed_X, transformed_np_X,
                       mean_diff_tol=0.0001, max_diff_tol=0.0001)


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
    (np_X, np_y), (X, y) = small_clf_dataset

    polyfeatures = PolynomialFeatures(degree=degree, order=order,
                                      interaction_only=interaction_only,
                                      include_bias=include_bias)
    transformed_X = polyfeatures.fit_transform(X)
    assert str(type(X)) == str(type(transformed_X))

    if isinstance(transformed_X, np.ndarray):
        if order == 'C':
            assert transformed_X.flags['C_CONTIGUOUS']
        elif order == 'F':
            assert transformed_X.flags['F_CONTIGUOUS']

    transformed_X = to_output_type(transformed_X, 'numpy')

    n_features = np_X.shape[1]

    start = 0 if include_bias else 1
    n_combinations = sum(ncr(n_features, i) for i in range(start, degree+1))

    n_outputs = transformed_X.shape[1]
    if interaction_only:
        assert n_outputs == n_combinations
    else:
        assert n_outputs > n_combinations
