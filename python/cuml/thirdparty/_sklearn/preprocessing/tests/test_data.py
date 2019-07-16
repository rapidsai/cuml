#
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
import cupy as cp
import cudf
import numpy as np
from numbers import Number

from sklearn.utils import gen_batches

# from sklearn.utils.testing import assert_raise_message
# from sklearn.utils.testing import assert_almost_equal
# from sklearn.utils.testing import clean_warning_registry
# from sklearn.utils.testing import assert_array_almost_equal
# from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import assert_less_equal
# from sklearn.utils.testing import assert_warns_message
# from sklearn.utils.testing import assert_no_warnings
from sklearn.utils.testing import assert_allclose
# from sklearn.utils.testing import assert_allclose_dense_sparse
from sklearn.utils.testing import assert_array_less


# from sklearn.utils.sparsefuncs import mean_variance_axis
from data import _handle_zeros_in_scale
from data import to_cupy

from data import StandardScaler
from data import scale
from data import MinMaxScaler
from data import minmax_scale
# from data import Binarizer
# from data import KernelCenterer
# from data import Normalizer
# from data import normalize
# from data import MaxAbsScaler
# from data import maxabs_scale
# from data import RobustScaler
# from data import robust_scale
# from data import add_dummy_feature

# from data import QuantileTransformer
# from data import quantile_transform
# from data import PolynomialFeatures
# from data import PowerTransformer
# from data import power_transform

from sklearn import datasets
iris = datasets.load_iris()

# Make some data to be used many times
rng = cp.random.RandomState(0)
n_features = 30
n_samples = 1000
offsets = rng.uniform(-1, 1, size=n_features)
scales = rng.uniform(1, 10, size=n_features)
X_2d = rng.randn(n_samples, n_features) * scales + offsets
X_1row = X_2d[0, :].reshape(1, n_features)
X_1col = X_2d[:, 0].reshape(n_samples, 1)
X_1row_cudf = cudf.DataFrame.from_gpu_matrix(X_1row.copy())
X_1col_cudf = cudf.DataFrame.from_gpu_matrix(X_1col.copy())


def assert_array_equal(a, b, tol=1e-4, with_sign=True):
    a, b = to_cparray(a), to_cparray(b)
    if not with_sign:
        a, b = cp.abs(a), cp.abs(b)
    res = cp.max(cp.abs(a-b)) < tol
    assert res.all()


def to_cparray(x):
    if isinstance(x, Number):
        return cp.array([x])
    elif isinstance(x, cp.ndarray):
        return x
    elif isinstance(x, cudf.DataFrame):
        return cp.array(x.as_gpu_matrix())
    elif isinstance(x, (cudf.Series, list)):
        return cp.array(x)
    else:
        raise TypeError('input of type {} is not cudf or cupy'.format(type))


def assert_correct_incr(i, batch_start, batch_stop, n, chunk_size,
                        n_samples_seen):
    if batch_stop != n:
        assert_array_equal((i + 1) * chunk_size, n_samples_seen)
    else:
        assert_array_equal(i * chunk_size + (batch_stop - batch_start),
                     n_samples_seen)


def test_standard_scaler_1d():
    # Test scaling of dataset along single axis
    for X in [X_1row, X_1col, X_1row_cudf, X_1col_cudf]:
        scaler = StandardScaler()
        X_scaled = scaler.fit(X).transform(X, copy=True)

        X, _ = to_cupy(X)
        X_scaled, _ = to_cupy(X_scaled)
        if X.shape[0] == 1:
            assert_array_equal(scaler.mean_, X.ravel())
            assert_array_equal(scaler.scale_, cp.ones(n_features))
            assert_array_equal(X_scaled.mean(axis=0),
                               cp.zeros_like(X_scaled.mean(axis=0)))
            assert_array_equal(X_scaled.std(axis=0),
                               cp.zeros_like(X_scaled.std(axis=0)))
        else:
            assert_array_equal(scaler.mean_, X.mean())
            assert_array_equal(scaler.scale_, X.std())
            assert_array_equal(X_scaled.mean(axis=0),
                               cp.zeros_like(X_scaled.mean(axis=0)))
            assert_array_equal(X_scaled.mean(axis=0), .0)
            assert_array_equal(X_scaled.std(axis=0), 1.)
        assert_array_equal(scaler.n_samples_seen_, X.shape[0])

        # check inverse transform
        X_scaled_back = scaler.inverse_transform(X_scaled)
        assert_array_equal(X_scaled_back, X)

    # Constant feature
    X = cp.ones((5, 1))
    scaler = StandardScaler()
    X_scaled = scaler.fit(X).transform(X, copy=True)
    assert_array_equal(scaler.mean_, 1.)
    assert_array_equal(scaler.scale_, 1.)
    assert_array_equal(X_scaled.mean(axis=0), .0)
    assert_array_equal(X_scaled.std(axis=0), .0)
    assert_array_equal(scaler.n_samples_seen_, X.shape[0])


def test_standard_scaler_dtype():
    # Ensure scaling does not affect dtype
    rng = cp.random.RandomState(0)
    n_samples = 10
    n_features = 3
    for dtype in [cp.float16, cp.float32, cp.float64]:
        X = rng.randn(n_samples, n_features).astype(dtype)
        scaler = StandardScaler()
        X_scaled = scaler.fit(X).transform(X)
        assert X.dtype == X_scaled.dtype
        assert scaler.mean_.dtype == np.float64
        assert scaler.scale_.dtype == np.float64


def test_scale_1d():
    # 1-d inputs
    X_arr = cp.array([1., 3., 5., 0.])
    X_cudf = cudf.from_dlpack(X_arr.copy().toDlpack())

    for X in [X_arr, X_cudf]:
        X_scaled, _ = to_cupy(scale(X))
        # after X_scaled rendered, convert X to cupy to compare with X_scaled
        X, _ = to_cupy(X)
        assert_array_equal(X_scaled.mean(), 0.0)
        assert_array_equal(X_scaled.std(), 1.0)
        assert_array_equal(scale(X, with_mean=False, with_std=False), X)


def test_scaler_2d_arrays():
    # Test scaling of 2d array along first axis
    rng = cp.random.RandomState(0)
    n_features = 5
    n_samples = 4
    X = rng.randn(n_samples, n_features)
    X[:, 0] = 0.0  # first feature is always of zero
    X_cudf = cudf.DataFrame.from_gpu_matrix(cp.asfortranarray(X))

    for X in [X, X_cudf]:
        scaler = StandardScaler()
        X_scaled = scaler.fit(X).transform(X, copy=True)

        X_scaled, _ = to_cupy(X_scaled)
        X, _ = to_cupy(X)
        assert not cp.any(cp.isnan(X_scaled))
        assert_array_equal(scaler.n_samples_seen_, n_samples)

        assert_array_equal(X_scaled.mean(axis=0), n_features * [0.0])
        assert_array_equal(X_scaled.std(axis=0), [0., 1., 1., 1., 1.])
        # Check that X has been copied
        assert X_scaled is not X

        # check inverse transform
        X_scaled_back = scaler.inverse_transform(X_scaled)
        assert X_scaled_back is not X
        assert X_scaled_back is not X_scaled
        assert_array_equal(X_scaled_back, X)

        X_scaled = scale(X, axis=1, with_std=False)
        assert not cp.any(cp.isnan(X_scaled))
        assert_array_equal(X_scaled.mean(axis=1), n_samples * [0.0])
        X_scaled = scale(X, axis=1, with_std=True)
        assert not cp.any(cp.isnan(X_scaled))
        assert_array_equal(X_scaled.mean(axis=1), n_samples * [0.0])
        assert_array_equal(X_scaled.std(axis=1), n_samples * [1.0])
        # Check that the data hasn't been modified
        assert X_scaled is not X

        X_scaled = scaler.fit(X).transform(X, copy=False)
        assert not cp.any(cp.isnan(X_scaled))
        assert_array_equal(X_scaled.mean(axis=0), n_features * [0.0])
        assert_array_equal(X_scaled.std(axis=0), [0., 1., 1., 1., 1.])
        # Check that X has not been copied
        assert X_scaled is X

        X = rng.randn(4, 5)
        X[:, 0] = 1.0  # first feature is a constant, non zero feature
        scaler = StandardScaler()
        X_scaled = scaler.fit(X).transform(X, copy=True)
        assert not cp.any(cp.isnan(X_scaled))
        assert_array_equal(X_scaled.mean(axis=0), n_features * [0.0])
        assert_array_equal(X_scaled.std(axis=0), [0., 1., 1., 1., 1.])
        # Check that X has not been copied
        assert X_scaled is not X


def test_scaler_float16_overflow():
    # Test if the scaler will not overflow on float16 numpy arrays
    rng = cp.random.RandomState(0)
    # float16 has a maximum of 65500.0. On the worst case 5 * 200000 is 100000
    # which is enough to overflow the data type
    X = rng.uniform(5, 10, [200000, 1]).astype(cp.float16)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Calculate the float64 equivalent to verify result
    X_scaled_f64 = StandardScaler().fit_transform(X.astype(cp.float64))

    # Overflow calculations may cause -inf, inf, or nan. Since there is no nan
    # icput, all of the outputs should be finite. This may be redundant since a
    # FloatingPointError exception will be thrown on overflow above.
    assert cp.all(cp.isfinite(X_scaled))

    # The normal distribution is very unlikely to go above 4. At 4.0-8.0 the
    # float16 precision is 2^-8 which is around 0.004. Thus only 2 decimals are
    # checked to account for precision differences.
    np.testing.assert_array_almost_equal(
        cp.asnumpy(X_scaled),
        cp.asnumpy(X_scaled_f64),
        decimal=2)


def test_handle_zeros_in_scale():
    s1 = cp.array([0, 1, 2, 3])
    s2 = _handle_zeros_in_scale(s1, copy=True)

    assert not s1[0] == s2[0]
    assert_array_equal(s1, cp.array([0, 1, 2, 3]))
    assert_array_equal(s2, cp.array([1, 1, 2, 3]))


def test_minmax_scaler_partial_fit():
    # Test if partial_fit run over many batches of size 1 and 50
    # gives the same results as fit
    X = X_2d
    n = X.shape[0]

    for chunk_size in [1, 2, 50, n, n + 42]:
        # Test mean at the end of the process
        scaler_batch = MinMaxScaler().fit(X)

        scaler_incr = MinMaxScaler()
        for batch in gen_batches(n_samples, chunk_size):
            scaler_incr = scaler_incr.partial_fit(X[batch])

        assert_array_equal(scaler_batch.data_min_,
                           scaler_incr.data_min_)
        assert_array_equal(scaler_batch.data_max_,
                           scaler_incr.data_max_)
        assert_array_equal(scaler_batch.n_samples_seen_,
                           scaler_incr.n_samples_seen_)
        assert_array_equal(scaler_batch.data_range_,
                           scaler_incr.data_range_)
        assert_array_equal(scaler_batch.scale_, scaler_incr.scale_)
        assert_array_equal(scaler_batch.min_, scaler_incr.min_)

        # Test std after 1 step
        batch0 = slice(0, chunk_size)
        scaler_batch = MinMaxScaler().fit(X[batch0])
        scaler_incr = MinMaxScaler().partial_fit(X[batch0])

        assert_array_equal(scaler_batch.data_min_,
                           scaler_incr.data_min_)
        assert_array_equal(scaler_batch.data_max_,
                           scaler_incr.data_max_)
        assert_array_equal(scaler_batch.n_samples_seen_,
                           scaler_incr.n_samples_seen_)
        assert_array_equal(scaler_batch.data_range_,
                           scaler_incr.data_range_)
        assert_array_equal(scaler_batch.scale_, scaler_incr.scale_)
        assert_array_equal(scaler_batch.min_, scaler_incr.min_)

        # Test std until the end of partial fits, and
        scaler_batch = MinMaxScaler().fit(X)
        scaler_incr = MinMaxScaler()  # Clean estimator
        for i, batch in enumerate(gen_batches(n_samples, chunk_size)):
            scaler_incr = scaler_incr.partial_fit(X[batch])
            assert_correct_incr(i, batch_start=batch.start,
                                batch_stop=batch.stop, n=n,
                                chunk_size=chunk_size,
                                n_samples_seen=scaler_incr.n_samples_seen_)


def test_standard_scaler_partial_fit():
    # Test if partial_fit run over many batches of size 1 and 50
    # gives the same results as fit
    X = X_2d
    n = X.shape[0]

    for chunk_size in [1, 2, 50, n, n + 42]:
        # Test mean at the end of the process
        scaler_batch = StandardScaler(with_std=False).fit(X)

        scaler_incr = StandardScaler(with_std=False)
        for batch in gen_batches(n_samples, chunk_size):
            scaler_incr = scaler_incr.partial_fit(X[batch])

        assert_array_equal(scaler_batch.mean_, scaler_incr.mean_)
        assert scaler_batch.var_ == scaler_incr.var_    # Nones
        assert_array_equal(scaler_batch.n_samples_seen_,
                           scaler_incr.n_samples_seen_)

        # Test std after 1 step
        batch0 = slice(0, chunk_size)
        scaler_incr = StandardScaler().partial_fit(X[batch0])
        if chunk_size == 1:
            assert_array_equal(cp.zeros(n_features, dtype=cp.float64),
                               scaler_incr.var_)
            assert_array_equal(cp.ones(n_features, dtype=cp.float64),
                               scaler_incr.scale_)
        else:
            assert_array_equal(cp.var(X[batch0], axis=0),
                               scaler_incr.var_)
            assert_array_equal(cp.std(X[batch0], axis=0),
                               scaler_incr.scale_)  # no constants

        # Test std until the end of partial fits, and
        scaler_batch = StandardScaler().fit(X)
        scaler_incr = StandardScaler()  # Clean estimator
        for i, batch in enumerate(gen_batches(n_samples, chunk_size)):
            scaler_incr = scaler_incr.partial_fit(X[batch])
            assert_correct_incr(i, batch_start=batch.start,
                                batch_stop=batch.stop, n=n,
                                chunk_size=chunk_size,
                                n_samples_seen=scaler_incr.n_samples_seen_)

        assert_array_equal(scaler_batch.var_, scaler_incr.var_)
        assert_array_equal(scaler_batch.n_samples_seen_,
                           scaler_incr.n_samples_seen_)


def test_standard_scaler_partial_fit_numerical_stability():
    # Test if the incremental computation introduces significative errors
    # for large datasets with values of large magniture
    rng = cp.random.RandomState(0)
    n_features = 2
    n_samples = 100
    offsets = rng.uniform(-1e15, 1e15, size=n_features)
    scales = rng.uniform(1e3, 1e6, size=n_features)
    X = rng.randn(n_samples, n_features) * scales + offsets

    scaler_batch = StandardScaler().fit(X)
    scaler_incr = StandardScaler()
    for chunk in X:
        scaler_incr = scaler_incr.partial_fit(chunk.reshape(1, n_features))

    # Regardless of abs values, they must not be more diff 6 significant digits
    tol = 10 ** (-6)
    assert_allclose(
        cp.asnumpy(scaler_incr.mean_),
        cp.asnumpy(scaler_batch.mean_), rtol=tol)
    assert_allclose(
        cp.asnumpy(scaler_incr.var_),
        cp.asnumpy(scaler_batch.var_), rtol=tol)
    assert_allclose(
        cp.asnumpy(scaler_incr.scale_),
        cp.asnumpy(scaler_batch.scale_), rtol=tol)
    # NOTE Be aware that for much larger offsets std is very unstable (last
    # assert) while mean is OK.


def test_standard_scaler_trasform_with_partial_fit():
    # Check some postconditions after applying partial_fit and transform
    X = X_2d[:100, :]

    scaler_incr = StandardScaler()
    for i, batch in enumerate(gen_batches(X.shape[0], 1)):

        X_sofar = X[:(i + 1), :]
        chunks_copy = X_sofar.copy()
        scaled_batch = StandardScaler().fit_transform(X_sofar)

        scaler_incr = scaler_incr.partial_fit(X[batch])
        scaled_incr = scaler_incr.transform(X_sofar)

        assert_array_equal(scaled_batch, scaled_incr)
        assert_array_equal(X_sofar, chunks_copy)     # No change
        right_input = scaler_incr.inverse_transform(scaled_incr)
        assert_array_equal(X_sofar, right_input)

        zero = np.zeros(X.shape[1])
        epsilon = np.finfo(float).eps
        assert_array_less(
            zero,
            cp.asnumpy(scaler_incr.var_) + epsilon)  # as less or equal
        assert_array_less(
            zero,
            cp.asnumpy(scaler_incr.scale_) + epsilon)
        # (i+1) because the Scaler has been already fitted
        assert_array_equal((i + 1), scaler_incr.n_samples_seen_)


def test_min_max_scaler_iris():
    X = cp.array(iris.data)
    scaler = MinMaxScaler()
    # default params
    X_trans = scaler.fit_transform(X)
    assert_array_equal(X_trans.min(axis=0), 0)
    assert_array_equal(X_trans.max(axis=0), 1)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_equal(X, X_trans_inv)

    # not default params: min=1, max=2
    scaler = MinMaxScaler(feature_range=(1, 2))
    X_trans = scaler.fit_transform(X)
    assert_array_equal(X_trans.min(axis=0), 1)
    assert_array_equal(X_trans.max(axis=0), 2)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_equal(X, X_trans_inv)

    # min=-.5, max=.6
    scaler = MinMaxScaler(feature_range=(-.5, .6))
    X_trans = scaler.fit_transform(X)
    assert_array_equal(X_trans.min(axis=0), -.5)
    assert_array_equal(X_trans.max(axis=0), .6)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_equal(X, X_trans_inv)

    # raises on invalid range
    scaler = MinMaxScaler(feature_range=(2, 1))
    with pytest.raises(ValueError):
        scaler.fit(X)


def test_min_max_scaler_zero_variance_features():
    # Check min max scaler on toy data with zero variance features
    X = cp.array([[0., 1., +0.5],
                  [0., 1., -0.1],
                  [0., 1., +1.1]])

    X_new =  cp.array([[+0., 2., 0.5],
                       [-1., 1., 0.0],
                       [+0., 1., 1.5]])

    # default params
    scaler = MinMaxScaler()
    X_trans = scaler.fit_transform(X)
    X_expected_0_1 = [[0., 0., 0.5],
                      [0., 0., 0.0],
                      [0., 0., 1.0]]
    assert_array_equal(X_trans, X_expected_0_1)
    X_trans_inv = scaler.inverse_transform(X_trans)
    assert_array_equal(X, X_trans_inv)

    X_trans_new = scaler.transform(X_new)
    X_expected_0_1_new = [[+0., 1., 0.500],
                          [-1., 0., 0.083],
                          [+0., 0., 1.333]]
    np.testing.assert_array_almost_equal(
        cp.asnumpy(X_trans_new),
        cp.asnumpy(X_expected_0_1_new), decimal=2)

    # not default params
    scaler = MinMaxScaler(feature_range=(1, 2))
    X_trans = scaler.fit_transform(X)
    X_expected_1_2 = [[1., 1., 1.5],
                      [1., 1., 1.0],
                      [1., 1., 2.0]]
    assert_array_equal(X_trans, X_expected_1_2)

    # function interface
    X_trans = minmax_scale(X)
    assert_array_equal(X_trans, X_expected_0_1)
    X_trans = minmax_scale(X, feature_range=(1, 2))
    assert_array_equal(X_trans, X_expected_1_2)


def test_minmax_scale_axis1():
    X = cp.array(iris.data)
    X_trans = minmax_scale(X, axis=1)
    assert_array_equal(cp.min(X_trans, axis=1), 0)
    assert_array_equal(cp.max(X_trans, axis=1), 1)


def test_min_max_scaler_1d():
    # Test scaling of dataset along single axis
    for X in [X_1row, X_1col, X_1row_cudf, X_1col_cudf]:

        scaler = MinMaxScaler(copy=True)
        X_scaled = scaler.fit(X).transform(X)
        X_scaled, _ = to_cupy(X_scaled)

        if X.shape[0] == 1:
            assert_array_equal(X_scaled.min(axis=0), cp.zeros(n_features))
            assert_array_equal(X_scaled.max(axis=0), cp.zeros(n_features))
        else:
            assert_array_equal(X_scaled.min(axis=0), .0)
            assert_array_equal(X_scaled.max(axis=0), 1.)
        assert scaler.n_samples_seen_ == X.shape[0]

        # check inverse transform
        X_scaled_back = scaler.inverse_transform(X_scaled)
        assert_array_equal(X_scaled_back, X)

    # Constant feature
    X = cp.ones((5, 1))
    scaler = MinMaxScaler()
    X_scaled = scaler.fit(X).transform(X)
    assert_greater_equal(cp.asnumpy(X_scaled.min()), 0.)
    assert_less_equal(cp.asnumpy(X_scaled.max()), 1.)
    assert scaler.n_samples_seen_ == X.shape[0]

    # Function interface
    X_1d = X_1row.ravel()
    min_ = X_1d.min()
    max_ = X_1d.max()
    assert_array_equal((X_1d - min_) / (max_ - min_),
                       minmax_scale(X_1d, copy=True))
