#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

from cuml.neighbors import KernelDensity, VALID_KERNELS, logsumexp_kernel
from sklearn.metrics import pairwise_distances as skl_pairwise_distances
from sklearn.neighbors._ball_tree import kernel_norm
import numpy as np
from hypothesis import given, settings, assume, strategies as st
from hypothesis.extra.numpy import arrays
import pytest
from sklearn.model_selection import GridSearchCV


# not in log probability space
def compute_kernel_naive(Y, X, kernel, metric, h, sample_weights):
    d = skl_pairwise_distances(Y, X, metric)
    norm = kernel_norm(h, X.shape[1], kernel)

    if kernel == "gaussian":
        k = np.exp(-0.5 * (d * d) / (h * h))
    elif kernel == "tophat":
        k = (d < h)
    elif kernel == "epanechnikov":
        k = ((1.0 - (d * d) / (h * h)) * (d < h))
    elif kernel == "exponential":
        k = (np.exp(-d / h))
    elif kernel == "linear":
        k = ((1 - d / h) * (d < h))
    elif kernel == "cosine":
        k = (np.cos(0.5 * np.pi * d / h) * (d < h))
    else:
        raise ValueError("kernel not recognized")
    return norm*np.average(k, -1, sample_weights)


@st.composite
def array_strategy(draw):
    n = draw(st.integers(1, 100))
    m = draw(st.integers(1, 100))
    dtype = draw(st.sampled_from([np.float64, np.float32]))
    rng = np.random.RandomState(34)
    X = rng.randn(n, m).astype(dtype)
    n_test = draw(st.integers(1, 100))
    X_test = rng.randn(n_test, m).astype(dtype)

    if draw(st.booleans()):
        sample_weights = None
    else:
        sample_weights = draw(arrays(dtype=np.float64, shape=n,
                                     elements=st.floats(0.1, 2.0),))

    return X, X_test, sample_weights


metrics_strategy = st.sampled_from(
    ['euclidean', 'manhattan',
     'chebyshev', 'minkowski', 'hamming', 'canberra'])


@settings(deadline=None, max_examples=100)
@given(array_strategy(), st.sampled_from(VALID_KERNELS),
       metrics_strategy, st.floats(0.2, 10))
def test_kernel_density(arrays, kernel, metric, bandwidth):
    X, X_test, sample_weights = arrays
    if kernel == 'cosine':
        # cosine is numerically unstable at high dimensions
        # for both cuml and sklearn
        assume(X.shape[1] <= 20)
    kde = KernelDensity(kernel=kernel, metric=metric,
                        bandwidth=bandwidth).fit(X,
                                                 sample_weight=sample_weights)
    cuml_prob = kde.score_samples(X)
    cuml_prob_test = kde.score_samples(X_test)

    if X.dtype == np.float64:
        ref_prob = compute_kernel_naive(
            X, X, kernel, metric, bandwidth, sample_weights)
        ref_prob_test = compute_kernel_naive(
            X_test, X, kernel, metric, bandwidth, sample_weights)
        tol = 1e-3
        assert np.allclose(np.exp(cuml_prob), ref_prob,
                           rtol=tol, atol=tol, equal_nan=True)
        assert np.allclose(np.exp(cuml_prob_test),
                           ref_prob_test, rtol=tol, atol=tol, equal_nan=True)


def test_logaddexp():
    X = np.array([[0.0, 0.0], [0.0, 0.0]])
    out = np.zeros(X.shape[0])
    logsumexp_kernel.forall(out.size)(X, out)
    assert np.allclose(out, np.logaddexp.reduce(X, axis=1))

    X = np.array([[3.0, 1.0], [0.2, 0.7]])
    logsumexp_kernel.forall(out.size)(X, out)
    assert np.allclose(out, np.logaddexp.reduce(X, axis=1))


@pytest.mark.xfail(
    reason="cuml's pairwise_distances does"
    "not process metric_params as expected")
def test_metric_params():
    X = np.array([[0.0, 1.0], [2.0, 0.5]])
    kde = KernelDensity(metric='minkowski', metric_params={'p': 1.0}
                        ).fit(X)
    kde2 = KernelDensity(metric='minkowski', metric_params={'p': 2.0}
                         ).fit(X)
    assert np.all(kde.score_samples(X) != kde2.score_samples(X))


def test_grid_search():
    rs = np.random.RandomState(3)
    X = rs.normal(size=(30, 5))
    params = {"bandwidth": np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(X)
