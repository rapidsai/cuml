#
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

import cupy as cp
import numpy as np
import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances as skl_pairwise_distances
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors._ball_tree import kernel_norm

import cuml
from cuml.common.exceptions import NotFittedError
from cuml.neighbors import VALID_KERNELS, KernelDensity
from cuml.neighbors.kernel_density import logaddexp_reduce
from cuml.testing.utils import as_type


# not in log probability space
def compute_kernel_naive(Y, X, kernel, metric, h, sample_weight):
    d = skl_pairwise_distances(Y, X, metric)
    norm = kernel_norm(h, X.shape[1], kernel)

    if kernel == "gaussian":
        k = np.exp(-0.5 * (d * d) / (h * h))
    elif kernel == "tophat":
        k = d < h
    elif kernel == "epanechnikov":
        k = (1.0 - (d * d) / (h * h)) * (d < h)
    elif kernel == "exponential":
        k = np.exp(-d / h)
    elif kernel == "linear":
        k = (1 - d / h) * (d < h)
    elif kernel == "cosine":
        k = np.cos(0.5 * np.pi * d / h) * (d < h)
    else:
        raise ValueError("kernel not recognized")
    return norm * np.average(k, -1, sample_weight)


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
        sample_weight = None
    else:
        sample_weight = draw(
            arrays(
                dtype=np.float64,
                shape=n,
                elements=st.floats(0.1, 2.0),
            )
        )
    type = draw(st.sampled_from(["numpy", "cupy", "cudf", "pandas"]))
    if type == "cupy":
        assume(n > 1 and n_test > 1)
    return as_type(type, X, X_test, sample_weight)


metrics_strategy = st.sampled_from(
    ["euclidean", "manhattan", "chebyshev", "minkowski", "hamming", "canberra"]
)


@settings(deadline=None)
@example(
    arrays=as_type(
        "numpy",
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[1.5, 2.5]]),
        None,
    ),
    kernel="gaussian",
    metric="euclidean",
    bandwidth=1.0,
)
@given(
    array_strategy(),
    st.sampled_from(VALID_KERNELS),
    metrics_strategy,
    st.floats(0.2, 10),
)
def test_kernel_density(arrays, kernel, metric, bandwidth):
    X, X_test, sample_weight = arrays
    X_np, X_test_np, sample_weight_np = as_type("numpy", *arrays)

    if kernel == "cosine":
        # cosine is numerically unstable at high dimensions
        # for both cuml and sklearn
        assume(X.shape[1] <= 20)
    kde = KernelDensity(
        kernel=kernel, metric=metric, bandwidth=bandwidth, output_type="cupy"
    )
    kde.fit(X, sample_weight=sample_weight)
    cuml_prob = kde.score_samples(X)
    cuml_prob_test = kde.score_samples(X_test)

    if X_np.dtype == np.float64:
        ref_prob = compute_kernel_naive(
            X_np, X_np, kernel, metric, bandwidth, sample_weight_np
        )
        ref_prob_test = compute_kernel_naive(
            X_test_np, X_np, kernel, metric, bandwidth, sample_weight_np
        )
        tol = 1e-3
        assert np.allclose(
            np.exp(as_type("numpy", cuml_prob), dtype="float64"),
            ref_prob,
            rtol=tol,
            atol=tol,
            equal_nan=True,
        )
        assert np.allclose(
            np.exp(as_type("numpy", cuml_prob_test), dtype="float64"),
            ref_prob_test,
            rtol=tol,
            atol=tol,
            equal_nan=True,
        )

    if kernel in ["gaussian", "tophat"] and metric == "euclidean":
        sample = kde.sample(100, random_state=32).get()
        nearest = skl_pairwise_distances(sample, X_np, metric=metric)
        nearest = nearest.min(axis=1)
        if kernel == "gaussian":
            from scipy.stats import chi

            # The euclidean distance of each sample from its cluster
            # follows a chi distribution (not squared) with DoF=dimension
            # and scale = bandwidth
            # Fail the test if the largest observed distance
            # is vanishingly unlikely
            assert chi.sf(nearest.max(), X.shape[1], scale=bandwidth) > 1e-8
        elif kernel == "tophat":
            assert np.all(nearest <= bandwidth)
    else:
        with pytest.raises(
            NotImplementedError,
            match=r"Only \['gaussian', 'tophat'\] kernels,"
            " and the euclidean metric are supported.",
        ):
            kde.sample(100)


@pytest.mark.parametrize(
    "kernel",
    ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"],
)
@pytest.mark.parametrize("fit_dtype", ["float32", "float64"])
@pytest.mark.parametrize("score_dtype", ["float32", "float64"])
def test_score_samples_output_type_and_dtype(kernel, fit_dtype, score_dtype):
    """Check that the output dtype and type of `score_samples` is correct"""
    X, _ = make_blobs(n_samples=200, n_features=10, centers=5, random_state=42)
    X_train, X_score = X[:100], X[100:]
    X_train = X_train.astype(fit_dtype)
    X_score = X_score.astype(score_dtype)
    kde = KernelDensity(kernel=kernel).fit(X_train)
    res = kde.score_samples(X_score)
    assert res.dtype == fit_dtype
    assert isinstance(res, np.ndarray)
    with cuml.using_output_type("cupy"):
        res = kde.score_samples(X_score)
    assert res.dtype == fit_dtype
    assert isinstance(res, cp.ndarray)


def test_logaddexp_reduce():
    X = np.array([[0.0, 0.0], [0.0, 0.0]])
    out = logaddexp_reduce(cp.asarray(X), axis=1).get()
    assert np.allclose(out, np.logaddexp.reduce(X, axis=1))

    X = np.array([[3.0, 1.0], [0.2, 0.7]])
    out = logaddexp_reduce(cp.asarray(X), axis=1).get()
    assert np.allclose(out, np.logaddexp.reduce(X, axis=1))


def test_metric_params():
    X = np.array([[0.0, 1.0], [2.0, 0.5]])
    kde = KernelDensity(metric="minkowski", metric_params={"p": 1.0}).fit(X)
    kde2 = KernelDensity(metric="minkowski", metric_params={"p": 2.0}).fit(X)
    assert not np.allclose(kde.score_samples(X), kde2.score_samples(X))


def test_grid_search():
    rs = np.random.RandomState(3)
    X = rs.normal(size=(30, 5))
    params = {"bandwidth": np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(X)


def test_not_fitted():
    rs = np.random.RandomState(3)
    kde = KernelDensity()
    X = rs.normal(size=(30, 5))
    with pytest.raises(NotFittedError):
        kde.score(X)
    with pytest.raises(NotFittedError):
        kde.sample(X)
    with pytest.raises(NotFittedError):
        kde.score_samples(X)


def test_bad_sample_weight_errors():
    kde = KernelDensity()
    X = np.array([[0.0, 1.0], [2.0, 0.5]])

    with pytest.raises(ValueError, match="Expected 2 rows but got 3 rows."):
        kde.fit(X, sample_weight=np.array([1, 2, 3]))

    with pytest.raises(
        ValueError, match="Expected 1 columns but got 2 columns."
    ):
        kde.fit(X, sample_weight=np.array([[1, 2], [3, 4]]))
