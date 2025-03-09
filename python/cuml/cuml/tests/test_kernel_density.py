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

from cuml.testing.utils import as_type
from sklearn.model_selection import GridSearchCV
import pytest
from hypothesis.extra.numpy import arrays
from hypothesis import example, given, settings, assume, strategies as st
from cuml.neighbors import KernelDensity, VALID_KERNELS, logsumexp_kernel
from cuml.common.exceptions import NotFittedError
from sklearn.metrics import pairwise_distances as skl_pairwise_distances
from sklearn.neighbors._ball_tree import kernel_norm
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")


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
    kde = KernelDensity(kernel=kernel, metric=metric, bandwidth=bandwidth)
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
            np.exp(as_type("numpy", cuml_prob)),
            ref_prob,
            rtol=tol,
            atol=tol,
            equal_nan=True,
        )
        assert np.allclose(
            np.exp(as_type("numpy", cuml_prob_test)),
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


def test_logaddexp():
    X = np.array([[0.0, 0.0], [0.0, 0.0]])
    out = np.zeros(X.shape[0])
    logsumexp_kernel.forall(out.size)(X, out)
    assert np.allclose(out, np.logaddexp.reduce(X, axis=1))

    X = np.array([[3.0, 1.0], [0.2, 0.7]])
    logsumexp_kernel.forall(out.size)(X, out)
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
