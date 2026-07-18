#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import warnings

import cupy as cp
import numpy as np
import pytest
import scipy.special
import sklearn.neighbors
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.datasets import make_blobs
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.metrics import pairwise_distances as skl_pairwise_distances
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors._ball_tree import kernel_norm

import cuml
from cuml.neighbors import VALID_KERNELS, KernelDensity
from cuml.testing.utils import as_numpy, as_type


def _cosine_kernel_norm(h, d):
    """Normalization constant for cosine kernel in d dimensions.

    Matches the recurrence in cuvs::distance for DensityKernelType::Cosine:
      I_0 = 2/pi
      I_1 = 2/pi - (2/pi)^2
      I_n = 2/pi - n*(n-1)*(2/pi)^2 * I_{n-2}  for n >= 2
      norm = 1 / (S_{d-1} * I_{d-1} * h^d)
    where S_{d-1} = 2*pi^(d/2) / Gamma(d/2).

    sklearn's kernel_norm returns NaN for cosine at d >= 4 due to a bug in
    its integration-by-parts formula, so we use this custom implementation.
    """
    two_over_pi = 2.0 / np.pi
    two_over_pi_sq = two_over_pi**2
    I_prev = two_over_pi  # I_0
    I_curr = two_over_pi - two_over_pi_sq  # I_1
    n = d - 1
    if n == 0:
        integral = I_prev
    else:
        for j in range(2, n + 1):
            I_next = two_over_pi - j * (j - 1) * two_over_pi_sq * I_prev
            I_prev, I_curr = I_curr, I_next
        integral = I_curr
    Sn = 2.0 * np.pi ** (d / 2.0) / scipy.special.gamma(d / 2.0)
    return 1.0 / (Sn * integral * h**d)


# not in log probability space
def compute_kernel_naive(Y, X, kernel, metric, h, sample_weight):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataConversionWarning)
        d = skl_pairwise_distances(Y, X, metric)
    if kernel == "cosine":
        norm = _cosine_kernel_norm(h, X.shape[1])
    else:
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
        k = np.maximum(np.cos(0.5 * np.pi * d / h), 1e-30) * (d < h)
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
    arrays=(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.5, 2.5]]), None),
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
    X_np, X_test_np, sample_weight_np = as_numpy(*arrays)

    if kernel == "cosine":
        # cosine is numerically unstable at high dimensions for both cuml
        # and the numpy reference.
        assume(X.shape[1] <= 15)
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
            np.exp(as_numpy(cuml_prob), dtype="float64"),
            ref_prob,
            rtol=tol,
            atol=tol,
            equal_nan=True,
        )
        assert np.allclose(
            np.exp(as_numpy(cuml_prob_test), dtype="float64"),
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
    elif kernel not in ["gaussian", "tophat"]:
        with pytest.raises(
            NotImplementedError,
            match=r"Only \['gaussian', 'tophat'\] kernels are supported.",
        ):
            kde.sample(100)


@pytest.mark.parametrize("bandwidth", ["scott", "silverman", 5.0])
@pytest.mark.parametrize("n_rows, n_cols", [(13, 17), (17, 13)])
def test_bandwidth(bandwidth, n_rows, n_cols):
    X, _ = make_blobs(n_samples=n_rows, n_features=n_cols)
    cu_model = cuml.KernelDensity(bandwidth=bandwidth).fit(X)
    sk_model = sklearn.neighbors.KernelDensity(bandwidth=bandwidth).fit(X)
    assert cu_model.bandwidth_ == sk_model.bandwidth_


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

    with pytest.raises(
        ValueError,
        match="inconsistent number of samples",
    ):
        kde.fit(X, sample_weight=np.array([1, 2, 3]))

    with pytest.raises(
        ValueError, match="Sample weights must be 1D array or scalar"
    ):
        kde.fit(X, sample_weight=np.array([[1, 2], [3, 4]]))


# -----------------------------------------------------------------------------
# Reference pairwise distances for metrics absent from sklearn.pairwise
# (must match the corresponding DistOp accumulate/finalize in kde.cu exactly)
# -----------------------------------------------------------------------------


def _hellinger_dist(X, Y):
    """sqrt(max(0, 1 - sum sqrt(xi * yi))) - matches DistOp<HellingerExpanded>."""
    sx = np.sqrt(np.maximum(X, 0.0))
    sy = np.sqrt(np.maximum(Y, 0.0))
    return np.sqrt(np.maximum(1.0 - sx @ sy.T, 0.0))


def _jensenshannon_dist(X, Y):
    """sqrt(0.5 * sum(a * log(a/m) + b * log(b/m))) - matches DistOp<JensenShannon>."""
    out = np.zeros((len(X), len(Y)))
    for i, a in enumerate(X):
        for j, b in enumerate(Y):
            m = 0.5 * (a + b)
            # Mirror device guards: log(0) -> 0
            logM = np.where(m > 0, np.log(np.where(m > 0, m, 1.0)), 0.0)
            logA = np.where(a > 0, np.log(np.where(a > 0, a, 1.0)), 0.0)
            logB = np.where(b > 0, np.log(np.where(b > 0, b, 1.0)), 0.0)
            acc = np.sum(-a * (logM - logA) + -b * (logM - logB))
            out[i, j] = np.sqrt(0.5 * max(float(acc), 0.0))
    return out


def _kldivergence_dist(X, Y):
    """sum a * log(a/b) for a,b > 0 - matches DistOp<KLDivergence>."""
    out = np.zeros((len(X), len(Y)))
    for i, a in enumerate(X):
        for j, b in enumerate(Y):
            mask = (a > 0) & (b > 0)
            out[i, j] = float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return out


def _kde_naive_custom(Y, X, kernel, dist_fn, h, sample_weight):
    """Like compute_kernel_naive but accepts a callable pairwise distance."""
    d = dist_fn(Y, X)
    if kernel == "cosine":
        norm = _cosine_kernel_norm(h, X.shape[1])
    else:
        norm = kernel_norm(h, X.shape[1], kernel)
    if kernel == "gaussian":
        k = np.exp(-0.5 * d * d / (h * h))
    elif kernel == "tophat":
        k = (d < h).astype(float)
    elif kernel == "epanechnikov":
        k = np.maximum(1.0 - d * d / (h * h), 0.0) * (d < h)
    elif kernel == "exponential":
        k = np.exp(-d / h)
    elif kernel == "linear":
        k = np.maximum(1.0 - d / h, 0.0) * (d < h)
    elif kernel == "cosine":
        k = np.maximum(np.cos(0.5 * np.pi * d / h), 1e-30) * (d < h)
    else:
        raise ValueError(kernel)
    return norm * np.average(k, axis=1, weights=sample_weight)


# Custom distance functions for metrics not in sklearn.pairwise_distances
_CUSTOM_DIST_FN = {
    "hellinger": _hellinger_dist,
    "jensenshannon": _jensenshannon_dist,
    "kldivergence": _kldivergence_dist,
}

# Metrics that require non-negative inputs
_NONNEG_METRICS = {"hellinger", "jensenshannon"}
# Metrics that require strictly positive inputs
_POSONLY_METRICS = {"kldivergence"}
# Metrics defined for binary {0,1} inputs (our DistOp matches sklearn only for binary)
_BINARY_METRICS = {"russellrao"}


def _make_metric_data(metric, n_train=40, n_query=8, d=4, seed=7):
    """Generate float64 test data appropriate for the given metric."""
    rng = np.random.RandomState(seed)
    if metric in _BINARY_METRICS:
        X = rng.randint(0, 2, size=(n_train, d)).astype(np.float64)
        Q = rng.randint(0, 2, size=(n_query, d)).astype(np.float64)
    elif metric in _POSONLY_METRICS:
        X = (
            rng.exponential(scale=1.0, size=(n_train, d)).astype(np.float64)
            + 0.1
        )
        Q = (
            rng.exponential(scale=1.0, size=(n_query, d)).astype(np.float64)
            + 0.1
        )
    elif metric in _NONNEG_METRICS:
        X = np.abs(rng.randn(n_train, d)).astype(np.float64) + 0.05
        Q = np.abs(rng.randn(n_query, d)).astype(np.float64) + 0.05
    else:
        X = rng.randn(n_train, d).astype(np.float64)
        Q = rng.randn(n_query, d).astype(np.float64)
    return X, Q


@pytest.mark.parametrize("kernel", VALID_KERNELS)
@pytest.mark.parametrize(
    "metric",
    [
        # sklearn-pairwise-compatible
        "euclidean",
        "manhattan",
        "chebyshev",
        "minkowski",
        "sqeuclidean",
        "canberra",
        "hamming",
        "cosine",
        "correlation",
        "russellrao",
        # custom reference required
        "hellinger",
        "jensenshannon",
        "kldivergence",
    ],
)
def test_all_kernels_all_metrics(metric, kernel):
    """Every metric x kernel combination produces output matching the reference.

    For metrics supported by sklearn.pairwise_distances the reference is
    compute_kernel_naive; for metrics absent from sklearn a matching numpy
    reference is used that mirrors the DistOp accumulate/finalize logic in
    kde.cu exactly.
    """
    X, Q = _make_metric_data(metric)
    h = 1.0

    kde = KernelDensity(kernel=kernel, metric=metric, bandwidth=h)
    kde.fit(X)
    cuml_log = as_numpy(kde.score_samples(Q))

    # -inf is valid (zero density when all train points are beyond the bandwidth);
    # only NaN indicates a real bug.
    assert not np.any(np.isnan(cuml_log)), (
        f"NaN output for metric={metric}, kernel={kernel}"
    )

    dist_fn = _CUSTOM_DIST_FN.get(metric)
    if dist_fn is not None:
        ref = _kde_naive_custom(Q, X, kernel, dist_fn, h, None)
    else:
        ref = compute_kernel_naive(Q, X, kernel, metric, h, None)

    # exp(-inf) == 0 == reference density, so this naturally handles the
    # all-zero-density case (compact-support kernels with small bandwidth).
    cuml_prob = np.exp(cuml_log)
    assert np.allclose(cuml_prob, ref, rtol=1e-3, atol=1e-3, equal_nan=True), (
        f"metric={metric}, kernel={kernel}: max err="
        f"{np.max(np.abs(cuml_prob - ref)):.4e}"
    )


def test_nan_euclidean_not_supported():
    """metric='nan_euclidean' is rejected with a clear error.

    The fused KDE kernel has no NaN-aware path, so we raise instead of
    silently producing wrong results.
    """
    rng = np.random.RandomState(0)
    X = rng.randn(20, 3).astype(np.float64)
    kde = KernelDensity(metric="nan_euclidean")
    with pytest.raises(NotImplementedError, match="nan_euclidean"):
        kde.fit(X)


def test_russellrao_coerces_non_binary_with_warning():
    """Non-binary inputs to metric='russellrao' are coerced to {0, 1} with a warning.

    Mirrors the long-standing behavior of cuml.metrics.pairwise_distances for
    this metric. Output must match a reference computed on the coerced data.
    """
    rng = np.random.RandomState(0)
    X = rng.uniform(-1.0, 2.0, size=(30, 4)).astype(np.float64)
    Q = rng.uniform(-1.0, 2.0, size=(5, 4)).astype(np.float64)

    kde = KernelDensity(kernel="gaussian", metric="russellrao", bandwidth=1.0)
    with pytest.warns(DataConversionWarning, match="converted to boolean"):
        kde.fit(X)
    with pytest.warns(DataConversionWarning, match="converted to boolean"):
        cuml_log = as_numpy(kde.score_samples(Q))

    X_bin = np.where(X != 0.0, 1.0, 0.0)
    Q_bin = np.where(Q != 0.0, 1.0, 0.0)
    ref = compute_kernel_naive(
        Q_bin, X_bin, "gaussian", "russellrao", 1.0, None
    )
    assert np.allclose(np.exp(cuml_log), ref, rtol=1e-4, atol=1e-4)


def test_russellrao_binary_no_warning():
    """Already-binary inputs to metric='russellrao' do not trigger a warning."""
    rng = np.random.RandomState(0)
    X = rng.randint(0, 2, size=(30, 4)).astype(np.float64)
    Q = rng.randint(0, 2, size=(5, 4)).astype(np.float64)
    kde = KernelDensity(kernel="gaussian", metric="russellrao", bandwidth=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataConversionWarning)
        kde.fit(X)
        kde.score_samples(Q)


def test_tiling_multipass():
    """Multi-pass tiling path (small n_query, large n_train) matches reference.

    When n_query is small enough that the 2-D grid / multi-pass reduction
    code path is taken the result must match the naive single-pass reference.
    """
    rng = np.random.RandomState(0)
    X_train = rng.randn(2000, 4).astype(np.float64)
    X_query = rng.randn(2, 4).astype(np.float64)

    kde = KernelDensity(kernel="gaussian", metric="euclidean", bandwidth=0.5)
    kde.fit(X_train)
    cuml_scores = as_numpy(kde.score_samples(X_query))

    ref = compute_kernel_naive(
        X_query, X_train, "gaussian", "euclidean", 0.5, None
    )
    assert np.allclose(np.exp(cuml_scores), ref, rtol=1e-3, atol=1e-3)
