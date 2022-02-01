from cuml.neighbors import KernelDensity, VALID_KERNELS, logsumexp_kernel
from sklearn.neighbors import KernelDensity as sklKernelDensity
import numpy as np
from hypothesis import given, settings, assume, strategies as st
from hypothesis.extra.numpy import arrays


@st.composite
def array_strategy(draw):
    n = draw(st.integers(1, 100))
    m = draw(st.integers(1, 100))
    dtype = draw(st.sampled_from([np.float64, np.float32]))
    X = draw(arrays(dtype=dtype, shape=(n, m),
             elements=st.floats(-5, 5, width=32),))
    X_test = draw(arrays(dtype=dtype, shape=(n, m),
                         elements=st.floats(-5, 5, width=32),))
    return X, X_test


metrics_strategy = st.sampled_from(
    ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'hamming', 'canberra'])


@settings(deadline=None)
@given(array_strategy(), st.sampled_from(VALID_KERNELS), metrics_strategy, st.floats(0.2, 5))
def test_kernel_density(arrays, kernel, metric, bandwidth):
    X, X_test = arrays
    if kernel == 'cosine':
        # cosine is numerically unstable at high dimensions
        # for both cuml and sklearn
        assume(X.shape[1] <= 20)
    kde = KernelDensity(kernel=kernel, metric=metric,
                        bandwidth=bandwidth).fit(X)
    skl_kde = sklKernelDensity(
        kernel=kernel, metric=metric, bandwidth=bandwidth).fit(X)
    # our algorithm returns float min instead of negative infinity for log(0) probability
    skl_prob = np.maximum(skl_kde.score_samples(X), np.finfo(X.dtype).min)
    skl_prob_test = np.maximum(skl_kde.score_samples(
        X_test), np.finfo(X.dtype).min)

    tol = 1e-2
    if X.dtype == np.float64:
        assert np.allclose(kde.score_samples(X), skl_prob,
                           rtol=tol, atol=tol, equal_nan=True)
        assert np.allclose(kde.score_samples(X_test),
                           skl_prob_test, rtol=tol, atol=tol, equal_nan=True)


def test_logaddexp():
    X = np.array([[0.0, 0.0],[0.0, 0.0]])
    out = np.zeros(X.shape[0])
    logsumexp_kernel.forall(out.size)(X, out)
    assert np.allclose(out, np.logaddexp.reduce(X, axis=1))

    X = np.array([[3.0, 1.0],[0.2, 0.7]])
    logsumexp_kernel.forall(out.size)(X, out)
    assert np.allclose(out, np.logaddexp.reduce(X, axis=1))