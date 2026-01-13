#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np
import pytest
from sklearn.covariance import LedoitWolf as SklearnLedoitWolf

import cuml
from cuml.covariance import LedoitWolf
from cuml.testing.utils import as_type


def _make_random_data():
    """Generate random data for testing."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 100, 10
    return rng.randn(n_samples, n_features)


def _make_correlated_data():
    """Generate correlated data where shrinkage is useful."""
    rng = np.random.RandomState(42)
    n_samples = 50
    base = rng.randn(n_samples, 5)
    return np.hstack([base + 0.1 * rng.randn(n_samples, 5) for _ in range(4)])


# Basic functionality tests


def test_fit_returns_self():
    X = _make_random_data()
    lw = LedoitWolf()
    result = lw.fit(X)
    assert result is lw


def test_fitted_attributes_exist():
    X = _make_random_data()
    lw = LedoitWolf().fit(X)

    assert hasattr(lw, "covariance_")
    assert hasattr(lw, "location_")
    assert hasattr(lw, "precision_")
    assert hasattr(lw, "shrinkage_")
    assert hasattr(lw, "n_features_in_")


def test_covariance_shape():
    X = _make_random_data()
    n_features = X.shape[1]
    lw = LedoitWolf().fit(X)

    cov = np.asarray(lw.covariance_)
    assert cov.shape == (n_features, n_features)


def test_location_shape():
    X = _make_random_data()
    n_features = X.shape[1]
    lw = LedoitWolf().fit(X)

    loc = np.asarray(lw.location_)
    assert loc.shape == (n_features,)


def test_precision_shape():
    X = _make_random_data()
    n_features = X.shape[1]
    lw = LedoitWolf(store_precision=True).fit(X)

    prec = np.asarray(lw.precision_)
    assert prec.shape == (n_features, n_features)


def test_shrinkage_in_valid_range():
    X = _make_random_data()
    lw = LedoitWolf().fit(X)
    assert 0 <= lw.shrinkage_ <= 1


def test_covariance_is_symmetric():
    X = _make_random_data()
    lw = LedoitWolf().fit(X)
    cov = np.asarray(lw.covariance_)
    np.testing.assert_allclose(cov, cov.T, rtol=1e-5)


def test_covariance_is_positive_definite():
    X = _make_random_data()
    lw = LedoitWolf().fit(X)
    cov = np.asarray(lw.covariance_)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 0)


# Tests comparing cuML implementation to sklearn


def test_covariance_matches_sklearn():
    X = _make_random_data()
    cu_lw = LedoitWolf().fit(X)
    sk_lw = SklearnLedoitWolf().fit(X)

    cu_cov = np.asarray(cu_lw.covariance_)
    sk_cov = sk_lw.covariance_

    np.testing.assert_allclose(cu_cov, sk_cov, rtol=1e-4, atol=1e-6)


def test_location_matches_sklearn():
    X = _make_random_data()
    cu_lw = LedoitWolf().fit(X)
    sk_lw = SklearnLedoitWolf().fit(X)

    cu_loc = np.asarray(cu_lw.location_)
    sk_loc = sk_lw.location_

    np.testing.assert_allclose(cu_loc, sk_loc, rtol=1e-5)


def test_shrinkage_matches_sklearn():
    X = _make_random_data()
    cu_lw = LedoitWolf().fit(X)
    sk_lw = SklearnLedoitWolf().fit(X)

    np.testing.assert_allclose(
        cu_lw.shrinkage_, sk_lw.shrinkage_, rtol=1e-4, atol=1e-6
    )


def test_precision_matches_sklearn():
    X = _make_random_data()
    cu_lw = LedoitWolf(store_precision=True).fit(X)
    sk_lw = SklearnLedoitWolf(store_precision=True).fit(X)

    cu_prec = np.asarray(cu_lw.precision_)
    sk_prec = sk_lw.precision_

    np.testing.assert_allclose(cu_prec, sk_prec, rtol=1e-4, atol=1e-6)


def test_score_matches_sklearn():
    X = _make_random_data()
    cu_lw = LedoitWolf().fit(X)
    sk_lw = SklearnLedoitWolf().fit(X)

    cu_score = cu_lw.score(X)
    sk_score = sk_lw.score(X)

    np.testing.assert_allclose(cu_score, sk_score, rtol=1e-4)


def test_correlated_data_matches_sklearn():
    X = _make_correlated_data()
    cu_lw = LedoitWolf().fit(X)
    sk_lw = SklearnLedoitWolf().fit(X)

    cu_cov = np.asarray(cu_lw.covariance_)
    sk_cov = sk_lw.covariance_

    np.testing.assert_allclose(cu_cov, sk_cov, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(
        cu_lw.shrinkage_, sk_lw.shrinkage_, rtol=1e-4, atol=1e-6
    )


# Tests for different parameter combinations


def test_assume_centered_true():
    X = _make_random_data()
    X_centered = X - X.mean(axis=0)

    cu_lw = LedoitWolf(assume_centered=True).fit(X_centered)
    sk_lw = SklearnLedoitWolf(assume_centered=True).fit(X_centered)

    cu_cov = np.asarray(cu_lw.covariance_)
    sk_cov = sk_lw.covariance_

    np.testing.assert_allclose(cu_cov, sk_cov, rtol=1e-4, atol=1e-6)

    cu_loc = np.asarray(cu_lw.location_)
    np.testing.assert_allclose(cu_loc, np.zeros_like(cu_loc))


def test_store_precision_false():
    X = _make_random_data()
    lw = LedoitWolf(store_precision=False).fit(X)

    assert lw.precision_ is None

    prec = lw.get_precision()
    assert prec is not None


def test_different_block_sizes():
    X = _make_random_data()
    lw1 = LedoitWolf(block_size=10).fit(X)
    lw2 = LedoitWolf(block_size=100).fit(X)
    lw3 = LedoitWolf(block_size=1000).fit(X)

    cov1 = np.asarray(lw1.covariance_)
    cov2 = np.asarray(lw2.covariance_)
    cov3 = np.asarray(lw3.covariance_)

    np.testing.assert_allclose(cov1, cov2, rtol=1e-5)
    np.testing.assert_allclose(cov2, cov3, rtol=1e-5)


# Tests for different data types


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtypes(dtype):
    X = _make_random_data().astype(dtype)
    lw = LedoitWolf().fit(X)

    cov = np.asarray(lw.covariance_)
    assert cov.dtype == dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dtype_matches_sklearn(dtype):
    X = _make_random_data().astype(dtype)

    cu_lw = LedoitWolf().fit(X)
    sk_lw = SklearnLedoitWolf().fit(X)

    cu_cov = np.asarray(cu_lw.covariance_)
    sk_cov = sk_lw.covariance_

    rtol = 1e-4 if dtype == np.float64 else 1e-3
    np.testing.assert_allclose(cu_cov, sk_cov, rtol=rtol, atol=1e-6)


# Tests for different input types


@pytest.mark.parametrize("input_type", ["numpy", "cupy", "cudf", "pandas"])
def test_input_types(input_type):
    X = as_type(input_type, _make_random_data())
    lw = LedoitWolf().fit(X)

    cov = lw.covariance_
    assert cov is not None


def test_cupy_input():
    X = _make_random_data()
    X_cp = cp.asarray(X)
    lw = LedoitWolf().fit(X_cp)

    cov = cp.asnumpy(cp.asarray(lw.covariance_))
    assert cov.shape == (X.shape[1], X.shape[1])


# Tests for individual methods


def test_get_precision():
    X = _make_random_data()
    lw = LedoitWolf(store_precision=True).fit(X)
    prec = lw.get_precision()

    stored_prec = cp.asnumpy(cp.asarray(lw.precision_))
    returned_prec = cp.asnumpy(cp.asarray(prec))
    np.testing.assert_allclose(returned_prec, stored_prec)


def test_get_precision_without_store():
    X = _make_random_data()
    lw = LedoitWolf(store_precision=False).fit(X)
    prec = lw.get_precision()

    cov = cp.asnumpy(cp.asarray(lw.covariance_))
    expected_prec = np.linalg.pinv(cov)
    returned_prec = cp.asnumpy(cp.asarray(prec))

    np.testing.assert_allclose(returned_prec, expected_prec, rtol=1e-5)


def test_mahalanobis():
    X = _make_random_data()
    lw = LedoitWolf().fit(X)
    mahal = lw.mahalanobis(X)

    mahal_np = np.asarray(mahal)
    assert mahal_np.shape == (X.shape[0],)
    assert np.all(mahal_np >= 0)


def test_mahalanobis_matches_sklearn():
    X = _make_random_data()
    cu_lw = LedoitWolf().fit(X)
    sk_lw = SklearnLedoitWolf().fit(X)

    cu_mahal = np.asarray(cu_lw.mahalanobis(X))
    sk_mahal = sk_lw.mahalanobis(X)

    np.testing.assert_allclose(cu_mahal, sk_mahal, rtol=1e-4)


def test_error_norm_frobenius():
    X = _make_random_data()
    lw = LedoitWolf().fit(X)
    cov = np.asarray(lw.covariance_)

    error = lw.error_norm(cov)
    np.testing.assert_allclose(error, 0.0, atol=1e-10)

    other_cov = cov + 0.1 * np.eye(cov.shape[0])
    error = lw.error_norm(other_cov)
    assert error > 0


def test_error_norm_spectral():
    X = _make_random_data()
    lw = LedoitWolf().fit(X)
    cov = np.asarray(lw.covariance_)

    error = lw.error_norm(cov, norm="spectral")
    np.testing.assert_allclose(error, 0.0, atol=1e-10)


def test_error_norm_invalid():
    X = _make_random_data()
    lw = LedoitWolf().fit(X)
    cov = np.asarray(lw.covariance_)

    with pytest.raises(ValueError, match="Invalid norm"):
        lw.error_norm(cov, norm="invalid")


# Tests for edge cases


def test_single_feature():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 1)

    lw = LedoitWolf().fit(X)
    cov = np.asarray(lw.covariance_)

    assert cov.shape == (1, 1)
    assert lw.shrinkage_ == 0.0


def test_two_features():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 2)

    cu_lw = LedoitWolf().fit(X)
    sk_lw = SklearnLedoitWolf().fit(X)

    cu_cov = np.asarray(cu_lw.covariance_)
    sk_cov = sk_lw.covariance_

    np.testing.assert_allclose(cu_cov, sk_cov, rtol=1e-4, atol=1e-6)


def test_more_features_than_samples():
    rng = np.random.RandomState(42)
    X = rng.randn(20, 50)

    cu_lw = LedoitWolf().fit(X)
    sk_lw = SklearnLedoitWolf().fit(X)

    cu_cov = np.asarray(cu_lw.covariance_)
    sk_cov = sk_lw.covariance_

    np.testing.assert_allclose(cu_cov, sk_cov, rtol=1e-3, atol=1e-5)


# Tests for output type handling


def test_output_type_numpy():
    X = _make_random_data()
    lw = LedoitWolf(output_type="numpy").fit(X)
    cov = lw.covariance_
    assert isinstance(cov, np.ndarray)


def test_output_type_cupy():
    X = _make_random_data()
    lw = LedoitWolf(output_type="cupy").fit(X)
    cov = lw.covariance_
    assert isinstance(cov, cp.ndarray)


def test_using_output_type_context():
    X = _make_random_data()
    lw = LedoitWolf().fit(X)

    with cuml.using_output_type("cupy"):
        cov = lw.covariance_
    assert isinstance(cov, cp.ndarray)


# Tests for sklearn interoperability


def test_from_sklearn():
    X = _make_random_data()
    sk_lw = SklearnLedoitWolf().fit(X)
    cu_lw = LedoitWolf.from_sklearn(sk_lw)

    cu_cov = np.asarray(cu_lw.covariance_)
    sk_cov = sk_lw.covariance_

    np.testing.assert_allclose(cu_cov, sk_cov, rtol=1e-5)


def test_as_sklearn():
    X = _make_random_data()
    cu_lw = LedoitWolf().fit(X)
    sk_lw = cu_lw.as_sklearn()

    cu_cov = np.asarray(cu_lw.covariance_)
    sk_cov = sk_lw.covariance_

    np.testing.assert_allclose(cu_cov, sk_cov, rtol=1e-5)


def test_get_params():
    lw = LedoitWolf(
        store_precision=False, assume_centered=True, block_size=500
    )
    params = lw.get_params()

    assert params["store_precision"] is False
    assert params["assume_centered"] is True
    assert params["block_size"] == 500


def test_set_params():
    lw = LedoitWolf()
    lw.set_params(store_precision=False, block_size=500)

    assert lw.store_precision is False
    assert lw.block_size == 500
