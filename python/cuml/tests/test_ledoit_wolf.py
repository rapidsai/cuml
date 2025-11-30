#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import numpy as np
import pytest
from sklearn.covariance import LedoitWolf as SklearnLedoitWolf

import cuml
from cuml.covariance import LedoitWolf
from cuml.testing.utils import as_type


@pytest.fixture
def random_data():
    """Generate random data for testing."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 100, 10
    X = rng.randn(n_samples, n_features)
    return X


@pytest.fixture
def correlated_data():
    """Generate correlated data where shrinkage is useful."""
    rng = np.random.RandomState(42)
    n_samples = 50
    # Create correlated features (20 features total: 4 blocks of 5)
    base = rng.randn(n_samples, 5)
    X = np.hstack([base + 0.1 * rng.randn(n_samples, 5) for _ in range(4)])
    return X


class TestLedoitWolfBasic:
    """Basic functionality tests."""

    def test_fit_returns_self(self, random_data):
        """Test that fit returns self."""
        lw = LedoitWolf()
        result = lw.fit(random_data)
        assert result is lw

    def test_fitted_attributes_exist(self, random_data):
        """Test that all fitted attributes are set after fit."""
        lw = LedoitWolf().fit(random_data)

        assert hasattr(lw, "covariance_")
        assert hasattr(lw, "location_")
        assert hasattr(lw, "precision_")
        assert hasattr(lw, "shrinkage_")
        assert hasattr(lw, "n_features_in_")

    def test_covariance_shape(self, random_data):
        """Test covariance matrix has correct shape."""
        n_features = random_data.shape[1]
        lw = LedoitWolf().fit(random_data)

        cov = np.asarray(lw.covariance_)
        assert cov.shape == (n_features, n_features)

    def test_location_shape(self, random_data):
        """Test location vector has correct shape."""
        n_features = random_data.shape[1]
        lw = LedoitWolf().fit(random_data)

        loc = np.asarray(lw.location_)
        assert loc.shape == (n_features,)

    def test_precision_shape(self, random_data):
        """Test precision matrix has correct shape."""
        n_features = random_data.shape[1]
        lw = LedoitWolf(store_precision=True).fit(random_data)

        prec = np.asarray(lw.precision_)
        assert prec.shape == (n_features, n_features)

    def test_shrinkage_in_valid_range(self, random_data):
        """Test that shrinkage is in [0, 1]."""
        lw = LedoitWolf().fit(random_data)
        assert 0 <= lw.shrinkage_ <= 1

    def test_covariance_is_symmetric(self, random_data):
        """Test that covariance matrix is symmetric."""
        lw = LedoitWolf().fit(random_data)
        cov = np.asarray(lw.covariance_)
        np.testing.assert_allclose(cov, cov.T, rtol=1e-5)

    def test_covariance_is_positive_definite(self, random_data):
        """Test that covariance matrix is positive definite."""
        lw = LedoitWolf().fit(random_data)
        cov = np.asarray(lw.covariance_)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)


class TestLedoitWolfVsSklearn:
    """Tests comparing cuML implementation to sklearn."""

    def test_covariance_matches_sklearn(self, random_data):
        """Test that covariance matches sklearn."""
        cu_lw = LedoitWolf().fit(random_data)
        sk_lw = SklearnLedoitWolf().fit(random_data)

        cu_cov = np.asarray(cu_lw.covariance_)
        sk_cov = sk_lw.covariance_

        np.testing.assert_allclose(cu_cov, sk_cov, rtol=1e-4, atol=1e-6)

    def test_location_matches_sklearn(self, random_data):
        """Test that location matches sklearn."""
        cu_lw = LedoitWolf().fit(random_data)
        sk_lw = SklearnLedoitWolf().fit(random_data)

        cu_loc = np.asarray(cu_lw.location_)
        sk_loc = sk_lw.location_

        np.testing.assert_allclose(cu_loc, sk_loc, rtol=1e-5)

    def test_shrinkage_matches_sklearn(self, random_data):
        """Test that shrinkage coefficient matches sklearn."""
        cu_lw = LedoitWolf().fit(random_data)
        sk_lw = SklearnLedoitWolf().fit(random_data)

        np.testing.assert_allclose(
            cu_lw.shrinkage_, sk_lw.shrinkage_, rtol=1e-4, atol=1e-6
        )

    def test_precision_matches_sklearn(self, random_data):
        """Test that precision matrix matches sklearn."""
        cu_lw = LedoitWolf(store_precision=True).fit(random_data)
        sk_lw = SklearnLedoitWolf(store_precision=True).fit(random_data)

        cu_prec = np.asarray(cu_lw.precision_)
        sk_prec = sk_lw.precision_

        np.testing.assert_allclose(cu_prec, sk_prec, rtol=1e-4, atol=1e-6)

    def test_score_matches_sklearn(self, random_data):
        """Test that score method matches sklearn."""
        cu_lw = LedoitWolf().fit(random_data)
        sk_lw = SklearnLedoitWolf().fit(random_data)

        # Score on training data
        cu_score = cu_lw.score(random_data)
        sk_score = sk_lw.score(random_data)

        np.testing.assert_allclose(cu_score, sk_score, rtol=1e-4)

    def test_correlated_data_matches_sklearn(self, correlated_data):
        """Test with correlated data where shrinkage is more important."""
        cu_lw = LedoitWolf().fit(correlated_data)
        sk_lw = SklearnLedoitWolf().fit(correlated_data)

        cu_cov = np.asarray(cu_lw.covariance_)
        sk_cov = sk_lw.covariance_

        np.testing.assert_allclose(cu_cov, sk_cov, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(
            cu_lw.shrinkage_, sk_lw.shrinkage_, rtol=1e-4, atol=1e-6
        )


class TestLedoitWolfParameters:
    """Tests for different parameter combinations."""

    def test_assume_centered_true(self, random_data):
        """Test assume_centered=True parameter."""
        # Center the data manually
        X_centered = random_data - random_data.mean(axis=0)

        cu_lw = LedoitWolf(assume_centered=True).fit(X_centered)
        sk_lw = SklearnLedoitWolf(assume_centered=True).fit(X_centered)

        cu_cov = np.asarray(cu_lw.covariance_)
        sk_cov = sk_lw.covariance_

        np.testing.assert_allclose(cu_cov, sk_cov, rtol=1e-4, atol=1e-6)

        # Location should be zeros
        cu_loc = np.asarray(cu_lw.location_)
        np.testing.assert_allclose(cu_loc, np.zeros_like(cu_loc))

    def test_store_precision_false(self, random_data):
        """Test store_precision=False parameter."""
        lw = LedoitWolf(store_precision=False).fit(random_data)

        # precision_ should be None
        assert lw.precision_ is None

        # get_precision() should still work
        prec = lw.get_precision()
        assert prec is not None

    def test_different_block_sizes(self, random_data):
        """Test that different block sizes give same results."""
        lw1 = LedoitWolf(block_size=10).fit(random_data)
        lw2 = LedoitWolf(block_size=100).fit(random_data)
        lw3 = LedoitWolf(block_size=1000).fit(random_data)

        cov1 = np.asarray(lw1.covariance_)
        cov2 = np.asarray(lw2.covariance_)
        cov3 = np.asarray(lw3.covariance_)

        np.testing.assert_allclose(cov1, cov2, rtol=1e-5)
        np.testing.assert_allclose(cov2, cov3, rtol=1e-5)


class TestLedoitWolfDtypes:
    """Tests for different data types."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtypes(self, random_data, dtype):
        """Test that estimator works with different dtypes."""
        X = random_data.astype(dtype)
        lw = LedoitWolf().fit(X)

        cov = np.asarray(lw.covariance_)
        assert cov.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtype_matches_sklearn(self, random_data, dtype):
        """Test dtype results match sklearn."""
        X = random_data.astype(dtype)

        cu_lw = LedoitWolf().fit(X)
        sk_lw = SklearnLedoitWolf().fit(X)

        cu_cov = np.asarray(cu_lw.covariance_)
        sk_cov = sk_lw.covariance_

        # Use appropriate tolerance for dtype
        rtol = 1e-4 if dtype == np.float64 else 1e-3
        np.testing.assert_allclose(cu_cov, sk_cov, rtol=rtol, atol=1e-6)


class TestLedoitWolfInputTypes:
    """Tests for different input types."""

    @pytest.mark.parametrize("input_type", ["numpy", "cupy", "cudf", "pandas"])
    def test_input_types(self, random_data, input_type):
        """Test that estimator works with different input types."""
        X = as_type(input_type, random_data)
        lw = LedoitWolf().fit(X)

        # Should be able to get covariance
        cov = lw.covariance_
        assert cov is not None

    def test_cupy_input(self, random_data):
        """Test with CuPy array input."""
        X_cp = cp.asarray(random_data)
        lw = LedoitWolf().fit(X_cp)

        cov = cp.asnumpy(cp.asarray(lw.covariance_))
        assert cov.shape == (random_data.shape[1], random_data.shape[1])


class TestLedoitWolfMethods:
    """Tests for individual methods."""

    def test_get_precision(self, random_data):
        """Test get_precision method."""
        lw = LedoitWolf(store_precision=True).fit(random_data)
        prec = lw.get_precision()

        # Should be the stored precision
        stored_prec = cp.asnumpy(cp.asarray(lw.precision_))
        returned_prec = cp.asnumpy(cp.asarray(prec))
        np.testing.assert_allclose(returned_prec, stored_prec)

    def test_get_precision_without_store(self, random_data):
        """Test get_precision when store_precision=False."""
        lw = LedoitWolf(store_precision=False).fit(random_data)
        prec = lw.get_precision()

        # Should compute precision on the fly
        cov = cp.asnumpy(cp.asarray(lw.covariance_))
        expected_prec = np.linalg.pinv(cov)
        returned_prec = cp.asnumpy(cp.asarray(prec))

        np.testing.assert_allclose(returned_prec, expected_prec, rtol=1e-5)

    def test_mahalanobis(self, random_data):
        """Test mahalanobis method."""
        lw = LedoitWolf().fit(random_data)
        mahal = lw.mahalanobis(random_data)

        mahal_np = np.asarray(mahal)
        assert mahal_np.shape == (random_data.shape[0],)
        assert np.all(mahal_np >= 0)  # Squared distances are non-negative

    def test_mahalanobis_matches_sklearn(self, random_data):
        """Test that mahalanobis matches sklearn."""
        cu_lw = LedoitWolf().fit(random_data)
        sk_lw = SklearnLedoitWolf().fit(random_data)

        cu_mahal = np.asarray(cu_lw.mahalanobis(random_data))
        sk_mahal = sk_lw.mahalanobis(random_data)

        np.testing.assert_allclose(cu_mahal, sk_mahal, rtol=1e-4)

    def test_error_norm_frobenius(self, random_data):
        """Test error_norm with Frobenius norm."""
        lw = LedoitWolf().fit(random_data)
        cov = np.asarray(lw.covariance_)

        # Error with itself should be 0
        error = lw.error_norm(cov)
        np.testing.assert_allclose(error, 0.0, atol=1e-10)

        # Error with different matrix should be > 0
        other_cov = cov + 0.1 * np.eye(cov.shape[0])
        error = lw.error_norm(other_cov)
        assert error > 0

    def test_error_norm_spectral(self, random_data):
        """Test error_norm with spectral norm."""
        lw = LedoitWolf().fit(random_data)
        cov = np.asarray(lw.covariance_)

        error = lw.error_norm(cov, norm="spectral")
        np.testing.assert_allclose(error, 0.0, atol=1e-10)

    def test_error_norm_invalid(self, random_data):
        """Test error_norm with invalid norm."""
        lw = LedoitWolf().fit(random_data)
        cov = np.asarray(lw.covariance_)

        with pytest.raises(ValueError, match="Invalid norm"):
            lw.error_norm(cov, norm="invalid")


class TestLedoitWolfEdgeCases:
    """Tests for edge cases."""

    def test_single_feature(self):
        """Test with single feature data."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 1)

        lw = LedoitWolf().fit(X)
        cov = np.asarray(lw.covariance_)

        assert cov.shape == (1, 1)
        assert lw.shrinkage_ == 0.0  # No shrinkage needed for 1 feature

    def test_two_features(self):
        """Test with two features."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)

        cu_lw = LedoitWolf().fit(X)
        sk_lw = SklearnLedoitWolf().fit(X)

        cu_cov = np.asarray(cu_lw.covariance_)
        sk_cov = sk_lw.covariance_

        np.testing.assert_allclose(cu_cov, sk_cov, rtol=1e-4, atol=1e-6)

    def test_more_features_than_samples(self):
        """Test with more features than samples (high-dimensional)."""
        rng = np.random.RandomState(42)
        X = rng.randn(20, 50)

        cu_lw = LedoitWolf().fit(X)
        sk_lw = SklearnLedoitWolf().fit(X)

        cu_cov = np.asarray(cu_lw.covariance_)
        sk_cov = sk_lw.covariance_

        np.testing.assert_allclose(cu_cov, sk_cov, rtol=1e-3, atol=1e-5)


class TestLedoitWolfOutputType:
    """Tests for output type handling."""

    def test_output_type_numpy(self, random_data):
        """Test numpy output type."""
        lw = LedoitWolf(output_type="numpy").fit(random_data)
        cov = lw.covariance_
        assert isinstance(cov, np.ndarray)

    def test_output_type_cupy(self, random_data):
        """Test cupy output type."""
        lw = LedoitWolf(output_type="cupy").fit(random_data)
        cov = lw.covariance_
        assert isinstance(cov, cp.ndarray)

    def test_using_output_type_context(self, random_data):
        """Test output type via context manager."""
        lw = LedoitWolf().fit(random_data)

        with cuml.using_output_type("cupy"):
            cov = lw.covariance_
        assert isinstance(cov, cp.ndarray)


class TestLedoitWolfInterop:
    """Tests for sklearn interoperability."""

    def test_from_sklearn(self, random_data):
        """Test conversion from sklearn model."""
        sk_lw = SklearnLedoitWolf().fit(random_data)
        cu_lw = LedoitWolf.from_sklearn(sk_lw)

        cu_cov = np.asarray(cu_lw.covariance_)
        sk_cov = sk_lw.covariance_

        np.testing.assert_allclose(cu_cov, sk_cov, rtol=1e-5)

    def test_as_sklearn(self, random_data):
        """Test conversion to sklearn model."""
        cu_lw = LedoitWolf().fit(random_data)
        sk_lw = cu_lw.as_sklearn()

        cu_cov = np.asarray(cu_lw.covariance_)
        sk_cov = sk_lw.covariance_

        np.testing.assert_allclose(cu_cov, sk_cov, rtol=1e-5)

    def test_get_params(self, random_data):
        """Test get_params method."""
        lw = LedoitWolf(
            store_precision=False, assume_centered=True, block_size=500
        )
        params = lw.get_params()

        assert params["store_precision"] is False
        assert params["assume_centered"] is True
        assert params["block_size"] == 500

    def test_set_params(self, random_data):
        """Test set_params method."""
        lw = LedoitWolf()
        lw.set_params(store_precision=False, block_size=500)

        assert lw.store_precision is False
        assert lw.block_size == 500
