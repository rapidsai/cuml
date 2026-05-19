# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from sklearn.covariance import EmpiricalCovariance


@pytest.fixture(scope="module")
def covariance_data():
    real_cov = np.array([[0.4, 0.2], [0.2, 0.8]])
    rng = np.random.RandomState(42)
    return rng.multivariate_normal(mean=[0, 0], cov=real_cov, size=200)


@pytest.mark.parametrize("store_precision", [True, False])
def test_empirical_covariance_store_precision(
    covariance_data, store_precision
):
    X = covariance_data
    cov = EmpiricalCovariance(store_precision=store_precision).fit(X)

    assert cov.covariance_.shape == (X.shape[1], X.shape[1])
    if store_precision:
        assert cov.precision_ is not None
        assert cov.precision_.shape == (X.shape[1], X.shape[1])
    else:
        assert cov.precision_ is None


@pytest.mark.parametrize("assume_centered", [True, False])
def test_empirical_covariance_assume_centered(
    covariance_data, assume_centered
):
    X = covariance_data
    cov = EmpiricalCovariance(assume_centered=assume_centered).fit(X)

    if assume_centered:
        np.testing.assert_array_equal(cov.location_, np.zeros(X.shape[1]))
    else:
        assert cov.location_.shape == (X.shape[1],)

    assert cov.covariance_.shape == (X.shape[1], X.shape[1])


def test_empirical_covariance_score(covariance_data):
    X = covariance_data
    cov = EmpiricalCovariance().fit(X)

    assert np.isfinite(cov.score(X))


def test_empirical_covariance_mahalanobis(covariance_data):
    X = covariance_data
    cov = EmpiricalCovariance().fit(X)
    mahal = cov.mahalanobis(X)

    assert mahal.shape == (X.shape[0],)
    assert np.all(mahal >= -1e-8)


def test_empirical_covariance_attributes(covariance_data):
    X = covariance_data
    cov = EmpiricalCovariance().fit(X)

    assert hasattr(cov, "covariance_")
    assert hasattr(cov, "location_")
    assert hasattr(cov, "precision_")
    assert cov.covariance_.shape == (X.shape[1], X.shape[1])
    assert cov.location_.shape == (X.shape[1],)
