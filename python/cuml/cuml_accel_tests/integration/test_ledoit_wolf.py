# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf


@pytest.fixture(scope="module")
def covariance_data():
    real_cov = np.array([[0.4, 0.2], [0.2, 0.8]])
    rng = np.random.RandomState(42)
    X = rng.multivariate_normal(mean=[0, 0], cov=real_cov, size=200)
    return X


@pytest.mark.parametrize("store_precision", [True, False])
def test_ledoit_wolf_store_precision(covariance_data, store_precision):
    X = covariance_data
    lw = LedoitWolf(store_precision=store_precision).fit(X)
    # Covariance should always be computed
    assert lw.covariance_.shape == (X.shape[1], X.shape[1])
    if store_precision:
        assert lw.precision_ is not None
        assert lw.precision_.shape == (X.shape[1], X.shape[1])
    else:
        assert lw.precision_ is None


@pytest.mark.parametrize("assume_centered", [True, False])
def test_ledoit_wolf_assume_centered(covariance_data, assume_centered):
    X = covariance_data
    if assume_centered:
        X_centered = X - X.mean(axis=0)
        lw = LedoitWolf(assume_centered=True).fit(X_centered)
        # Location should be zeros when assume_centered=True
        np.testing.assert_array_equal(lw.location_, np.zeros(X.shape[1]))
    else:
        lw = LedoitWolf(assume_centered=False).fit(X)
        # Location should be the mean of the data
        assert lw.location_.shape == (X.shape[1],)
    # Shrinkage should be in valid range
    assert 0 <= lw.shrinkage_ <= 1


@pytest.mark.parametrize("block_size", [500, 1000, 2000])
def test_ledoit_wolf_block_size(covariance_data, block_size):
    X = covariance_data
    lw = LedoitWolf(block_size=block_size).fit(X)
    # block_size is a memory optimization, results should be consistent
    assert lw.covariance_.shape == (X.shape[1], X.shape[1])
    assert 0 <= lw.shrinkage_ <= 1


def test_ledoit_wolf_score(covariance_data):
    X = covariance_data
    lw = LedoitWolf().fit(X)
    score = lw.score(X)
    # Score is log-likelihood, should be a finite number
    assert np.isfinite(score)


def test_ledoit_wolf_attributes(covariance_data):
    X = covariance_data
    lw = LedoitWolf().fit(X)
    # Check all expected attributes exist and have correct shapes
    assert hasattr(lw, "covariance_")
    assert hasattr(lw, "location_")
    assert hasattr(lw, "shrinkage_")
    assert lw.covariance_.shape == (X.shape[1], X.shape[1])
    assert lw.location_.shape == (X.shape[1],)
    assert isinstance(lw.shrinkage_, float)
