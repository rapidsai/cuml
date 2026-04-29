# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def test_standard_scaler():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    model = StandardScaler().fit(X)

    assert model.mean_.shape == (X.shape[1],)
    assert model.var_.shape == (X.shape[1],)
    assert model.scale_.shape == (X.shape[1],)

    # Transform and check shape
    X_transformed = model.transform(X)
    assert X_transformed.shape == X.shape

    # Check that transformed data has ~0 mean and ~1 std
    np.testing.assert_allclose(X_transformed.mean(axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(X_transformed.std(axis=0), 1, atol=1e-7)

    # Check inverse transform
    X_inverse = model.inverse_transform(X_transformed)
    assert X_inverse.shape == X.shape
    np.testing.assert_allclose(X_inverse, X, atol=1e-6)


def test_min_max_scaler():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    model = MinMaxScaler().fit(X)

    assert model.min_.shape == (X.shape[1],)
    assert model.scale_.shape == (X.shape[1],)
    assert model.data_min_.shape == (X.shape[1],)
    assert model.data_max_.shape == (X.shape[1],)
    assert model.data_range_.shape == (X.shape[1],)

    # Transform and check shape
    X_transformed = model.transform(X)
    assert X_transformed.shape == X.shape

    # Check that transformed data is scaled appropriately
    assert (X_transformed.min(axis=0) >= 0).all()
    assert (X_transformed.max(axis=0) <= 1).all()

    # Check inverse transform
    X_inverse = model.inverse_transform(X_transformed)
    assert X_inverse.shape == X.shape
    np.testing.assert_allclose(X_inverse, X, atol=1e-6)


@pytest.mark.parametrize("cls", [StandardScaler, MinMaxScaler])
def test_partial_fit(cls):
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

    model = StandardScaler().fit(X)

    model2 = StandardScaler()
    model2.partial_fit(X[:25])
    assert model2.n_samples_seen_ == 25
    model2.partial_fit(X[25:])
    assert model2.n_samples_seen_ == X.shape[0]

    sol = model.transform(X)
    res = model2.transform(X)
    np.testing.assert_allclose(sol, res)
