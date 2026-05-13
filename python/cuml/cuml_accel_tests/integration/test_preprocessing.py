# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.preprocessing import (
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    PolynomialFeatures,
    StandardScaler,
)


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


def test_max_abs_scaler():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    model = MaxAbsScaler().fit(X)

    assert model.scale_.shape == (X.shape[1],)
    assert model.max_abs_.shape == (X.shape[1],)

    # Transform and check shape
    X_transformed = model.transform(X)
    assert X_transformed.shape == X.shape

    # Check that transformed data is scaled appropriately
    assert (np.abs(X_transformed) <= 1).all()

    # Check inverse transform
    X_inverse = model.inverse_transform(X_transformed)
    assert X_inverse.shape == X.shape
    np.testing.assert_allclose(X_inverse, X, atol=1e-6)


@pytest.mark.parametrize("cls", [StandardScaler, MinMaxScaler, MaxAbsScaler])
def test_scaler_partial_fit(cls):
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

    model = cls().fit(X)

    model2 = cls()
    model2.partial_fit(X[:25])
    assert model2.n_samples_seen_ == 25
    model2.partial_fit(X[25:])
    assert model2.n_samples_seen_ == X.shape[0]

    sol = model.transform(X)
    res = model2.transform(X)
    np.testing.assert_allclose(sol, res)


def test_polynomial_features():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

    model = PolynomialFeatures().fit(X)
    assert isinstance(model.powers_, np.ndarray)
    out = model.transform(X)
    assert isinstance(out, np.ndarray)

    model.set_output(transform="pandas")
    out_df = model.transform(X)
    assert isinstance(out_df, pd.DataFrame)


def test_label_encoder():
    y = np.array(["a", "b", "a", "b"])
    enc = LabelEncoder()
    y2 = enc.fit_transform(y)
    np.testing.assert_array_equal(y2, np.array([0, 1, 0, 1]))
    np.testing.assert_array_equal(enc.classes_, np.array(["a", "b"]))
    y3 = enc.inverse_transform(y2)
    np.testing.assert_array_equal(y3, y)
