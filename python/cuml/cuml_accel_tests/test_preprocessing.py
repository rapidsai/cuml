# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from sklearn.datasets import make_blobs
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    StandardScaler,
    TargetEncoder,
)

from cuml.accel.core import logger

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_log_level():
    """Reset logger back to warn after every test."""
    yield
    logger.set_level("warn")


@pytest.fixture
def get_logs(capsys):
    """Return a callable that yields the cuml.accel debug lines captured so far."""

    def _get_logs():
        out, _ = capsys.readouterr()
        return [
            line for line in out.split("\n") if line.startswith("[cuml.accel]")
        ]

    return _get_logs


@pytest.fixture
def X_float64():
    X, _ = make_blobs(n_samples=200, n_features=5, random_state=42)
    return X.astype(np.float64)


# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_standard_scaler_fit_transform(dtype, get_logs):
    logger.set_level("debug")
    X, _ = make_blobs(n_samples=200, n_features=5, random_state=42)
    X = X.astype(dtype)

    scaler = StandardScaler()
    X_tr = scaler.fit(X).transform(X)

    assert X_tr.shape == X.shape
    assert isinstance(X_tr, np.ndarray)
    np.testing.assert_allclose(X_tr.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(X_tr.std(axis=0), 1, atol=1e-5)

    logs = get_logs()
    assert any(
        "StandardScaler.fit` input data moved to GPU" in line for line in logs
    )


def test_standard_scaler_inverse_transform(X_float64):
    scaler = StandardScaler().fit(X_float64)
    X_tr = scaler.transform(X_float64)
    X_back = scaler.inverse_transform(X_tr)
    np.testing.assert_allclose(X_back, X_float64, atol=1e-6)


def test_standard_scaler_fit_transform_matches_fit_then_transform(X_float64):
    X = X_float64
    scaler1 = StandardScaler()
    X_ft = scaler1.fit_transform(X)

    scaler2 = StandardScaler()
    X_expected = scaler2.fit(X).transform(X)

    np.testing.assert_allclose(X_ft, X_expected, atol=1e-10)


def test_standard_scaler_partial_fit(X_float64):
    X = X_float64
    half = len(X) // 2

    scaler_full = StandardScaler().fit(X)

    scaler_partial = StandardScaler()
    scaler_partial.partial_fit(X[:half])
    scaler_partial.partial_fit(X[half:])

    np.testing.assert_allclose(
        scaler_partial.mean_, scaler_full.mean_, atol=1e-6
    )
    np.testing.assert_allclose(
        scaler_partial.var_, scaler_full.var_, atol=1e-6
    )


def test_standard_scaler_fallback_sample_weight(X_float64, get_logs):
    logger.set_level("debug")
    X = X_float64
    w = np.ones(len(X))
    scaler = StandardScaler()
    scaler.fit(X, sample_weight=w)
    X_tr = scaler.transform(X)

    assert X_tr.shape == X.shape
    logs = get_logs()
    assert any("not optimized: unsupported input" in line for line in logs)


def test_standard_scaler_fallback_dataframe(X_float64, get_logs):
    logger.set_level("debug")
    df = pd.DataFrame(X_float64)

    scaler_cpu = StandardScaler().fit(X_float64)
    X_expected = scaler_cpu.transform(X_float64)

    scaler = StandardScaler()
    scaler.fit(df)
    X_tr = scaler.transform(df)

    assert X_tr.shape == X_float64.shape
    np.testing.assert_allclose(X_tr, X_expected, atol=1e-10)
    logs = get_logs()
    assert any("not optimized: unsupported input" in line for line in logs)


def test_standard_scaler_fallback_sparse(X_float64, get_logs):
    logger.set_level("debug")
    X_sparse = scipy.sparse.csr_matrix(X_float64)

    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_sparse)
    X_tr = scaler.transform(X_sparse)

    assert X_tr.shape == X_float64.shape
    logs = get_logs()
    assert any("not optimized: unsupported input" in line for line in logs)


def test_standard_scaler_fallback_float16(get_logs):
    logger.set_level("debug")
    X, _ = make_blobs(n_samples=100, n_features=4, random_state=42)
    X = X.astype(np.float16)

    scaler = StandardScaler()
    scaler.fit(X)
    X_tr = scaler.transform(X)

    assert X_tr.shape == X.shape
    logs = get_logs()
    assert any("not optimized: unsupported input" in line for line in logs)


def test_standard_scaler_fallback_set_output_pandas(X_float64, get_logs):
    logger.set_level("debug")
    scaler = StandardScaler().set_output(transform="pandas")
    scaler.fit(X_float64)
    X_tr = scaler.transform(X_float64)

    assert isinstance(X_tr, pd.DataFrame)
    logs = get_logs()
    assert any(
        "not optimized: non-default output container" in line for line in logs
    )


# ---------------------------------------------------------------------------
# MinMaxScaler
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("feature_range", [(0, 1), (-1, 1)])
def test_min_max_scaler_fit_transform(feature_range, X_float64, get_logs):
    logger.set_level("debug")
    X = X_float64

    scaler = MinMaxScaler(feature_range=feature_range)
    X_tr = scaler.fit(X).transform(X)

    assert X_tr.shape == X.shape
    assert isinstance(X_tr, np.ndarray)
    # Fitted data should reach the range boundaries
    np.testing.assert_allclose(X_tr.min(axis=0), feature_range[0], atol=1e-6)
    np.testing.assert_allclose(X_tr.max(axis=0), feature_range[1], atol=1e-6)

    logs = get_logs()
    assert any(
        "MinMaxScaler.fit` input data moved to GPU" in line for line in logs
    )


def test_min_max_scaler_inverse_transform(X_float64):
    X = X_float64
    scaler = MinMaxScaler().fit(X)
    X_tr = scaler.transform(X)
    X_back = scaler.inverse_transform(X_tr)
    np.testing.assert_allclose(X_back, X, atol=1e-6)


def test_min_max_scaler_fit_transform_method(X_float64):
    X = X_float64
    scaler1 = MinMaxScaler()
    X_ft = scaler1.fit_transform(X)

    scaler2 = MinMaxScaler()
    X_expected = scaler2.fit(X).transform(X)

    np.testing.assert_allclose(X_ft, X_expected, atol=1e-10)


def test_min_max_scaler_partial_fit(X_float64):
    X = X_float64
    half = len(X) // 2

    scaler_full = MinMaxScaler().fit(X)

    scaler_partial = MinMaxScaler()
    scaler_partial.partial_fit(X[:half])
    scaler_partial.partial_fit(X[half:])

    np.testing.assert_allclose(
        scaler_partial.data_min_, scaler_full.data_min_, atol=1e-10
    )
    np.testing.assert_allclose(
        scaler_partial.data_max_, scaler_full.data_max_, atol=1e-10
    )


def test_min_max_scaler_fallback_dataframe(X_float64, get_logs):
    logger.set_level("debug")
    df = pd.DataFrame(X_float64)

    scaler_cpu = MinMaxScaler().fit(X_float64)
    X_expected = scaler_cpu.transform(X_float64)

    scaler = MinMaxScaler()
    scaler.fit(df)
    X_tr = scaler.transform(df)

    assert X_tr.shape == X_float64.shape
    np.testing.assert_allclose(X_tr, X_expected, atol=1e-10)
    logs = get_logs()
    assert any("not optimized: unsupported input" in line for line in logs)


def test_min_max_scaler_fallback_set_output_pandas(X_float64, get_logs):
    logger.set_level("debug")
    scaler = MinMaxScaler().set_output(transform="pandas")
    scaler.fit(X_float64)
    X_tr = scaler.transform(X_float64)

    assert isinstance(X_tr, pd.DataFrame)
    logs = get_logs()
    assert any(
        "not optimized: non-default output container" in line for line in logs
    )


# ---------------------------------------------------------------------------
# LabelEncoder
# ---------------------------------------------------------------------------


def test_label_encoder_fit_transform(get_logs):
    logger.set_level("debug")
    y = np.array([3, 1, 2, 1, 3, 2, 0, 0])

    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)

    assert y_enc.shape == y.shape
    assert isinstance(y_enc, np.ndarray)
    np.testing.assert_array_equal(enc.classes_, np.array([0, 1, 2, 3]))

    logs = get_logs()
    assert any(
        "LabelEncoder.fit_transform` input data moved to GPU" in line
        for line in logs
    )


def test_label_encoder_fit_then_transform(get_logs):
    logger.set_level("debug")
    y = np.array([10, 20, 30, 10, 20])

    enc = LabelEncoder()
    enc.fit(y)
    y_enc = enc.transform(y)

    assert y_enc.shape == y.shape
    np.testing.assert_array_equal(np.sort(np.unique(y_enc)), [0, 1, 2])
    logs = get_logs()
    assert any(
        "LabelEncoder.transform` input data moved to GPU" in line
        for line in logs
    )


def test_label_encoder_inverse_transform():
    y = np.array([0, 1, 2, 1, 0, 2])
    enc = LabelEncoder().fit(y)
    y_enc = enc.transform(y)
    y_back = enc.inverse_transform(y_enc)
    np.testing.assert_array_equal(y_back, y)


def test_label_encoder_fallback_string_y(get_logs):
    logger.set_level("debug")
    y = np.array(["cat", "dog", "cat", "fish", "dog"])

    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)

    assert y_enc.shape == y.shape
    np.testing.assert_array_equal(enc.classes_, ["cat", "dog", "fish"])
    logs = get_logs()
    assert any("not optimized: unsupported input" in line for line in logs)


def test_label_encoder_fallback_string_fitted_transform(get_logs):
    """Encoder fit on strings falls back for any transform because classes_.dtype is non-numeric."""
    logger.set_level("debug")
    y_strings = np.array(["a", "b", "c", "a", "b"])
    enc = LabelEncoder().fit(y_strings)

    # classes_ has string dtype — GPU path rejects the call
    y_enc = enc.transform(y_strings)
    assert y_enc.shape == y_strings.shape

    logs = get_logs()
    assert any("not optimized: unsupported input" in line for line in logs)


# ---------------------------------------------------------------------------
# TargetEncoder (ProxyBase / override path)
# ---------------------------------------------------------------------------


def _make_target_encoder_data():
    """Return a float64 X (n, 2) and continuous y for TargetEncoder."""
    rng = np.random.default_rng(0)
    X = rng.integers(0, 5, size=(100, 2)).astype(float)
    y = rng.standard_normal(100)
    return X, y


def test_target_encoder_fit_transform():
    X, y = _make_target_encoder_data()

    enc = TargetEncoder(random_state=0)
    X_enc = enc.fit_transform(X, y)

    assert X_enc.shape == X.shape
    assert isinstance(X_enc, np.ndarray)
    # GPU path was taken: GPU model exists, CPU has no fitted attrs yet
    assert enc._gpu is not None
    assert not hasattr(enc._cpu, "n_features_in_")


def test_target_encoder_transform_after_fit():
    X, y = _make_target_encoder_data()
    X_new = X[:10]

    enc = TargetEncoder(random_state=0).fit(X, y)
    X_enc = enc.transform(X_new)

    assert X_enc.shape == X_new.shape
    assert isinstance(X_enc, np.ndarray)


def test_target_encoder_fallback_multiclass_y():
    X, y = _make_target_encoder_data()
    y_class = (y > 0).astype(int)
    # Make it a 3-class problem — triggers UnsupportedOnGPU
    y_class[0] = 2

    enc = TargetEncoder(random_state=0)
    X_enc = enc.fit_transform(X, y_class)

    # sklearn expands to n_features * n_classes columns for multiclass targets
    assert X_enc.shape[0] == X.shape[0]
    # CPU fallback
    assert enc._gpu is None


def test_target_encoder_fallback_random_state_object():
    X, y = _make_target_encoder_data()

    enc = TargetEncoder(random_state=np.random.RandomState(0))
    enc.fit_transform(X, y)

    # RandomState object triggers CPU fallback
    assert enc._gpu is None


def test_target_encoder_fallback_custom_categories():
    X, y = _make_target_encoder_data()
    # Provide explicit category lists — cuML doesn't support this
    categories = [list(range(5)), list(range(5))]

    enc = TargetEncoder(random_state=0, categories=categories)
    enc.fit_transform(X, y)

    assert enc._gpu is None


def test_target_encoder_fallback_object_dtype_X():
    X, y = _make_target_encoder_data()
    X_obj = X.astype(object)

    enc = TargetEncoder(random_state=0)
    enc.fit_transform(X_obj, y)

    assert enc._gpu is None
