#
# Copyright (c) 2025, NVIDIA CORPORATION.
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
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

from cuml.internals import logger


def _create_test_data():
    """Create consistent test data for all logging tests."""
    return np.random.randn(10, 5), np.random.randn(10)


def _create_ridge_estimator():
    """Create a Ridge estimator with consistent parameters."""
    return Ridge(random_state=42)


def _create_kmeans_estimator():
    """Create a KMeans estimator with consistent parameters."""
    return KMeans(n_clusters=3, random_state=42)


def _get_log_entries(captured_output):
    """Extract non-empty log entries from captured output."""
    return [line for line in captured_output.split("\n") if line]


@pytest.mark.parametrize(
    "log_level,expected_messages",
    [
        (
            logger.level_enum.warn,
            [],
        ),
        (
            logger.level_enum.info,
            ["[cuml.accel] Successfully accelerated 'Ridge.fit()' call"],
        ),
        (
            logger.level_enum.debug,
            [
                "[cuml.accel] Initialized estimator 'Ridge' for GPU acceleration",
                "[cuml.accel] Successfully accelerated 'Ridge.fit()' call",
            ],
        ),
    ],
)
def test_gpu_init_and_fit_logging(capsys, log_level, expected_messages):
    "Test GPU initialization and fitting at different logging levels."
    # Setup
    ridge = _create_ridge_estimator()
    X, y = _create_test_data()

    # Execute
    with logger.set_level(log_level):
        ridge.fit(X, y)

    # Assert
    captured = capsys.readouterr()
    log_entries = _get_log_entries(captured.out)

    assert len(log_entries) == len(expected_messages)

    for expected_message in expected_messages:
        assert any(expected_message in entry for entry in log_entries), (
            f"Expected message '{expected_message}' not found in log entries: "
            f"{log_entries}"
        )


def test_parameter_synchronization_logging(capsys):
    """Test logging when parameters are synchronized from CPU to GPU."""
    ridge = _create_ridge_estimator()
    X, y = _create_test_data()

    # First fit to initialize GPU estimator
    ridge.fit(X, y)

    # Clear captured output
    capsys.readouterr()

    # Change parameter to trigger sync
    with logger.set_level(logger.level_enum.debug):
        ridge.alpha = 2.0
        ridge.fit(X, y)

    captured = capsys.readouterr()
    log_entries = _get_log_entries(captured.out)

    # Should see parameter sync message
    sync_message = "[cuml.accel] Synced parameters from CPU to GPU for 'Ridge'"
    assert any(sync_message in entry for entry in log_entries), (
        f"Expected parameter sync message not found in log entries: "
        f"{log_entries}"
    )


def test_parameter_sync_failure_logging(capsys):
    """Test logging when parameter synchronization fails."""
    # Create KMeans with unsupported init parameter to trigger sync failure

    def init_callable(X, n_clusters, random_state):
        return np.random.rand(n_clusters, X.shape[1])

    kmeans = KMeans(n_clusters=3, init=init_callable, random_state=42)
    X, _ = _create_test_data()

    with logger.set_level(logger.level_enum.warn):
        kmeans.fit(X)

    captured = capsys.readouterr()
    log_entries = _get_log_entries(captured.out)

    # Should see initialization failure message (not parameter sync)
    failure_message = (
        "[cuml.accel] Failed to initialize 'KMeans' with GPU acceleration"
    )
    assert any(failure_message in entry for entry in log_entries), (
        f"Expected initialization failure message not found in log entries: "
        f"{log_entries}"
    )


def test_attribute_synchronization_logging(capsys):
    """Test logging when fit attributes are synchronized from GPU to CPU."""
    ridge = _create_ridge_estimator()
    X, y = _create_test_data()

    # Fit to initialize GPU estimator
    ridge.fit(X, y)

    # Clear captured output
    capsys.readouterr()

    # Access fit attributes to trigger sync
    with logger.set_level(logger.level_enum.debug):
        _ = ridge.coef_

    captured = capsys.readouterr()
    log_entries = _get_log_entries(captured.out)

    # Should see attribute sync message
    sync_message = (
        "[cuml.accel] Synced fit attributes from GPU to CPU for 'Ridge'"
    )
    assert any(sync_message in entry for entry in log_entries), (
        f"Expected attribute sync message not found in log entries: "
        f"{log_entries}"
    )


def test_gpu_initialization_failure_logging(capsys):
    """Test logging when GPU initialization fails."""
    # Create Ridge with multi-dimensional y to trigger initialization failure
    ridge = _create_ridge_estimator()
    X, y = _create_test_data()
    y_2d = y.reshape(-1, 1)  # Make y 2D

    with logger.set_level(logger.level_enum.warn):
        ridge.fit(X, y_2d)

    captured = capsys.readouterr()
    log_entries = _get_log_entries(captured.out)

    # Should see acceleration failure message (not initialization)
    assert (
        len(log_entries) == 1
    ), f"Expected 1 log entry, but found: {log_entries}"
    failure_message = "[cuml.accel] Failed to accelerate 'Ridge.fit()'"
    assert any(failure_message in entry for entry in log_entries), (
        f"Expected acceleration failure message not found in log entries: "
        f"{log_entries}"
    )


def test_method_acceleration_success_logging(capsys):
    """Test logging when method acceleration succeeds."""
    ridge = _create_ridge_estimator()
    X, y = _create_test_data()

    # First fit successfully
    ridge.fit(X, y)

    # Clear captured output
    capsys.readouterr()

    # Try to call a method that should succeed on GPU
    with logger.set_level(logger.level_enum.info):
        # This should work and show successful acceleration
        ridge.predict(X)

    captured = capsys.readouterr()
    log_entries = _get_log_entries(captured.out)

    # Should see successful acceleration message
    success_message = (
        "[cuml.accel] Successfully accelerated 'Ridge.predict()' call"
    )
    assert any(success_message in entry for entry in log_entries), (
        f"Expected successful acceleration message not found in log entries: "
        f"{log_entries}"
    )


def test_sparse_input_fallback_logging(capsys):
    """Test logging when sparse inputs cause fallback to CPU."""
    ridge = _create_ridge_estimator()
    X, y = _create_test_data()

    # Create sparse matrix
    X_sparse = csr_matrix(X)

    with logger.set_level(logger.level_enum.debug):
        ridge.fit(X_sparse, y)

    captured = capsys.readouterr()
    log_entries = _get_log_entries(captured.out)

    # Should see sparse input fallback message
    fallback_message = (
        "[cuml.accel] Unable to accelerate 'Ridge.fit()' call: "
        "Sparse inputs are not supported"
    )
    assert any(fallback_message in entry for entry in log_entries), (
        f"Expected sparse input fallback message not found in log entries: "
        f"{log_entries}"
    )


def test_cpu_execution_logging(capsys):
    """Test logging when methods are executed on CPU."""
    ridge = _create_ridge_estimator()
    X, y = _create_test_data()

    # Fit with multi-dimensional y to force CPU fallback
    y_2d = y.reshape(-1, 1)
    ridge.fit(X, y_2d)

    # Clear captured output
    capsys.readouterr()

    # Call predict which should execute on CPU
    with logger.set_level(logger.level_enum.debug):
        ridge.predict(X)

    captured = capsys.readouterr()
    log_entries = _get_log_entries(captured.out)

    # Should see CPU execution message
    cpu_message = "[cuml.accel] Executing 'Ridge.predict()' on CPU"
    assert any(cpu_message in entry for entry in log_entries), (
        f"Expected CPU execution message not found in log entries: "
        f"{log_entries}"
    )


def test_multiple_operations_logging_sequence(capsys):
    """Test logging sequence for multiple operations."""
    ridge = _create_ridge_estimator()
    X, y = _create_test_data()

    with logger.set_level(logger.level_enum.debug):
        # 1. Initial fit (should log initialization and successful acceleration)
        ridge.fit(X, y)

        # 2. Access attributes (should log attribute sync)
        _ = ridge.coef_

        # 3. Change parameters (should log parameter sync)
        ridge.alpha = 2.0
        ridge.fit(X, y)

        # 4. Predict (should log successful acceleration)
        ridge.predict(X)

    captured = capsys.readouterr()
    log_entries = _get_log_entries(captured.out)

    # Verify all expected messages are present
    expected_messages = [
        "[cuml.accel] Initialized estimator 'Ridge' for GPU acceleration",
        "[cuml.accel] Successfully accelerated 'Ridge.fit()' call",
        "[cuml.accel] Synced fit attributes from GPU to CPU for 'Ridge'",
        "[cuml.accel] Synced parameters from CPU to GPU for 'Ridge'",
        "[cuml.accel] Successfully accelerated 'Ridge.predict()' call",
    ]

    for expected_message in expected_messages:
        assert any(expected_message in entry for entry in log_entries), (
            f"Expected message '{expected_message}' not found in log entries: "
            f"{log_entries}"
        )
