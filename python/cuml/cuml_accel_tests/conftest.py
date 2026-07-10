# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Configure ``cuml.accel`` for its integration tests.

The upstream tests use the ``cuml.accel`` pytest plugin explicitly.
"""

import os

import pytest

from cuml.accel import install
from cuml.accel.core import logger

# Install the accelerator
install(log_level="error")


@pytest.fixture(scope="session", autouse=True)
def restore_log_level():
    """Restore normal logging after xdist workers have started."""
    logger.set_level("warn")
    os.environ["CUML_ACCEL_LOG_LEVEL"] = "warn"


# Ignore the upstream directory, those tests need to be invoked separately
collect_ignore = ["upstream"]
