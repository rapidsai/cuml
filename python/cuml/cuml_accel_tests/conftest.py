# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# This conftest is only used by the integration tests, not by the upstream
# tests. The upstream tests use the cuml.accel plugin explicitly.

from cuml.accel import install

# Install the accelerator
install()

# Ignore the upstream directory, those tests need to be invoked separately
collect_ignore = ["upstream"]
