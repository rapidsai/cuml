#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from cuml.accel.core import enabled, install
from cuml.accel.estimator_proxy import is_proxy
from cuml.accel.magics import load_ipython_extension
from cuml.accel.profilers import profile
from cuml.accel.pytest_plugin import (
    pytest_addoption,
    pytest_collection_modifyitems,
    pytest_load_initial_conftests,
)

__all__ = (
    "enabled",
    "install",
    "is_proxy",
    "load_ipython_extension",
    "profile",
    "pytest_addoption",
    "pytest_collection_modifyitems",
    "pytest_load_initial_conftests",
)
