# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide the CPU scikit-learn benchmark backend."""

from .base import Backend


class SklearnBackend(Backend):
    """Execute CPU estimators using the default backend hooks."""

    pass
