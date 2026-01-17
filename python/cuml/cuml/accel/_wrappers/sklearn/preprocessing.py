#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cuml.preprocessing
from cuml.accel.estimator_proxy import ProxyBase

__all__ = ("TargetEncoder",)


class TargetEncoder(ProxyBase):
    _gpu_class = cuml.preprocessing.TargetEncoder
