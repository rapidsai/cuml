#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cuml.covariance
from cuml.accel.estimator_proxy import ProxyBase

__all__ = ("LedoitWolf",)


class LedoitWolf(ProxyBase):
    _gpu_class = cuml.covariance.LedoitWolf
