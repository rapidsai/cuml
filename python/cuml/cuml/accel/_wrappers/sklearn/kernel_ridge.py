#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cuml.kernel_ridge
from cuml.accel.estimator_proxy import ProxyBase

__all__ = ("KernelRidge",)


class KernelRidge(ProxyBase):
    _gpu_class = cuml.kernel_ridge.KernelRidge
