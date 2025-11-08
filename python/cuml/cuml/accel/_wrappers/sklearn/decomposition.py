#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cuml.decomposition
from cuml.accel.estimator_proxy import ProxyBase

__all__ = ("PCA", "TruncatedSVD")


class PCA(ProxyBase):
    _gpu_class = cuml.decomposition.PCA


class TruncatedSVD(ProxyBase):
    _gpu_class = cuml.decomposition.TruncatedSVD
