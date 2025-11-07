#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import sklearn
from packaging.version import Version

import cuml.decomposition
from cuml.accel.estimator_proxy import ProxyBase

__all__ = ("PCA", "TruncatedSVD")


class PCA(ProxyBase):
    _gpu_class = (
        cuml.decomposition._PCAWithUBasedSignFlipEnabled
        if Version(sklearn.__version__) < Version("1.5.0")
        else cuml.decomposition.PCA
    )


class TruncatedSVD(ProxyBase):
    _gpu_class = (
        cuml.decomposition._TruncatedSVDWithUBasedSignFlipEnabled
        if Version(sklearn.__version__) < Version("1.5.0")
        else cuml.decomposition.TruncatedSVD
    )
