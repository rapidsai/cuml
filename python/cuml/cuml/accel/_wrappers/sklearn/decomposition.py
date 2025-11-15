#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import sklearn
from packaging.version import Version

import cuml.decomposition
from cuml.accel.estimator_proxy import ProxyBase

__all__ = ("PCA", "TruncatedSVD")

# In sklearn 1.5 the sign flipping behavior changed. For sklearn < 1.5 we
# enable the old behavior.
SKLEARN_15 = Version(sklearn.__version__) >= Version("1.5.0")


class PCA(ProxyBase):
    _gpu_class = cuml.decomposition.PCA

    if not SKLEARN_15:

        def _gpu_fit(self, X, y=None):
            self._gpu._u_based_sign_flip = True
            return self._gpu.fit(X, y)

        def _gpu_fit_transform(self, X, y=None):
            self._gpu._u_based_sign_flip = True
            return self._gpu.fit_transform(X, y)


class TruncatedSVD(ProxyBase):
    _gpu_class = cuml.decomposition.TruncatedSVD

    if not SKLEARN_15:

        def _gpu_fit(self, X, y=None):
            self._gpu._u_based_sign_flip = True
            return self._gpu.fit(X, y)

        def _gpu_fit_transform(self, X, y=None):
            self._gpu._u_based_sign_flip = True
            return self._gpu.fit_transform(X, y)
