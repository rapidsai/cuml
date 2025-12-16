#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import sklearn
from packaging.version import Version

import cuml.cluster
from cuml.accel.estimator_proxy import ProxyBase
from cuml.internals.interop import UnsupportedOnGPU

__all__ = ("KMeans", "DBSCAN")


class KMeans(ProxyBase):
    _gpu_class = cuml.cluster.KMeans

    def _gpu_fit_transform(self, X, y=None, sample_weight=None):
        # Fixes signature mismatch with cuml.KMeans. Can be removed after #6741.
        return self._gpu.fit_transform(X, y=y, sample_weight=sample_weight)

    if Version(sklearn.__version__) < Version("1.5.0"):

        def _gpu_predict(self, X, sample_weight="deprecated"):
            # `sample_weight` was deprecated in 1.3 and removed in 1.5.
            if sample_weight == "deprecated":
                return self._gpu.predict(X)
            else:
                raise UnsupportedOnGPU("`sample_weight` is unsupported")

    def _init_centroids(self, *args, **kwargs):
        # Exposed for use by the sklearn test suite
        return self._cpu._init_centroids(*args, **kwargs)


class DBSCAN(ProxyBase):
    _gpu_class = cuml.cluster.DBSCAN

    def _gpu_fit(self, X, y=None, sample_weight=None):
        # Fixes signature mismatch with cuml.DBSCAN. Can be removed after #6741.
        return self._gpu.fit(X, y=y, sample_weight=sample_weight)

    def _gpu_fit_predict(self, X, y=None, sample_weight=None):
        # Fixes signature mismatch with cuml.DBSCAN. Can be removed after #6741.
        return self._gpu.fit_predict(X, y=y, sample_weight=sample_weight)
