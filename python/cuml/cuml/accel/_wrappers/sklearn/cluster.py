#
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cuml.cluster
from cuml.accel.estimator_proxy import ProxyBase

__all__ = ("KMeans", "DBSCAN")


class KMeans(ProxyBase):
    _gpu_class = cuml.cluster.KMeans

    def _gpu_fit_transform(self, X, y=None, sample_weight=None):
        # Fixes signature mismatch with cuml.KMeans. Can be removed after #6741.
        return self._gpu.fit_transform(X, y=y, sample_weight=sample_weight)

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
