#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import cuml.manifold
from cuml.accel.estimator_proxy import ProxyBase

__all__ = ("UMAP",)


class UMAP(ProxyBase):
    _gpu_class = cuml.manifold.UMAP

    def _gpu_fit(self, X, y=None, force_all_finite=True, **kwargs):
        # **kwargs is here for signature compatibility - umap.UMAP has them,
        # but ignores all but the ones named here.
        # TODO: cuml.UMAP currently doesn't handle non-finite inputs.
        # force_alL_finite is in here for _signature_ compatibility
        # with umap.UMAP, but we don't properly implement it (yet).
        return self._gpu.fit(X, y=y)

    def _gpu_fit_transform(self, X, y=None, force_all_finite=True, **kwargs):
        # **kwargs is here for signature compatibility - umap.UMAP has them,
        # but ignores all but the ones named here.
        return self._gpu.fit_transform(X, y=y)

    def _gpu_transform(self, X, force_all_finite=True):
        return self._gpu.transform(X)
