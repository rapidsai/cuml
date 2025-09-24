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

import cuml.manifold
from cuml.accel.estimator_proxy import ProxyBase
from cuml.internals.interop import UnsupportedOnGPU

__all__ = ("SpectralEmbedding", "TSNE")


class SpectralEmbedding(ProxyBase):
    _gpu_class = cuml.manifold.SpectralEmbedding
    _not_implemented_attributes = frozenset(("affinity_matrix_",))

    def _gpu_fit(self, X, y=None):
        shape = getattr(X, "shape", ())
        if len(shape) < 2 or shape[1] < 2:
            raise UnsupportedOnGPU(
                "X with < 2 features is not supported on GPU"
            )
        return self._gpu.fit(X, y=y)


class TSNE(ProxyBase):
    _gpu_class = cuml.manifold.TSNE
