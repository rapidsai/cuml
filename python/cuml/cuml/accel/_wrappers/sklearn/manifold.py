#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
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
