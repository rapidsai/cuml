#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from packaging.version import Version

import cuml.manifold
from cuml.accel.estimator_proxy import ProxyBase
from cuml.internals.interop import UnsupportedOnGPU

try:
    import umap as _umap_module

    _UMAP_LT_058 = Version(_umap_module.__version__) < Version("0.5.8")
except ImportError:
    _UMAP_LT_058 = False

__all__ = ("UMAP",)


class UMAP(ProxyBase):
    _gpu_class = cuml.manifold.UMAP

    if _UMAP_LT_058:
        # We support the old signature for backwards compatibility prior to
        # umap-learn 0.5.8

        def _gpu_fit(self, X, y=None, force_all_finite=True, **kwargs):
            return self._gpu.fit(X, y=y)

        def _gpu_fit_transform(
            self, X, y=None, force_all_finite=True, **kwargs
        ):
            return self._gpu.fit_transform(X, y=y)

        def _gpu_transform(self, X, force_all_finite=True):
            return self._gpu.transform(X)

    else:

        def _gpu_fit(self, X, y=None, ensure_all_finite=True, **kwargs):
            # **kwargs is here for signature compatibility - umap.UMAP has them,
            # but ignores all but the ones named here.
            if ensure_all_finite is not True:
                raise UnsupportedOnGPU(
                    f"{ensure_all_finite=!r} is not supported"
                )
            return self._gpu.fit(X, y=y)

        def _gpu_fit_transform(
            self, X, y=None, ensure_all_finite=True, **kwargs
        ):
            # **kwargs is here for signature compatibility - umap.UMAP has them,
            # but ignores all but the ones named here.
            if ensure_all_finite is not True:
                raise UnsupportedOnGPU(
                    f"{ensure_all_finite=!r} is not supported"
                )
            return self._gpu.fit_transform(X, y=y)

        def _gpu_transform(self, X, ensure_all_finite=True):
            if ensure_all_finite is not True:
                raise UnsupportedOnGPU(
                    f"{ensure_all_finite=!r} is not supported"
                )
            return self._gpu.transform(X)

    def _gpu_inverse_transform(self, X):
        return self._gpu.inverse_transform(X)
