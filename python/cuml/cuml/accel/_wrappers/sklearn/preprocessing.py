#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np

import cuml.preprocessing
from cuml.accel.estimator_proxy import ProxyBase
from cuml.internals.interop import UnsupportedOnGPU

__all__ = ("TargetEncoder",)


class TargetEncoder(ProxyBase):
    _gpu_class = cuml.preprocessing.TargetEncoder

    def _gpu_fit(self, X, y, **kwargs):
        # cupy doesn't support object dtype (strings), fall back to CPU
        if hasattr(X, "dtype") and X.dtype == np.object_:
            raise UnsupportedOnGPU("TargetEncoder with object dtype")
        return self._gpu.fit(X, y, **kwargs)

    def _gpu_fit_transform(self, X, y, **kwargs):
        # cupy doesn't support object dtype (strings), fall back to CPU
        if hasattr(X, "dtype") and X.dtype == np.object_:
            raise UnsupportedOnGPU("TargetEncoder with object dtype")
        return self._gpu.fit_transform(X, y, **kwargs)

    def _gpu_transform(self, X):
        # cupy doesn't support object dtype (strings), fall back to CPU
        if hasattr(X, "dtype") and X.dtype == np.object_:
            raise UnsupportedOnGPU("TargetEncoder with object dtype")
        return self._gpu.transform(X)
