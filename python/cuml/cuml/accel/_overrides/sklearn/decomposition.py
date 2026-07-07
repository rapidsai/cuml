#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cuml.decomposition
from cuml.accel.estimator_proxy import ProxyBase

__all__ = ("IncrementalPCA", "PCA", "TruncatedSVD")


class PCA(ProxyBase):
    _gpu_class = cuml.decomposition.PCA


class TruncatedSVD(ProxyBase):
    _gpu_class = cuml.decomposition.TruncatedSVD


class IncrementalPCA(ProxyBase):
    _gpu_class = cuml.decomposition.IncrementalPCA

    def _gpu_fit_transform(self, X, y=None, **fit_params):
        if fit_params:
            param = next(iter(fit_params))
            raise TypeError(
                "IncrementalPCA.fit() got an unexpected keyword argument "
                f"{param!r}"
            )
        return self._gpu.fit_transform(X, y)

    def _gpu_partial_fit(self, X, y=None, check_input=True):
        return self._gpu.partial_fit(X, y, check_input=check_input)
