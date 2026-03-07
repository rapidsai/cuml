#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import cuml.manifold
from cuml.accel.estimator_proxy import ProxyBase
from sklearn.utils.validation import check_array

__all__ = ("UMAP",)

class UMAP(ProxyBase):
    """
    GPU-accelerated UMAP proxy estimator with input validation and memory optimization.
    """
    _gpu_class = cuml.manifold.UMAP

    def _gpu_fit(self, X, y=None, force_all_finite=True, **kwargs):
        """Fit the UMAP model with optimized float32 input validation."""
        # Validate and optimize memory: convert to float32 and C-order
        X = check_array(
            X, 
            accept_sparse="csr", 
            force_all_finite=force_all_finite,
            dtype=np.float32,
            order="C"
        )

        return self._gpu.fit(X, y=y, **kwargs)

    def _gpu_fit_transform(self, X, y=None, force_all_finite=True, **kwargs):
        """Fit and transform the UMAP model with optimized float32 input validation."""
        # Validate and optimize memory: convert to float32 and C-order
        X = check_array(
            X, 
            accept_sparse="csr", 
            force_all_finite=force_all_finite,
            dtype=np.float32,
            order="C"
        )

        return self._gpu.fit_transform(X, y=y, **kwargs)

    def _gpu_transform(self, X, force_all_finite=True):
        """Transform the data with optimized float32 input validation."""
        # Validate and optimize memory
        X = check_array(
            X, 
            accept_sparse="csr", 
            force_all_finite=force_all_finite,
            dtype=np.float32,
            order="C"
        )

        return self._gpu.transform(X)

    def _gpu_inverse_transform(self, X):
        """Inverse transform the data."""
        return self._gpu.inverse_transform(X)