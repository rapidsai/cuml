#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cuml.manifold
from cuml.accel.estimator_proxy import ProxyBase
from cuml.thirdparty_adapters.adapters import check_array

__all__ = ("UMAP",)


class UMAP(ProxyBase):
    """
    GPU-accelerated UMAP proxy estimator for input validation and fit.
    """
    _gpu_class = cuml.manifold.UMAP

    def _gpu_fit(self, X, y=None, force_all_finite=True, **kwargs):
        """Fit the UMAP model with GPU-accelerated input validation."""
        del kwargs  # Handle intentionally unused kwargs for signature compatibility

        X = check_array(
            X, accept_sparse="csr", force_all_finite=force_all_finite
        )

        return self._gpu.fit(X, y=y)

    def _gpu_fit_transform(self, X, y=None, force_all_finite=True, **kwargs):
        """Fit and transform the data with GPU-accelerated input validation."""
        del kwargs  # Handle intentionally unused kwargs for signature compatibility

        X = check_array(
            X, accept_sparse="csr", force_all_finite=force_all_finite
        )

        return self._gpu.fit_transform(X, y=y)

    def _gpu_transform(self, X, force_all_finite=True, **kwargs):
        """Transform the data with GPU-accelerated input validation."""
        del kwargs  # Handle intentionally unused kwargs for signature compatibility

        X = check_array(
            X, accept_sparse="csr", force_all_finite=force_all_finite
        )

        return self._gpu.transform(X)

    def _gpu_inverse_transform(self, X, force_all_finite=True, **kwargs):
        """Inverse transform the data with GPU-accelerated input validation."""
        del kwargs  # Handle intentionally unused kwargs for signature compatibility

        X = check_array(
            X, accept_sparse="csr", force_all_finite=force_all_finite
        )

        return self._gpu.inverse_transform(X)