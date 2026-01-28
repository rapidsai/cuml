#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np

import cuml.preprocessing
from cuml.accel.estimator_proxy import ProxyBase
from cuml.internals.interop import UnsupportedOnGPU

__all__ = ("TargetEncoder",)


def _check_unsupported_inputs(X, y, cpu_model):
    """Check if inputs are supported on GPU.

    Raises UnsupportedOnGPU for unsupported cases to trigger CPU fallback.
    """
    from sklearn.utils.multiclass import type_of_target

    # Check for custom categories (cuML doesn't support sklearn's categories param)
    if hasattr(cpu_model, "categories") and cpu_model.categories != "auto":
        raise UnsupportedOnGPU(
            "Custom categories parameter not supported on GPU"
        )

    # Check for multiclass targets (sklearn uses one-hot encoding internally)
    target_type = type_of_target(y)
    if target_type == "multiclass":
        raise UnsupportedOnGPU(
            "Multiclass targets not supported on GPU "
            "(sklearn uses one-hot encoding internally)"
        )


class TargetEncoder(ProxyBase):
    _gpu_class = cuml.preprocessing.TargetEncoder

    def _gpu_fit(self, X, y, **kwargs):
        """Fit with independent mode for sklearn compatibility.

        sklearn's TargetEncoder always encodes features independently,
        so we force independent mode when using cuml.accel.
        """
        # Check for unsupported inputs (triggers CPU fallback)
        _check_unsupported_inputs(X, y, self._cpu)

        # Ensure independent mode is set for sklearn compatibility
        self._gpu.multi_feature_mode = "independent"
        return self._gpu.fit(X, y, **kwargs)

    def _gpu_fit_transform(self, X, y, **kwargs):
        """Fit-transform with independent mode for sklearn compatibility.

        sklearn's TargetEncoder always encodes features independently,
        so we force independent mode when using cuml.accel.
        """
        # Check for unsupported inputs (triggers CPU fallback)
        _check_unsupported_inputs(X, y, self._cpu)

        # Ensure independent mode is set for sklearn compatibility
        self._gpu.multi_feature_mode = "independent"
        return self._gpu.fit_transform(X, y, **kwargs)

    def _gpu_get_feature_names_out(self, input_features=None):
        """Return feature names for output features.

        sklearn's TargetEncoder returns input feature names (one output per input).
        For cuML in combination mode, we return a single column name.
        """
        n_features_out = getattr(self._gpu, "_n_features_out", 1)
        if input_features is not None:
            return np.asarray(input_features[:n_features_out], dtype=object)
        # Generate default names like sklearn does
        return np.array(
            [f"targetencoder{i}" for i in range(n_features_out)], dtype=object
        )
