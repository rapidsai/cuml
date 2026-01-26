#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np

import cuml.preprocessing
from cuml.accel.estimator_proxy import ProxyBase

__all__ = ("TargetEncoder",)


class TargetEncoder(ProxyBase):
    _gpu_class = cuml.preprocessing.TargetEncoder

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
