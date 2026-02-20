#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupyx.scipy.sparse as cupy_sparse
import numpy as np
from scipy import sparse as sp_sparse

import cuml.preprocessing
from cuml.accel.estimator_proxy import ProxyBase
from cuml.internals.interop import UnsupportedOnGPU

__all__ = ("StandardScaler", "TargetEncoder")


def _check_standardscaler_unsupported_inputs(X, **kwargs):
    """Check if inputs are supported by cuML's StandardScaler on GPU.

    Raises UnsupportedOnGPU for unsupported cases to trigger CPU fallback.
    """
    if kwargs.get("sample_weight") is not None:
        raise UnsupportedOnGPU("sample_weight is not supported")

    # Reject complex, object, and float16 dtypes
    if hasattr(X, "dtype"):
        if np.issubdtype(X.dtype, np.complexfloating):
            raise UnsupportedOnGPU("complex dtype is not supported")
        if X.dtype == np.object_:
            raise UnsupportedOnGPU("object dtype is not supported")
        if X.dtype == np.float16:
            raise UnsupportedOnGPU("float16 dtype is not supported")

    # Check for sparse matrices with unsupported properties
    if sp_sparse.issparse(X):
        if np.issubdtype(X.dtype, np.integer):
            raise UnsupportedOnGPU(
                "sparse matrix with integer dtype is not supported"
            )
        # cuML's StandardScaler algorithm only supports CSR/CSC formats.
        if X.format not in ("csr", "csc"):
            raise UnsupportedOnGPU(
                f"sparse matrix format '{X.format}' is not supported"
            )
    elif cupy_sparse.issparse(X):
        if np.issubdtype(X.dtype, np.integer):
            raise UnsupportedOnGPU(
                "sparse matrix with integer dtype is not supported"
            )
        # cuML's StandardScaler algorithm only supports CSR/CSC formats.
        if X.format not in ("csr", "csc"):
            raise UnsupportedOnGPU(
                f"sparse matrix format '{X.format}' is not supported"
            )


class StandardScaler(ProxyBase):
    _gpu_class = cuml.preprocessing.StandardScaler

    def _gpu_fit(self, X, y=None, sample_weight=None):
        kwargs = {"sample_weight": sample_weight}
        _check_standardscaler_unsupported_inputs(X, **kwargs)
        return self._gpu.fit(X, y)

    def _gpu_fit_transform(self, X, y=None, **fit_params):
        _check_standardscaler_unsupported_inputs(X, **fit_params)
        return self._gpu.fit_transform(X, y, **fit_params)

    def _gpu_partial_fit(self, X, y=None, sample_weight=None):
        """partial_fit not supported on GPU - always fall back to CPU."""
        raise UnsupportedOnGPU("partial_fit not supported on GPU")


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

    # Check for RandomState objects - cuML uses different CV fold assignment
    # algorithm, so we can't reproduce exact sklearn behavior with RandomState.
    # Integer seeds work fine, but RandomState objects require CPU fallback
    # to ensure statistical tests that depend on exact CV splits pass.
    if hasattr(cpu_model, "random_state") and isinstance(
        cpu_model.random_state, np.random.RandomState
    ):
        raise UnsupportedOnGPU(
            "RandomState objects require CPU fallback for exact sklearn "
            "CV fold assignment behavior (use integer seed for GPU)"
        )

    # Check for object dtype X with float values - cudf can't handle this
    if hasattr(X, "dtype") and X.dtype == np.object_:
        raise UnsupportedOnGPU(
            "Object dtype arrays with numeric values not supported on GPU"
        )

    # Check for object dtype y - cudf can't handle float objects
    if hasattr(y, "dtype") and y.dtype == np.object_:
        raise UnsupportedOnGPU(
            "Object dtype target arrays not supported on GPU"
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
        result = self._gpu.fit(X, y, **kwargs)

        # Sync sklearn-expected attributes directly to the CPU model.
        if hasattr(self._gpu, "feature_names_in_"):
            self._cpu.feature_names_in_ = self._gpu.feature_names_in_
        if hasattr(self._gpu, "n_features_in_"):
            self._cpu.n_features_in_ = self._gpu.n_features_in_

        return result

    def _gpu_fit_transform(self, X, y, **kwargs):
        """Fit-transform with independent mode for sklearn compatibility.

        sklearn's TargetEncoder always encodes features independently,
        so we force independent mode when using cuml.accel.
        """
        # Check for unsupported inputs (triggers CPU fallback)
        _check_unsupported_inputs(X, y, self._cpu)

        # Ensure independent mode is set for sklearn compatibility
        self._gpu.multi_feature_mode = "independent"
        result = self._gpu.fit_transform(X, y, **kwargs)

        # Sync sklearn-expected attributes directly to the CPU model.
        if hasattr(self._gpu, "feature_names_in_"):
            self._cpu.feature_names_in_ = self._gpu.feature_names_in_
        if hasattr(self._gpu, "n_features_in_"):
            self._cpu.n_features_in_ = self._gpu.n_features_in_

        return result

    def _gpu_transform(self, X):
        # Perform sklearn's feature name validation using the CPU model.
        # reset=False: only validate feature names/count against fit()
        # dtype=None: skip dtype validation (TargetEncoder accepts categorical data)
        try:
            from sklearn.utils.validation import validate_data

            validate_data(self._cpu, X=X, reset=False, skip_check_array=True)
        except ImportError:
            # sklearn 1.5.* and earlier: use the method on the estimator
            self._cpu._validate_data(X=X, reset=False, dtype=None)
        return self._gpu.transform(X)

    def _gpu_get_feature_names_out(self, input_features=None):
        """Return feature names for output features.

        sklearn's TargetEncoder returns input feature names (one output per input).
        For cuML in combination mode, we return a single column name.
        """
        n_features_out = getattr(self._gpu, "_n_features_out", 1)
        if input_features is not None:
            return np.asarray(input_features[:n_features_out], dtype=object)
        # Use feature_names_in_ if available (set during fit with DataFrame input)
        if hasattr(self._gpu, "feature_names_in_"):
            return np.asarray(
                self._gpu.feature_names_in_[:n_features_out], dtype=object
            )
        # Generate default names like sklearn does
        return np.array(
            [f"targetencoder{i}" for i in range(n_features_out)], dtype=object
        )
