# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp


def compute_weights(distances, weights):
    """
    Compute weights from distances for KNN algorithms.

    Parameters
    ----------
    distances : cupy.ndarray or array-like
        The input distances (n_samples, k).

    weights : {'uniform', 'distance'} or callable
        The kind of weighting used.

    Returns
    -------
    weights_arr : cupy.ndarray
        Weights of shape (n_samples, k).
        For uniform weights, returns array of 1/k (normalized).
        For distance weights, returns raw inverse distances (not normalized).
        For callable, returns raw result of callable(distances) (not normalized).
    """
    # Convert to cupy array if needed and ensure 2D
    if not isinstance(distances, cp.ndarray):
        distances = cp.asarray(distances)

    # Ensure distances is 2D
    if distances.ndim == 1:
        distances = distances.reshape(-1, 1)
    elif distances.ndim != 2:
        raise ValueError(
            f"distances must be 1D or 2D, got shape {distances.shape}"
        )

    if weights in (None, "uniform"):
        # Uniform weights: all neighbors contribute equally
        n_neighbors = distances.shape[1]
        return cp.full_like(distances, 1.0 / n_neighbors, dtype=cp.float32)
    elif weights == "distance":
        # Distance weights: inverse of distance (raw, not normalized)
        # Match sklearn behavior: if any neighbor has distance 0, only those
        # neighbors contribute (with equal weight)

        # Compute 1/distance (this will produce inf for zero distances)
        raw_weights = (1.0 / distances).astype(cp.float32)

        # Handle infinite weights (from zero distances)
        inf_mask = cp.isinf(raw_weights)
        inf_row = cp.any(inf_mask, axis=1)

        # For rows with any infinite weight, use binary mask:
        # 1.0 for zero-distance neighbors, 0.0 for others
        if cp.any(inf_row):
            raw_weights[inf_row] = inf_mask[inf_row].astype(cp.float32)

        return raw_weights
    elif callable(weights):
        # Custom callable weights (raw, not normalized)
        raw_weights = weights(distances).astype(cp.float32)
        # Return raw weights - normalization will be done in C++ kernel
        return raw_weights
    else:
        raise ValueError(
            f"weights must be 'uniform', 'distance', or a callable, got {weights}"
        )
