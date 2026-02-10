# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import typing as t

import cupy as cp
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Reference UMAP implementation
from umap.umap_ import nearest_neighbors as umap_nearest_neighbors


def compute_knn_metrics(
    knn_graph_a,
    knn_graph_b,
) -> t.Tuple[float, float]:
    """
    Compute average neighbor recall and mean absolute distance error between two KNN results.

    Parameters
    ----------
    knn_graph_a : Tuple[np.ndarray, np.ndarray]
        Tuple of (distances, indices) for method A, each with shape (n_samples, n_neighbors).
    knn_graph_b : Tuple[np.ndarray, np.ndarray]
        Tuple of (distances, indices) for method B, each with shape (n_samples, n_neighbors).

    Returns
    -------
    avg_recall : float
        Average recall across all samples, i.e., the fraction of shared neighbors per row.
    mae_dist : float
        Mean absolute error of distances for intersecting neighbors across all samples.
    """
    recalls: t.List[float] = []
    distance_abs_errors: t.List[float] = []

    dists_a, inds_a = knn_graph_a
    dists_b, inds_b = knn_graph_b

    for i in range(inds_a.shape[0]):
        # Full neighbor rows (possibly including self)
        row_inds_a_full = inds_a[i]
        row_inds_b_full = inds_b[i]

        # Exclude self from both neighbor lists for fair comparison
        row_inds_a = [int(x) for x in row_inds_a_full if int(x) != i]
        row_inds_b = [int(x) for x in row_inds_b_full if int(x) != i]

        set_a = set(row_inds_a)
        set_b = set(row_inds_b)
        intersect = set_a & set_b

        # Use the size of A's neighborhood (after removing self) as denominator
        denom = max(1, len(row_inds_a))
        recalls.append(len(intersect) / float(denom))

        if not intersect:
            continue

        # Map index -> original distance position (in the unfiltered rows)
        pos_a = {
            int(idx): j
            for j, idx in enumerate(row_inds_a_full)
            if int(idx) != i
        }
        pos_b = {
            int(idx): j
            for j, idx in enumerate(row_inds_b_full)
            if int(idx) != i
        }
        for idx in intersect:
            da = float(dists_a[i, pos_a[idx]])
            db = float(dists_b[i, pos_b[idx]])
            distance_abs_errors.append(abs(da - db))

    avg_recall = float(np.mean(recalls)) if recalls else 0.0
    mae_dist = (
        float(np.mean(distance_abs_errors)) if distance_abs_errors else 0.0
    )

    return avg_recall, mae_dist


def _build_knn_with_umap(
    X: t.Union[np.ndarray, cp.ndarray],
    k: int,
    metric: str,
    backend: str,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Compute kNN using UMAP's nearest_neighbors.

    Returns (knn_dists, knn_indices) as NumPy arrays.
    """
    X_np = cp.asnumpy(X) if isinstance(X, cp.ndarray) else np.asarray(X)

    if backend == "bruteforce":
        nn = NearestNeighbors(
            n_neighbors=k, metric=metric, algorithm="brute", n_jobs=-1
        )
        nn.fit(X_np)
        knn_dists, knn_indices = nn.kneighbors(X_np, return_distance=True)
        return knn_dists.astype(np.float32, copy=False), knn_indices

    if backend == "nn_descent":
        angular = metric == "angular"
        knn_indices, knn_dists, _ = umap_nearest_neighbors(
            X_np,
            n_neighbors=k,
            metric=metric,
            metric_kwds={},
            angular=angular,
            random_state=np.random.RandomState(42),
            low_memory=True,
            use_pynndescent=True,
            n_jobs=-1,
            verbose=False,
        )
        return knn_dists, knn_indices

    raise ValueError(f"Unknown backend: {backend}")
