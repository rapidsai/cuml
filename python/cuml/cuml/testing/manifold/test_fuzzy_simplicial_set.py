#
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import pathlib
import typing as t

import cupy as cp
import numpy as np
import pytest
import umap
from cuvs.neighbors import brute_force, nn_descent
from scipy.sparse import csr_matrix

# Reference UMAP implementation
from umap.umap_ import fuzzy_simplicial_set as ref_fuzzy_simplicial_set

# cuML implementation
from cuml.manifold.simpl_set import (
    fuzzy_simplicial_set as cu_fuzzy_simplicial_set,
)
from cuml.testing.manifold.umap_quality_checks import (
    compute_edge_jaccard,
    compute_fuzzy_kl_sym,
)

pytestmark = [pytest.mark.slow]


_DATASET_SPECS: t.Dict[str, t.Dict[str, t.Any]] = {
    "deep-image-96-angular": {"metric": "cosine"},
    "fashion-mnist-784-euclidean": {"metric": "euclidean"},
    "gist-960-euclidean": {"metric": "euclidean"},
    "glove-25-angular": {"metric": "cosine"},
    "mnist-784-euclidean": {"metric": "euclidean"},
    "sift-128-euclidean": {"metric": "euclidean"},
}


def _env_data_dir() -> pathlib.Path:
    env = os.environ.get("DATASET_DIR")
    if not env:
        raise ValueError(
            "Set DATASET_DIR to the root directory containing datasets (e.g., directories with base.fbin).\n"
            "Datasets can be downloaded with the following command `python -m cuvs_bench.get_dataset --dataset <dataset_name> --normalize`\n"
            "For more information, see https://docs.rapids.ai/api/cuvs/nightly/cuvs_bench"
        )
    p = pathlib.Path(env)
    if not p.exists():
        raise ValueError(f"Data dir not found: {p}")
    return p


def read_fbin(fname: pathlib.Path) -> np.ndarray:
    shape = np.fromfile(fname, dtype=np.uint32, count=2)
    # if float(shape[0]) * shape[1] * 4 > 2_000_000_000:
    #    data = np.memmap(fname, dtype=np.float32, offset=8, mode="r").reshape(shape)
    # else:
    data = np.fromfile(fname, dtype=np.float32, offset=8).reshape(shape)
    return data


def _load_dataset(name: str) -> np.ndarray:
    """Load dataset as float32 NumPy array."""
    data_root = _env_data_dir()

    fbin_path = []
    # Heuristic: deep-image-96-angular -> deep-image-96-inner
    if "-angular" in name:
        dir_name = name.replace("-angular", "-inner")
    else:
        dir_name = name

    fbin_path = data_root / dir_name / "base.fbin"

    if fbin_path.exists():
        arr = read_fbin(fbin_path)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        return arr

    raise ValueError(
        f"Dataset '{name}' not found. Looked under: {fbin_path}\n"
        f"Please run `python -m cuvs_bench.get_dataset --dataset {name} --normalize`"
    )


def _build_knn_with_cuvs(
    X: cp.ndarray, k: int, metric: str, backend: str
) -> t.Tuple[np.ndarray, np.ndarray]:
    if backend == "bruteforce":
        index = brute_force.build(X, metric=metric)
        knn_dists, knn_indices = brute_force.search(index, X, k)
        return cp.asnumpy(knn_dists), cp.asnumpy(knn_indices)

    if backend == "nn_descent":
        params = nn_descent.IndexParams(metric=metric)
        index = nn_descent.build(params, X)
        knn_indices = cp.asarray(index.graph[:, :k])
        knn_dists = cp.asarray(index.distances[:, :k])
        return cp.asnumpy(knn_dists), cp.asnumpy(knn_indices)

    raise ValueError(f"Unknown backend: {backend}")


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
    angular = metric in ("cosine", "angular")
    use_pynndescent = backend == "nn_descent"
    knn_indices, knn_dists, _ = umap.umap_.nearest_neighbors(
        X_np,
        n_neighbors=k,
        metric=metric,
        metric_kwds={},
        angular=angular,
        random_state=np.random.RandomState(42),
        low_memory=True,
        use_pynndescent=use_pynndescent,
        n_jobs=-1,
        verbose=False,
    )
    return knn_dists, knn_indices


@pytest.mark.parametrize(
    "dataset_name",
    [
        "deep-image-96-angular",
        "fashion-mnist-784-euclidean",
        "gist-960-euclidean",
        "glove-25-angular",
        "mnist-784-euclidean",
        "sift-128-euclidean",
    ],
)
@pytest.mark.parametrize("n_neighbors", [10, 30])
@pytest.mark.parametrize("backend", ["bruteforce", "nn_descent"])
def test_knn_cuvs_vs_umap_equivalence(
    dataset_name: str, n_neighbors: int, backend: str
):
    ds_spec = _DATASET_SPECS.get(dataset_name, {})
    metric = ds_spec.get("metric")

    X_np = _load_dataset(dataset_name)
    X_cp = cp.asarray(X_np)

    # cuVS kNN (GPU -> NumPy)
    dists_cuvs, inds_cuvs = _build_knn_with_cuvs(
        X_cp, k=n_neighbors, metric=metric, backend=backend
    )

    # UMAP kNN (CPU)
    dists_umap, inds_umap = _build_knn_with_umap(
        X_np, k=n_neighbors, metric=metric, backend=backend
    )

    # Sanity: shapes
    assert dists_cuvs.shape == dists_umap.shape == (X_np.shape[0], n_neighbors)
    assert inds_cuvs.shape == inds_umap.shape == (X_np.shape[0], n_neighbors)

    # Compute per-row recall (overlap of neighbor index sets)
    recalls = []
    distance_abs_errors = []
    for i in range(X_np.shape[0]):
        row_inds_cuvs = inds_cuvs[i]
        row_inds_umap = inds_umap[i]

        set_cuvs = set(int(x) for x in row_inds_cuvs)
        set_umap = set(int(x) for x in row_inds_umap)
        intersect = set_cuvs & set_umap

        recalls.append(len(intersect) / float(n_neighbors))

        if not intersect:
            continue

        # Map index -> distance for matched neighbors in both results
        pos_cuvs = {int(idx): j for j, idx in enumerate(row_inds_cuvs)}
        pos_umap = {int(idx): j for j, idx in enumerate(row_inds_umap)}
        for idx in intersect:
            dc = float(dists_cuvs[i, pos_cuvs[idx]])
            du = float(dists_umap[i, pos_umap[idx]])
            distance_abs_errors.append(abs(dc - du))

    avg_recall = float(np.mean(recalls)) if recalls else 0.0
    mae_dist = (
        float(np.mean(distance_abs_errors)) if distance_abs_errors else 0.0
    )

    # Tolerances: stricter for bruteforce, looser for NN-descent approx
    if backend == "bruteforce":
        recall_tol = 0.995
        # Slightly looser for cosine than euclidean due to numerical nuances
        dist_tol = 5e-5 if metric == "cosine" else 1e-5
    else:
        recall_tol = 0.95
        dist_tol = 5e-3 if metric == "cosine" else 1e-3

    assert avg_recall >= recall_tol, (
        f"Neighbor recall too low: {avg_recall:.3f} < {recall_tol:.3f} "
        f"for {dataset_name}, metric={metric}, backend={backend}"
    )

    # Distance check only over matched pairs; skip if none matched
    if distance_abs_errors:
        assert mae_dist <= dist_tol, (
            f"Mean abs distance error too high: {mae_dist:.3e} > {dist_tol:.3e} "
            f"for {dataset_name}, metric={metric}, backend={backend}"
        )


@pytest.mark.parametrize(
    "dataset_name",
    [
        "deep-image-96-angular",
        "fashion-mnist-784-euclidean",
        "gist-960-euclidean",
        "glove-25-angular",
        "mnist-784-euclidean",
        "sift-128-euclidean",
    ],
)
@pytest.mark.parametrize("n_neighbors", [10, 30])
@pytest.mark.parametrize("backend", ["bruteforce", "nn_descent"])
def test_fuzzy_simplicial_set_quality(
    dataset_name: str, n_neighbors: int, backend: str
):
    ds_spec = _DATASET_SPECS.get(dataset_name, {})
    metric = ds_spec.get("metric")

    X_np = _load_dataset(dataset_name)
    X_cp = cp.asarray(X_np)

    # Build kNN using cuVS on GPU
    knn_dists_np, knn_inds_np = _build_knn_with_cuvs(
        X_cp, k=n_neighbors, metric=metric, backend=backend
    )

    # Reference fuzzy graph (CPU / SciPy sparse)
    ref_graph, _, _ = ref_fuzzy_simplicial_set(
        X_np,
        n_neighbors=n_neighbors,
        random_state=42,
        metric=metric,
        knn_indices=knn_inds_np,
        knn_dists=knn_dists_np,
    )
    ref_graph = csr_matrix(ref_graph)

    # cuML fuzzy graph (GPU -> CPU SciPy sparse)
    cu_graph = cu_fuzzy_simplicial_set(
        X_cp,
        n_neighbors=n_neighbors,
        random_state=42,
        metric=metric,
        knn_indices=cp.asarray(knn_inds_np),
        knn_dists=cp.asarray(knn_dists_np),
    )
    # cuML returns a cupy sparse or raft handle; convert via .get() if available
    if hasattr(cu_graph, "get"):
        cu_graph = cu_graph.get()
    cu_graph = csr_matrix(cu_graph)

    # Symmetric KL divergence between the two fuzzy graphs
    kl_sym = compute_fuzzy_kl_sym(ref_graph, cu_graph)

    # Jaccard over undirected edges
    jacc = compute_edge_jaccard(ref_graph, cu_graph, eps=0.0)

    # Row-sum L1: average absolute difference of total membership mass per node
    row_l1 = float(
        np.mean(
            np.abs(
                np.asarray(ref_graph.sum(axis=1)).ravel()
                - np.asarray(cu_graph.sum(axis=1)).ravel()
            )
        )
    )

    # Simple, global tolerances
    kl_tol = "1e-2"
    j_tol = 0.90
    row_l1_tol = 1e-3

    # Assertions focused on matching the reference
    assert np.isfinite(kl_sym), "KL not finite"
    assert (
        kl_sym <= kl_tol
    ), f"KL(sym) too high: {kl_sym:.3e} > {kl_tol:.3e} for {dataset_name}, metric={metric}, backend={backend}"
    assert (
        jacc >= j_tol
    ), f"Edge Jaccard too low (jacc={jacc:.3f} < tol={j_tol:.3f}) for {dataset_name}, metric={metric}, backend={backend}"
    assert (
        row_l1 <= row_l1_tol
    ), f"Row-sum L1 too high (row_l1={row_l1:.3e} > tol={row_l1_tol:.3e}) for {dataset_name}, metric={metric}, backend={backend}"
