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

import typing as t

import cudf
import cugraph
import cupy as cp
import numpy as np
from cuvs.common import MultiGpuResources
from cuvs.distance import pairwise_distance
from cuvs.neighbors import all_neighbors, brute_force, nn_descent
from scipy.linalg import orthogonal_procrustes
from scipy.sparse import csr_matrix

# from scipy.sparse.csgraph import shortest_path  # replaced by cuGraph for speed
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors

# Reference UMAP implementation
from umap.umap_ import nearest_neighbors as umap_nearest_neighbors
from umap.umap_ import spectral_layout

from cuml.manifold import SpectralEmbedding

# cuML implementation
from cuml.manifold.simpl_set import (
    fuzzy_simplicial_set as cu_fuzzy_simplicial_set,
)
from cuml.metrics import trustworthiness


def compute_knn_metrics(
    knn_graph_a,
    knn_graph_b,
    n_neighbors: int,
) -> t.Tuple[float, float]:
    """
    Compute average neighbor recall and mean absolute distance error between two KNN results.

    Parameters
    ----------
    knn_graph_a : Tuple[np.ndarray, np.ndarray]
        Tuple of (distances, indices) for method A, each with shape (n_samples, n_neighbors).
    knn_graph_b : Tuple[np.ndarray, np.ndarray]
        Tuple of (distances, indices) for method B, each with shape (n_samples, n_neighbors).
    n_neighbors : int
        Number of neighbors per sample (k).

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


def compare_spectral_embeddings(
    fuzzy_graph_cpu, n_components=2, n_neighbors=15, random_state=42
):
    """Compare UMAP's spectral_layout with cuML's SpectralEmbedding.

    Parameters
    ----------
    fuzzy_graph_cpu : csr_matrix
        Precomputed fuzzy simplicial set graph (CPU)
    n_components : int, default=2
        Number of embedding dimensions
    n_neighbors : int, default=15
        Number of neighbors (ignored when affinity="precomputed")
    random_state : int, default=42
        Random state for reproducibility

    Returns
    -------
    dict with keys: ref_embedding, cu_embedding, rmse, correlations, stats
    """
    # Reference UMAP spectral layout
    ref_spectral_init = spectral_layout(
        data=None,
        graph=fuzzy_graph_cpu,
        dim=n_components,
        random_state=np.random.RandomState(random_state),
    )

    # cuML SpectralEmbedding
    se = SpectralEmbedding(
        affinity="precomputed",
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    cu_spectral_init = se.fit_transform(fuzzy_graph_cpu)

    # Convert to numpy arrays
    ref_init_np = (
        cp.asnumpy(ref_spectral_init)
        if isinstance(ref_spectral_init, cp.ndarray)
        else np.asarray(ref_spectral_init, dtype=np.float32)
    )
    cu_init_np = (
        cp.asnumpy(cu_spectral_init)
        if isinstance(cu_spectral_init, cp.ndarray)
        else np.asarray(cu_spectral_init, dtype=np.float32)
    )

    # Validate shapes
    expected_shape = (fuzzy_graph_cpu.shape[0], n_components)
    if ref_init_np.shape != expected_shape:
        raise ValueError(
            f"Reference embedding shape {ref_init_np.shape} != expected {expected_shape}"
        )
    if cu_init_np.shape != expected_shape:
        raise ValueError(
            f"cuML embedding shape {cu_init_np.shape} != expected {expected_shape}"
        )

    # Center and normalize
    def center_and_normalize(arr):
        centered = arr - arr.mean(axis=0, keepdims=True)
        std = np.std(centered, axis=0, keepdims=True)
        std = np.where(std > 1e-8, std, 1.0)
        return centered / std

    ref_norm = center_and_normalize(ref_init_np)
    cu_norm = center_and_normalize(cu_init_np)

    # Compute metrics
    rmse = procrustes_rmse(ref_norm, cu_norm)

    correlations = []
    for dim in range(n_components):
        ref_col, cu_col = ref_norm[:, dim], cu_norm[:, dim]
        if np.std(ref_col) < 1e-10 or np.std(cu_col) < 1e-10:
            corr = 0.0
        else:
            corr = np.corrcoef(ref_col, cu_col)[0, 1]
            if not np.isfinite(corr):
                corr = 0.0
        correlations.append(corr)

    # Compute statistics
    ref_stats = {
        "mean": np.mean(ref_norm, axis=0).tolist(),
        "std": np.std(ref_norm, axis=0).tolist(),
        "min": np.min(ref_norm, axis=0).tolist(),
        "max": np.max(ref_norm, axis=0).tolist(),
    }
    cu_stats = {
        "mean": np.mean(cu_norm, axis=0).tolist(),
        "std": np.std(cu_norm, axis=0).tolist(),
        "min": np.min(cu_norm, axis=0).tolist(),
        "max": np.max(cu_norm, axis=0).tolist(),
    }

    return {
        "ref_embedding": ref_norm,
        "cu_embedding": cu_norm,
        "rmse": rmse,
        "correlations": correlations,
        "stats": {"ref": ref_stats, "cu": cu_stats},
    }


def continuity_score(
    hr_indices: np.ndarray,
    lr_indices: np.ndarray,
    n_total: int,
    k_including_self: int,
) -> float:
    """Compute continuity where k includes the point itself.

    Converts to the standard continuity definition that excludes self by using k' = k-1
    for both high- and low-dimensional rankings and applies the usual normalization.
    """
    k_excluding_self = k_including_self - 1
    if k_excluding_self <= 0:
        return 1.0

    penalty_sum = 0.0
    for i, (hr_row, lr_row) in enumerate(zip(hr_indices, lr_indices)):
        lr_order_excl = [int(idx) for idx in lr_row if int(idx) != i]
        lr_topk_excl = set(lr_order_excl[:k_excluding_self])
        lr_rank_map = {idx: pos + 1 for pos, idx in enumerate(lr_order_excl)}

        hr_order_excl = [int(idx) for idx in hr_row if int(idx) != i]
        hr_topk_excl = set(hr_order_excl[:k_excluding_self])

        missing_in_low = hr_topk_excl - lr_topk_excl
        for j in missing_in_low:
            r_excl = lr_rank_map.get(j, n_total)
            # Accumulate raw penalties; global normalization is applied below
            penalty_sum += r_excl - k_excluding_self

    norm = 2.0 / (
        n_total * k_excluding_self * (2 * n_total - 3 * k_excluding_self - 1)
    )
    return 1.0 - norm * penalty_sum


def _align_edge_weights(g1, g2, reduce: str = "max"):
    """Align undirected edge weights of two fuzzy graphs and return arrays (p, q).

    For each graph, build a mapping from an undirected edge (min(i,j), max(i,j))
    to a single weight. If both directions exist, combine them via `reduce`:
    - "max": take the maximum of the two directed weights (UMAP-style fuzzy union)
    - any other value: sum the directed weights
    """

    def to_coo(m):
        if hasattr(m, "get"):
            m = m.get()
        return m.tocoo()

    c1, c2 = to_coo(g1), to_coo(g2)

    def build_dict(c):
        d = {}
        for i, j, v in zip(c.row, c.col, c.data):
            if i == j:
                continue
            a, b = int(i), int(j)
            k = (min(a, b), max(a, b))
            if reduce == "max":
                d[k] = max(float(v), d.get(k, 0.0))
            else:
                d[k] = float(v) + d.get(k, 0.0)
        return d

    d1, d2 = build_dict(c1), build_dict(c2)
    edges = d1.keys() | d2.keys()
    p = np.array([d1.get(e, 0.0) for e in edges])
    q = np.array([d2.get(e, 0.0) for e in edges])
    mask = np.isfinite(p) & np.isfinite(q)
    return p[mask], q[mask]


def compute_fuzzy_kl_divergence(
    g1, g2, eps: float = 1e-8, average: bool = False
) -> float:
    """KL divergence KL(P||Q) between aligned Bernoulli edge weights of g1 and g2.

    Returns the sum over edges by default; set average=True for mean per-edge KL.
    """
    p, q = _align_edge_weights(g1, g2)
    if p.size == 0:
        return 0.0
    p = np.clip(p, eps, 1 - eps).astype(np.float64, copy=False)
    q = np.clip(q, eps, 1 - eps).astype(np.float64, copy=False)
    per_edge = p * (np.log(p) - np.log(q)) + (1.0 - p) * (
        np.log1p(-p) - np.log1p(-q)
    )
    per_edge = per_edge[np.isfinite(per_edge)]
    if per_edge.size == 0:
        return 0.0
    return float(np.mean(per_edge) if average else np.sum(per_edge))


def compute_fuzzy_kl_sym(
    g1, g2, eps: float = 1e-8, average: bool = False
) -> float:
    """Symmetric KL divergence: KL(P||Q) + KL(Q||P)."""
    return compute_fuzzy_kl_divergence(
        g1, g2, eps=eps, average=average
    ) + compute_fuzzy_kl_divergence(g2, g1, eps=eps, average=average)


def compute_fuzzy_js_divergence(
    g1, g2, eps: float = 1e-8, average: bool = False
) -> float:
    """Jensen–Shannon divergence between aligned Bernoulli edge weights.

    Returns the sum over edges by default; set average=True for mean per-edge JSD.
    """
    p, q = _align_edge_weights(g1, g2)
    if p.size == 0:
        return 0.0
    p = np.clip(p, eps, 1 - eps).astype(np.float64, copy=False)
    q = np.clip(q, eps, 1 - eps).astype(np.float64, copy=False)
    m = 0.5 * (p + q)

    def kl(a, b):
        return np.sum(
            a * (np.log(a) - np.log(b))
            + (1.0 - a) * (np.log1p(-a) - np.log1p(-b))
        )

    js = 0.5 * (kl(p, m) + kl(q, m))
    return float(js / p.size) if average else float(js)


def compute_edge_jaccard(g1, g2, eps: float = 0.0) -> float:
    """Jaccard index over undirected edges with weight > eps in g1 and g2."""

    def to_edge_set(m):
        if hasattr(m, "get"):
            m = m.get()
        c = m.tocoo()
        s = set()
        for i, j, v in zip(c.row, c.col, c.data):
            if i == j or float(v) <= eps:
                continue
            a, b = int(i), int(j)
            s.add((min(a, b), max(a, b)))
        return s

    e1, e2 = to_edge_set(g1), to_edge_set(g2)
    if not e1 and not e2:
        return 1.0
    union = len(e1 | e2)
    if union == 0:
        return 0.0
    inter = len(e1 & e2)
    return inter / union


def compute_fuzzy_simplicial_set_metrics(ref_fss_graph, cu_fss_graph):
    # Symmetric KL divergence between the two fuzzy graphs
    kl_sym = compute_fuzzy_kl_sym(ref_fss_graph, cu_fss_graph)

    # Jaccard over undirected edges (ignore near-zero weights for stability)
    jacc = compute_edge_jaccard(ref_fss_graph, cu_fss_graph, eps=1e-6)

    # Row-sum relative L1: average relative difference of total membership mass per node
    sums_ref = np.asarray(ref_fss_graph.sum(axis=1)).ravel()
    sums_cu = np.asarray(cu_fss_graph.sum(axis=1)).ravel()
    denom = np.maximum(np.abs(sums_ref), 1e-12)
    row_l1 = float(np.mean(np.abs(sums_ref - sums_cu) / denom))

    return kl_sym, jacc, row_l1


def procrustes_rmse(A, B):
    A0 = A - A.mean(axis=0, keepdims=True)
    B0 = B - B.mean(axis=0, keepdims=True)
    A0 /= np.linalg.norm(A0)
    B0 /= np.linalg.norm(B0)
    R, _ = orthogonal_procrustes(B0, A0)
    Balign = B0 @ R
    err = np.linalg.norm(A0 - Balign) / np.sqrt(A0.shape[0])
    return float(err)


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


def _build_knn_with_cuvs(
    X: t.Union[cp.ndarray, np.ndarray], k: int, metric: str, backend: str
) -> t.Tuple[np.ndarray, np.ndarray]:
    # Ensure X is on device for non-all_neighbors backends
    if backend != "all_neighbors":
        X_device = cp.asarray(X) if isinstance(X, np.ndarray) else X

    if backend == "bruteforce":
        index = brute_force.build(X_device, metric=metric)
        knn_dists, knn_indices = brute_force.search(index, X_device, k)
        return cp.asnumpy(knn_dists), cp.asnumpy(knn_indices)

    if backend == "nn_descent":
        params = nn_descent.IndexParams(metric=metric)
        index = nn_descent.build(params, X_device)
        knn_indices = cp.asarray(index.graph[:, :k])
        knn_dists = cp.asarray(index.distances[:, :k])
        return cp.asnumpy(knn_dists), cp.asnumpy(knn_indices)

    if backend == "all_neighbors":
        n_rows = X.shape[0]

        nn_descent_params = nn_descent.IndexParams(
            metric=metric,
            graph_degree=64,
            intermediate_graph_degree=128,
            max_iterations=20,
            termination_threshold=0.0001,
        )

        # Create all_neighbors parameters
        params = all_neighbors.AllNeighborsParams(
            algo="nn_descent",
            overlap_factor=2,
            n_clusters=8,
            metric=metric,
            nn_descent_params=nn_descent_params,
        )

        # Create MultiGpuResources for SNMG-only operation
        resources = MultiGpuResources()

        # Ensure data is on host for multi-GPU all_neighbors
        X_host = X if isinstance(X, np.ndarray) else cp.asnumpy(X)

        # Build all-neighbors graph with SNMG
        indices, distances = all_neighbors.build(
            X_host,
            k,
            params,
            distances=cp.empty((n_rows, k), dtype=cp.float32),
            resources=resources,
        )

        return cp.asnumpy(distances), cp.asnumpy(indices)

    raise ValueError(f"Unknown backend: {backend}")


def _build_symmetric_csr_from_knn(
    knn_indices: np.ndarray, knn_dists: np.ndarray, n: int
) -> csr_matrix:
    """Build a symmetric CSR graph from knn arrays, dropping self-edges."""
    rows = np.repeat(np.arange(n), knn_indices.shape[1])
    cols = knn_indices.ravel()
    data = knn_dists.ravel()

    # drop self-edges if present
    mask = rows != cols
    rows, cols, data = rows[mask], cols[mask], data[mask]

    # symmetrize (undirected)
    rows2 = np.concatenate([rows, cols])
    cols2 = np.concatenate([cols, rows])
    data2 = np.concatenate([data, data])

    return csr_matrix((data2, (rows2, cols2)), shape=(n, n))


def _compute_geodesic_correlations(
    Y_cp: cp.ndarray,
    hd_inds_np: np.ndarray,
    hd_dists_np: np.ndarray,
    n: int,
) -> t.Tuple[float, float]:
    rng = np.random.RandomState(42)
    subset_size = int(max(2000, 0.1 * n))
    subset_size = min(subset_size, n)
    subset = rng.choice(n, size=subset_size, replace=False)

    # Build undirected weighted edge list from kNN (drop self-edges and symmetrize)
    rows = np.repeat(np.arange(n), hd_inds_np.shape[1])
    cols = hd_inds_np.ravel()
    data = hd_dists_np.ravel().astype(np.float32, copy=False)

    mask = rows != cols
    rows, cols, data = rows[mask], cols[mask], data[mask]

    # Restrict to vertex-induced subgraph
    in_sub = np.zeros(n, dtype=bool)
    in_sub[subset] = True
    keep = in_sub[rows] & in_sub[cols]

    # Map to local [0, subset_size) ids
    orig_to_local = -np.ones(n, dtype=np.int32)
    orig_to_local[subset] = np.arange(subset_size, dtype=np.int32)
    src_sub = orig_to_local[rows[keep]]
    dst_sub = orig_to_local[cols[keep]]
    w_sub = data[keep]

    # Symmetrize
    src_all = np.concatenate([src_sub, dst_sub])
    dst_all = np.concatenate([dst_sub, src_sub])
    w_all = np.concatenate([w_sub, w_sub]).astype(np.float32, copy=False)

    # cuGraph on GPU
    edgelist = cudf.DataFrame(
        {
            "src": cp.asarray(src_all),
            "dst": cp.asarray(dst_all),
            "w": cp.asarray(w_all),
        }
    )
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(
        edgelist,
        source="src",
        destination="dst",
        edge_attr="w",
        renumber=False,
    )

    # Sources inside the subgraph (heuristic applied to subset)
    num_sources = int(max(256, 0.05 * subset_size))
    num_sources = max(1, min(subset_size - 1, num_sources))
    sources_local = rng.choice(subset_size, size=num_sources, replace=False)

    # Compute distances from multiple sources with per-source SSSP (version-compatible)
    high_geo = np.full((num_sources, subset_size), np.inf, dtype=np.float32)
    for r, s in enumerate(sources_local):
        res = cugraph.sssp(G, source=int(s))
        v_np = res["vertex"].to_numpy()
        d_np = res["distance"].to_numpy()
        if d_np.dtype != np.float32:
            d_np = d_np.astype(np.float32, copy=False)
        high_geo[r, v_np] = d_np

    # Low-d Euclidean distances on GPU, restricted to the same subset/sources
    Y_cp_sub = Y_cp[subset]
    low_d_cp = pairwise_distance(
        Y_cp_sub[sources_local], Y_cp_sub, metric="euclidean"
    )
    low_d = cp.asnumpy(low_d_cp).astype(np.float32, copy=False)

    # Build mask: exclude self-distances for each source row and non-finite geodesics
    valid = np.isfinite(high_geo)
    for r, s in enumerate(sources_local):
        valid[r, s] = False

    sp, _ = spearmanr(high_geo[valid], low_d[valid])
    pe, _ = pearsonr(high_geo[valid], low_d[valid])
    return float(sp), float(pe)


def compute_simplicial_set_embedding_metrics(
    high_dim_data,
    embedding,
    k,
    metric,
    skip_topology_preservation=False,
):
    """
    Assess the quality of UMAP embeddings with GPU acceleration.

    Metric families:
    1) Local Structure Preservation: trustworthiness, continuity
    2) Global Structure Preservation: geodesic Spearman/Pearson correlations (DEMaP = Pearson)
    3) Fuzzy Simplicial Sets Cross-Entropy: KL divergences between high-d and low-d fuzzy graphs
    4) Topology Preservation via Persistent Homology: persistence diagrams and Betti numbers (H0, H1)

    Parameters
    ----------
    high_dim_data : array-like or cupy.ndarray, shape (n_samples, n_features)
    embedding : array-like or cupy.ndarray, shape (n_samples, n_components)
    k : int
        Number of neighbors (includes self for continuity; helper adjusts internally).
    metric : str
        Metric used for high-dimensional KNN (e.g., "euclidean", "cosine").
    skip_topology_preservation : bool
        If True, skip persistent homology metrics.

    Notes
    -----
    Geodesic correlations are computed from a subsample of sources and a subset of the graph for speed,
    using max(256, floor(0.05 * n)) sources (capped at n-1) with a fixed internal seed for reproducibility.

    Returns
    -------
    dict
    """
    # Ensure GPU arrays
    X_cp = cp.asarray(high_dim_data, dtype=cp.float32)
    Y_cp = cp.asarray(embedding, dtype=cp.float32)
    n = int(X_cp.shape[0])

    metrics = {}

    # 1) Local Structure Preservation
    # 1.a) Trustworthiness (GPU)
    trust = trustworthiness(X_cp, Y_cp, n_neighbors=k)
    metrics["trustworthiness"] = float(trust)

    # 1.b) Continuity: high-d and low-d KNN via cuVS; continuity_score expects NumPy indices; k includes self
    hd_dists_np, hd_inds_np = _build_knn_with_cuvs(
        X_cp, k=k, metric=metric, backend="bruteforce"
    )
    ld_dists_np, ld_inds_np = _build_knn_with_cuvs(
        Y_cp, k=k, metric="euclidean", backend="bruteforce"
    )
    cont = continuity_score(hd_inds_np, ld_inds_np, n, k)
    metrics["continuity"] = float(cont)

    # 2) Global Structure Preservation
    sp, pe = _compute_geodesic_correlations(Y_cp, hd_inds_np, hd_dists_np, n)
    metrics["geodesic_spearman_correlation"] = float(sp)
    metrics["geodesic_pearson_correlation"] = float(pe)
    metrics["demap"] = float(pe)

    # 3) Fuzzy Simplicial Sets KL divergences (GPU)
    hg = cu_fuzzy_simplicial_set(
        X_cp,
        n_neighbors=k,
        random_state=42,
        metric=metric,
        knn_indices=cp.asarray(hd_inds_np),
        knn_dists=cp.asarray(hd_dists_np),
    )
    lg = cu_fuzzy_simplicial_set(
        Y_cp,
        n_neighbors=k,
        random_state=42,
        metric="euclidean",
        knn_indices=cp.asarray(ld_inds_np),
        knn_dists=cp.asarray(ld_dists_np),
    )
    metrics["fuzzy_kl_divergence"] = float(compute_fuzzy_kl_divergence(hg, lg))
    metrics["fuzzy_sym_kl_divergence"] = float(compute_fuzzy_kl_sym(hg, lg))

    # 4) Topology Preservation via Persistent Homology (CPU via ripser)

    if not skip_topology_preservation:
        try:
            from ripser import ripser
        except ImportError:
            raise ImportError(
                "Warning: ripser library not found. Persistence homology metrics require this library."
            )

        X_np = cp.asnumpy(X_cp)
        Y_np = cp.asnumpy(Y_cp)
        high_pd = ripser(X_np, maxdim=1)["dgms"]
        low_pd = ripser(Y_np, maxdim=1)["dgms"]
        metrics["high_pd"] = high_pd
        metrics["low_pd"] = low_pd

        # Betti numbers (subtract infinite bar from H0)
        betti_high_h0 = len(high_pd[0]) - 1
        betti_high_h1 = len(high_pd[1]) if len(high_pd) > 1 else 0
        betti_low_h0 = len(low_pd[0]) - 1
        betti_low_h1 = len(low_pd[1]) if len(low_pd) > 1 else 0

        metrics["betti_h0_high"] = int(betti_high_h0)
        metrics["betti_h1_high"] = int(betti_high_h1)
        metrics["betti_h0_low"] = int(betti_low_h0)
        metrics["betti_h1_low"] = int(betti_low_h1)

    return metrics
