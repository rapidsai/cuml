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

import cupy as cp
import numpy as np
from cuvs.distance import pairwise_distance
from cuvs.neighbors import brute_force, nn_descent
from scipy.linalg import orthogonal_procrustes
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr, spearmanr
from umap.umap_ import find_ab_params

from cuml.manifold.simpl_set import (
    fuzzy_simplicial_set as cu_fuzzy_simplicial_set,
)
from cuml.metrics import trustworthiness


def compute_knn_metrics(
    inds_a: np.ndarray,
    dists_a: np.ndarray,
    inds_b: np.ndarray,
    dists_b: np.ndarray,
    n_neighbors: int,
) -> t.Tuple[float, float]:
    """
    Compute average neighbor recall and mean absolute distance error between two KNN results.

    Parameters
    ----------
    inds_a, dists_a : np.ndarray
        Neighbor indices and distances for method A with shape (n_samples, n_neighbors).
    inds_b, dists_b : np.ndarray
        Neighbor indices and distances for method B with shape (n_samples, n_neighbors).
    n_neighbors : int
        Number of neighbors per sample (k).

    Returns
    -------
    (avg_recall, mae_dist) : Tuple[float, float]
        Average recall across rows and mean absolute error of distances for intersecting neighbors.
    """
    recalls: t.List[float] = []
    distance_abs_errors: t.List[float] = []

    for i in range(inds_a.shape[0]):
        row_inds_a = inds_a[i]
        row_inds_b = inds_b[i]

        set_a = set(int(x) for x in row_inds_a)
        set_b = set(int(x) for x in row_inds_b)
        intersect = set_a & set_b

        recalls.append(len(intersect) / float(n_neighbors))

        if not intersect:
            continue

        # Map index -> distance position for matched neighbors in both results
        pos_a = {int(idx): j for j, idx in enumerate(row_inds_a)}
        pos_b = {int(idx): j for j, idx in enumerate(row_inds_b)}
        for idx in intersect:
            da = float(dists_a[i, pos_a[idx]])
            db = float(dists_b[i, pos_b[idx]])
            distance_abs_errors.append(abs(da - db))

    avg_recall = float(np.mean(recalls)) if recalls else 0.0
    mae_dist = (
        float(np.mean(distance_abs_errors)) if distance_abs_errors else 0.0
    )

    return avg_recall, mae_dist


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
            penalty_sum += (r_excl - k_excluding_self) / (
                n_total - k_excluding_self - 1
            )

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

    # Jaccard over undirected edges
    jacc = compute_edge_jaccard(ref_fss_graph, cu_fss_graph, eps=0.0)

    # Row-sum L1: average absolute difference of total membership mass per node
    row_l1 = float(
        np.mean(
            np.abs(
                np.asarray(ref_fss_graph.sum(axis=1)).ravel()
                - np.asarray(cu_fss_graph.sum(axis=1)).ravel()
            )
        )
    )

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


def compute_simplicial_set_embedding_metrics(
    high_dim_data, embedding, k, metric, skip_topolgy_preservation=False
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

    Returns
    -------
    dict
    """
    # Ensure GPU arrays
    X_cp = (
        high_dim_data
        if isinstance(high_dim_data, cp.ndarray)
        else cp.asarray(high_dim_data, dtype=cp.float32)
    )
    Y_cp = (
        embedding
        if isinstance(embedding, cp.ndarray)
        else cp.asarray(embedding, dtype=cp.float32)
    )
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
    # 2.a) Geodesic distances from symmetric high-d KNN graph
    high_knn_graph = _build_symmetric_csr_from_knn(hd_inds_np, hd_dists_np, n)
    high_geo = shortest_path(high_knn_graph, directed=False).astype(
        np.float32, copy=False
    )

    # 2.b) Low-d full Euclidean distance matrix on GPU
    low_d_cp = pairwise_distance(Y_cp, Y_cp, metric="euclidean")
    low_d = cp.asnumpy(low_d_cp).astype(np.float32, copy=False)

    mask = np.isfinite(high_geo)
    sp, _ = spearmanr(high_geo[mask], low_d[mask])
    pe, _ = pearsonr(high_geo[mask].ravel(), low_d[mask].ravel())
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

    if not skip_topolgy_preservation:
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


def run_umap_pipeline(X, implementation, **umap_params):
    """
    Compute UMAP embeddings and assess quality.

    Parameters:
    - X: input data
    - implementation: "reference" or "cuml"
    - umap_params: dictionary of UMAP parameters

    Returns:
    - knn_graph: KNN graph
    - fuzzy_graph: fuzzy simplicial set
    - spectral_init: spectral initialization from fuzzy graph
    - embedding: UMAP embedding
    """

    if implementation == "reference":
        from umap.spectral import spectral_layout
        from umap.umap_ import (
            fuzzy_simplicial_set,
            nearest_neighbors,
            simplicial_set_embedding,
        )
    elif implementation == "cuml":
        from cuml.manifold.simpl_set import (
            fuzzy_simplicial_set,
            simplicial_set_embedding,
        )
        from cuml.neighbors import NearestNeighbors

        # from cuml.manifold.spectral_embedding import spectral_embedding
    else:
        raise ValueError(f"Invalid implementation: {implementation}")

    print(f"Running {implementation} UMAP pipeline...")

    # ------------------------------------------------------------------
    # Extract UMAP-related parameters (with defaults)
    # ------------------------------------------------------------------
    n_neighbors = umap_params.get("k", umap_params.get("n_neighbors", 15))
    min_dist = umap_params.get("min_dist", 0.1)
    spread = umap_params.get("spread", 1.0)
    n_components = umap_params.get("n_components", 2)
    n_epochs = umap_params.get("n_epochs", 500)
    init = umap_params.get("init", "spectral")
    negative_sample_rate = umap_params.get("negative_sample_rate", 5)
    gamma = umap_params.get("gamma", 1.0)
    metric = umap_params.get("metric", "euclidean")
    random_state = umap_params.get("random_state", np.random.RandomState(42))
    # Parameters that control the attraction/repulsion curve.
    a, b = find_ab_params(spread=spread, min_dist=min_dist)

    # ------------------------------------------------------------------
    # STEP 1: Nearest-neighbors search (high-dimensional space)
    # ------------------------------------------------------------------
    print("  Computing k-nearest neighbors (KNN) ...")

    if implementation == "reference":
        # Use UMAP's nearest neighbor descent algorithm
        knn_indices, knn_dists, _ = nearest_neighbors(
            X,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds={},
            angular=False,
            random_state=random_state,
        )
    else:
        # Use cuML's NearestNeighbors
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn.fit(X)
        knn_dists, knn_indices = nn.kneighbors(
            X, n_neighbors=n_neighbors, return_distance=True
        )

    knn_graph = (knn_dists, knn_indices)

    # ------------------------------------------------------------------
    # STEP 2: Fuzzy simplicial set construction using pre-computed KNN
    # ------------------------------------------------------------------
    print("  Computing fuzzy simplicial set ...")
    fuzzy_results = fuzzy_simplicial_set(
        X,
        n_neighbors=n_neighbors,
        random_state=random_state,
        metric=metric,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )

    if implementation == "reference":
        fuzzy_graph = fuzzy_results[0]
    else:
        fuzzy_graph = fuzzy_results

    # ------------------------------------------------------------------
    # STEP 2.5: Spectral initialization from fuzzy graph
    # ------------------------------------------------------------------
    print("  Computing spectral initialization from fuzzy graph ...")
    spectral_init = None
    if implementation == "reference":
        spectral_init = spectral_layout(
            data=X,
            graph=fuzzy_graph,
            dim=n_components,
            random_state=random_state,
        )
    else:
        """
        spectral_init = spectral_embedding(
            X,
            n_components=n_components,
            random_state=random_state,
            norm_laplacian=True,
            drop_first=True
        )
        """
        # run simplicial_set_embedding for 1 epoch to get spectral initialization
        spectral_init = simplicial_set_embedding(
            X,
            graph=fuzzy_graph,
            n_components=n_components,
            initial_alpha=1.0,
            a=a,
            b=b,
            gamma=gamma,
            negative_sample_rate=negative_sample_rate,
            n_epochs=1,
            init="spectral",
            random_state=random_state,
            metric=metric,
        )

    # ------------------------------------------------------------------
    # STEP 3: Low-dimensional embedding via simplicial_set_embedding
    # ------------------------------------------------------------------
    print("  Computing 2-D embedding from fuzzy simplicial set ...")

    # Use spectral initialization if init is "spectral"
    embedding_init = (
        spectral_init
        if spectral_init is not None and init == "spectral"
        else init
    )

    if implementation == "reference":
        additional_params = {
            "metric_kwds": {},
            "densmap": False,
            "densmap_kwds": {},
            "output_dens": False,
            "output_metric": metric,
            "output_metric_kwds": {},
        }
    else:
        additional_params = {}

    embedding_results = simplicial_set_embedding(
        X,
        graph=fuzzy_graph,
        n_components=n_components,
        initial_alpha=1.0,
        a=a,
        b=b,
        gamma=gamma,
        negative_sample_rate=negative_sample_rate,
        n_epochs=n_epochs,
        init=embedding_init,
        random_state=random_state,
        metric=metric,
        **additional_params,
    )

    if implementation == "reference":
        embedding = embedding_results[0]
    else:
        embedding = embedding_results

    return knn_graph, fuzzy_graph, spectral_init, embedding


def run_implementation(X, implementation, **umap_params):
    knn_graph, fuzzy_graph, spectral_init, embedding = run_umap_pipeline(
        X, implementation=implementation, **umap_params
    )
    metrics = compute_simplicial_set_embedding_metrics(
        X, embedding, k=umap_params["n_neighbors"]
    )
    return knn_graph, fuzzy_graph, spectral_init, embedding, metrics


def compare_implementations(X, **umap_params):
    (
        ref_knn_graph,
        ref_fuzzy_graph,
        ref_spectral_init,
        ref_embedding,
        ref_metrics,
    ) = run_implementation(X, implementation="reference", **umap_params)
    (
        knn_graph,
        fuzzy_graph,
        spectral_init,
        embedding,
        metrics,
    ) = run_implementation(X, implementation="cuml", **umap_params)

    metrics["knn_recall"] = compute_knn_metrics(ref_knn_graph[1], knn_graph[1])
    metrics["fuzzy_cross_entropy_refwise"] = compute_fuzzy_kl_divergence(
        ref_fuzzy_graph, fuzzy_graph
    )

    return {
        "ref_knn_graph": ref_knn_graph,
        "ref_fuzzy_graph": ref_fuzzy_graph,
        "ref_spectral_init": ref_spectral_init,
        "ref_embedding": ref_embedding,
        "ref_metrics": ref_metrics,
        "knn_graph": knn_graph,
        "fuzzy_graph": fuzzy_graph,
        "spectral_init": spectral_init,
        "embedding": embedding,
        "metrics": metrics,
    }
