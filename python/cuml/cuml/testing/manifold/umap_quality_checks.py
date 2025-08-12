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

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from umap.umap_ import find_ab_params


def compute_knn_recall(ground_truth, retrieved):
    assert ground_truth.shape == retrieved.shape

    n, k = ground_truth.shape
    correct = np.array(
        [np.isin(retrieved[i], ground_truth[i]).sum() for i in range(n)]
    )
    recalls = correct / k
    return recalls.mean()


def compute_fuzzy_cross_entropy(g_high, g_low, eps=1e-8):
    """Compute cross-entropy between two fuzzy simplicial set graphs.

    The graphs are sparse adjacency matrices whose entries are membership
    strengths in [0, 1]. We align edges by their (unordered) vertex pair
    so that weights are compared correctly even if the two graphs contain
    different neighbour relationships.

    Parameters
    ----------
    g_high, g_low : scipy.sparse.csr_matrix or cupy.sparse matrix (square)
        Fuzzy simplicial set graphs of the high-dimensional data and the
        low-dimensional embedding respectively. Can be either NumPy/SciPy
        or CuPy sparse matrices.
    eps : float
        Small constant to avoid log(0). Default 1e-8.
    """

    def to_scipy_coo(sparse_matrix):
        if hasattr(sparse_matrix, "get"):
            sparse_matrix = sparse_matrix.get()
        return sparse_matrix.tocoo()

    coo_h = to_scipy_coo(g_high)
    coo_l = to_scipy_coo(g_low)

    # Build dictionaries keyed by unordered edge (min, max)
    h_dict = {
        (min(int(i), int(j)), max(int(i), int(j))): v
        for i, j, v in zip(coo_h.row, coo_h.col, coo_h.data)
        if i != j
    }
    l_dict = {
        (min(int(i), int(j)), max(int(i), int(j))): v
        for i, j, v in zip(coo_l.row, coo_l.col, coo_l.data)
        if i != j
    }

    # Union of all edges appearing in either graph
    all_edges = set(h_dict.keys()).union(l_dict.keys())

    # Gather aligned weight arrays
    p_vals = np.array([h_dict.get(edge, 0.0) for edge in all_edges])
    q_vals = np.array([l_dict.get(edge, 0.0) for edge in all_edges])

    # Filter out any non-finite entries to avoid NaNs propagating in calculations
    finite_mask = np.isfinite(p_vals) & np.isfinite(q_vals)
    if not np.any(finite_mask):
        return 0.0
    p_vals = p_vals[finite_mask]
    q_vals = q_vals[finite_mask]

    # Simple but robust clipping
    p_vals = np.clip(p_vals, eps, 1 - eps)
    q_vals = np.clip(q_vals, eps, 1 - eps)

    # Check if distributions are very similar to avoid NaN
    if np.allclose(p_vals, q_vals, rtol=1e-10, atol=1e-12):
        return 0.0

    # Cross-entropy computation in float64 with per-edge masking for stability
    p64 = p_vals.astype(np.float64, copy=False)
    q64 = q_vals.astype(np.float64, copy=False)
    logp = np.log(p64)
    logq = np.log(q64)
    log1mp = np.log1p(-p64)
    log1mq = np.log1p(-q64)
    per_edge = p64 * (logp - logq) + (1.0 - p64) * (log1mp - log1mq)
    per_edge = per_edge[np.isfinite(per_edge)]
    if per_edge.size == 0:
        return 0.0
    return float(np.sum(per_edge))


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


def evaluate_umap_embeddings(high_dim_data, embedding, k=15):
    """
    Assess the quality of UMAP embeddings based on various metrics.

    Parameters:
    - high_dim_data: np.array, high-dimensional input data (n_samples, n_features)
    - embedding: np.array, low-dimensional UMAP embedding (n_samples, n_components, typically 2 or 3)
    - k: int, number of neighbors for local metrics and graphs (default=15)

    Returns:
    - dict: Dictionary containing computed metrics
    """
    try:
        from ripser import ripser
    except ImportError:
        raise ImportError(
            "Warning: ripser library not found. Persistence homology metrics require this library."
        )

    n_samples = high_dim_data.shape[0]
    metrics = {}

    # 1. Local Structure Preservation

    # Trustworthiness
    trust = trustworthiness(high_dim_data, embedding, n_neighbors=k)
    metrics["trustworthiness"] = trust

    # Continuity measures how many high-dim neighbors are preserved in low-dim
    high_dist = pairwise_distances(high_dim_data)
    low_dist = pairwise_distances(embedding)

    high_ranks = np.argsort(high_dist, axis=1)[:, 1 : k + 1]  # Exclude self
    low_ranks = np.argsort(low_dist, axis=1)[:, 1 : k + 1]

    continuity_sum = 0
    for i in range(n_samples):
        missing_neighbors = set(high_ranks[i]) - set(low_ranks[i])
        for j in missing_neighbors:
            rank_in_low = np.where(low_ranks[i] == j)[0]
            if len(rank_in_low) == 0:
                rank_in_low = n_samples  # Worst case
            else:
                rank_in_low = rank_in_low[0] + 1
            continuity_sum += (rank_in_low - k) / (n_samples - k - 1)

    norm = 2 / (n_samples * k * (2 * n_samples - 3 * k - 1))
    continuity = 1 - norm * continuity_sum
    metrics["continuity"] = continuity

    # 2. Global Structure Preservation

    # Geodesic Distance Correlation
    # Compute graph-based distances in high-dim (approximate geodesic)
    high_knn_graph = kneighbors_graph(
        high_dim_data, n_neighbors=k, mode="distance"
    )
    high_geodesic_dist = shortest_path(high_knn_graph, directed=False)

    # Euclidean in low-dim
    low_euclidean_dist = pairwise_distances(embedding)

    # Flatten and correlate (using Spearman for rank correlation)
    high_flat = high_geodesic_dist.flatten()
    low_flat = low_euclidean_dist.flatten()
    valid_mask = np.isfinite(high_flat) & np.isfinite(low_flat)
    spearman_corr, _ = spearmanr(high_flat[valid_mask], low_flat[valid_mask])
    pearson_corr, _ = pearsonr(high_flat[valid_mask], low_flat[valid_mask])
    metrics["geodesic_spearman_correlation"] = spearman_corr
    metrics["geodesic_pearson_correlation"] = pearson_corr

    # DEMaP (simple normalized correlation across distances)
    metrics["demap"] = pearson_corr

    # 3. Fuzzy Simplicial Sets Cross-Entropy

    # Reconstruct fuzzy simplicial sets
    from umap.umap_ import fuzzy_simplicial_set

    high_graph, _, _ = fuzzy_simplicial_set(
        high_dim_data, n_neighbors=k, random_state=42, metric="euclidean"
    )
    high_graph = csr_matrix(high_graph)

    low_graph, _, _ = fuzzy_simplicial_set(
        embedding, n_neighbors=k, random_state=42, metric="euclidean"
    )
    low_graph = csr_matrix(low_graph)

    metrics["fuzzy_cross_entropy"] = compute_fuzzy_cross_entropy(
        high_graph, low_graph
    )

    # 4. Topology Preservation via Persistent Homology
    # Compute persistence diagrams for high and low dim using ripser
    high_pd = ripser(high_dim_data, maxdim=1)["dgms"]
    low_pd = ripser(embedding, maxdim=1)["dgms"]
    metrics["high_pd"] = high_pd
    metrics["low_pd"] = low_pd

    # Compare Betti numbers (simple count of persistent features)
    betti_high_h0 = len(high_pd[0]) - 1  # Subtract infinite bar
    betti_high_h1 = len(high_pd[1])
    betti_low_h0 = len(low_pd[0]) - 1
    betti_low_h1 = len(low_pd[1])

    metrics["betti_h0_high"] = betti_high_h0
    metrics["betti_h1_high"] = betti_high_h1
    metrics["betti_h0_low"] = betti_low_h0
    metrics["betti_h1_low"] = betti_low_h1

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
    metrics = evaluate_umap_embeddings(
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

    metrics["knn_recall"] = compute_knn_recall(ref_knn_graph[1], knn_graph[1])
    metrics["fuzzy_cross_entropy_refwise"] = compute_fuzzy_cross_entropy(
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
