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
from scipy.sparse import csr_matrix

# Reference UMAP implementation
from umap.umap_ import find_ab_params
from umap.umap_ import fuzzy_simplicial_set as ref_fuzzy_simplicial_set
from umap.umap_ import simplicial_set_embedding as ref_simplicial_set_embedding

# cuML implementation
from cuml.manifold.simpl_set import (
    fuzzy_simplicial_set as cu_fuzzy_simplicial_set,
)
from cuml.manifold.simpl_set import (
    simplicial_set_embedding as cu_simplicial_set_embedding,
)
from cuml.testing.manifold.umap_metrics import (
    _build_knn_with_cuvs,
    _build_knn_with_umap,
    compute_fuzzy_simplicial_set_metrics,
    compute_knn_metrics,
    compute_simplicial_set_embedding_metrics,
    procrustes_rmse,
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


# Precompute parameters and human-readable ids for the parametrized fixture
_CU_KNN_PARAMS = [
    (name, k, backend)
    for name in [
        "deep-image-96-angular",
        "fashion-mnist-784-euclidean",
        "gist-960-euclidean",
        "glove-25-angular",
        "mnist-784-euclidean",
        "sift-128-euclidean",
    ]
    for k in (10, 30)
    for backend in ("bruteforce", "nn_descent")
]
_CU_KNN_IDS = [
    f"{name}-k{k}-{backend}" for (name, k, backend) in _CU_KNN_PARAMS
]


@pytest.fixture(scope="module", params=_CU_KNN_PARAMS, ids=_CU_KNN_IDS)
def cu_knn_graph_fixture(request):
    dataset_name, k, backend = request.param
    ds_spec = _DATASET_SPECS.get(dataset_name, {})
    metric = ds_spec.get("metric")

    X_np = _load_dataset(dataset_name)
    X_cp = cp.asarray(X_np)

    # Precompute cuVS KNN for the selected backend
    knn_dists_np, knn_inds_np = _build_knn_with_cuvs(
        X_cp, k=k, metric=metric, backend=backend
    )

    return {
        "dataset_name": dataset_name,
        "metric": metric,
        "X_np": X_np,
        "X_cp": X_cp,
        "k": k,
        "backend": backend,
        "knn_dists_np": knn_dists_np,
        "knn_inds_np": knn_inds_np,
    }


@pytest.fixture(scope="module")
def cu_fuzzy_fixture(cu_knn_graph_fixture):
    d = cu_knn_graph_fixture
    metric = d["metric"]
    X_cp = d["X_cp"]
    k = d["k"]

    # Skip fuzzy graph/embedding tests for nn_descent backend while retaining it for KNN tests
    if d.get("backend") == "nn_descent":
        pytest.skip("Skipping cu_fuzzy_fixture for nn_descent backend")

    # cuML fuzzy graph (GPU native + CPU SciPy for ref embedding step) for the selected backend
    cu_graph_gpu = cu_fuzzy_simplicial_set(
        X_cp,
        n_neighbors=k,
        random_state=42,
        metric=metric,
        knn_indices=cp.asarray(d["knn_inds_np"]),
        knn_dists=cp.asarray(d["knn_dists_np"]),
    )
    cu_graph_cpu = (
        cu_graph_gpu.get() if hasattr(cu_graph_gpu, "get") else cu_graph_gpu
    )
    cu_graph_cpu = csr_matrix(cu_graph_cpu)

    return {
        **d,
        "cu_graph_gpu": cu_graph_gpu,
        "cu_graph_cpu": cu_graph_cpu,
    }


def test_knn(cu_knn_graph_fixture):
    d = cu_knn_graph_fixture
    dataset_name = d["dataset_name"]
    metric = d["metric"]
    X_np = d["X_np"]
    n_neighbors = d["k"]

    # cuVS KNN (precomputed in fixture for the selected backend)
    dists_cuvs, inds_cuvs = d["knn_dists_np"], d["knn_inds_np"]

    # UMAP KNN (CPU) computed on-the-fly with the same backend
    dists_umap, inds_umap = _build_knn_with_umap(
        X_np, k=n_neighbors, metric=metric, backend=d["backend"]
    )

    # Sanity: shapes
    assert dists_cuvs.shape == dists_umap.shape == (X_np.shape[0], n_neighbors)
    assert inds_cuvs.shape == inds_umap.shape == (X_np.shape[0], n_neighbors)

    # Compute KNN metrics via helper
    avg_recall, mae_dist = compute_knn_metrics(
        (dists_umap, inds_umap), (dists_cuvs, inds_cuvs), n_neighbors
    )

    # Tolerances: stricter for bruteforce, looser for NN-descent approx
    if d["backend"] == "bruteforce":
        recall_tol = 0.995
        dist_tol = 5e-5 if metric == "cosine" else 1e-5
    else:
        recall_tol = 0.95
        dist_tol = 5e-3 if metric == "cosine" else 1e-3

    assert avg_recall >= recall_tol, (
        f"Neighbor recall too low: {avg_recall:.3f} < {recall_tol:.3f} "
        f"for {dataset_name}, metric={metric}, backend={d['backend']}"
    )

    assert mae_dist <= dist_tol, (
        f"Mean abs distance error too high: {mae_dist:.3e} > {dist_tol:.3e} "
        f"for {dataset_name}, metric={metric}, backend={d['backend']}"
    )


def test_fuzzy_simplicial_set(cu_fuzzy_fixture):
    d = cu_fuzzy_fixture
    dataset_name = d["dataset_name"]
    metric = d["metric"]

    X_np = d["X_np"]
    n_neighbors = d["k"]

    # Use precomputed KNN for the selected backend
    knn_dists_np, knn_inds_np = d["knn_dists_np"], d["knn_inds_np"]

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

    # cuML fuzzy graph (GPU native + CPU SciPy for ref embedding step) precomputed in fixture
    cu_graph = d["cu_graph_cpu"]

    # Fuzzy simplicial set from similar KNN graphs should match very closely
    assert (
        ref_graph.todense() - cu_graph.todense()
    ).sum() < 1e-2, "Fuzzy graph mismatch"

    # Compute fuzzy simplicial set metrics : KL, Jaccard, Row-sum L1
    kl_sym, jacc, row_l1 = compute_fuzzy_simplicial_set_metrics(
        ref_graph, cu_graph
    )

    # Simple, global tolerances
    kl_tol = 1e-2
    j_tol = 0.90
    row_l1_tol = 1e-3

    # Assertions focused on matching the reference
    assert np.isfinite(kl_sym), "KL not finite"
    assert (
        kl_sym <= kl_tol
    ), f"KL(sym) too high: {kl_sym:.3e} > {kl_tol:.3e} for {dataset_name}, metric={metric}, backend={d['backend']}"
    assert (
        jacc >= j_tol
    ), f"Edge Jaccard too low (jacc={jacc:.3f} < tol={j_tol:.3f}) for {dataset_name}, metric={metric}, backend={d['backend']}"
    assert (
        row_l1 <= row_l1_tol
    ), f"Row-sum L1 too high (row_l1={row_l1:.3e} > tol={row_l1_tol:.3e}) for {dataset_name}, metric={metric}, backend={d['backend']}"


# Curated parameter sets spanning key UMAP embedding knobs (kept small for CI)
EMBED_PARAM_SETS = [
    pytest.param(
        dict(
            learning_rate=0.5,
            min_dist=0.1,
            spread=1.0,
            n_epochs=100,
            negative_sample_rate=5,
            gamma=1.0,
            init="spectral",
            n_components=2,
        ),
        id="baseline-short",
    ),
    pytest.param(
        dict(
            learning_rate=1.0,
            min_dist=0.1,
            spread=1.0,
            n_epochs=300,
            negative_sample_rate=5,
            gamma=1.0,
            init="spectral",
            n_components=2,
        ),
        id="baseline-long",
    ),
    pytest.param(
        dict(
            learning_rate=2.0,
            min_dist=0.0,
            spread=0.9,
            n_epochs=200,
            negative_sample_rate=10,
            gamma=1.5,
            init="random",
            n_components=2,
        ),
        id="aggressive-tight",
    ),
    pytest.param(
        dict(
            learning_rate=0.5,
            min_dist=0.5,
            spread=1.5,
            n_epochs=200,
            negative_sample_rate=5,
            gamma=1.0,
            init="spectral",
            n_components=2,
        ),
        id="spread-out",
    ),
    pytest.param(
        dict(
            learning_rate=1.0,
            min_dist=0.001,
            spread=1.0,
            n_epochs=400,
            negative_sample_rate=5,
            gamma=1.0,
            init="spectral",
            n_components=2,
        ),
        id="very-tight-min-dist",
    ),
]


@pytest.mark.parametrize("params", EMBED_PARAM_SETS)
def test_simplicial_set_embedding(cu_fuzzy_fixture, params):
    d = cu_fuzzy_fixture
    X_np, X_cp = d["X_np"], d["X_cp"]
    metric = d["metric"]
    k = d["k"]
    graph_gpu = d["cu_graph_gpu"]
    graph_cpu = d["cu_graph_cpu"]

    a, b = find_ab_params(spread=params["spread"], min_dist=params["min_dist"])

    # Reference embedding (CPU) on cuML fuzzy graph (CPU)
    ref_res = ref_simplicial_set_embedding(
        X_np,
        graph=graph_cpu,
        n_components=params["n_components"],
        initial_alpha=params["learning_rate"],
        a=a,
        b=b,
        gamma=params["gamma"],
        negative_sample_rate=params["negative_sample_rate"],
        n_epochs=params["n_epochs"],
        init=params["init"],
        random_state=np.random.RandomState(42),
        metric=metric,
        metric_kwds={},
        densmap=False,
        densmap_kwds={},
        output_dens=False,
        output_metric=metric,
        output_metric_kwds={},
    )
    ref_emb = ref_res[0]

    # cuML embedding (GPU) on the same cuML fuzzy graph (GPU)
    cu_emb = cu_simplicial_set_embedding(
        X_cp,
        graph=graph_gpu,
        n_components=params["n_components"],
        initial_alpha=params["learning_rate"],
        a=a,
        b=b,
        gamma=params["gamma"],
        negative_sample_rate=params["negative_sample_rate"],
        n_epochs=params["n_epochs"],
        init=params["init"],
        random_state=42,
        metric=metric,
    )
    cu_emb = cp.asnumpy(cu_emb) if isinstance(cu_emb, cp.ndarray) else cu_emb

    # Evaluate metrics using helper (GPU-accelerated), and keep Procrustes RMSE local
    ref_emb_cp = cp.asarray(ref_emb, dtype=cp.float32)
    cu_emb_cp = cp.asarray(cu_emb, dtype=cp.float32)

    metrics_ref = compute_simplicial_set_embedding_metrics(
        X_cp, ref_emb_cp, k=k, metric=metric, skip_topology_preservation=True
    )
    metrics_cu = compute_simplicial_set_embedding_metrics(
        X_cp, cu_emb_cp, k=k, metric=metric, skip_topology_preservation=True
    )

    trust_ref = metrics_ref["trustworthiness"]
    trust_cu = metrics_cu["trustworthiness"]

    cont_ref = metrics_ref["continuity"]
    cont_cu = metrics_cu["continuity"]

    sp_ref = metrics_ref["geodesic_spearman_correlation"]
    sp_cu = metrics_cu["geodesic_spearman_correlation"]
    pe_ref = metrics_ref["geodesic_pearson_correlation"]
    pe_cu = metrics_cu["geodesic_pearson_correlation"]

    xent_ref = metrics_ref["fuzzy_kl_divergence"]
    xent_cu = metrics_cu["fuzzy_kl_divergence"]
    kl_ref = metrics_ref["fuzzy_sym_kl_divergence"]
    kl_cu = metrics_cu["fuzzy_sym_kl_divergence"]

    rmse = procrustes_rmse(ref_emb, cu_emb)

    # Moderate thresholds: small degradations allowed
    mod_trust = 0.05
    mod_cont = 0.05
    mod_corr = 0.15
    mod_rel_kl = 0.15
    mod_rmse = 0.12

    # Severe thresholds: clearly unacceptable degradations
    sev_trust = 0.10
    sev_cont = 0.10
    sev_corr = 0.30
    sev_rel_kl = 0.35
    sev_rmse = 0.20

    # Compute deficits (positive means cuML is worse than reference)
    trust_def = max(0.0, trust_ref - trust_cu)
    cont_def = max(0.0, cont_ref - cont_cu)
    sp_def = max(0.0, sp_ref - sp_cu)
    pe_def = max(0.0, pe_ref - pe_cu)
    xent_rel_increase = max(
        0.0, (xent_cu - xent_ref) / max(abs(xent_ref), 1e-12)
    )
    kl_rel_increase = max(0.0, (kl_cu - kl_ref) / max(abs(kl_ref), 1e-12))

    moderate_issues = []
    severe_issues = []

    # Trustworthiness
    if trust_def > sev_trust:
        severe_issues.append(
            f"trustworthiness deficit {trust_def:.3f} (cu={trust_cu:.4f}, ref={trust_ref:.4f})"
        )
    elif trust_def > mod_trust:
        moderate_issues.append(
            f"trustworthiness deficit {trust_def:.3f} (cu={trust_cu:.4f}, ref={trust_ref:.4f})"
        )

    # Continuity
    if cont_def > sev_cont:
        severe_issues.append(
            f"continuity deficit {cont_def:.3f} (cu={cont_cu:.4f}, ref={cont_ref:.4f})"
        )
    elif cont_def > mod_cont:
        moderate_issues.append(
            f"continuity deficit {cont_def:.3f} (cu={cont_cu:.4f}, ref={cont_ref:.4f})"
        )

    # Geodesic correlations
    if sp_def > sev_corr:
        severe_issues.append(
            f"spearman correlation deficit {sp_def:.3f} (cu={sp_cu:.4f}, ref={sp_ref:.4f})"
        )
    elif sp_def > mod_corr:
        moderate_issues.append(
            f"spearman correlation deficit {sp_def:.3f} (cu={sp_cu:.4f}, ref={sp_ref:.4f})"
        )

    if pe_def > sev_corr:
        severe_issues.append(
            f"pearson correlation deficit {pe_def:.3f} (cu={pe_cu:.4f}, ref={pe_ref:.4f})"
        )
    elif pe_def > mod_corr:
        moderate_issues.append(
            f"pearson correlation deficit {pe_def:.3f} (cu={pe_cu:.4f}, ref={pe_ref:.4f})"
        )

    # Fuzzy KL and symmetric KL (relative change)
    if xent_rel_increase > sev_rel_kl:
        severe_issues.append(
            f"fuzzy KL relative increase {xent_rel_increase:.3f} (cu={xent_cu:.4e}, ref={xent_ref:.4e})"
        )
    elif xent_rel_increase > mod_rel_kl:
        moderate_issues.append(
            f"fuzzy KL relative increase {xent_rel_increase:.3f} (cu={xent_cu:.4e}, ref={xent_ref:.4e})"
        )

    if kl_rel_increase > sev_rel_kl:
        severe_issues.append(
            f"symmetric KL relative increase {kl_rel_increase:.3f} (cu={kl_cu:.4e}, ref={kl_ref:.4e})"
        )
    elif kl_rel_increase > mod_rel_kl:
        moderate_issues.append(
            f"symmetric KL relative increase {kl_rel_increase:.3f} (cu={kl_cu:.4e}, ref={kl_ref:.4e})"
        )

    # Procrustes RMSE
    if rmse > sev_rmse:
        severe_issues.append(f"rmse {rmse:.3f} > {sev_rmse:.3f}")
    elif rmse > mod_rmse:
        moderate_issues.append(f"rmse {rmse:.3f} > {mod_rmse:.3f}")

    # Holistic decision rule:
    # - Fail if multiple severe degradations, or RMSE alone is severe
    # - Or if many moderate degradations accumulate
    should_fail = False
    if len(severe_issues) >= 2 or rmse > sev_rmse:
        should_fail = True
    elif len(moderate_issues) >= 4:
        should_fail = True

    if should_fail:
        details = (
            f"Severe issues: {severe_issues} | Moderate issues: {moderate_issues} "
            f"on {d['dataset_name']} (k={k}, params={params})"
        )
        assert False, details
