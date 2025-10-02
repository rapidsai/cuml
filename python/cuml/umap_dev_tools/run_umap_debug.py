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

import argparse

import cupy as cp
import numpy as np
from toy_datasets import generate_datasets
from umap.umap_ import find_ab_params
from umap.umap_ import fuzzy_simplicial_set as ref_fuzzy_simplicial_set
from umap.umap_ import simplicial_set_embedding as ref_simplicial_set_embedding
from umap.umap_ import spectral_layout
from umap_metrics import (
    _build_knn_with_cuvs,
    _build_knn_with_umap,
    compute_fuzzy_simplicial_set_metrics,
    compute_knn_metrics,
    compute_simplicial_set_embedding_metrics,
)
from web_results_generation import generate_web_report

from cuml.manifold.simpl_set import (
    fuzzy_simplicial_set as cu_fuzzy_simplicial_set,
)
from cuml.manifold.simpl_set import (
    simplicial_set_embedding as cu_simplicial_set_embedding,
)


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
        fuzzy_simplicial_set = ref_fuzzy_simplicial_set
        simplicial_set_embedding = ref_simplicial_set_embedding
    elif implementation == "cuml":
        fuzzy_simplicial_set = cu_fuzzy_simplicial_set
        simplicial_set_embedding = cu_simplicial_set_embedding
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
    learning_rate = umap_params.get("learning_rate", 1.0)
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

    # Force brute-force backend for both implementations
    X = X.astype(np.float32, copy=False)
    knn_backend = "bruteforce"
    if implementation == "reference":
        knn_dists, knn_indices = _build_knn_with_umap(
            X, k=n_neighbors, metric=metric, backend=knn_backend
        )
    else:
        X_cp = X if isinstance(X, cp.ndarray) else cp.asarray(X)
        knn_dists, knn_indices = _build_knn_with_cuvs(
            X_cp, k=n_neighbors, metric=metric, backend=knn_backend
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
            fuzzy_graph,
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
            initial_alpha=0,
            a=a,
            b=b,
            gamma=gamma,
            negative_sample_rate=0,
            n_epochs=1,
            init="spectral",
            random_state=random_state,
            metric=metric,
        )

    # ------------------------------------------------------------------
    # STEP 3: Low-dimensional embedding via simplicial_set_embedding
    # ------------------------------------------------------------------
    print("  Computing embedding from fuzzy simplicial set ...")

    if implementation == "reference":
        additional_params = {
            "metric_kwds": {},
            "densmap": False,
            "densmap_kwds": {},
            "output_dens": False,
            "output_metric": "euclidean",
            "output_metric_kwds": {},
        }
    else:
        additional_params = {}

    embedding_results = simplicial_set_embedding(
        X,
        graph=fuzzy_graph,
        n_components=n_components,
        initial_alpha=learning_rate,
        a=a,
        b=b,
        gamma=gamma,
        negative_sample_rate=negative_sample_rate,
        n_epochs=n_epochs,
        init=init,
        random_state=random_state,
        metric=metric,
        **additional_params,
    )

    if implementation == "reference":
        embedding = embedding_results[0]
    else:
        embedding = embedding_results
        fuzzy_graph = fuzzy_graph.get()

    return knn_graph, fuzzy_graph, spectral_init, embedding


def run_implementation(X, implementation, **umap_params):
    knn_graph, fuzzy_graph, spectral_init, embedding = run_umap_pipeline(
        X, implementation=implementation, **umap_params
    )
    metrics = compute_simplicial_set_embedding_metrics(
        X,
        embedding,
        k=umap_params["n_neighbors"],
        metric=umap_params["metric"],
    )
    return knn_graph, fuzzy_graph, spectral_init, embedding, metrics


def compare_implementations(X, **umap_params):
    (
        ref_knn_graph,
        ref_fuzzy_graph,
        ref_spectral_init,
        ref_embedding,
    ) = run_umap_pipeline(X, implementation="reference", **umap_params)
    knn_graph, fuzzy_graph, spectral_init, embedding = run_umap_pipeline(
        X, implementation="cuml", **umap_params
    )

    ref_metrics = compute_simplicial_set_embedding_metrics(
        X,
        ref_embedding,
        k=umap_params["n_neighbors"],
        metric=umap_params["metric"],
    )
    metrics = compute_simplicial_set_embedding_metrics(
        X,
        embedding,
        k=umap_params["n_neighbors"],
        metric=umap_params["metric"],
    )

    avg_knn_recall, mae_knn_dist = compute_knn_metrics(
        ref_knn_graph, knn_graph, umap_params["n_neighbors"]
    )
    kl_sym, jacc, row_l1 = compute_fuzzy_simplicial_set_metrics(
        ref_fuzzy_graph, fuzzy_graph
    )
    metrics["kl_sym"] = kl_sym
    metrics["jacc"] = jacc
    metrics["row_l1"] = row_l1
    metrics["avg_knn_recall"] = avg_knn_recall
    metrics["mae_knn_dist"] = mae_knn_dist

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


def print_metrics(metrics, name):
    """Print formatted metrics for a dataset."""

    def _fmt(v):
        try:
            return f"{float(v):.4f}"
        except Exception:
            return "N/A"

    print(f"\n{'='*50}")
    print(f"METRICS FOR {name.upper()}")
    print(f"{'='*50}")

    print("Local Structure Preservation:")
    print(f"  Trustworthiness: {_fmt(metrics.get('trustworthiness'))}")
    print(f"  Continuity: {_fmt(metrics.get('continuity'))}")

    print("\nGlobal Structure Preservation:")
    print(
        f"  Geodesic Spearman Correlation: {_fmt(metrics.get('geodesic_spearman_correlation'))}"
    )
    print(
        f"  Geodesic Pearson Correlation: {_fmt(metrics.get('geodesic_pearson_correlation'))}"
    )
    print(f"  DEMaP: {_fmt(metrics.get('demap'))}")

    print("\nFuzzy Simplicial Set:")
    print(
        f"  KL divergence (high vs low): {_fmt(metrics.get('fuzzy_kl_divergence'))}"
    )
    print(
        f"  Symmetric KL divergence (high vs low): {_fmt(metrics.get('fuzzy_sym_kl_divergence'))}"
    )

    # Print comparison metrics if available (when comparing implementations)
    has_comp = any(
        k in metrics
        for k in ("kl_sym", "jacc", "row_l1", "avg_knn_recall", "mae_knn_dist")
    )
    if has_comp:
        print("\nComparison with Reference Implementation:")
        if "avg_knn_recall" in metrics:
            print(f"  Avg KNN Recall: {_fmt(metrics.get('avg_knn_recall'))}")
        if "mae_knn_dist" in metrics:
            print(f"  KNN Distance MAE: {_fmt(metrics.get('mae_knn_dist'))}")
        if "kl_sym" in metrics:
            print(
                f"  Fuzzy Graph Symmetric KL (ref vs cuML): {_fmt(metrics.get('kl_sym'))}"
            )
        if "jacc" in metrics:
            print(
                f"  Fuzzy Graph Edge Jaccard (ref vs cuML): {_fmt(metrics.get('jacc'))}"
            )
        if "row_l1" in metrics:
            print(
                f"  Fuzzy Graph Row-sum L1 (ref vs cuML): {_fmt(metrics.get('row_l1'))}"
            )

    if "betti_h0_high" in metrics:
        print("\nTopological Features (Betti Numbers):")
        print(f"  High-dim H0: {metrics.get('betti_h0_high', 'N/A')}")
        print(f"  High-dim H1: {metrics.get('betti_h1_high', 'N/A')}")
        print(f"  Low-dim H0: {metrics.get('betti_h0_low', 'N/A')}")
        print(f"  Low-dim H1: {metrics.get('betti_h1_low', 'N/A')}")


def get_available_datasets():
    """Get list of available dataset names."""
    datasets = generate_datasets()
    return list(datasets.keys())


def parse_args():
    """Parse command line arguments."""
    available_datasets = get_available_datasets()
    dataset_options = "', '".join(available_datasets)

    parser = argparse.ArgumentParser(
        description="UMAP Quality Assessment Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--implementation",
        "--impl",
        choices=["reference", "cuml", "both"],
        default="both",
        help="Which UMAP implementation(s) to run",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help=f"Dataset to run assessment on. Options: 'all', '{dataset_options}'",
    )

    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit",
    )

    parser.add_argument(
        "--web-report",
        action="store_true",
        default=False,
        help="Generate web report (default: disabled)",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="Metric for high-dimensional KNN (e.g., 'euclidean', 'cosine').",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Handle list-datasets option
    if args.list_datasets:
        available_datasets = get_available_datasets()
        print("Available datasets:")
        for i, dataset in enumerate(available_datasets, 1):
            print(f"  {i:2d}. {dataset}")
        return

    print("UMAP Quality Assessment Script")
    print("=" * 50)
    print(f"Implementation: {args.implementation}")
    print(f"Dataset: {args.dataset}")
    print(f"Web report: {'enabled' if args.web_report else 'disabled'}")
    print("=" * 50)

    # Generate datasets
    all_datasets = generate_datasets()

    # Filter datasets based on selection
    if args.dataset.lower() == "all":
        datasets = all_datasets
        print(f"Running assessment on all {len(datasets)} datasets")
    else:
        # Case-insensitive dataset matching
        dataset_found = None
        for dataset_name in all_datasets.keys():
            if dataset_name.lower() == args.dataset.lower():
                dataset_found = dataset_name
                break

        if dataset_found:
            datasets = {dataset_found: all_datasets[dataset_found]}
            print(f"Running assessment on dataset: {dataset_found}")
        else:
            available_datasets = list(all_datasets.keys())
            print(f"Error: Dataset '{args.dataset}' not found.")
            print(f"Available datasets: {', '.join(available_datasets)}")
            print("Tip: Use --list-datasets to see all available options")
            return

    # Process each dataset
    embeddings = {}
    spectral_inits = {}
    all_metrics = {}

    n_neighbors = 15

    # Create a copy of the items to avoid dictionary modification during iteration
    dataset_items = list(datasets.items())

    for name, (X, colors) in dataset_items:
        print(f"\nProcessing dataset: {name}")

        if args.implementation == "both":
            # Run comparison between both implementations
            comparison = compare_implementations(
                X, n_neighbors=n_neighbors, metric=args.metric
            )

            print_metrics(comparison["ref_metrics"], f"{name} (reference)")
            print_metrics(comparison["metrics"], f"{name} (cuml)")

            # For web report, create grouped entries for both implementations
            if args.web_report:
                # Store both implementations under the original dataset name
                embeddings[name] = {
                    "reference": comparison["ref_embedding"],
                    "cuml": comparison["embedding"],
                }
                spectral_inits[name] = {
                    "reference": comparison["ref_spectral_init"],
                    "cuml": comparison["spectral_init"],
                }
                all_metrics[name] = {
                    "reference": comparison["ref_metrics"],
                    "cuml": comparison["metrics"],
                }
            else:
                # If not generating web report, just store cuML results as before
                embeddings[name] = comparison["embedding"]
                spectral_inits[name] = comparison["spectral_init"]
                all_metrics[name] = comparison["metrics"]

        else:
            # Run single implementation
            (
                knn_graph,
                fuzzy_graph,
                spectral_init,
                embedding,
            ) = run_umap_pipeline(
                X,
                implementation=args.implementation,
                n_neighbors=n_neighbors,
                metric=args.metric,
            )
            metrics = compute_simplicial_set_embedding_metrics(
                X, embedding, k=n_neighbors, metric=args.metric
            )

            print_metrics(metrics, f"{name} ({args.implementation})")

            embeddings[name] = embedding
            spectral_inits[name] = spectral_init
            all_metrics[name] = metrics

    # Generate web page if requested
    if args.web_report:
        print(f"\n{'='*50}")
        print("GENERATING WEB PAGE...")
        print(f"{'='*50}")

        html_content = generate_web_report(
            datasets, embeddings, all_metrics, spectral_inits
        )

        # Save HTML file
        filename = "umap_quality_assessment_results.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"\n{'='*50}")
        print("ANALYSIS COMPLETE!")
        print(f"Results saved as '{filename}'")
        print(
            "Open this file in your web browser to view the interactive results."
        )
        if args.implementation == "both":
            print(
                "The web report includes results from both Reference and cuML implementations."
            )
        print(f"{'='*50}")
    else:
        print(f"\n{'='*50}")
        print("ANALYSIS COMPLETE!")
        print("Web report generation was disabled.")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
