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

from toy_datasets import generate_datasets
from umap_quality_checks import compare_implementations, run_implementation
from web_results_generation import generate_web_report


def print_metrics(metrics, name):
    """Print formatted metrics for a dataset."""
    print(f"\n{'='*50}")
    print(f"METRICS FOR {name.upper()}")
    print(f"{'='*50}")

    print("Local Structure Preservation:")
    print(f"  Trustworthiness: {metrics.get('trustworthiness', 'N/A'):.4f}")
    print(f"  Continuity: {metrics.get('continuity', 'N/A'):.4f}")

    print("\nGlobal Structure Preservation:")
    print(
        f"  Geodesic Spearman Correlation: {metrics.get('geodesic_spearman_correlation', 'N/A'):.4f}"
    )
    print(
        f"  Geodesic Pearson Correlation: {metrics.get('geodesic_pearson_correlation', 'N/A'):.4f}"
    )
    print(f"  DEMaP: {metrics.get('demap', 'N/A'):.4f}")

    print("\nFuzzy Simplicial Set:")
    print(f"  Cross-entropy: {metrics.get('fuzzy_cross_entropy', 'N/A'):.4f}")

    # Print comparison metrics if available (when comparing implementations)
    if "knn_recall" in metrics or "fuzzy_cross_entropy_refwise" in metrics:
        print("\nComparison with Reference Implementation:")
        if "knn_recall" in metrics:
            print(f"  KNN Recall: {metrics['knn_recall']:.4f}")
        if "fuzzy_cross_entropy_refwise" in metrics:
            print(
                f"  Fuzzy Cross-entropy (vs. reference): {metrics['fuzzy_cross_entropy_refwise']:.4f}"
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
            comparison = compare_implementations(X, n_neighbors=n_neighbors)

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
                metrics,
            ) = run_implementation(
                X, implementation=args.implementation, n_neighbors=n_neighbors
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
