#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""CLI argument parser for the cuML benchmark runner."""

import argparse


def build_parser():
    """Build and return the argument parser for the benchmark CLI."""
    parser = argparse.ArgumentParser(
        prog="run_benchmarks",
        description=r"""
        Command-line benchmark runner, logging results to
        stdout and/or CSV.

        This tool supports both GPU (cuML) and CPU-only (scikit-learn) modes.
        Use --skip-gpu to run only CPU benchmarks.
        Use --skip-cpu to run only GPU benchmarks.

        Examples:
          # Simple logistic regression (GPU + CPU if cuML installed)
          python run_benchmarks.py --dataset classification LogisticRegression

          # CPU-only benchmarking
          python run_benchmarks.py --skip-gpu --dataset classification LogisticRegression

          # Compare impact of RF parameters and data sets for multiclass
          python run_benchmarks.py --dataset classification  \
                --max-rows 100000 --min-rows 10000 \
                --dataset-param-sweep n_classes=[2,8] \
                --cuml-param-sweep n_bins=[4,16] n_estimators=[10,100] \
                --csv results.csv \
                RandomForestClassifier

          # Run a bunch of clustering and dimensionality reduction algorithms
          # (Because `--input-dimensions` takes a varying number of args, you
          # need the extra `--` to separate it from the algorithm names
          python run_benchmarks.py --dataset blobs \
                --max-rows 20000 --min-rows 20000 --num-sizes 1 \
                --input-dimensions 16 256 \
                -- DBSCAN KMeans TSNE PCA UMAP

          # Use a real dataset at its default size
          python run_benchmarks.py --dataset higgs --default-size \
                RandomForestClassifier LogisticRegression

        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100000,
        help="Evaluate at most max_row samples",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=10000,
        help="Evaluate at least min_rows samples",
    )
    parser.add_argument(
        "--num-sizes",
        type=int,
        default=2,
        help="Number of different sizes to test",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=None,
        metavar="N",
        help="Shortcut for --min-rows N --max-rows N --num-sizes 1",
    )
    parser.add_argument("--num-features", type=int, default=-1)
    parser.add_argument(
        "--quiet", "-q", action="store_false", dest="verbose", default=True
    )
    parser.add_argument("--csv", nargs="?")
    parser.add_argument("--dataset", default="blobs")
    parser.add_argument(
        "--skip-cpu",
        action="store_true",
        help="Skip CPU/scikit-learn benchmarks",
    )
    parser.add_argument(
        "--skip-gpu", action="store_true", help="Skip GPU/cuML benchmarks"
    )
    parser.add_argument("--input-type", default="numpy")
    parser.add_argument(
        "--test-split",
        default=0.1,
        type=float,
        help="Fraction of input data used for testing (between 0.0 and 1.0)",
    )
    parser.add_argument(
        "--input-dimensions",
        default=[64, 256, 512],
        nargs="+",
        type=int,
        help="Data dimension sizes (may provide multiple sizes)",
    )
    parser.add_argument(
        "--param-sweep",
        nargs="*",
        type=str,
        help="""Parameter values to vary, in the form:
                key=val_list, where val_list may be a comma-separated list""",
    )
    parser.add_argument(
        "--cuml-param-sweep",
        nargs="*",
        type=str,
        help="""Parameter values to vary for cuML only, in the form:
                key=val_list, where val_list may be a comma-separated list""",
    )
    parser.add_argument(
        "--cpu-param-sweep",
        nargs="*",
        type=str,
        help="""Parameter values to vary for CPU only, in the form:
                key=val_list, where val_list may be a comma-separated list""",
    )
    parser.add_argument(
        "--dataset-param-sweep",
        nargs="*",
        type=str,
        help="""Parameter values to vary for dataset generator, in the form
                key=val_list, where val_list may be a comma-separated list""",
    )
    parser.add_argument(
        "--default-size",
        action="store_true",
        help="Only run datasets at default size",
    )
    parser.add_argument(
        "--raise-on-error",
        action="store_true",
        help="Throw exception on a failed benchmark",
    )
    parser.add_argument(
        "--print-algorithms",
        action="store_true",
        help="Print the list of all available algorithms and exit",
    )
    parser.add_argument(
        "--print-datasets",
        action="store_true",
        help="Print the list of all available datasets and exit",
    )
    parser.add_argument(
        "--print-status",
        action="store_true",
        help="Print GPU/CPU status and exit",
    )
    parser.add_argument(
        "algorithms",
        nargs="*",
        help="List of algorithms to run, or omit to run all",
    )
    parser.add_argument("--n-reps", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp64"],
        default="fp32",
        help="Precision of the dataset to benchmark with",
    )
    parser.add_argument(
        "--rmm-allocator",
        choices=["cuda", "managed", "prefetched"],
        default="cuda",
        help="RMM memory resource to use (default: cuda). Ignored if --skip-gpu.",
    )
    return parser
