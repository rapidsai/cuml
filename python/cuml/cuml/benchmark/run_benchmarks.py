#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Command-line ML benchmark runner (core logic).

This module holds the main benchmark logic. Entry points:
  - Full mode:     python -m cuml.benchmark
  - Standalone:    python run_benchmark.py  (from this directory)
  - Direct script: python run_benchmarks.py  (from this directory)

Modes:
  1. GPU mode: When cuML is installed, benchmarks both cuML (GPU) and
     scikit-learn (CPU) implementations.
  2. CPU-only mode: When cuML is not installed, benchmarks only CPU
     implementations (scikit-learn, umap-learn, etc.).
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np

# Import benchmark modules - supports both package and standalone execution
# without polluting sys.path on normal package import
try:
    from cuml.benchmark import algorithms, datagen, runners
    from cuml.benchmark.gpu_check import (
        HAS_CUML,
        get_status_string,
        is_gpu_available,
    )
except ImportError:
    # Standalone execution (benchmark/ directory or cuML not installed)
    _benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    if _benchmark_dir not in sys.path:
        sys.path.insert(0, _benchmark_dir)
    import algorithms  # noqa: E402
    import datagen  # noqa: E402
    import runners  # noqa: E402
    from gpu_check import (  # noqa: E402
        HAS_CUML,
        get_status_string,
        is_gpu_available,
    )

# Conditional RMM import (RMM is a cuML dependency)
rmm = None
if HAS_CUML:
    import rmm

PrecisionMap = {
    "fp32": np.float32,
    "fp64": np.float64,
}


def extract_param_overrides(params_to_sweep):
    """
    Parameters
    ----------
      params_to_sweep : list[str]
        list of string key=[value] pairs, where values are to be interpreted
        as a json-style array. E.g. 'n_estimators=[10,100,1000]'

    Returns
    ---------
      List of dicts of params to evaluate. Always contains as least one dict.
    """
    import itertools

    if not params_to_sweep:
        return [{}]

    # Expand each arg into a list of (key,value) tuples
    single_param_lists = []
    for p in params_to_sweep:
        key, val_string = p.split("=")
        vals = val_string.split(",")
        # use json loads to convert to correct data type
        for idx, val in enumerate(vals):
            try:
                vals[idx] = json.loads(val)
            except ValueError:
                pass

        single_param_lists.append([(key, val) for val in vals])

    # Create dicts with the cartesian product of all arg-based lists
    tuple_list = itertools.product(*single_param_lists)
    dict_list = [dict(tl) for tl in tuple_list]
    return dict_list


def setup_rmm_allocator(allocator_type):
    """Setup RMM allocator if GPU is available.

    Parameters
    ----------
    allocator_type : str
        One of 'cuda', 'managed', 'prefetched'

    Returns
    -------
    bool
        True if RMM was set up, False otherwise
    """
    if not is_gpu_available() or rmm is None:
        return False

    if allocator_type == "cuda":
        dev_resource = rmm.mr.CudaMemoryResource()
        rmm.mr.set_current_device_resource(dev_resource)
        print("Using CUDA Memory Resource...")
    elif allocator_type == "managed":
        managed_resource = rmm.mr.ManagedMemoryResource()
        rmm.mr.set_current_device_resource(managed_resource)
        print("Using Managed Memory Resource...")
    elif allocator_type == "prefetched":
        upstream_mr = rmm.mr.ManagedMemoryResource()
        prefetch_mr = rmm.mr.PrefetchResourceAdaptor(upstream_mr)
        rmm.mr.set_current_device_resource(prefetch_mr)
        print("Using Prefetched Managed Memory Resource...")
    else:
        raise ValueError(f"Unknown RMM allocator type: {allocator_type}")

    return True


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


def run_benchmark(args):
    """Execute the benchmark run from parsed CLI arguments."""
    if args.skip_gpu and args.skip_cpu:
        raise ValueError(
            "Cannot use both --skip-gpu and --skip-cpu; no benchmarks would run. "
            "Use --skip-gpu for CPU-only or --skip-cpu for GPU-only."
        )

    # Print status information
    print(f"Benchmark mode: {get_status_string()}")

    if args.print_status:
        sys.exit()

    # Determine whether to run GPU benchmarks
    run_gpu = is_gpu_available() and not args.skip_gpu

    # Setup RMM allocator if running GPU benchmarks
    if run_gpu:
        setup_rmm_allocator(args.rmm_allocator)

    args.dtype = PrecisionMap[args.dtype]

    if args.print_algorithms:
        for algo in algorithms.all_algorithms():
            print(algo.name)
        sys.exit()

    if args.print_datasets:
        for dataset in datagen.all_datasets().keys():
            print(dataset)
        sys.exit()

    if not 0.0 <= args.test_split <= 1.0:
        raise ValueError(
            "test_split: got %f, want a value between 0.0 and 1.0"
            % args.test_split
        )

    # Validate input type when not running GPU benchmarks
    if not run_gpu:
        cpu_valid_types = ["numpy", "pandas"]
        if args.input_type not in cpu_valid_types:
            warnings.warn(
                f"--input-type={args.input_type} not available without GPU. "
                f"Switching to 'numpy'. Available types: {cpu_valid_types}"
            )
            args.input_type = "numpy"

    bench_rows = np.logspace(
        np.log10(args.min_rows),
        np.log10(args.max_rows),
        num=args.num_sizes,
        dtype=np.int32,
    )

    bench_dims = args.input_dimensions

    if args.num_rows is not None:
        bench_rows = [args.num_rows]

    if args.num_features > 0:
        bench_dims = [args.num_features]
    if args.default_size:
        bench_rows = [0]
        bench_dims = [0]

    param_override_list = extract_param_overrides(args.param_sweep)
    cuml_param_override_list = extract_param_overrides(args.cuml_param_sweep)
    cpu_param_override_list = extract_param_overrides(args.cpu_param_sweep)
    dataset_param_override_list = extract_param_overrides(
        args.dataset_param_sweep
    )

    if args.algorithms:
        algos_to_run = []
        for name in args.algorithms:
            algo = algorithms.algorithm_by_name(name)
            if not algo:
                # List available algorithms for user
                available = [a.name for a in algorithms.all_algorithms()]
                raise ValueError(
                    f"No '{name}' algorithm found. Available algorithms: {available}"
                )
            algos_to_run.append(algo)
    else:
        # Run all by default
        algos_to_run = algorithms.all_algorithms()

    results_df = runners.run_variations(
        algos_to_run,
        dataset_name=args.dataset,
        bench_rows=bench_rows,
        bench_dims=bench_dims,
        input_type=args.input_type,
        test_fraction=args.test_split,
        param_override_list=param_override_list,
        cuml_param_override_list=cuml_param_override_list,
        cpu_param_override_list=cpu_param_override_list,
        dataset_param_override_list=dataset_param_override_list,
        dtype=args.dtype,
        run_cpu=(not args.skip_cpu),
        run_cuml=run_gpu,
        raise_on_error=args.raise_on_error,
        n_reps=args.n_reps,
    )

    if args.csv:
        results_df.to_csv(args.csv)
        print("Saved results to %s" % args.csv)


if __name__ == "__main__":
    run_benchmark(build_parser().parse_args())
