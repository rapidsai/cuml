#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Command-line ML benchmark runner (core logic).

This module holds the main benchmark logic. Entry points:
  - Full mode:     python -m cuml.benchmark
  - Standalone:    python run_benchmarks.py  (from this directory)

Modes:
  1. GPU mode: When cuML is installed, benchmarks both cuML (GPU) and
     scikit-learn (CPU) implementations.
  2. CPU-only mode: When cuML is not installed, benchmarks only CPU
     implementations (scikit-learn, umap-learn, etc.).
"""

import json
import os
import sys
import warnings

import numpy as np

# Import benchmark modules - supports both package and standalone execution
# without polluting sys.path on normal package import
try:
    from cuml.benchmark import algorithms, datagen, runners
    from cuml.benchmark.cli import build_parser
    from cuml.benchmark.gpu_check import (
        get_status_string,
        is_cuml_available,
        is_gpu_available,
    )
except ImportError:
    # Standalone execution (benchmark/ directory or cuML not installed)
    # Add benchmark directory to sys.path
    _benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    if _benchmark_dir not in sys.path:
        sys.path.insert(0, _benchmark_dir)
    assert any("cuml/benchmark" in p for p in sys.path)

    import algorithms  # noqa: E402
    import datagen  # noqa: E402
    import runners  # noqa: E402
    from cli import build_parser  # noqa: E402
    from gpu_check import (  # noqa: E402
        get_status_string,
        is_cuml_available,
        is_gpu_available,
    )

# Conditional RMM import (RMM is a cuML dependency)
rmm = None
if is_cuml_available():
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


def _validate_args(args, run_gpu):
    """Validate CLI args: skip flags, test_split range, and CPU-only input_type."""
    if args.skip_gpu and args.skip_cpu:
        raise ValueError(
            "Cannot use both --skip-gpu and --skip-cpu; no benchmarks would run. "
            "Use --skip-gpu for CPU-only or --skip-cpu for GPU-only."
        )
    if not 0.0 <= args.test_split <= 1.0:
        raise ValueError(
            "test_split: got %f, want a value between 0.0 and 1.0"
            % args.test_split
        )
    if not run_gpu:
        cpu_valid_types = ["numpy", "pandas"]
        if args.input_type not in cpu_valid_types:
            warnings.warn(
                f"--input-type={args.input_type} not available without GPU. "
                f"Switching to 'numpy'. Available types: {cpu_valid_types}"
            )
            args.input_type = "numpy"


def _handle_print_commands(args):
    """Handle --print-status, --print-algorithms, --print-datasets; exits if used."""
    if args.print_status:
        sys.exit()
    if args.print_algorithms:
        for algo in algorithms.all_algorithms():
            print(algo.name)
        sys.exit()
    if args.print_datasets:
        for dataset in datagen.all_datasets().keys():
            print(dataset)
        sys.exit()


def _resolve_bench_dimensions(args):
    """Compute bench_rows and bench_dims from args; apply overrides. Returns (bench_rows, bench_dims)."""
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
    return bench_rows, bench_dims


def _build_param_override_lists(args):
    """Build the four param override lists from sweep args. Returns a dict of lists."""
    return {
        "param_override_list": extract_param_overrides(args.param_sweep),
        "cuml_param_override_list": extract_param_overrides(
            args.cuml_param_sweep
        ),
        "cpu_param_override_list": extract_param_overrides(
            args.cpu_param_sweep
        ),
        "dataset_param_override_list": extract_param_overrides(
            args.dataset_param_sweep
        ),
    }


def _resolve_algorithms(args):
    """Resolve algorithm names from args to list of algorithm objects."""
    if args.algorithms:
        algos_to_run = []
        for name in args.algorithms:
            algo = algorithms.algorithm_by_name(name)
            if not algo:
                available = [a.name for a in algorithms.all_algorithms()]
                raise ValueError(
                    f"No '{name}' algorithm found. Available algorithms: {available}"
                )
            algos_to_run.append(algo)
        return algos_to_run
    return list(algorithms.all_algorithms())


def _save_results(results_df, csv_path):
    """Write results DataFrame to CSV and print confirmation if path given."""
    if csv_path:
        results_df.to_csv(csv_path)
        print("Saved results to %s" % csv_path)


def run_benchmark(args):
    """Execute the benchmark run from parsed CLI arguments."""
    print(f"Benchmark mode: {get_status_string()}")
    _handle_print_commands(args)  # early exit paths

    run_gpu = is_gpu_available() and not args.skip_gpu
    if run_gpu:
        setup_rmm_allocator(args.rmm_allocator)
    args.dtype = PrecisionMap[args.dtype]

    _validate_args(args, run_gpu)

    bench_rows, bench_dims = _resolve_bench_dimensions(args)
    param_lists = _build_param_override_lists(args)
    algos_to_run = _resolve_algorithms(args)

    results_df = runners.run_variations(
        algos_to_run,
        dataset_name=args.dataset,
        bench_rows=bench_rows,
        bench_dims=bench_dims,
        input_type=args.input_type,
        test_fraction=args.test_split,
        dtype=args.dtype,
        run_cpu=(not args.skip_cpu),
        run_cuml=run_gpu,
        raise_on_error=args.raise_on_error,
        n_reps=args.n_reps,
        **param_lists,
    )
    _save_results(results_df, args.csv)


def main(argv=None):
    """Parse arguments and run the benchmark. Returns exit code."""
    args = build_parser().parse_args(argv)
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
