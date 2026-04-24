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
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

# Import benchmark modules - supports both package and standalone execution
# without polluting sys.path on normal package import
try:
    from cuml.benchmark import algorithms, datagen, runners
    from cuml.benchmark.cli import build_parser
    from cuml.benchmark.config import load_and_resolve_config
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
    from config import load_and_resolve_config  # noqa: E402
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
SUPPORTED_BACKENDS = {"accel", "cpu", "gpu"}


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


def _parse_backends(backends):
    if backends is None:
        return None
    if isinstance(backends, str):
        values = [part.strip() for part in backends.split(",") if part.strip()]
    else:
        values = list(backends)
    if not values:
        raise ValueError("--backends must include at least one backend")

    normalized = []
    seen = set()
    for backend in values:
        backend = backend.lower()
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                f"Supported backends: {sorted(SUPPORTED_BACKENDS)}"
            )
        if backend not in seen:
            seen.add(backend)
            normalized.append(backend)
    return normalized


def _selected_backends(args, explicit_options=None, default_backends=None):
    explicit_options = explicit_options or set()
    default_backends = default_backends or ["cpu", "gpu"]

    if "--backends" in explicit_options or getattr(args, "backends", None):
        backends = _parse_backends(args.backends)
    else:
        backends = list(default_backends)

    if args.skip_cpu or "--skip-cpu" in explicit_options:
        backends = [backend for backend in backends if backend != "cpu"]
    if args.skip_gpu or "--skip-gpu" in explicit_options:
        backends = [backend for backend in backends if backend != "gpu"]

    if not backends:
        raise ValueError(
            "No benchmark backends selected. Use --backends or skip flags "
            "so at least one backend remains."
        )
    return backends


def _validate_args(args, run_gpu, selected_backends):
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
    if "gpu" not in selected_backends or not run_gpu:
        cpu_valid_types = ["numpy", "pandas"]
        if args.input_type not in cpu_valid_types:
            warnings.warn(
                f"--input-type={args.input_type} not available without GPU. "
                f"Switching to 'numpy'. Available types: {cpu_valid_types}",
                stacklevel=2,
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


def _extract_explicit_options(argv):
    """Return the set of CLI option names explicitly provided by the user."""
    explicit = set()
    option_aliases = {"-q": "--quiet"}
    parsing_options = True

    for token in argv:
        if parsing_options and token == "--":
            parsing_options = False
            continue
        if not parsing_options:
            continue
        if token.startswith("--"):
            explicit.add(token.split("=", 1)[0])
        elif token in option_aliases:
            explicit.add(option_aliases[token])

    return explicit


def _resolve_dtype(dtype):
    """Normalize a dtype string into the dtype object expected by runners."""
    if dtype is None:
        return PrecisionMap["fp32"]
    if dtype in PrecisionMap:
        return PrecisionMap[dtype]
    return dtype


def _validate_benchmark_inputs(test_split, input_type, run_gpu):
    """Validate per-benchmark inputs and normalize CPU-only input types."""
    if not 0.0 <= test_split <= 1.0:
        raise ValueError(
            "test_split: got %f, want a value between 0.0 and 1.0" % test_split
        )

    if run_gpu:
        return input_type

    cpu_valid_types = ["numpy", "pandas"]
    if input_type not in cpu_valid_types:
        warnings.warn(
            f"--input-type={input_type} not available without GPU. "
            f"Switching to 'numpy'. Available types: {cpu_valid_types}",
            stacklevel=2,
        )
        return "numpy"
    return input_type


def _has_size_override(explicit_options):
    """Whether the user explicitly requested size-related CLI overrides."""
    return any(
        option in explicit_options
        for option in (
            "--min-rows",
            "--max-rows",
            "--num-sizes",
            "--num-rows",
            "--num-features",
            "--input-dimensions",
            "--default-size",
        )
    )


def _config_param_lists(entry, args, explicit_options):
    """Return param override lists for a resolved config entry."""
    if "--param-sweep" in explicit_options:
        param_override_list = extract_param_overrides(args.param_sweep)
    else:
        param_override_list = entry["param_override_list"]

    if "--cuml-param-sweep" in explicit_options:
        cuml_param_override_list = extract_param_overrides(
            args.cuml_param_sweep
        )
    else:
        cuml_param_override_list = entry["cuml_param_override_list"]

    if "--cpu-param-sweep" in explicit_options:
        cpu_param_override_list = extract_param_overrides(args.cpu_param_sweep)
    else:
        cpu_param_override_list = entry["cpu_param_override_list"]

    if "--dataset-param-sweep" in explicit_options:
        dataset_param_override_list = extract_param_overrides(
            args.dataset_param_sweep
        )
    else:
        dataset_param_override_list = entry["dataset_param_override_list"]

    return {
        "param_override_list": param_override_list,
        "cuml_param_override_list": cuml_param_override_list,
        "cpu_param_override_list": cpu_param_override_list,
        "dataset_param_override_list": dataset_param_override_list,
    }


def _dtype_to_config_value(dtype):
    if dtype is np.float32:
        return "fp32"
    if dtype is np.float64:
        return "fp64"
    return dtype


def _run_accel_variations(
    *,
    algorithm_name,
    dataset_name,
    bench_rows,
    bench_dims,
    input_type,
    test_fraction,
    dtype,
    raise_on_error,
    n_reps,
    param_lists,
):
    """Run a benchmark variation in a subprocess with cuml.accel enabled."""
    with tempfile.NamedTemporaryFile(suffix=".csv") as csv_file:
        payload = {
            "algorithm": algorithm_name,
            "dataset_name": dataset_name,
            "bench_rows": bench_rows,
            "bench_dims": bench_dims,
            "input_type": input_type,
            "test_fraction": test_fraction,
            "dtype": _dtype_to_config_value(dtype),
            "raise_on_error": raise_on_error,
            "n_reps": n_reps,
            "param_lists": param_lists,
            "csv_path": csv_file.name,
        }
        script = f"""
import json
import numpy as np

from cuml.benchmark import algorithms, runners

payload = json.loads({json.dumps(json.dumps(payload))})
dtype = {{"fp32": np.float32, "fp64": np.float64}}.get(
    payload["dtype"], payload["dtype"]
)
algo = algorithms.algorithm_by_name(payload["algorithm"])
if algo is None:
    raise ValueError(f"Unknown algorithm: {{payload['algorithm']}}")

results_df = runners.run_variations(
    [algo],
    dataset_name=payload["dataset_name"],
    bench_rows=payload["bench_rows"],
    bench_dims=payload["bench_dims"],
    input_type=payload["input_type"],
    test_fraction=payload["test_fraction"],
    dtype=dtype,
    run_cpu=True,
    run_cuml=False,
    raise_on_error=payload["raise_on_error"],
    n_reps=payload["n_reps"],
    **payload["param_lists"],
)
if "cpu_time" in results_df:
    results_df["accel_time"] = results_df["cpu_time"]
    results_df["cpu_time"] = 0.0
if "cpu_acc" in results_df:
    results_df["accel_acc"] = results_df["cpu_acc"]
    results_df["cpu_acc"] = 0.0
results_df["backend"] = "accel"
results_df.to_csv(payload["csv_path"], index=False)
"""
        completed = subprocess.run(
            [sys.executable, "-m", "cuml.accel", "-"],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            message = (
                "cuml.accel benchmark subprocess failed with exit code "
                f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )
            if raise_on_error:
                raise RuntimeError(message)
            warnings.warn(message, stacklevel=2)
            return pd.DataFrame()
        return pd.read_csv(csv_file.name)


def _resolved_entry_dimensions(entry, args, explicit_options):
    """Resolve benchmark dimensions for a config-backed benchmark entry."""
    if entry["shape_pairs"] is not None:
        base_rows = [shape["rows"] for shape in entry["shape_pairs"]]
        base_dims = [shape["features"] for shape in entry["shape_pairs"]]
    else:
        base_rows = entry["bench_rows"]
        base_dims = entry["bench_dims"]

    if "--default-size" in explicit_options:
        return None, [0], [0]

    row_override_flags = {
        "--min-rows",
        "--max-rows",
        "--num-sizes",
        "--num-rows",
    }
    if "--num-rows" in explicit_options:
        base_rows = [args.num_rows]
    elif row_override_flags.intersection(explicit_options):
        base_rows = np.logspace(
            np.log10(args.min_rows),
            np.log10(args.max_rows),
            num=args.num_sizes,
            dtype=np.int32,
        )

    if "--num-features" in explicit_options and args.num_features > 0:
        base_dims = [args.num_features]
    elif "--input-dimensions" in explicit_options:
        base_dims = args.input_dimensions

    if _has_size_override(explicit_options):
        return None, base_rows, base_dims

    if entry["shape_pairs"] is not None:
        return entry["shape_pairs"], None, None

    return None, base_rows, base_dims


def _run_config_benchmarks(args, explicit_options):
    """Execute benchmarks resolved from a YAML config file."""
    resolved = load_and_resolve_config(
        args.config,
        profile=args.profile,
        algorithm_filter=args.algorithms,
    )

    benchmark_entries = resolved["benchmarks"]
    if not benchmark_entries:
        raise ValueError(
            "No benchmark entries remain after applying the selected config "
            "profile and filters."
        )

    entry_backends = [
        _selected_backends(
            args, explicit_options, default_backends=entry["backends"]
        )
        for entry in benchmark_entries
    ]
    allow_gpu_runs = is_gpu_available()
    if allow_gpu_runs and any(
        "gpu" in backends for backends in entry_backends
    ):
        setup_rmm_allocator(args.rmm_allocator)

    all_results = []
    for entry, selected_backends in zip(benchmark_entries, entry_backends):
        algo = algorithms.algorithm_by_name(entry["algorithm"])
        shape_pairs, bench_rows, bench_dims = _resolved_entry_dimensions(
            entry, args, explicit_options
        )

        input_type = entry["input_type"]
        if "--input-type" in explicit_options:
            input_type = args.input_type

        test_split = entry["test_split"]
        if "--test-split" in explicit_options:
            test_split = args.test_split

        dtype = entry["dtype"]
        if "--dtype" in explicit_options:
            dtype = args.dtype

        n_reps = entry["n_reps"]
        if "--n-reps" in explicit_options:
            n_reps = args.n_reps

        run_cpu = "cpu" in selected_backends
        run_accel = "accel" in selected_backends
        run_cuml = "gpu" in selected_backends and allow_gpu_runs

        if not run_cpu and not run_cuml and not run_accel:
            raise ValueError(
                f"Benchmark '{entry['benchmark_id']}' has no runnable backends "
                "after applying CLI overrides and GPU availability."
            )

        raise_on_error = entry["raise_on_error"]
        if "--raise-on-error" in explicit_options:
            raise_on_error = args.raise_on_error

        dataset_name = entry["dataset"]
        if "--dataset" in explicit_options:
            dataset_name = args.dataset

        input_type = _validate_benchmark_inputs(
            test_split, input_type, run_cuml
        )
        accel_input_type = _validate_benchmark_inputs(
            test_split, input_type, False
        )
        dtype = _resolve_dtype(dtype)
        param_lists = _config_param_lists(entry, args, explicit_options)

        variation_shapes = shape_pairs or [
            {"rows": rows, "features": dims}
            for rows in bench_rows
            for dims in bench_dims
        ]

        for shape in variation_shapes:
            if run_cpu or run_cuml:
                results_df = runners.run_variations(
                    [algo],
                    dataset_name=dataset_name,
                    bench_rows=[shape["rows"]],
                    bench_dims=[shape["features"]],
                    input_type=input_type,
                    test_fraction=test_split,
                    dtype=dtype,
                    run_cpu=run_cpu,
                    run_cuml=run_cuml,
                    raise_on_error=raise_on_error,
                    n_reps=n_reps,
                    **param_lists,
                )
                results_df["benchmark_id"] = entry["benchmark_id"]
                results_df["config_path"] = resolved["config_path"]
                results_df["suite_name"] = resolved["suite_name"]
                results_df["suite_tier"] = resolved["suite_tier"]
                results_df["profile"] = resolved["profile"]
                all_results.append(results_df)

            if run_accel:
                accel_results_df = _run_accel_variations(
                    algorithm_name=entry["algorithm"],
                    dataset_name=dataset_name,
                    bench_rows=[shape["rows"]],
                    bench_dims=[shape["features"]],
                    input_type=accel_input_type,
                    test_fraction=test_split,
                    dtype=dtype,
                    raise_on_error=raise_on_error,
                    n_reps=n_reps,
                    param_lists=param_lists,
                )
                if not accel_results_df.empty:
                    accel_results_df["benchmark_id"] = entry["benchmark_id"]
                    accel_results_df["config_path"] = resolved["config_path"]
                    accel_results_df["suite_name"] = resolved["suite_name"]
                    accel_results_df["suite_tier"] = resolved["suite_tier"]
                    accel_results_df["profile"] = resolved["profile"]
                    all_results.append(accel_results_df)

    if not all_results:
        raise ValueError("No benchmark results were produced.")
    return pd.concat(all_results, ignore_index=True)


def run_benchmark(args, explicit_options=None):
    """Execute the benchmark run from parsed CLI arguments."""
    print(f"Benchmark mode: {get_status_string()}")
    _handle_print_commands(args)  # early exit paths
    explicit_options = explicit_options or set()

    if args.config:
        results_df = _run_config_benchmarks(args, explicit_options)
        _save_results(results_df, args.csv)
        return

    selected_backends = _selected_backends(args, explicit_options)
    run_gpu = is_gpu_available() and "gpu" in selected_backends
    if run_gpu:
        setup_rmm_allocator(args.rmm_allocator)
    args.dtype = _resolve_dtype(args.dtype)

    _validate_args(args, run_gpu, selected_backends)

    bench_rows, bench_dims = _resolve_bench_dimensions(args)
    param_lists = _build_param_override_lists(args)
    algos_to_run = _resolve_algorithms(args)

    all_results = []
    if "cpu" in selected_backends or run_gpu:
        all_results.append(
            runners.run_variations(
                algos_to_run,
                dataset_name=args.dataset,
                bench_rows=bench_rows,
                bench_dims=bench_dims,
                input_type=args.input_type,
                test_fraction=args.test_split,
                dtype=args.dtype,
                run_cpu=("cpu" in selected_backends),
                run_cuml=run_gpu,
                raise_on_error=args.raise_on_error,
                n_reps=args.n_reps,
                **param_lists,
            )
        )
    if "accel" in selected_backends:
        accel_input_type = _validate_benchmark_inputs(
            args.test_split, args.input_type, False
        )
        for algo in algos_to_run:
            accel_results_df = _run_accel_variations(
                algorithm_name=algo.name,
                dataset_name=args.dataset,
                bench_rows=list(bench_rows),
                bench_dims=list(bench_dims),
                input_type=accel_input_type,
                test_fraction=args.test_split,
                dtype=args.dtype,
                raise_on_error=args.raise_on_error,
                n_reps=args.n_reps,
                param_lists=param_lists,
            )
            if not accel_results_df.empty:
                all_results.append(accel_results_df)

    if not all_results:
        raise ValueError("No benchmark results were produced.")
    results_df = pd.concat(all_results, ignore_index=True)
    _save_results(results_df, args.csv)


def main(argv=None):
    """Parse arguments and run the benchmark. Returns exit code."""
    argv = argv if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)
    run_benchmark(args, explicit_options=_extract_explicit_options(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
