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
import math
import os
import platform
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timezone

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
SUPPORTED_BACKENDS = {"cpu", "gpu"}


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
        key, val_string = p.split("=", 1)
        try:
            vals = json.loads(val_string)
        except ValueError:
            vals = val_string.split(",")

        if not isinstance(vals, list):
            vals = [vals]

        # use json loads to convert comma-separated values to correct data type
        for idx, val in enumerate(vals):
            if not isinstance(val, str):
                continue
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


def _json_safe(value):
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _git_output(args):
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _read_first_cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return platform.processor() or None
    return platform.processor() or None


def _read_total_memory_bytes():
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _gpu_hardware_from_nvml():
    try:
        import pynvml
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        devices = []
        for index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            if isinstance(name, bytes):
                name = name.decode()
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            devices.append(
                {
                    "index": index,
                    "name": name,
                    "uuid": uuid,
                    "total_memory_bytes": int(memory.total),
                    "compute_capability": f"{major}.{minor}",
                }
            )
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode()
        return {
            "count": device_count,
            "devices": devices,
            "driver_version": driver_version,
        }
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _gpu_hardware_from_nvidia_smi():
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None

    devices = []
    driver_version = None
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        index, name, uuid, memory_mib, driver_version = parts[:5]
        try:
            memory_bytes = int(float(memory_mib) * 1024 * 1024)
            index = int(index)
        except ValueError:
            continue
        devices.append(
            {
                "index": index,
                "name": name,
                "uuid": uuid,
                "total_memory_bytes": memory_bytes,
                "compute_capability": None,
            }
        )
    return {
        "count": len(devices),
        "devices": devices,
        "driver_version": driver_version,
    }


def _collect_detected_hardware():
    gpu = _gpu_hardware_from_nvml() or _gpu_hardware_from_nvidia_smi()
    if gpu is None:
        gpu = {"count": 0, "devices": [], "driver_version": None}
    return {
        "gpu": {"detected": gpu},
        "cpu": {
            "detected": {
                "model": _read_first_cpu_model(),
                "logical_cores": os.cpu_count(),
                "physical_cores": None,
                "architecture": platform.machine(),
            }
        },
        "memory": {
            "detected": {
                "total_memory_bytes": _read_total_memory_bytes(),
            }
        },
        "os": {
            "detected": {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "kernel": platform.version(),
            }
        },
    }


def _hardware_overrides(args):
    hardware = {}
    hardware_label = getattr(args, "hardware_label", None)
    if hardware_label:
        hardware["label"] = hardware_label

    gpu_effective = {}
    if getattr(args, "hardware_gpu_name", None):
        gpu_effective["name"] = args.hardware_gpu_name
    if getattr(args, "hardware_gpu_memory_gb", None) is not None:
        gpu_effective["total_memory_bytes"] = int(
            args.hardware_gpu_memory_gb * 1e9
        )
    if gpu_effective:
        hardware.setdefault("gpu", {})["effective"] = gpu_effective

    cpu_effective = {}
    if getattr(args, "hardware_cpu_name", None):
        cpu_effective["model"] = args.hardware_cpu_name
    if getattr(args, "hardware_cpu_cores", None) is not None:
        cpu_effective["logical_cores"] = args.hardware_cpu_cores
    if cpu_effective:
        hardware.setdefault("cpu", {})["effective"] = cpu_effective

    return hardware


def _merge_hardware_metadata(detected, overrides):
    merged = dict(detected)
    for section, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(section), dict):
            section_value = dict(merged[section])
            section_value.update(value)
            merged[section] = section_value
        else:
            merged[section] = value
    return merged


def _run_json_command(command):
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return None, str(exc)
    if completed.returncode != 0:
        return None, completed.stderr.strip() or completed.stdout.strip()
    try:
        return json.loads(completed.stdout), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def _conda_package_snapshot(packages):
    return [
        {
            "name": package.get("name"),
            "version": package.get("version"),
            "build": package.get("build_string"),
            "channel": package.get("channel"),
        }
        for package in packages
    ]


def _pip_package_snapshot(packages):
    return [
        {
            "name": package.get("name"),
            "version": package.get("version"),
        }
        for package in packages
    ]


def _collect_package_snapshot():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        packages, error = _run_json_command(["conda", "list", "--json"])
        if packages is not None:
            return {
                "package_snapshot_source": "conda",
                "conda_prefix": conda_prefix,
                "packages": _conda_package_snapshot(packages),
            }
        conda_error = error
    else:
        conda_error = None

    packages, error = _run_json_command(
        [sys.executable, "-m", "pip", "list", "--format=json"]
    )
    if packages is not None:
        return {
            "package_snapshot_source": "pip",
            "conda_prefix": conda_prefix,
            "packages": _pip_package_snapshot(packages),
            "conda_error": conda_error,
        }
    return {
        "package_snapshot_source": None,
        "conda_prefix": conda_prefix,
        "packages": [],
        "conda_error": conda_error,
        "pip_error": error,
    }


def _collect_run_metadata(args):
    git_sha = _git_output(["rev-parse", "HEAD"])
    git_status = _git_output(["status", "--porcelain"])
    hardware = _merge_hardware_metadata(
        _collect_detected_hardware(), _hardware_overrides(args)
    )
    return {
        "result_schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": {
            "argv": list(getattr(args, "_argv", sys.argv[1:])),
            "cwd": os.getcwd(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "cuml": {
            "version": getattr(sys.modules.get("cuml"), "__version__", None),
            "git_sha": git_sha,
            "git_dirty": bool(git_status) if git_status is not None else None,
        },
        "runtime": {
            "status": get_status_string(),
            "gpu_available": is_gpu_available(),
        },
        "environment": _collect_package_snapshot(),
        "hardware": hardware,
        "config": {
            "path": getattr(args, "config", None),
            "profile": getattr(args, "profile", None),
            "backends": getattr(args, "backends", None),
        },
    }


def _backend_result(row, backend):
    time_column = {
        "gpu": "cuml_time",
        "cpu": "cpu_time",
    }[backend]
    acc_column = {
        "gpu": "cuml_acc",
        "cpu": "cpu_acc",
    }[backend]

    time_sec = _positive_value(row, time_column)
    if time_sec is None:
        requested_backends = row.get("requested_backends")
        if isinstance(requested_backends, str):
            requested_backends = [
                item for item in requested_backends.split(",") if item
            ]
        if (
            isinstance(requested_backends, (list, tuple))
            and backend in requested_backends
        ):
            reason = "no successful timing recorded"
            if backend == "gpu" and not row.get("gpu_available", True):
                reason = "GPU unavailable"
            return {"status": "skipped", "reason": reason}
        return None
    result = {"status": "success", "time_sec": time_sec}
    accuracy = _positive_value(row, acc_column)
    if accuracy is not None:
        result["accuracy"] = accuracy
    return result


def _declared_params(row):
    metadata_columns = {
        "algo",
        "input",
        "cuml_time",
        "cpu_time",
        "cuml_acc",
        "cpu_acc",
        "cuml_params",
        "cpu_params",
        "speedup",
        "n_samples",
        "n_features",
        "benchmark_id",
        "config_path",
        "suite_name",
        "suite_tier",
        "profile",
        "dataset",
        "operation",
        "dtype",
        "n_reps",
        "backend",
        "requested_backends",
        "gpu_available",
    }
    params = {}
    for column, value in row.items():
        if column in metadata_columns or pd.isna(value):
            continue
        params[column] = _json_safe(value)
    return params


def _result_record(row, dtype):
    rows = _json_safe(row.get("n_samples"))
    features = _json_safe(row.get("n_features"))
    size_gb = _estimated_input_size_gb(rows, features, dtype)
    backends = {}
    for backend in ("gpu", "cpu"):
        backend_result = _backend_result(row, backend)
        if backend_result is not None:
            backends[backend] = backend_result

    record = {
        "benchmark_id": _json_safe(row.get("benchmark_id")),
        "algorithm": _json_safe(row.get("algo")),
        "dataset": _json_safe(row.get("dataset")),
        "operation": _json_safe(row.get("operation")),
        "shape": {
            "rows": rows,
            "features": features,
            "estimated_input_size_bytes": (
                None if math.isnan(size_gb) else int(size_gb * 1e9)
            ),
            "estimated_input_size_gb": None if math.isnan(size_gb) else size_gb,
        },
        "data": {
            "input_type": _json_safe(row.get("input")),
            "dtype": _json_safe(row.get("dtype", _dtype_to_config_value(dtype))),
            "n_reps": _json_safe(row.get("n_reps")),
        },
        "params": {
            "declared": _declared_params(row),
            "effective": {
                "gpu": _json_safe(row.get("cuml_params")),
                "cpu": _json_safe(row.get("cpu_params")),
            },
        },
        "backends": backends,
    }
    return record


def _results_to_json_records(results_df, dtype):
    grouped = _coalesce_progress_rows(results_df)
    return [
        _result_record(row, dtype)
        for _, row in grouped.iterrows()
    ]


def _write_json_atomic(path, payload):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=".benchmark-",
            suffix=".json",
            delete=False,
        ) as fh:
            temp_path = fh.name
            json.dump(_json_safe(payload), fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise


def _save_json_results(results_df, output_path, args, dtype):
    if not output_path:
        return
    payload = {
        "results": _results_to_json_records(results_df, dtype),
        "metadata": _collect_run_metadata(args),
    }
    _write_json_atomic(output_path, payload)
    print("Saved JSON results to %s" % output_path)


def _human_count(value):
    """Format counts compactly for progress output."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "nan"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(value))


def _format_seconds(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value <= 0 or math.isnan(value):
        return None
    if value >= 10:
        return f"{value:.1f}s"
    if value >= 1:
        return f"{value:.2f}s"
    if value >= 0.001:
        return f"{value:.3f}s"
    return f"{value * 1000:.2f}ms"


def _format_seconds_cell(value):
    return _format_seconds(value) or "-"


def _dtype_itemsize(dtype):
    try:
        return np.dtype(dtype).itemsize
    except TypeError:
        return np.dtype(_resolve_dtype(dtype)).itemsize


def _estimated_input_size_gb(rows, features, dtype):
    try:
        return float(rows) * float(features) * _dtype_itemsize(dtype) / 1e9
    except (TypeError, ValueError):
        return float("nan")


def _ratio(numerator, denominator):
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return None
    if numerator <= 0 or denominator <= 0 or math.isnan(numerator) or math.isnan(denominator):
        return None
    return denominator / numerator


def _positive_value(row, key):
    if key not in row:
        return None
    try:
        value = float(row[key])
    except (TypeError, ValueError):
        return None
    if value <= 0 or math.isnan(value):
        return None
    return value


def _progress_group_key(row):
    key_columns = [
        "benchmark_id",
        "algo",
        "n_samples",
        "n_features",
    ]
    ignored_columns = {
        "input",
        "cuml_time",
        "cpu_time",
        "cuml_acc",
        "cpu_acc",
        "cuml_params",
        "cpu_params",
        "speedup",
        "backend",
    }
    key = [row.get(column, None) for column in key_columns]
    for column in sorted(set(row.index) - set(key_columns) - ignored_columns):
        value = row.get(column)
        if pd.notna(value):
            key.append((column, value))
    return tuple(key)


def _coalesce_progress_rows(results_df):
    rows = []
    for _, group in results_df.groupby(
        results_df.apply(_progress_group_key, axis=1),
        sort=False,
        dropna=False,
    ):
        base = group.iloc[0].copy()
        for column in (
            "cuml_time",
            "cpu_time",
            "cuml_acc",
            "cpu_acc",
        ):
            if column not in group:
                continue
            values = [
                value
                for value in group[column]
                if pd.notna(value) and _positive_value({column: value}, column)
            ]
            if values:
                base[column] = values[0]
        if "input" in group:
            inputs = [str(value) for value in group["input"].dropna().unique()]
            base["input"] = ",".join(inputs)
        rows.append(base)
    return pd.DataFrame(rows)


def _progress_line(row, index, total, dtype):
    algo = row.get("algo", "")
    rows = row.get("n_samples", "")
    features = row.get("n_features", "")
    size_gb = _estimated_input_size_gb(rows, features, dtype)

    cpu_time = _positive_value(row, "cpu_time")
    gpu_time = _positive_value(row, "cuml_time")

    ratio_parts = []
    for label, numerator, denominator in (
        ("gpu_speedup", gpu_time, cpu_time),
    ):
        value = _ratio(numerator, denominator)
        if value is not None:
            ratio_parts.append(f"{label}={value:.2f}x")

    metric = ""
    for metric in ("cuml_acc", "cpu_acc"):
        value = _positive_value(row, metric)
        if value is not None:
            metric = f"acc={value:.4f}"
            break

    ratios = " ".join(ratio_parts)
    details = " ".join(part for part in (ratios, metric) if part)

    return (
        f"{f'[{index}/{total}]':>9}  "
        f"{str(algo):<26.26}  "
        f"{(_human_count(rows) + ' x ' + _human_count(features)):>14}  "
        f"{('~' + format(size_gb, '.2f') + ' GB') if not math.isnan(size_gb) else '':>10}  "
        f"{_format_seconds_cell(gpu_time):>9}  "
        f"{_format_seconds_cell(cpu_time):>9}  "
        f"{details}"
    ).rstrip()


def _progress_header():
    return (
        f"{'progress':>9}  "
        f"{'algorithm':<26}  "
        f"{'shape':>14}  "
        f"{'data':>10}  "
        f"{'gpu_time':>9}  "
        f"{'cpu_time':>9}  "
        f"{'details'}"
    )


def _print_progress_rows(results_df, start_index, total, dtype, verbose=True):
    if not verbose or results_df.empty:
        return start_index
    if start_index == 1:
        header = _progress_header()
        print(header)
        print("-" * len(header))
    for _, row in _coalesce_progress_rows(results_df).iterrows():
        print(_progress_line(row, start_index, total, dtype))
        start_index += 1
    return start_index


def _median_ratio_by_algorithm(results_df, numerator_column, denominator_column):
    if numerator_column not in results_df or denominator_column not in results_df:
        return []
    rows = []
    for algo, group in results_df.groupby("algo", dropna=False):
        ratios = []
        for _, row in group.iterrows():
            value = _ratio(row.get(numerator_column), row.get(denominator_column))
            if value is not None:
                ratios.append(value)
        if ratios:
            rows.append((str(algo), float(np.median(ratios))))
    return rows


def _print_summary(results_df, csv_path=None, verbose=True):
    if not verbose:
        return
    completed = len(results_df)
    print("")
    print("Summary:")
    print(f"  completed: {completed}")
    if csv_path:
        print(f"  results: {csv_path}")

    largest_gpu = sorted(
        _median_ratio_by_algorithm(results_df, "cuml_time", "cpu_time"),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    smallest_gpu = sorted(
        _median_ratio_by_algorithm(results_df, "cuml_time", "cpu_time"),
        key=lambda item: item[1],
    )[:5]

    if largest_gpu:
        print("  largest gpu acceleration vs cpu:")
        for algo, value in largest_gpu:
            print(f"    {algo}: {value:.2f}x")
    if smallest_gpu:
        print("  smallest gpu acceleration vs cpu:")
        for algo, value in smallest_gpu:
            print(f"    {algo}: {value:.2f}x")


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


def _cpu_compatible_input_type(input_type):
    """Return an input type that sklearn/CPU estimators can consume."""
    if input_type in ("numpy", "pandas"):
        return input_type
    return "numpy"


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


def _variation_shapes(shape_pairs, bench_rows, bench_dims):
    return shape_pairs or [
        {"rows": rows, "features": dims}
        for rows in bench_rows
        for dims in bench_dims
    ]


def _param_combination_count(param_lists):
    count = 1
    for values in param_lists.values():
        count *= len(values)
    return count


def _planned_legacy_result_count(algos_to_run, bench_rows, bench_dims, param_lists):
    return (
        len(algos_to_run)
        * len(bench_rows)
        * len(bench_dims)
        * _param_combination_count(param_lists)
    )


def _planned_config_result_count(benchmark_entries, entry_backends, args, explicit_options):
    total = 0
    for entry, _ in zip(benchmark_entries, entry_backends):
        shape_pairs, bench_rows, bench_dims = _resolved_entry_dimensions(
            entry, args, explicit_options
        )
        param_lists = _config_param_lists(entry, args, explicit_options)
        total += (
            len(_variation_shapes(shape_pairs, bench_rows, bench_dims))
            * _param_combination_count(param_lists)
        )
    return total


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

    progress_total = _planned_config_result_count(
        benchmark_entries, entry_backends, args, explicit_options
    )
    progress_index = 1
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

        run_cuml = "gpu" in selected_backends and allow_gpu_runs

        if not run_cpu and not run_cuml:
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
        cpu_input_type = _cpu_compatible_input_type(input_type)
        dtype = _resolve_dtype(dtype)
        param_lists = _config_param_lists(entry, args, explicit_options)

        variation_shapes = _variation_shapes(shape_pairs, bench_rows, bench_dims)

        for shape in variation_shapes:
            shape_results = []
            run_groups = []
            if run_cpu and run_cuml and cpu_input_type != input_type:
                run_groups.extend(
                    [
                        (False, True, input_type),
                        (True, False, cpu_input_type),
                    ]
                )
            else:
                run_groups.append((run_cpu, run_cuml, input_type))

            for group_run_cpu, group_run_cuml, group_input_type in run_groups:
                if not group_run_cpu and not group_run_cuml:
                    continue
                results_df = runners.run_variations(
                    [algo],
                    dataset_name=dataset_name,
                    bench_rows=[shape["rows"]],
                    bench_dims=[shape["features"]],
                    input_type=group_input_type,
                    test_fraction=test_split,
                    dtype=dtype,
                    run_cpu=group_run_cpu,
                    run_cuml=group_run_cuml,
                    raise_on_error=raise_on_error,
                    n_reps=n_reps,
                    **param_lists,
                )
                results_df["benchmark_id"] = entry["benchmark_id"]
                results_df["config_path"] = resolved["config_path"]
                results_df["suite_name"] = resolved["suite_name"]
                results_df["suite_tier"] = resolved["suite_tier"]
                results_df["profile"] = resolved["profile"]
                results_df["dataset"] = dataset_name
                results_df["operation"] = entry["operation"]
                results_df["dtype"] = _dtype_to_config_value(dtype)
                results_df["n_reps"] = n_reps
                results_df["requested_backends"] = ",".join(selected_backends)
                results_df["gpu_available"] = allow_gpu_runs
                all_results.append(results_df)
                shape_results.append(results_df)

            if shape_results:
                progress_index = _print_progress_rows(
                    pd.concat(shape_results, ignore_index=True),
                    progress_index,
                    progress_total,
                    dtype,
                    verbose=args.verbose,
                )

    if not all_results:
        raise ValueError("No benchmark results were produced.")
    results_df = pd.concat(all_results, ignore_index=True)
    _print_summary(results_df, args.csv, verbose=args.verbose)
    return results_df


def run_benchmark(args, explicit_options=None):
    """Execute the benchmark run from parsed CLI arguments."""
    print(f"Benchmark mode: {get_status_string()}")
    _handle_print_commands(args)  # early exit paths
    explicit_options = explicit_options or set()

    if args.config:
        results_df = _run_config_benchmarks(args, explicit_options)
        _save_json_results(results_df, args.output, args, args.dtype)
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
    progress_total = _planned_legacy_result_count(
        algos_to_run, bench_rows, bench_dims, param_lists
    )
    progress_index = 1

    def progress_callback(record):
        nonlocal progress_index
        progress_df = pd.DataFrame(
            [
                {
                    **record,
                    "dataset": args.dataset,
                    "operation": None,
                    "dtype": _dtype_to_config_value(args.dtype),
                    "n_reps": args.n_reps,
                    "requested_backends": ",".join(selected_backends),
                    "gpu_available": run_gpu,
                }
            ]
        )
        progress_index = _print_progress_rows(
            progress_df,
            progress_index,
            progress_total,
            args.dtype,
            verbose=args.verbose,
        )

    all_results = []
    if "cpu" in selected_backends or run_gpu:
        results_df = runners.run_variations(
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
            progress_callback=progress_callback,
            **param_lists,
        )
        results_df["dataset"] = args.dataset
        results_df["operation"] = None
        results_df["dtype"] = _dtype_to_config_value(args.dtype)
        results_df["n_reps"] = args.n_reps
        results_df["requested_backends"] = ",".join(selected_backends)
        results_df["gpu_available"] = run_gpu
        all_results.append(results_df)

    if not all_results:
        raise ValueError("No benchmark results were produced.")
    results_df = pd.concat(all_results, ignore_index=True)
    _print_summary(results_df, args.csv, verbose=args.verbose)
    _save_json_results(results_df, args.output, args, args.dtype)
    _save_results(results_df, args.csv)


def main(argv=None):
    """Parse arguments and run the benchmark. Returns exit code."""
    argv = argv if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)
    args._argv = list(argv)
    run_benchmark(args, explicit_options=_extract_explicit_options(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
