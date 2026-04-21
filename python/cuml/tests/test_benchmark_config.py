#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from argparse import Namespace
from pathlib import Path

import pandas as pd

from cuml.benchmark.config import load_and_resolve_config
from cuml.benchmark.run_benchmarks import _run_config_benchmarks, main


def _make_args(**overrides):
    args = {
        "config": "/tmp/benchmark.yaml",
        "profile": None,
        "algorithms": [],
        "dataset": "blobs",
        "input_type": "numpy",
        "test_split": 0.1,
        "dtype": "fp32",
        "n_reps": 1,
        "skip_cpu": False,
        "skip_gpu": False,
        "raise_on_error": False,
        "rmm_allocator": "cuda",
        "min_rows": 10,
        "max_rows": 100,
        "num_sizes": 3,
        "num_rows": None,
        "num_features": -1,
        "input_dimensions": [16, 32],
        "default_size": False,
        "param_sweep": None,
        "cuml_param_sweep": None,
        "cpu_param_sweep": None,
        "dataset_param_sweep": None,
        "csv": None,
    }
    args.update(overrides)
    return Namespace(**args)


def test_load_and_resolve_config_default_profile_filters_single_gpu_manifest():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "cuml"
        / "benchmark"
        / "configs"
        / "single_gpu.yaml"
    )
    resolved = load_and_resolve_config(
        str(config_path)
    )

    benchmark_ids = {entry["benchmark_id"] for entry in resolved["benchmarks"]}

    assert resolved["suite_name"] == "single_gpu"
    assert resolved["profile"] == "default"
    assert all(entry["input_type"] == "cupy" for entry in resolved["benchmarks"])
    assert "logreg_fit_narrow_default" in benchmark_ids
    assert "logreg_fit_medium_default" in benchmark_ids
    assert "logreg_fit_wide_default" in benchmark_ids
    assert "elasticnet_fit_narrow_default" in benchmark_ids
    assert "tsne_fit_wide_default" in benchmark_ids
    assert "elasticnet_fit_narrow_extended" not in benchmark_ids
    assert "tsne_fit_wide_nightly" not in benchmark_ids


def test_load_and_resolve_config_single_gpu_profiles_preserve_algorithm_set():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "cuml"
        / "benchmark"
        / "configs"
        / "single_gpu.yaml"
    )

    default_resolved = load_and_resolve_config(str(config_path), profile="default")
    extended_resolved = load_and_resolve_config(
        str(config_path), profile="extended"
    )
    nightly_resolved = load_and_resolve_config(str(config_path), profile="nightly")

    default_algorithms = {
        entry["algorithm"] for entry in default_resolved["benchmarks"]
    }
    extended_algorithms = {
        entry["algorithm"] for entry in extended_resolved["benchmarks"]
    }
    nightly_algorithms = {
        entry["algorithm"] for entry in nightly_resolved["benchmarks"]
    }
    default_ids = {
        entry["benchmark_id"] for entry in default_resolved["benchmarks"]
    }

    assert default_algorithms == extended_algorithms == nightly_algorithms
    assert "logreg_fit_narrow_default" in default_ids
    assert "logreg_fit_medium_default" in default_ids
    assert "logreg_fit_wide_default" in default_ids


def test_load_and_resolve_config_expands_shape_pairs_and_param_grid(tmp_path):
    config_path = tmp_path / "shapes.yaml"
    config_path.write_text(
        """
version: 1

suite:
  name: shapes
  tier: test
  description: shape-pair coverage

defaults:
  dataset: classification
  input_type: numpy
  dtype: fp32
  n_reps: 2
  random_state: 42
  test_split: 0.2
  run_cpu: true
  run_gpu: false
  raise_on_error: true

benchmarks:
  - id: shaped_logreg
    algorithm: LogisticRegression
    operation: fit
    shapes:
      - rows: 100
        features: 8
      - rows: 250
        features: 16
    param_grid:
      C: [0.25, 1.0]
""".strip(),
        encoding="utf-8",
    )

    resolved = load_and_resolve_config(str(config_path))
    [entry] = resolved["benchmarks"]

    assert entry["shape_pairs"] == [
        {"rows": 100, "features": 8},
        {"rows": 250, "features": 16},
    ]
    assert entry["bench_rows"] is None
    assert entry["bench_dims"] is None
    assert entry["param_override_list"] == [{"C": 0.25}, {"C": 1.0}]


def test_run_config_benchmarks_uses_shape_pairs_without_cartesian_product(
    monkeypatch,
):
    calls = []

    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.load_and_resolve_config",
        lambda *args, **kwargs: {
            "config_path": "/tmp/shapes.yaml",
            "suite_name": "test-suite",
            "suite_tier": "test",
            "profile": "default",
            "benchmarks": [
                {
                    "benchmark_id": "shape-bench",
                    "algorithm": "LogisticRegression",
                    "dataset": "classification",
                    "input_type": "numpy",
                    "dtype": "fp32",
                    "n_reps": 2,
                    "random_state": 42,
                    "test_split": 0.2,
                    "run_cpu": True,
                    "run_gpu": False,
                    "raise_on_error": True,
                    "default_size": False,
                    "shape_pairs": [
                        {"rows": 100, "features": 8},
                        {"rows": 250, "features": 16},
                    ],
                    "bench_rows": None,
                    "bench_dims": None,
                    "operation": "fit",
                    "comparison": None,
                    "param_override_list": [{}],
                    "cuml_param_override_list": [{}],
                    "cpu_param_override_list": [{}],
                    "dataset_param_override_list": [{}],
                    "tags": ["test"],
                    "enabled": True,
                    "skip_reason": None,
                    "metadata": {},
                }
            ],
        },
    )
    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.algorithms.algorithm_by_name",
        lambda name: {"name": name},
    )
    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.is_gpu_available", lambda: False
    )
    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.runners.run_variations",
        lambda algos, **kwargs: calls.append(kwargs)
        or pd.DataFrame([{"algo": "LogisticRegression"}]),
    )

    results = _run_config_benchmarks(_make_args(), explicit_options=set())

    assert [(call["bench_rows"], call["bench_dims"]) for call in calls] == [
        ([100], [8]),
        ([250], [16]),
    ]
    assert list(results["benchmark_id"]) == ["shape-bench", "shape-bench"]
    assert set(["config_path", "suite_name", "suite_tier", "profile"]).issubset(
        results.columns
    )


def test_run_config_benchmarks_applies_only_explicit_cli_overrides(monkeypatch):
    calls = []
    setup_calls = []

    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.load_and_resolve_config",
        lambda *args, **kwargs: {
            "config_path": "/tmp/overrides.yaml",
            "suite_name": "test-suite",
            "suite_tier": "test",
            "profile": "default",
            "benchmarks": [
                {
                    "benchmark_id": "override-bench",
                    "algorithm": "LogisticRegression",
                    "dataset": "classification",
                    "input_type": "numpy",
                    "dtype": "fp32",
                    "n_reps": 3,
                    "random_state": 42,
                    "test_split": 0.1,
                    "run_cpu": True,
                    "run_gpu": True,
                    "raise_on_error": True,
                    "default_size": False,
                    "shape_pairs": None,
                    "bench_rows": [200],
                    "bench_dims": [8],
                    "operation": "fit",
                    "comparison": None,
                    "param_override_list": [{"C": 0.5}],
                    "cuml_param_override_list": [{}],
                    "cpu_param_override_list": [{}],
                    "dataset_param_override_list": [{}],
                    "tags": ["test"],
                    "enabled": True,
                    "skip_reason": None,
                    "metadata": {},
                }
            ],
        },
    )
    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.algorithms.algorithm_by_name",
        lambda name: {"name": name},
    )
    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.is_gpu_available", lambda: True
    )
    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.setup_rmm_allocator",
        lambda allocator: setup_calls.append(allocator),
    )
    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.runners.run_variations",
        lambda algos, **kwargs: calls.append(kwargs)
        or pd.DataFrame([{"algo": "LogisticRegression"}]),
    )

    args = _make_args(
        dataset="blobs",
        num_rows=50,
        param_sweep=["C=2.0"],
    )
    _run_config_benchmarks(
        args,
        explicit_options={"--num-rows", "--param-sweep", "--skip-gpu"},
    )

    [call] = calls
    assert call["dataset_name"] == "classification"
    assert call["bench_rows"] == [50]
    assert call["bench_dims"] == [8]
    assert call["param_override_list"] == [{"C": 2.0}]
    assert call["run_cuml"] is False
    assert setup_calls == []


def test_main_runs_config_smoke_manifest_end_to_end(monkeypatch, tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "cuml"
        / "benchmark"
        / "configs"
        / "test.yaml"
    )
    csv_path = tmp_path / "benchmark-results.csv"
    calls = []

    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.algorithms.algorithm_by_name",
        lambda name: {"name": name},
    )
    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.is_gpu_available", lambda: False
    )
    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.runners.run_variations",
        lambda algos, **kwargs: calls.append((algos, kwargs))
        or pd.DataFrame([{"algo": algos[0]["name"]}]),
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--profile",
            "default",
            "--skip-gpu",
            "--csv",
            str(csv_path),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 2
    assert all(kwargs["run_cuml"] is False for _, kwargs in calls)
    assert [kwargs["bench_rows"] for _, kwargs in calls] == [[200], [200]]
    assert [kwargs["bench_dims"] for _, kwargs in calls] == [[8], [8]]

    results = pd.read_csv(csv_path)
    assert set(["benchmark_id", "config_path", "suite_name", "suite_tier", "profile"]).issubset(
        results.columns
    )
    assert set(results["benchmark_id"]) == {
        "test_logreg_fit",
        "test_scaler_fittransform",
    }
