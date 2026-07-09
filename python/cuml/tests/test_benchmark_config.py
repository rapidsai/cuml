#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from cuml.benchmark.config import (
    BenchmarkConfigError,
    load_and_resolve_config,
    load_config_file,
)
from cuml.benchmark.run_benchmarks import (
    _run_config_benchmarks,
    extract_param_overrides,
    main,
    run_benchmark,
)


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
        "backends": None,
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
        "output": None,
        "print_algorithms": False,
        "print_datasets": False,
        "print_status": False,
        "verbose": True,
        "metadata_override": [],
    }
    args.update(overrides)
    return Namespace(**args)


def test_extract_param_overrides_accepts_json_style_lists():
    assert extract_param_overrides(
        ["n_classes=[2,8]", "n_estimators=[10,100]"]
    ) == [
        {"n_classes": 2, "n_estimators": 10},
        {"n_classes": 2, "n_estimators": 100},
        {"n_classes": 8, "n_estimators": 10},
        {"n_classes": 8, "n_estimators": 100},
    ]


def test_checked_in_benchmark_manifests_match_msgspec_schema():
    pytest.importorskip("msgspec")
    pytest.importorskip("yaml")
    configs_dir = (
        Path(__file__).resolve().parents[1] / "cuml" / "benchmark" / "configs"
    )

    manifest_paths = sorted(configs_dir.glob("*.yaml"))
    assert manifest_paths

    for manifest_path in manifest_paths:
        load_config_file(str(manifest_path))


def test_load_and_resolve_config_default_profile_filters_single_gpu_manifest():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "cuml"
        / "benchmark"
        / "configs"
        / "single_gpu.yaml"
    )
    resolved = load_and_resolve_config(str(config_path))

    benchmark_ids = {entry["benchmark_id"] for entry in resolved["benchmarks"]}

    assert resolved["suite_name"] == "single_gpu"
    assert resolved["profile"] == "default"
    assert all(
        entry["input_type"] == "cupy" for entry in resolved["benchmarks"]
    )
    assert "logreg_fit_narrow_default" in benchmark_ids
    assert "logreg_fit_medium_default" in benchmark_ids
    assert "logreg_fit_wide_default" in benchmark_ids
    assert "elasticnet_fit_narrow_default" in benchmark_ids
    assert "tsne_fit_wide_default" in benchmark_ids
    assert "elasticnet_fit_narrow_nightly" not in benchmark_ids
    assert "tsne_fit_wide_nightly" not in benchmark_ids


def test_load_and_resolve_config_single_gpu_profiles_preserve_algorithm_set():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "cuml"
        / "benchmark"
        / "configs"
        / "single_gpu.yaml"
    )

    default_resolved = load_and_resolve_config(
        str(config_path), profile="default"
    )
    nightly_resolved = load_and_resolve_config(
        str(config_path), profile="nightly"
    )

    default_algorithms = {
        entry["algorithm"] for entry in default_resolved["benchmarks"]
    }
    nightly_algorithms = {
        entry["algorithm"] for entry in nightly_resolved["benchmarks"]
    }
    default_ids = {
        entry["benchmark_id"] for entry in default_resolved["benchmarks"]
    }

    assert default_algorithms == nightly_algorithms
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
  backends: [cpu]
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


def test_load_and_resolve_config_expands_compact_variants(tmp_path):
    config_path = tmp_path / "compact.yaml"
    config_path.write_text(
        """
version: 1

suite:
  name: compact
  tier: test
  description: compact benchmark coverage

profiles:
  default:
    include_tags: [default]
  nightly:
    include_tags: [nightly]

defaults:
  dataset: classification
  input_type: numpy
  dtype: fp32
  n_reps: 2
  random_state: 42
  test_split: 0.2
  backends: [cpu]
  raise_on_error: true

benchmarks:
  - id: compact_logreg_fit
    algorithm: LogisticRegression
    operation: fit
    tags: [linear]
    variants:
      narrow:
        features: [8]
        tiers:
          default:
            rows: [100]
          nightly:
            rows: [200]
      wide:
        features: [16]
        tags: [fat]
        tiers:
          default:
            rows: [50]
""".strip(),
        encoding="utf-8",
    )

    resolved = load_and_resolve_config(str(config_path), profile="default")

    assert [entry["benchmark_id"] for entry in resolved["benchmarks"]] == [
        "compact_logreg_fit_narrow_default",
        "compact_logreg_fit_wide_default",
    ]
    assert resolved["benchmarks"][0]["bench_rows"] == [100]
    assert resolved["benchmarks"][0]["bench_dims"] == [8]
    assert resolved["benchmarks"][0]["tags"] == [
        "linear",
        "narrow",
        "default",
    ]
    assert resolved["benchmarks"][1]["tags"] == [
        "linear",
        "wide",
        "fat",
        "default",
    ]


@pytest.mark.parametrize(
    ("yaml_text", "error_match"),
    [
        (
            """
version: true
suite:
  name: bool-version
  tier: test
  description: bool version
benchmarks:
  - algorithm: LogisticRegression
    dataset: classification
    input_type: numpy
    dtype: fp32
    n_reps: 1
    test_split: 0.1
    rows: [100]
    features: [8]
""".strip(),
            "Config field 'version' must be an integer",
        ),
        (
            """
version: 2
suite:
  name: version-two
  tier: test
  description: unsupported version
benchmarks:
  - algorithm: LogisticRegression
    dataset: classification
    input_type: numpy
    dtype: fp32
    n_reps: 1
    test_split: 0.1
    rows: [100]
    features: [8]
""".strip(),
            "Unsupported config version 2",
        ),
        (
            """
version: 1
suite:
  name: bool-rows
  tier: test
  description: bool rows
defaults:
  dataset: classification
  input_type: numpy
  dtype: fp32
  n_reps: 1
  test_split: 0.1
benchmarks:
  - algorithm: LogisticRegression
    rows: [true]
    features: [8]
""".strip(),
            "Field 'benchmarks\\[0\\]\\.rows' must contain only integers",
        ),
        (
            """
version: 1
suite:
  name: bool-shapes
  tier: test
  description: bool shapes
defaults:
  dataset: classification
  input_type: numpy
  dtype: fp32
  n_reps: 1
  test_split: 0.1
benchmarks:
  - algorithm: LogisticRegression
    shapes:
      - rows: true
        features: 8
""".strip(),
            "Field 'benchmarks\\[0\\]\\.shapes\\[0\\]' must contain integer 'rows' and 'features'",
        ),
        (
            """
version: 1
suite:
  name: bad-test-split
  tier: test
  description: bad test split
defaults:
  dataset: classification
  input_type: numpy
  dtype: fp32
  n_reps: 1
  test_split: 1.5
benchmarks:
  - algorithm: LogisticRegression
    rows: [100]
    features: [8]
""".strip(),
            "defaults field 'test_split' must be between 0.0 and 1.0",
        ),
    ],
)
def test_load_and_resolve_config_rejects_boolean_numeric_values(
    tmp_path, yaml_text, error_match
):
    config_path = tmp_path / "bool-numeric.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match=error_match):
        load_and_resolve_config(str(config_path))


@pytest.mark.parametrize(
    ("defaults_block", "expected_field"),
    [
        ("", "dataset"),
        ("  dataset: classification\n", "input_type"),
        ("  dataset: classification\n  input_type: numpy\n", "dtype"),
        (
            "  dataset: classification\n  input_type: numpy\n  dtype: fp32\n",
            "n_reps",
        ),
        (
            "  dataset: classification\n"
            "  input_type: numpy\n"
            "  dtype: fp32\n"
            "  n_reps: 2\n",
            "test_split",
        ),
    ],
)
def test_load_and_resolve_config_requires_runtime_fields_after_defaults(
    tmp_path, defaults_block, expected_field
):
    config_path = tmp_path / "missing-required.yaml"
    config_path.write_text(
        (
            "version: 1\n\n"
            "suite:\n"
            "  name: missing-required\n"
            "  tier: test\n"
            "  description: missing field coverage\n\n"
            "defaults:\n"
            f"{defaults_block}"
            "  run_cpu: true\n"
            "  run_gpu: false\n\n"
            "benchmarks:\n"
            "  - id: shaped_logreg\n"
            "    algorithm: LogisticRegression\n"
            "    operation: fit\n"
            "    rows: [100]\n"
            "    features: [8]\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        BenchmarkConfigError,
        match=rf"must define .*'{expected_field}'.*after applying defaults",
    ):
        load_and_resolve_config(str(config_path))


def test_load_and_resolve_config_rejects_profile_with_zero_matches(tmp_path):
    config_path = tmp_path / "empty-profile.yaml"
    config_path.write_text(
        """
version: 1

suite:
  name: empty-profile
  tier: test
  description: empty profile coverage

profiles:
  default:
    include_tags: [default-profile]

defaults:
  dataset: classification
  input_type: numpy
  dtype: fp32
  n_reps: 1
  test_split: 0.1
  backends: [cpu]

benchmarks:
  - id: logreg
    algorithm: LogisticRegression
    operation: fit
    rows: [100]
    features: [8]
    tags: [other-tag]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        BenchmarkConfigError,
        match="Profile 'default' did not match any benchmark entries",
    ):
        load_and_resolve_config(str(config_path))


def test_load_and_resolve_config_algorithm_filter_is_case_insensitive():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "cuml"
        / "benchmark"
        / "configs"
        / "single_gpu.yaml"
    )

    resolved = load_and_resolve_config(
        str(config_path), algorithm_filter=["logisticregression"]
    )

    assert resolved["benchmarks"]
    assert {entry["algorithm"] for entry in resolved["benchmarks"]} == {
        "LogisticRegression"
    }


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
                    "backends": ["cpu"],
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
    assert set(
        ["config_path", "suite_name", "suite_tier", "profile"]
    ).issubset(results.columns)


def test_run_config_benchmarks_applies_only_explicit_cli_overrides(
    monkeypatch,
):
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
                    "backends": ["cpu", "gpu"],
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


def test_run_config_benchmarks_backends_cli_override(monkeypatch):
    calls = []
    setup_calls = []

    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.load_and_resolve_config",
        lambda *args, **kwargs: {
            "config_path": "/tmp/backends.yaml",
            "suite_name": "test-suite",
            "suite_tier": "test",
            "profile": "default",
            "benchmarks": [
                {
                    "benchmark_id": "backend-bench",
                    "algorithm": "LogisticRegression",
                    "dataset": "classification",
                    "input_type": "numpy",
                    "dtype": "fp32",
                    "n_reps": 1,
                    "random_state": 42,
                    "test_split": 0.1,
                    "backends": ["cpu", "gpu"],
                    "run_cpu": True,
                    "run_gpu": True,
                    "raise_on_error": True,
                    "default_size": False,
                    "shape_pairs": None,
                    "bench_rows": [200],
                    "bench_dims": [8],
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

    _run_config_benchmarks(
        _make_args(backends="gpu"), explicit_options={"--backends"}
    )

    [call] = calls
    assert call["run_cpu"] is False
    assert call["run_cuml"] is True
    assert setup_calls == ["cuda"]


def test_run_config_benchmarks_splits_mixed_backends_for_gpu_native_inputs(
    monkeypatch,
):
    calls = []

    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.load_and_resolve_config",
        lambda *args, **kwargs: {
            "config_path": "/tmp/backends.yaml",
            "suite_name": "test-suite",
            "suite_tier": "test",
            "profile": "default",
            "benchmarks": [
                {
                    "benchmark_id": "mixed-backend-bench",
                    "algorithm": "LogisticRegression",
                    "dataset": "classification",
                    "input_type": "cupy",
                    "dtype": "fp32",
                    "n_reps": 1,
                    "random_state": 42,
                    "test_split": 0.1,
                    "backends": ["cpu", "gpu"],
                    "run_cpu": True,
                    "run_gpu": True,
                    "raise_on_error": True,
                    "default_size": False,
                    "shape_pairs": None,
                    "bench_rows": [200],
                    "bench_dims": [8],
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
        "cuml.benchmark.run_benchmarks.is_gpu_available", lambda: True
    )
    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.setup_rmm_allocator",
        lambda allocator: None,
    )
    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.runners.run_variations",
        lambda algos, **kwargs: calls.append(kwargs)
        or pd.DataFrame([{"algo": "LogisticRegression"}]),
    )

    _run_config_benchmarks(_make_args(), explicit_options=set())

    assert [
        (call["input_type"], call["run_cpu"], call["run_cuml"])
        for call in calls
    ] == [
        ("cupy", False, True),
        ("numpy", True, False),
    ]


@pytest.mark.parametrize(
    ("arg_name", "expected_run_cpu", "expected_run_cuml"),
    [
        ("skip_gpu", True, False),
        ("skip_cpu", False, True),
    ],
)
def test_run_benchmark_config_mode_respects_skip_flags_without_explicit_options(
    monkeypatch, arg_name, expected_run_cpu, expected_run_cuml
):
    calls = []
    setup_calls = []

    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.load_and_resolve_config",
        lambda *args, **kwargs: {
            "config_path": "/tmp/config.yaml",
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
                    "n_reps": 1,
                    "random_state": 42,
                    "test_split": 0.1,
                    "run_cpu": True,
                    "run_gpu": True,
                    "backends": ["cpu", "gpu"],
                    "raise_on_error": True,
                    "default_size": False,
                    "shape_pairs": None,
                    "bench_rows": [200],
                    "bench_dims": [8],
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

    args = _make_args(config="/tmp/config.yaml", **{arg_name: True})
    run_benchmark(args)

    [call] = calls
    assert call["run_cpu"] is expected_run_cpu
    assert call["run_cuml"] is expected_run_cuml
    assert bool(setup_calls) is expected_run_cuml


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
    assert set(
        ["benchmark_id", "config_path", "suite_name", "suite_tier", "profile"]
    ).issubset(results.columns)
    assert set(results["benchmark_id"]) == {
        "test_logreg_fit",
        "test_scaler_fittransform",
    }


def test_main_writes_grouped_json_output(monkeypatch, tmp_path):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "cuml"
        / "benchmark"
        / "configs"
        / "test.yaml"
    )
    output_path = tmp_path / "benchmark-results.json"

    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.algorithms.algorithm_by_name",
        lambda name: {"name": name},
    )
    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.is_gpu_available", lambda: False
    )
    monkeypatch.setattr(
        "cuml.benchmark.run_benchmarks.runners.run_variations",
        lambda algos, **kwargs: pd.DataFrame(
            [
                {
                    "algo": algos[0]["name"],
                    "input": kwargs["input_type"],
                    "cpu_time": 0.5,
                    "cpu_acc": 0.9,
                    "cpu_params": {"C": 2.0, "solver": "lbfgs"},
                    "n_samples": kwargs["bench_rows"][0],
                    "n_features": kwargs["bench_dims"][0],
                    **kwargs["param_override_list"][0],
                }
            ]
        ),
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--profile",
            "default",
            "--skip-gpu",
            "--output",
            str(output_path),
            "--metadata-override",
            "hardware.label=test-node",
            "--metadata-override",
            "hardware.gpu.effective.name=Test GPU",
            "--metadata-override",
            "hardware.gpu.effective.total_memory_bytes=80000000000",
            "--metadata-override",
            "hardware.cpu.effective.model=Test CPU",
            "--metadata-override",
            "hardware.cpu.effective.logical_cores=16",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["result_schema_version"] == 1
    assert payload["metadata"]["config"]["profile"] == "default"
    assert payload["metadata"]["hardware"]["label"] == "test-node"
    assert payload["metadata"]["hardware"]["gpu"]["effective"] == {
        "name": "Test GPU",
        "total_memory_bytes": 80000000000,
    }
    assert payload["metadata"]["hardware"]["cpu"]["effective"] == {
        "model": "Test CPU",
        "logical_cores": 16,
    }
    assert [result["benchmark_id"] for result in payload["results"]] == [
        "test_logreg_fit",
        "test_scaler_fittransform",
    ]
    assert payload["results"][0]["backends"]["cpu"]["time_sec"] == 0.5
    assert payload["results"][0]["backends"]["cpu"]["accuracy"] == 0.9
    assert payload["results"][0]["params"]["declared"] == {}
    assert payload["results"][0]["params"]["effective"]["cpu"] == {
        "C": 2.0,
        "solver": "lbfgs",
    }


def test_json_result_preserves_declared_params():
    from cuml.benchmark.run_benchmarks import _result_record

    row = pd.Series(
        {
            "benchmark_id": "param-bench",
            "algo": "LogisticRegression",
            "dataset": "classification",
            "operation": "fit",
            "input": "numpy",
            "n_samples": 100,
            "n_features": 8,
            "cpu_time": 0.1,
            "C": 2.0,
            "max_iter": 50,
        }
    )

    result = _result_record(row, "fp32")

    assert result["params"]["declared"] == {"C": 2.0, "max_iter": 50}


def test_json_result_marks_requested_missing_backend_as_skipped():
    from cuml.benchmark.run_benchmarks import _result_record

    row = pd.Series(
        {
            "benchmark_id": "skip-bench",
            "algo": "LogisticRegression",
            "dataset": "classification",
            "operation": "fit",
            "input": "numpy",
            "n_samples": 100,
            "n_features": 8,
            "cpu_time": 0.1,
            "requested_backends": "cpu,gpu",
            "gpu_available": False,
        }
    )

    result = _result_record(row, "fp32")

    assert result["backends"]["cpu"]["status"] == "success"
    assert result["backends"]["gpu"] == {
        "status": "skipped",
        "reason": "GPU unavailable",
    }


def test_json_result_preserves_zero_accuracy():
    from cuml.benchmark.run_benchmarks import _result_record

    row = pd.Series(
        {
            "benchmark_id": "zero-acc-bench",
            "algo": "LogisticRegression",
            "dataset": "classification",
            "operation": "fit",
            "input": "numpy",
            "n_samples": 100,
            "n_features": 8,
            "cpu_time": 0.1,
            "cpu_acc": 0.0,
        }
    )

    result = _result_record(row, "fp32")

    assert result["backends"]["cpu"]["accuracy"] == 0.0


def test_progress_line_omits_metric_when_accuracy_is_missing():
    from cuml.benchmark.run_benchmarks import _progress_line

    row = pd.Series(
        {
            "algo": "LogisticRegression",
            "n_samples": 100,
            "n_features": 8,
            "cpu_time": 0.1,
        }
    )

    line = _progress_line(row, 1, 1, "fp32")

    assert "cpu_acc" not in line
    assert "acc=" not in line


def test_write_json_atomic_replaces_existing_file(tmp_path):
    from cuml.benchmark.run_benchmarks import _write_json_atomic

    output_path = tmp_path / "results.json"
    output_path.write_text('{"old": true}\n', encoding="utf-8")

    _write_json_atomic(str(output_path), {"new": True})

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"new": True}


def test_write_json_atomic_preserves_existing_file_on_failure(
    monkeypatch, tmp_path
):
    from cuml.benchmark import run_benchmarks

    output_path = tmp_path / "results.json"
    output_path.write_text('{"old": true}\n', encoding="utf-8")

    def fail_dump(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(run_benchmarks.json, "dump", fail_dump)

    with pytest.raises(RuntimeError, match="boom"):
        run_benchmarks._write_json_atomic(str(output_path), {"new": True})

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"old": True}
    assert not list(tmp_path.glob(".benchmark-*.json"))


def test_collect_package_snapshot_prefers_conda(monkeypatch):
    from cuml.benchmark import _metadata

    monkeypatch.setenv("CONDA_PREFIX", "/tmp/conda-env")
    monkeypatch.setattr(
        _metadata,
        "_run_json_command",
        lambda command: (
            [
                {
                    "name": "cuml",
                    "version": "26.06",
                    "build_string": "cuda13_py313_0",
                    "channel": "rapidsai-nightly",
                    "base_url": "https://conda.anaconda.org/rapidsai-nightly",
                    "platform": "linux-64",
                }
            ],
            None,
        ),
    )

    snapshot = _metadata.collect_package_snapshot()

    assert snapshot == {
        "package_snapshot_source": "conda",
        "conda_prefix": "/tmp/conda-env",
        "packages": [
            {
                "name": "cuml",
                "version": "26.06",
                "build": "cuda13_py313_0",
                "channel": "rapidsai-nightly",
            }
        ],
    }


def test_collect_package_snapshot_falls_back_to_pip(monkeypatch):
    from cuml.benchmark import _metadata

    monkeypatch.setenv("CONDA_PREFIX", "/tmp/conda-env")

    def fake_run_json_command(command):
        if command[:2] == ["conda", "list"]:
            return None, "conda failed"
        return [
            {
                "name": "cuml",
                "version": "26.06",
                "editable_project_location": "/tmp",
            }
        ], None

    monkeypatch.setattr(_metadata, "_run_json_command", fake_run_json_command)

    snapshot = _metadata.collect_package_snapshot()

    assert snapshot == {
        "package_snapshot_source": "pip",
        "conda_prefix": "/tmp/conda-env",
        "packages": [{"name": "cuml", "version": "26.06"}],
        "conda_error": "conda failed",
    }
