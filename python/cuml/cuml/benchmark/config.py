#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""YAML benchmark config loading and resolution."""

from __future__ import annotations

import itertools
import os
import sys
from copy import deepcopy
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency issue
    raise RuntimeError(
        "PyYAML is required to load benchmark config files."
    ) from exc

# Supports both package and standalone execution
try:
    from cuml.benchmark import algorithms
except ImportError:
    if not any("cuml/benchmark" in p for p in sys.path):
        raise
    import algorithms  # noqa: E402


TOP_LEVEL_KEYS = {"version", "suite", "profiles", "defaults", "benchmarks"}
SUITE_KEYS = {"name", "tier", "description"}
PROFILE_KEYS = {"include_tags"}
BENCHMARK_KEYS = {
    "id",
    "algorithm",
    "dataset",
    "input_type",
    "dtype",
    "n_reps",
    "random_state",
    "test_split",
    "run_cpu",
    "run_gpu",
    "raise_on_error",
    "default_size",
    "shapes",
    "rows",
    "features",
    "operation",
    "params",
    "cuml_params",
    "cpu_params",
    "dataset_params",
    "param_grid",
    "cuml_param_grid",
    "cpu_param_grid",
    "dataset_param_grid",
    "comparison",
    "tags",
    "enabled",
    "skip_reason",
    "metadata",
}
DICT_FIELDS = {
    "params",
    "cuml_params",
    "cpu_params",
    "dataset_params",
    "comparison",
    "metadata",
}
GRID_FIELDS = {
    "param_grid",
    "cuml_param_grid",
    "cpu_param_grid",
    "dataset_param_grid",
}
MERGEABLE_DICT_FIELDS = {
    "params",
    "cuml_params",
    "cpu_params",
    "dataset_params",
    "comparison",
    "metadata",
}
SCALAR_FIELDS = {
    "id",
    "algorithm",
    "dataset",
    "input_type",
    "dtype",
    "n_reps",
    "random_state",
    "test_split",
    "run_cpu",
    "run_gpu",
    "raise_on_error",
    "default_size",
    "operation",
    "enabled",
    "skip_reason",
}
ALLOWED_OPERATIONS = {
    "fit",
    "predict",
    "transform",
    "fit_transform",
    "fit_predict",
    "fit_kneighbors",
    "kneighbors",
}


class BenchmarkConfigError(ValueError):
    """Raised when a benchmark config file is invalid."""


def load_config_file(config_path: str) -> dict[str, Any]:
    """Load a YAML benchmark config file."""
    abs_path = os.path.abspath(config_path)
    with open(abs_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raise BenchmarkConfigError(f"Config file is empty: {abs_path}")
    if not isinstance(raw, dict):
        raise BenchmarkConfigError(
            f"Top-level YAML document must be a mapping: {abs_path}"
        )

    validate_config(raw)
    return raw


def resolve_config(
    raw_config: dict[str, Any],
    *,
    config_path: str | None = None,
    profile: str | None = None,
    algorithm_filter: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Resolve a raw config dict into the normalized config model."""
    validate_config(raw_config)

    suite = raw_config["suite"]
    defaults = deepcopy(raw_config.get("defaults", {}))
    profiles = deepcopy(raw_config.get("profiles", {}))
    selected_profile = _select_profile(profiles, profile)

    benchmark_entries = []
    for entry in raw_config.get("benchmarks", []):
        resolved_entry = _apply_defaults(defaults, entry)
        _validate_post_defaults_entry(resolved_entry)
        if not resolved_entry.get("enabled", True):
            continue

        benchmark_entries.append(_normalize_resolved_entry(resolved_entry))

    benchmark_entries = _apply_profile_selection(
        benchmark_entries, profiles, selected_profile
    )
    benchmark_entries = _apply_algorithm_filter(
        benchmark_entries, algorithm_filter
    )

    return {
        "config_path": os.path.abspath(config_path) if config_path else None,
        "suite_name": suite["name"],
        "suite_tier": suite["tier"],
        "profile": selected_profile,
        "suite_metadata": {"description": suite["description"]},
        "benchmarks": benchmark_entries,
    }


def load_and_resolve_config(
    config_path: str,
    *,
    profile: str | None = None,
    algorithm_filter: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load a config file from disk and resolve it."""
    raw = load_config_file(config_path)
    return resolve_config(
        raw,
        config_path=config_path,
        profile=profile,
        algorithm_filter=algorithm_filter,
    )


def validate_config(raw_config: dict[str, Any]) -> None:
    """Validate the full config structure."""
    unknown_top_level = set(raw_config) - TOP_LEVEL_KEYS
    if unknown_top_level:
        raise BenchmarkConfigError(
            f"Unknown top-level config keys: {sorted(unknown_top_level)}"
        )

    version = raw_config.get("version")
    if not isinstance(version, int):
        raise BenchmarkConfigError("Config field 'version' must be an integer")

    suite = raw_config.get("suite")
    if not isinstance(suite, dict):
        raise BenchmarkConfigError("Config field 'suite' must be a mapping")
    unknown_suite = set(suite) - SUITE_KEYS
    if unknown_suite:
        raise BenchmarkConfigError(
            f"Unknown suite keys: {sorted(unknown_suite)}"
        )
    for key in ("name", "tier", "description"):
        if not isinstance(suite.get(key), str) or not suite[key]:
            raise BenchmarkConfigError(
                f"Suite field '{key}' must be a non-empty string"
            )

    defaults = raw_config.get("defaults", {})
    if not isinstance(defaults, dict):
        raise BenchmarkConfigError("Config field 'defaults' must be a mapping")
    _validate_default_or_entry(
        defaults, context="defaults", require_algorithm=False
    )

    profiles = raw_config.get("profiles", {})
    if not isinstance(profiles, dict):
        raise BenchmarkConfigError("Config field 'profiles' must be a mapping")
    if profiles:
        if "default" not in profiles:
            raise BenchmarkConfigError(
                "A 'default' profile is required when profiles are defined"
            )
        for profile_name, profile_def in profiles.items():
            if not isinstance(profile_def, dict):
                raise BenchmarkConfigError(
                    f"Profile '{profile_name}' must be a mapping"
                )
            unknown_profile = set(profile_def) - PROFILE_KEYS
            if unknown_profile:
                raise BenchmarkConfigError(
                    f"Unknown keys in profile '{profile_name}': "
                    f"{sorted(unknown_profile)}"
                )
            include_tags = profile_def.get("include_tags")
            if not isinstance(include_tags, list) or not include_tags:
                raise BenchmarkConfigError(
                    f"Profile '{profile_name}' must define a non-empty "
                    "'include_tags' list"
                )
            for tag in include_tags:
                if not isinstance(tag, str) or not tag:
                    raise BenchmarkConfigError(
                        f"Profile '{profile_name}' contains an invalid tag"
                    )

    benchmarks = raw_config.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise BenchmarkConfigError(
            "Config field 'benchmarks' must be a non-empty list"
        )

    seen_ids = set()
    for idx, entry in enumerate(benchmarks):
        if not isinstance(entry, dict):
            raise BenchmarkConfigError(
                f"Benchmark entry at index {idx} must be a mapping"
            )
        _validate_default_or_entry(
            entry, context=f"benchmarks[{idx}]", require_algorithm=True
        )
        entry_id = entry.get("id")
        if entry_id is not None:
            if entry_id in seen_ids:
                raise BenchmarkConfigError(
                    f"Duplicate benchmark id '{entry_id}'"
                )
            seen_ids.add(entry_id)


def _validate_default_or_entry(
    entry: dict[str, Any], *, context: str, require_algorithm: bool
) -> None:
    unknown_keys = set(entry) - BENCHMARK_KEYS
    if unknown_keys:
        raise BenchmarkConfigError(
            f"Unknown keys in {context}: {sorted(unknown_keys)}"
        )

    if require_algorithm:
        algorithm_name = entry.get("algorithm")
        if not isinstance(algorithm_name, str) or not algorithm_name:
            raise BenchmarkConfigError(
                f"{context} must define a non-empty 'algorithm'"
            )
        if algorithms.algorithm_by_name(algorithm_name) is None:
            raise BenchmarkConfigError(
                f"{context} references unknown or unavailable algorithm "
                f"'{algorithm_name}'"
            )

    if "id" in entry and (
        not isinstance(entry["id"], str) or not entry["id"].strip()
    ):
        raise BenchmarkConfigError(
            f"{context} field 'id' must be a non-empty string"
        )

    if "operation" in entry and entry["operation"] not in ALLOWED_OPERATIONS:
        raise BenchmarkConfigError(
            f"{context} has unsupported operation '{entry['operation']}'"
        )

    for field in DICT_FIELDS:
        if field in entry and not isinstance(entry[field], dict):
            raise BenchmarkConfigError(
                f"{context} field '{field}' must be a mapping"
            )

    for field in GRID_FIELDS:
        if field not in entry:
            continue
        value = entry[field]
        if not isinstance(value, dict):
            raise BenchmarkConfigError(
                f"{context} field '{field}' must be a mapping"
            )
        for key, values in value.items():
            if not isinstance(key, str) or not key:
                raise BenchmarkConfigError(
                    f"{context} field '{field}' has an invalid key"
                )
            if not isinstance(values, list) or not values:
                raise BenchmarkConfigError(
                    f"{context} field '{field}.{key}' must be a non-empty list"
                )

    for fixed_field, grid_field in (
        ("params", "param_grid"),
        ("cuml_params", "cuml_param_grid"),
        ("cpu_params", "cpu_param_grid"),
        ("dataset_params", "dataset_param_grid"),
    ):
        fixed_keys = set(entry.get(fixed_field, {}))
        grid_keys = set(entry.get(grid_field, {}))
        if fixed_keys & grid_keys:
            overlap = sorted(fixed_keys & grid_keys)
            raise BenchmarkConfigError(
                f"{context} has overlapping keys in '{fixed_field}' and "
                f"'{grid_field}': {overlap}"
            )

    if "tags" in entry:
        _validate_tag_list(entry["tags"], field_name=f"{context}.tags")

    if "enabled" in entry and not isinstance(entry["enabled"], bool):
        raise BenchmarkConfigError(
            f"{context} field 'enabled' must be a boolean"
        )
    if "run_cpu" in entry and not isinstance(entry["run_cpu"], bool):
        raise BenchmarkConfigError(
            f"{context} field 'run_cpu' must be a boolean"
        )
    if "run_gpu" in entry and not isinstance(entry["run_gpu"], bool):
        raise BenchmarkConfigError(
            f"{context} field 'run_gpu' must be a boolean"
        )
    if "raise_on_error" in entry and not isinstance(
        entry["raise_on_error"], bool
    ):
        raise BenchmarkConfigError(
            f"{context} field 'raise_on_error' must be a boolean"
        )
    if "default_size" in entry and not isinstance(entry["default_size"], bool):
        raise BenchmarkConfigError(
            f"{context} field 'default_size' must be a boolean"
        )

    for numeric_field in ("n_reps", "random_state"):
        if numeric_field in entry and not isinstance(
            entry[numeric_field], int
        ):
            raise BenchmarkConfigError(
                f"{context} field '{numeric_field}' must be an integer"
            )
    if "test_split" in entry and not isinstance(
        entry["test_split"], (int, float)
    ):
        raise BenchmarkConfigError(
            f"{context} field 'test_split' must be numeric"
        )

    if "rows" in entry:
        _normalize_int_list(entry["rows"], field_name=f"{context}.rows")
    if "features" in entry:
        _normalize_int_list(
            entry["features"], field_name=f"{context}.features"
        )
    if "shapes" in entry:
        _normalize_shapes(entry["shapes"], field_name=f"{context}.shapes")

    if "shapes" in entry and ("rows" in entry or "features" in entry):
        raise BenchmarkConfigError(
            f"{context} cannot define 'shapes' together with 'rows'/'features'"
        )
    if entry.get("default_size") and (
        "rows" in entry or "features" in entry or "shapes" in entry
    ):
        raise BenchmarkConfigError(
            f"{context} cannot define 'default_size: true' together with "
            "'rows', 'features', or 'shapes'"
        )


def _apply_defaults(
    defaults: dict[str, Any], entry: dict[str, Any]
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}

    for field in SCALAR_FIELDS:
        if field in defaults:
            resolved[field] = deepcopy(defaults[field])
        if field in entry:
            resolved[field] = deepcopy(entry[field])

    for field in MERGEABLE_DICT_FIELDS:
        merged = {}
        if isinstance(defaults.get(field), dict):
            merged.update(deepcopy(defaults[field]))
        if isinstance(entry.get(field), dict):
            merged.update(deepcopy(entry[field]))
        if merged:
            resolved[field] = merged

    for field in GRID_FIELDS:
        merged = {}
        if isinstance(defaults.get(field), dict):
            merged.update(deepcopy(defaults[field]))
        if isinstance(entry.get(field), dict):
            merged.update(deepcopy(entry[field]))
        if merged:
            resolved[field] = merged

    default_tags = deepcopy(defaults.get("tags", []))
    entry_tags = deepcopy(entry.get("tags", []))
    if default_tags or entry_tags:
        resolved["tags"] = _merge_tags(default_tags, entry_tags)

    for field in ("rows", "features", "shapes"):
        if field in defaults:
            resolved[field] = deepcopy(defaults[field])
        if field in entry:
            resolved[field] = deepcopy(entry[field])

    if "algorithm" in entry:
        resolved["algorithm"] = entry["algorithm"]

    return resolved


def _validate_post_defaults_entry(entry: dict[str, Any]) -> None:
    _validate_default_or_entry(
        entry,
        context=f"benchmark '{entry.get('id', entry['algorithm'])}'",
        require_algorithm=True,
    )

    benchmark_name = entry.get("id", entry["algorithm"])

    for field in ("dataset", "input_type", "dtype"):
        if not isinstance(entry.get(field), str) or not entry[field]:
            raise BenchmarkConfigError(
                f"Benchmark '{benchmark_name}' must define a non-empty "
                f"'{field}' after applying defaults"
            )

    if not isinstance(entry.get("n_reps"), int):
        raise BenchmarkConfigError(
            f"Benchmark '{benchmark_name}' must define integer 'n_reps' "
            "after applying defaults"
        )

    if not isinstance(entry.get("test_split"), (int, float)):
        raise BenchmarkConfigError(
            f"Benchmark '{benchmark_name}' must define numeric 'test_split' "
            "after applying defaults"
        )

    if not entry.get("run_cpu", True) and not entry.get("run_gpu", True):
        raise BenchmarkConfigError(
            f"Benchmark '{benchmark_name}' cannot "
            "disable both CPU and GPU execution"
        )


def _normalize_resolved_entry(entry: dict[str, Any]) -> dict[str, Any]:
    fixed_param_overrides = _build_override_list(
        entry.get("params"), entry.get("param_grid")
    )
    fixed_cuml_overrides = _build_override_list(
        entry.get("cuml_params"), entry.get("cuml_param_grid")
    )
    fixed_cpu_overrides = _build_override_list(
        entry.get("cpu_params"), entry.get("cpu_param_grid")
    )
    fixed_dataset_overrides = _build_override_list(
        entry.get("dataset_params"), entry.get("dataset_param_grid")
    )

    default_size = bool(entry.get("default_size", False))
    shape_pairs = None
    bench_rows = None
    bench_dims = None

    if "shapes" in entry:
        shape_pairs = _normalize_shapes(entry["shapes"], field_name="shapes")
    elif default_size:
        bench_rows = [0]
        bench_dims = [0]
    else:
        bench_rows = _normalize_int_list(
            entry.get("rows"), field_name="rows", allow_none=False
        )
        bench_dims = _normalize_int_list(
            entry.get("features"), field_name="features", allow_none=False
        )

    return {
        "benchmark_id": entry.get("id", entry["algorithm"]),
        "algorithm": entry["algorithm"],
        "dataset": entry.get("dataset"),
        "input_type": entry.get("input_type"),
        "dtype": entry.get("dtype"),
        "n_reps": entry.get("n_reps"),
        "random_state": entry.get("random_state"),
        "test_split": entry.get("test_split"),
        "run_cpu": entry.get("run_cpu", True),
        "run_gpu": entry.get("run_gpu", True),
        "raise_on_error": entry.get("raise_on_error", False),
        "default_size": default_size,
        "shape_pairs": shape_pairs,
        "bench_rows": bench_rows,
        "bench_dims": bench_dims,
        "operation": entry.get("operation"),
        "comparison": deepcopy(entry.get("comparison")),
        "param_override_list": fixed_param_overrides,
        "cuml_param_override_list": fixed_cuml_overrides,
        "cpu_param_override_list": fixed_cpu_overrides,
        "dataset_param_override_list": fixed_dataset_overrides,
        "tags": deepcopy(entry.get("tags", [])),
        "enabled": entry.get("enabled", True),
        "skip_reason": entry.get("skip_reason"),
        "metadata": deepcopy(entry.get("metadata", {})),
    }


def _select_profile(
    profiles: dict[str, Any], requested_profile: str | None
) -> str | None:
    if not profiles:
        if requested_profile is not None:
            raise BenchmarkConfigError(
                f"Profile '{requested_profile}' requested, but the config "
                "does not define any profiles"
            )
        return None

    selected = requested_profile or "default"
    if selected not in profiles:
        available = sorted(profiles.keys())
        raise BenchmarkConfigError(
            f"Unknown profile '{selected}'. Available profiles: {available}"
        )
    return selected


def _apply_profile_selection(
    benchmark_entries: list[dict[str, Any]],
    profiles: dict[str, Any],
    selected_profile: str | None,
) -> list[dict[str, Any]]:
    if selected_profile is None:
        return benchmark_entries

    include_tags = set(profiles[selected_profile]["include_tags"])
    return [
        entry
        for entry in benchmark_entries
        if include_tags.intersection(entry.get("tags", []))
    ]


def _apply_algorithm_filter(
    benchmark_entries: list[dict[str, Any]],
    algorithm_filter: list[str] | tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    if not algorithm_filter:
        return benchmark_entries

    wanted = set(algorithm_filter)
    filtered = [e for e in benchmark_entries if e["algorithm"] in wanted]
    if not filtered:
        raise BenchmarkConfigError(
            "Algorithm filter did not match any resolved benchmark entries: "
            f"{sorted(wanted)}"
        )
    return filtered


def _build_override_list(
    fixed_values: dict[str, Any] | None,
    grid_values: dict[str, list[Any]] | None,
) -> list[dict[str, Any]]:
    fixed = deepcopy(fixed_values or {})
    grid = deepcopy(grid_values or {})

    if not fixed and not grid:
        return [{}]
    if not grid:
        return [fixed]

    grid_keys = list(grid.keys())
    combinations = itertools.product(*(grid[key] for key in grid_keys))

    result = []
    for combo in combinations:
        combo_dict = dict(zip(grid_keys, combo))
        result.append({**fixed, **combo_dict})
    return result


def _normalize_int_list(
    value: Any,
    *,
    field_name: str,
    allow_none: bool = False,
) -> list[int] | None:
    if value is None:
        if allow_none:
            return None
        raise BenchmarkConfigError(f"Field '{field_name}' is required")

    if isinstance(value, int):
        return [value]
    if not isinstance(value, list) or not value:
        raise BenchmarkConfigError(
            f"Field '{field_name}' must be an integer or a non-empty list "
            "of integers"
        )
    if not all(isinstance(v, int) for v in value):
        raise BenchmarkConfigError(
            f"Field '{field_name}' must contain only integers"
        )
    return value


def _normalize_shapes(value: Any, *, field_name: str) -> list[dict[str, int]]:
    if not isinstance(value, list) or not value:
        raise BenchmarkConfigError(
            f"Field '{field_name}' must be a non-empty list"
        )

    shape_pairs = []
    for idx, shape in enumerate(value):
        if not isinstance(shape, dict):
            raise BenchmarkConfigError(
                f"Field '{field_name}[{idx}]' must be a mapping"
            )
        if set(shape) != {"rows", "features"}:
            raise BenchmarkConfigError(
                f"Field '{field_name}[{idx}]' must define exactly 'rows' and "
                "'features'"
            )
        rows = shape["rows"]
        features = shape["features"]
        if not isinstance(rows, int) or not isinstance(features, int):
            raise BenchmarkConfigError(
                f"Field '{field_name}[{idx}]' must contain integer 'rows' "
                "and 'features'"
            )
        shape_pairs.append({"rows": rows, "features": features})

    return shape_pairs


def _validate_tag_list(value: Any, *, field_name: str) -> None:
    if not isinstance(value, list) or not value:
        raise BenchmarkConfigError(
            f"Field '{field_name}' must be a non-empty list of strings"
        )
    for tag in value:
        if not isinstance(tag, str) or not tag:
            raise BenchmarkConfigError(
                f"Field '{field_name}' must contain only non-empty strings"
            )


def _merge_tags(default_tags: list[Any], entry_tags: list[Any]) -> list[str]:
    if default_tags:
        _validate_tag_list(default_tags, field_name="defaults.tags")
    if entry_tags:
        _validate_tag_list(entry_tags, field_name="benchmark.tags")

    seen = set()
    merged = []
    for tag in default_tags + entry_tags:
        if tag not in seen:
            seen.add(tag)
            merged.append(tag)
    return merged


__all__ = [
    "BenchmarkConfigError",
    "load_and_resolve_config",
    "load_config_file",
    "resolve_config",
    "validate_config",
]
