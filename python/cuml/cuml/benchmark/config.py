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
    import msgspec as _msgspec
except ImportError:  # pragma: no cover - optional YAML config dependency
    _msgspec = None

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
    "backends",
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
    "variants",
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
    "backends",
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
    "fit_score_samples",
    "kneighbors",
}
ALLOWED_BACKENDS = {"cpu", "gpu"}
SIZE_FIELDS = {"default_size", "shapes", "rows", "features"}
COMPACT_VARIANT_KEYS = (BENCHMARK_KEYS - {"id", "algorithm", "variants"}) | {
    "id_suffix",
    "tiers",
}
COMPACT_TIER_KEYS = (BENCHMARK_KEYS - {"id", "algorithm", "variants"}) | {
    "id_suffix"
}
_YAML_CONFIG_INSTALL_MESSAGE = (
    "YAML benchmark configs require PyYAML and msgspec. "
    "Install them with `conda install -c conda-forge pyyaml msgspec` "
    "or `python -m pip install pyyaml msgspec`."
)


def _build_benchmark_manifest_model(msgspec):
    Shape = msgspec.defstruct(
        "Shape",
        [("rows", int), ("features", int)],
        forbid_unknown_fields=True,
        module=__name__,
    )
    common_fields = [
        ("id", str | None, None),
        ("algorithm", str | None, None),
        ("dataset", str | None, None),
        ("input_type", str | None, None),
        ("dtype", str | None, None),
        ("n_reps", int | None, None),
        ("random_state", int | None, None),
        ("test_split", int | float | None, None),
        ("run_cpu", bool | None, None),
        ("run_gpu", bool | None, None),
        ("backends", list[str] | None, None),
        ("raise_on_error", bool | None, None),
        ("default_size", bool | None, None),
        ("shapes", list[Shape] | None, None),
        ("rows", list[int] | None, None),
        ("features", list[int] | None, None),
        ("operation", str | None, None),
        ("params", dict[str, Any] | None, None),
        ("cuml_params", dict[str, Any] | None, None),
        ("cpu_params", dict[str, Any] | None, None),
        ("dataset_params", dict[str, Any] | None, None),
        ("param_grid", dict[str, list[Any]] | None, None),
        ("cuml_param_grid", dict[str, list[Any]] | None, None),
        ("cpu_param_grid", dict[str, list[Any]] | None, None),
        ("dataset_param_grid", dict[str, list[Any]] | None, None),
        ("comparison", dict[str, Any] | None, None),
        ("tags", list[str] | None, None),
        ("enabled", bool | None, None),
        ("skip_reason", str | None, None),
        ("metadata", dict[str, Any] | None, None),
    ]
    compact_fields = [
        field for field in common_fields if field[0] not in {"id", "algorithm"}
    ]
    Tier = msgspec.defstruct(
        "Tier",
        compact_fields + [("id_suffix", str | None, None)],
        forbid_unknown_fields=True,
        module=__name__,
    )
    Variant = msgspec.defstruct(
        "Variant",
        compact_fields
        + [
            ("id_suffix", str | None, None),
            ("tiers", dict[str, Tier] | None, None),
        ],
        forbid_unknown_fields=True,
        module=__name__,
    )
    CommonFields = msgspec.defstruct(
        "CommonFields",
        common_fields,
        forbid_unknown_fields=True,
        module=__name__,
    )
    Benchmark = msgspec.defstruct(
        "Benchmark",
        common_fields + [("variants", dict[str, Variant] | None, None)],
        forbid_unknown_fields=True,
        module=__name__,
    )
    Suite = msgspec.defstruct(
        "Suite",
        [("name", str), ("tier", str), ("description", str)],
        forbid_unknown_fields=True,
        module=__name__,
    )
    Profile = msgspec.defstruct(
        "Profile",
        [("include_tags", list[str])],
        forbid_unknown_fields=True,
        module=__name__,
    )
    return msgspec.defstruct(
        "BenchmarkManifest",
        [
            ("version", int),
            ("suite", Suite),
            ("benchmarks", list[Benchmark]),
            ("profiles", dict[str, Profile] | None, None),
            ("defaults", CommonFields | None, None),
        ],
        forbid_unknown_fields=True,
        module=__name__,
    )


BenchmarkManifest = (
    _build_benchmark_manifest_model(_msgspec) if _msgspec is not None else None
)


class BenchmarkConfigError(ValueError):
    """Raised when a benchmark config file is invalid."""


def _load_yaml_module():
    try:
        import yaml
    except ImportError as exc:
        raise BenchmarkConfigError(_YAML_CONFIG_INSTALL_MESSAGE) from exc
    return yaml


def _load_msgspec_module():
    if _msgspec is None:
        raise BenchmarkConfigError(_YAML_CONFIG_INSTALL_MESSAGE)
    return _msgspec


def _validate_config_structure(raw_config: dict[str, Any]) -> None:
    """Validate manifest structure with typed msgspec models."""
    msgspec = _load_msgspec_module()
    manifest_model = BenchmarkManifest or _build_benchmark_manifest_model(
        msgspec
    )

    try:
        msgspec.convert(raw_config, type=manifest_model)
    except msgspec.ValidationError as exc:
        raise BenchmarkConfigError(
            f"Benchmark config does not match the manifest schema: {exc}"
        ) from exc


def benchmark_manifest_json_schema() -> dict[str, Any]:
    """Return a JSON Schema generated from the msgspec manifest model."""
    msgspec = _load_msgspec_module()
    manifest_model = BenchmarkManifest or _build_benchmark_manifest_model(
        msgspec
    )
    return msgspec.json.schema(manifest_model)


def _is_int_value(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_numeric_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_config_file(config_path: str) -> dict[str, Any]:
    """Load a YAML benchmark config file."""
    yaml = _load_yaml_module()
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
    _validate_config_structure(raw)
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
    expanded_benchmarks = _expand_benchmark_entries(
        raw_config.get("benchmarks", [])
    )

    benchmark_entries = []
    for entry in expanded_benchmarks:
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
    if not _is_int_value(version):
        raise BenchmarkConfigError("Config field 'version' must be an integer")
    if version != 1:
        raise BenchmarkConfigError(
            f"Unsupported config version {version}. Supported versions: [1]"
        )

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
        if "variants" in entry:
            _validate_compact_variants(entry, context=f"benchmarks[{idx}]")
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
    if "backends" in entry:
        _normalize_backends(entry["backends"], context=context)

    for numeric_field in ("n_reps", "random_state"):
        if numeric_field in entry and not _is_int_value(entry[numeric_field]):
            raise BenchmarkConfigError(
                f"{context} field '{numeric_field}' must be an integer"
            )
    if "test_split" in entry and not _is_numeric_value(entry["test_split"]):
        raise BenchmarkConfigError(
            f"{context} field 'test_split' must be numeric"
        )
    if "test_split" in entry and not 0.0 <= entry["test_split"] <= 1.0:
        raise BenchmarkConfigError(
            f"{context} field 'test_split' must be between 0.0 and 1.0"
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


def _validate_compact_variants(entry: dict[str, Any], *, context: str) -> None:
    if any(field in entry for field in SIZE_FIELDS):
        raise BenchmarkConfigError(
            f"{context} cannot define 'variants' together with benchmark-level "
            "'rows', 'features', 'shapes', or 'default_size'"
        )
    if "id" not in entry:
        raise BenchmarkConfigError(
            f"{context} must define 'id' when using compact 'variants'"
        )

    variants = entry.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise BenchmarkConfigError(
            f"{context} field 'variants' must be a non-empty mapping"
        )

    for variant_name, variant_def in variants.items():
        if not isinstance(variant_name, str) or not variant_name:
            raise BenchmarkConfigError(
                f"{context}.variants contains an invalid variant name"
            )
        if not isinstance(variant_def, dict):
            raise BenchmarkConfigError(
                f"{context}.variants.{variant_name} must be a mapping"
            )
        unknown_variant = set(variant_def) - COMPACT_VARIANT_KEYS
        if unknown_variant:
            raise BenchmarkConfigError(
                f"Unknown keys in {context}.variants.{variant_name}: "
                f"{sorted(unknown_variant)}"
            )

        variant_id_suffix = variant_def.get("id_suffix")
        if variant_id_suffix is not None and (
            not isinstance(variant_id_suffix, str) or not variant_id_suffix
        ):
            raise BenchmarkConfigError(
                f"{context}.variants.{variant_name}.id_suffix must be a "
                "non-empty string"
            )

        variant_tiers = variant_def.get("tiers")
        if not isinstance(variant_tiers, dict) or not variant_tiers:
            raise BenchmarkConfigError(
                f"{context}.variants.{variant_name}.tiers must be a non-empty mapping"
            )

        variant_overrides = {
            key: value
            for key, value in variant_def.items()
            if key not in {"tiers", "id_suffix"}
        }
        _validate_default_or_entry(
            variant_overrides,
            context=f"{context}.variants.{variant_name}",
            require_algorithm=False,
        )

        for tier_name, tier_def in variant_tiers.items():
            if not isinstance(tier_name, str) or not tier_name:
                raise BenchmarkConfigError(
                    f"{context}.variants.{variant_name}.tiers contains an invalid tier name"
                )
            if not isinstance(tier_def, dict):
                raise BenchmarkConfigError(
                    f"{context}.variants.{variant_name}.tiers.{tier_name} must be a mapping"
                )
            unknown_tier = set(tier_def) - COMPACT_TIER_KEYS
            if unknown_tier:
                raise BenchmarkConfigError(
                    f"Unknown keys in {context}.variants.{variant_name}.tiers.{tier_name}: "
                    f"{sorted(unknown_tier)}"
                )

            tier_id_suffix = tier_def.get("id_suffix")
            if tier_id_suffix is not None and (
                not isinstance(tier_id_suffix, str) or not tier_id_suffix
            ):
                raise BenchmarkConfigError(
                    f"{context}.variants.{variant_name}.tiers.{tier_name}.id_suffix "
                    "must be a non-empty string"
                )

            _validate_default_or_entry(
                {
                    key: value
                    for key, value in tier_def.items()
                    if key != "id_suffix"
                },
                context=f"{context}.variants.{variant_name}.tiers.{tier_name}",
                require_algorithm=False,
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


def _expand_benchmark_entries(
    benchmarks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expanded = []
    seen_ids = set()

    for entry in benchmarks:
        if "variants" not in entry:
            _append_expanded_entry(expanded, seen_ids, deepcopy(entry))
            continue

        base_entry = {
            k: deepcopy(v) for k, v in entry.items() if k != "variants"
        }
        base_id = base_entry["id"]
        for variant_name, variant_def in entry["variants"].items():
            variant_overrides = {
                key: deepcopy(value)
                for key, value in variant_def.items()
                if key not in {"tiers", "id_suffix"}
            }
            variant_overrides["tags"] = _named_tags(
                variant_name, variant_overrides.get("tags")
            )
            variant_resolved = _apply_defaults(base_entry, variant_overrides)
            variant_suffix = variant_def.get("id_suffix", variant_name)

            for tier_name, tier_def in variant_def["tiers"].items():
                tier_overrides = {
                    key: deepcopy(value)
                    for key, value in tier_def.items()
                    if key != "id_suffix"
                }
                tier_overrides["tags"] = _named_tags(
                    tier_name, tier_overrides.get("tags")
                )
                expanded_entry = _apply_defaults(
                    variant_resolved, tier_overrides
                )
                tier_suffix = tier_def.get("id_suffix", tier_name)
                expanded_entry["id"] = (
                    f"{base_id}_{variant_suffix}_{tier_suffix}"
                )
                _append_expanded_entry(expanded, seen_ids, expanded_entry)

    return expanded


def _append_expanded_entry(
    expanded: list[dict[str, Any]],
    seen_ids: set[str],
    entry: dict[str, Any],
) -> None:
    entry_id = entry.get("id")
    if entry_id is not None:
        if entry_id in seen_ids:
            raise BenchmarkConfigError(f"Duplicate benchmark id '{entry_id}'")
        seen_ids.add(entry_id)
    expanded.append(entry)


def _named_tags(name: str, explicit_tags: Any) -> list[str]:
    if explicit_tags is None:
        return [name]
    if not isinstance(explicit_tags, list):
        return [name, explicit_tags]
    return _merge_tags([name], explicit_tags)


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

    if not _is_int_value(entry.get("n_reps")):
        raise BenchmarkConfigError(
            f"Benchmark '{benchmark_name}' must define integer 'n_reps' "
            "after applying defaults"
        )

    if not _is_numeric_value(entry.get("test_split")):
        raise BenchmarkConfigError(
            f"Benchmark '{benchmark_name}' must define numeric 'test_split' "
            "after applying defaults"
        )
    if not 0.0 <= entry["test_split"] <= 1.0:
        raise BenchmarkConfigError(
            f"Benchmark '{benchmark_name}' must define 'test_split' between "
            "0.0 and 1.0 after applying defaults"
        )

    backends = _resolved_entry_backends(entry)
    if not backends:
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

    backends = _resolved_entry_backends(entry)

    return {
        "benchmark_id": entry.get("id", entry["algorithm"]),
        "algorithm": entry["algorithm"],
        "dataset": entry.get("dataset"),
        "input_type": entry.get("input_type"),
        "dtype": entry.get("dtype"),
        "n_reps": entry.get("n_reps"),
        "random_state": entry.get("random_state"),
        "test_split": entry.get("test_split"),
        "backends": backends,
        "run_cpu": "cpu" in backends,
        "run_gpu": "gpu" in backends,
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


def _resolved_entry_backends(entry: dict[str, Any]) -> list[str]:
    if "backends" in entry:
        return _normalize_backends(entry["backends"], context="benchmark")

    backends = []
    if entry.get("run_cpu", True):
        backends.append("cpu")
    if entry.get("run_gpu", True):
        backends.append("gpu")
    return backends


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
    filtered = [
        entry
        for entry in benchmark_entries
        if include_tags.intersection(entry.get("tags", []))
    ]
    if not filtered:
        raise BenchmarkConfigError(
            f"Profile '{selected_profile}' did not match any benchmark entries"
        )
    return filtered


def _apply_algorithm_filter(
    benchmark_entries: list[dict[str, Any]],
    algorithm_filter: list[str] | tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    if not algorithm_filter:
        return benchmark_entries

    wanted = {name.lower() for name in algorithm_filter}
    filtered = [
        e for e in benchmark_entries if e["algorithm"].lower() in wanted
    ]
    if not filtered:
        raise BenchmarkConfigError(
            "Algorithm filter did not match any resolved benchmark entries: "
            f"{sorted(algorithm_filter)}"
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

    if _is_int_value(value):
        return [value]
    if not isinstance(value, list) or not value:
        raise BenchmarkConfigError(
            f"Field '{field_name}' must be an integer or a non-empty list "
            "of integers"
        )
    if not all(_is_int_value(v) for v in value):
        raise BenchmarkConfigError(
            f"Field '{field_name}' must contain only integers"
        )
    return value


def _normalize_backends(value: Any, *, context: str) -> list[str]:
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        values = value
    else:
        raise BenchmarkConfigError(
            f"{context} field 'backends' must be a non-empty list of backend names"
        )

    if not values:
        raise BenchmarkConfigError(
            f"{context} field 'backends' must be a non-empty list of backend names"
        )

    normalized = []
    seen = set()
    for backend in values:
        if not isinstance(backend, str) or not backend:
            raise BenchmarkConfigError(
                f"{context} field 'backends' must contain only non-empty strings"
            )
        backend = backend.lower()
        if backend not in ALLOWED_BACKENDS:
            raise BenchmarkConfigError(
                f"{context} field 'backends' contains unsupported backend "
                f"'{backend}'. Supported backends: {sorted(ALLOWED_BACKENDS)}"
            )
        if backend not in seen:
            seen.add(backend)
            normalized.append(backend)
    return normalized


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
        if not _is_int_value(rows) or not _is_int_value(features):
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
