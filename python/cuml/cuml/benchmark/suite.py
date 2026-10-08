# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and resolve benchmark suite manifests and built-in suites."""

from __future__ import annotations

import copy
import importlib.resources
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

from ._serialization import case_artifact_fields
from .datasets import resolve_dataset, resolve_dtypes
from .identity import result_id
from .providers import PROVIDERS, Provider, get_provider

if TYPE_CHECKING:
    from ._schema import CaseManifest, ProfileManifest


class SuiteError(ValueError):
    """Indicate an invalid or unsupported benchmark suite configuration."""

    pass


TRAINING_OPERATIONS = frozenset({"fit", "fit_predict", "fit_transform"})
OPERATIONS = TRAINING_OPERATIONS | frozenset(
    {
        "predict",
        "transform",
        "kneighbors",
        "score_samples",
    }
)
BUILTIN_SUITES = frozenset({"estimators"})


def _manifest_schema() -> ModuleType:
    """Import manifest validation support or report a missing dependency."""
    try:
        from . import _schema
    except ImportError as exc:
        raise SuiteError(
            "Benchmark suite validation requires msgspec. Install it with "
            "`conda install -c conda-forge msgspec` or "
            "`python -m pip install msgspec`."
        ) from exc
    return _schema


def suite_manifest_json_schema() -> dict[str, Any]:
    """Return the JSON Schema for benchmark suite manifests."""
    return _manifest_schema().suite_manifest_json_schema()


def _mapping(value: Any, where: str) -> dict[str, Any]:
    """Validate that a manifest value is a mapping."""
    if not isinstance(value, dict):
        raise SuiteError(f"{where} must be a mapping")
    return value


def _load_suite_document(suite_path: Path) -> Any:
    """Read a YAML suite document with actionable loading errors."""
    try:
        import yaml
    except ImportError as exc:
        raise SuiteError(
            "YAML benchmark suites require PyYAML. Install it with "
            "`conda install -c conda-forge pyyaml` or "
            "`python -m pip install pyyaml`."
        ) from exc
    try:
        return yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SuiteError(f"unable to load suite {suite_path}: {exc}") from exc


@dataclass(frozen=True, kw_only=True)
class ResolvedCase:
    """Store validated workload settings and execution counts for one case."""

    estimator: str
    dataset: str
    operation: str
    generated_rows: int
    training_rows: int
    measured_rows: int
    features: int
    parameters: dict[str, Any]
    dataset_parameters: dict[str, Any]
    dtypes: dict[str, str]
    input_format: str
    input_selection: tuple[str, ...]
    fit_input_selection: tuple[str, ...] | None
    warmups: int
    repetitions: int
    timeout_sec: float | None

    @property
    def lifecycle(self) -> Literal["fit", "inference"]:
        """Return the operation's fitting or inference lifecycle."""
        return "fit" if self.operation in TRAINING_OPERATIONS else "inference"

    def to_artifact_fields(self) -> dict[str, Any]:
        """Serialize the case's workload fields for a result artifact."""
        return case_artifact_fields(self)

    @property
    def id(self) -> str:
        """Return the content-derived workload identifier."""
        return result_id(self.to_artifact_fields())

    @property
    def label(self) -> str:
        """Return a compact label derived from the workload identifier."""
        return f"case-{self.id.removeprefix('sha256:')[:20]}"


def resolve_case(
    request: dict[str, Any], profile: dict[str, Any]
) -> ResolvedCase:
    """Validate and resolve a case request using profile settings.

    Parameters
    ----------
    request : dict
        Case manifest fields.
    profile : dict
        Execution counts, size scale, and timeout settings.
    """
    schema = _manifest_schema()
    try:
        item = schema.convert_case(_mapping(request, "case request"))
        settings = schema.convert_profile(_mapping(profile, "profile request"))
    except schema.msgspec.ValidationError as exc:
        raise SuiteError(f"invalid case or profile request: {exc}") from exc
    return _resolve_case(item, settings)


def _resolve_case(
    request: CaseManifest, profile: ProfileManifest
) -> ResolvedCase:
    """Resolve typed case and profile settings into a validated workload."""
    schema = _manifest_schema()
    if request.operation not in OPERATIONS:
        raise SuiteError(f"invalid operation {request.operation!r}")
    is_fit = request.operation in TRAINING_OPERATIONS
    dataset = request.dataset
    shape = dataset.shape
    if is_fit and shape.train_rows is not schema.msgspec.UNSET:
        raise SuiteError("train_rows is only valid for inference operations")
    if is_fit and request.fit_input_selection is not None:
        raise SuiteError(
            "fit_input_selection is only valid for inference operations"
        )
    if not is_fit and request.fit_input_selection is None:
        raise SuiteError(
            "fit_input_selection is required for inference operations"
        )
    for name in ("input_selection", "fit_input_selection"):
        value = getattr(request, name)
        if name == "fit_input_selection" and value is None:
            continue
        if value not in [["X"], ["y"], ["X", "y"]]:
            raise SuiteError(f"{name} must be X, y, or X followed by y")

    if not is_fit and shape.train_rows is schema.msgspec.UNSET:
        raise SuiteError("train_rows is required for inference operations")

    rows = max(32, round(shape.rows * profile.size_scale))
    if is_fit:
        training_rows = measured_rows = generated_rows = rows
    else:
        training_rows = max(1, round(shape.train_rows * profile.size_scale))
        measured_rows = rows
        generated_rows = training_rows + measured_rows
    dtype = dataset.dtype
    if isinstance(dtype, schema.DtypeMapping):
        dtype = {"X": dtype.X, "y": dtype.y}
    try:
        parameters = resolve_dataset(
            dataset.kind,
            shape.features,
            dataset.parameters,
            dtype,
            dataset.format,
        )
    except ValueError as exc:
        raise SuiteError(f"dataset: {exc}") from exc
    return ResolvedCase(
        estimator=request.estimator,
        dataset=dataset.kind,
        operation=request.operation,
        generated_rows=generated_rows,
        training_rows=training_rows,
        measured_rows=measured_rows,
        features=shape.features,
        parameters=copy.deepcopy(request.parameters),
        dataset_parameters=copy.deepcopy(parameters),
        dtypes=resolve_dtypes(dtype),
        input_format=dataset.format,
        input_selection=tuple(request.input_selection),
        fit_input_selection=None
        if request.fit_input_selection is None
        else tuple(request.fit_input_selection),
        warmups=profile.warmups,
        repetitions=profile.repetitions,
        timeout_sec=profile.timeout_sec
        if request.timeout_sec is schema.msgspec.UNSET
        else request.timeout_sec,
    )


@dataclass(frozen=True)
class Suite:
    """Store one provider's resolved workloads and execution profile."""

    path: str | Path
    name: str
    provider: str
    profile_name: str
    cases: tuple[ResolvedCase, ...]

    @property
    def provider_spec(self) -> Provider:
        """Resolve this run's Python-defined provider."""
        return get_provider(self.provider)


@dataclass(frozen=True)
class SuitePlan:
    """Store a suite's selected, ordered single-provider runs."""

    path: str | Path
    name: str
    profile_name: str
    runs: tuple[Suite, ...]


def _providers(values: list[str], where: str) -> None:
    """Reject unknown and duplicate provider declarations."""
    if len(set(values)) != len(values):
        raise SuiteError(f"{where}: duplicate providers")
    for value in values:
        if value not in PROVIDERS:
            raise SuiteError(f"{where}: invalid provider {value!r}")


def load_suite(
    path: str | Path,
    profile: str | None = None,
    providers: list[str] | None = None,
) -> SuitePlan:
    """Load and validate a suite manifest for the selected profile.

    Parameters
    ----------
    path : str or Path
        YAML suite manifest path.
    profile : str, optional
        Profile name; defaults to standard.
    providers : list of str, optional
        Subset of declared providers; defaults to all in manifest order.
    """
    suite_path = Path(path).resolve()
    schema = _manifest_schema()
    try:
        manifest = schema.convert_manifest(_load_suite_document(suite_path))
    except schema.msgspec.ValidationError as exc:
        raise SuiteError(f"invalid suite {suite_path}: {exc}") from exc
    _providers(manifest.providers, "suite")
    declared = manifest.providers
    selected = declared if providers is None else providers
    _providers(selected, "selection")
    if not selected or not set(selected) <= set(declared):
        raise SuiteError(
            "selection must be a nonempty subset of suite providers"
        )
    selected = [name for name in declared if name in selected]
    profile_name = profile or "standard"
    if profile_name not in manifest.profiles:
        available = ", ".join(sorted(manifest.profiles))
        raise SuiteError(
            f"unknown profile {profile_name!r}; available profiles: {available}"
        )
    profile_data = manifest.profiles[profile_name]
    resolved = {name: [] for name in selected}
    seen = {name: set() for name in selected}
    for index, item in enumerate(manifest.cases):
        where = f"case {index}"
        applicable = item.providers or declared
        _providers(applicable, where)
        if not set(applicable) <= set(declared):
            raise SuiteError(
                f"{where}: providers must be a subset of the suite"
            )
        try:
            case = _resolve_case(item, profile_data)
        except SuiteError as exc:
            raise SuiteError(f"{where}: {exc}") from exc
        for provider in selected:
            if provider not in applicable:
                continue
            provider_spec = get_provider(provider)
            if case.estimator not in provider_spec.catalog:
                raise SuiteError(
                    f"{where}: estimator {case.estimator!r} is incompatible with {provider!r}"
                )
            if case.id in seen[provider]:
                raise SuiteError(
                    f"{where}: duplicate case identity {case.id!r}"
                )
            seen[provider].add(case.id)
            resolved[provider].append(case)
    runs = []
    for provider, cases in resolved.items():
        if not cases:
            raise SuiteError(f"provider {provider!r} has no applicable cases")
        try:
            get_provider(provider).backend.validate_profile(
                profile_name, profile_data.warmups
            )
        except SuiteError as exc:
            raise SuiteError(
                f"{provider}, profile {profile_name!r}: {exc}"
            ) from exc
        runs.append(
            Suite(
                suite_path,
                manifest.name,
                provider,
                profile_name,
                tuple(cases),
            )
        )
    return SuitePlan(suite_path, manifest.name, profile_name, tuple(runs))


@contextmanager
def _suite_reference_path(
    reference: str | Path,
) -> Iterator[tuple[Path, str | None]]:
    """Resolve a suite reference to a file path and optional built-in name."""
    reference_text = str(reference)
    builtin_name = reference_text
    if builtin_name in BUILTIN_SUITES and Path(reference_text).parent == Path(
        "."
    ):
        resource = importlib.resources.files("cuml.benchmark.suites").joinpath(
            f"{builtin_name}.yaml"
        )
        with importlib.resources.as_file(resource) as resource_path:
            yield resource_path, builtin_name
    else:
        yield Path(reference).resolve(), None


def suite_profile_names(reference: str | Path) -> tuple[str, ...]:
    """Return the sorted profile names defined by a suite.

    Parameters
    ----------
    reference : str or Path
        Built-in suite name or YAML manifest path.
    """
    with _suite_reference_path(reference) as (suite_path, _):
        raw = _load_suite_document(suite_path)
    raw = _mapping(raw, "suite")
    schema = _manifest_schema()
    try:
        profiles = schema.convert_profiles(raw.get("profiles"))
    except schema.msgspec.ValidationError as exc:
        raise SuiteError(f"invalid profiles: {exc}") from exc
    return tuple(sorted(profiles))


def load_suite_reference(
    reference: str | Path,
    profile: str | None = None,
    providers: list[str] | None = None,
) -> SuitePlan:
    """Load a built-in suite or manifest for the selected profile.

    Parameters
    ----------
    reference : str or Path
        Built-in suite name or YAML manifest path.
    profile : str, optional
        Profile name; defaults to standard.
    providers : list of str, optional
        Subset of declared providers; defaults to all in manifest order.
    """
    with _suite_reference_path(reference) as (suite_path, builtin_name):
        suite = load_suite(suite_path, profile, providers)
    if builtin_name is not None:
        path = f"builtin:{builtin_name}"
        return replace(
            suite,
            path=path,
            runs=tuple(replace(run, path=path) for run in suite.runs),
        )
    return suite
