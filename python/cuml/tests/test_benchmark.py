# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Critical contracts for benchmark workloads, measurements, and recovery."""

from __future__ import annotations

import ast
import contextlib
import copy
import datetime as dt
import importlib
import inspect
import json
import multiprocessing
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import jsonschema
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import sparse

from cuml import benchmark
from cuml.benchmark import _runner as coordinator
from cuml.benchmark import _subprocess as runner
from cuml.benchmark import cli, harness
from cuml.benchmark.backends import cuml as cuml_backend
from cuml.benchmark.backends import get_backend
from cuml.benchmark.backends.accel import ACCEL_EXTENSION
from cuml.benchmark.backends.base import Backend
from cuml.benchmark.datasets import generate_data
from cuml.benchmark.identity import canonical_json, result_id
from cuml.benchmark.providers import PROVIDERS, Provider, get_provider
from cuml.benchmark.providers.base import EstimatorSpec
from cuml.benchmark.schemas import benchmark_result_schema
from cuml.benchmark.suite import (
    BUILTIN_SUITES,
    Suite,
    SuiteError,
    load_suite,
    load_suite_reference,
    resolve_case,
    suite_manifest_json_schema,
)

SUITES = Path(__file__).resolve().parents[1] / "cuml" / "benchmark" / "suites"
PROFILE = {"warmups": 1, "repetitions": 2, "size_scale": 1}
PACKAGE = {
    "name": "scikit-learn",
    "version": "test",
    "build": None,
    "source": None,
}
ACCEL_BENCHMARK_EXCLUSIONS = {
    "IsolationForest": "No benchmark registry or suite entry is defined yet.",
}
BENCHMARK_EXCLUSIONS = {
    "CD": "Low-level solver API is outside the estimator benchmark suite.",
    "IsolationForest": "No benchmark registry entry is defined yet.",
    "Lars": "No benchmark registry entry is defined yet.",
    "MBSGDClassifier": "No benchmark registry or suite entry is defined yet.",
    "MBSGDRegressor": "No benchmark registry or suite entry is defined yet.",
    "QN": "Low-level solver API is outside the estimator benchmark suite.",
    "SGD": "Low-level solver API is outside the estimator benchmark suite.",
}


def _request(**changes):
    return {
        "estimator": "PCA",
        "operation": "fit",
        "input_selection": ["X"],
        "parameters": {"n_components": 2},
        "dataset": {"kind": "matrix", "shape": {"rows": 64, "features": 8}},
        **changes,
    }


def _suite(*cases, provider="scikit-learn"):
    return Suite("test", "test", provider, "standard", tuple(cases))


def _benchmark(case, provider="scikit-learn", **kwargs):
    return harness._benchmark_case(
        _suite(case, provider=provider),
        case,
        implementation_record=PACKAGE,
        **kwargs,
    )


def _write_suite(tmp_path, document):
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _validate_artifact(artifact):
    schema = json.loads(benchmark_result_schema().read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(artifact)
    for result in artifact["results"]:
        assert result["id"] == result_id(result)
        assert (
            result["implementation"] in artifact["run"]["software"]["packages"]
        )


@pytest.fixture
def document():
    return {
        "version": 2,
        "name": "test",
        "providers": ["scikit-learn"],
        "profiles": {"standard": dict(PROFILE)},
        "cases": [_request()],
    }


def test_provider_qualified_hdbscan_workers(document, tmp_path):
    """Two CPU providers share a workload, not an estimator binding."""
    document["providers"] = ["scikit-learn", "hdbscan"]
    document["cases"] = [
        _request(estimator="HDBSCAN", parameters={"min_cluster_size": 5})
    ]
    path = _write_suite(tmp_path, document)
    jsonschema.validate(document, suite_manifest_json_schema())
    plan = load_suite(path)
    assert [run.provider for run in plan.runs] == document["providers"]
    assert plan.runs[0].cases[0].id == plan.runs[1].cases[0].id
    assert (
        plan.runs[0].provider_spec.backend
        is plan.runs[1].provider_spec.backend
    )
    assert [
        run.provider_spec.estimator_spec("HDBSCAN").module for run in plan.runs
    ] == ["sklearn.cluster", "hdbscan"]
    selected = load_suite(path, providers=["hdbscan", "scikit-learn"])
    assert selected == plan
    assert (
        cli.main(
            [
                "--suite",
                str(path),
                "--provider",
                "scikit-learn",
                "--provider",
                "hdbscan",
                "--output",
                str(tmp_path / "results"),
            ]
        )
        == 0
    )
    for provider in document["providers"]:
        artifact = json.loads(
            (tmp_path / "results" / f"{provider}.json").read_text()
        )
        _validate_artifact(artifact)
        result = artifact["results"][0]
        assert result["outcome"]["status"] == "success"
        assert result["implementation"]["name"] == provider
        extension = result["extensions"][harness.EXTENSION]
        assert extension["provider"] == provider
        run_extension = artifact["run"]["extensions"][harness.EXTENSION]
        assert run_extension["provider"] == provider


@pytest.mark.parametrize(
    "providers",
    [
        {"candidate": {"provider": "hdbscan", "execution": "cpu"}},
        [],
        ["cpu"],
        ["reference"],
    ],
)
def test_invalid_provider_declarations_schema(document, tmp_path, providers):
    document["providers"] = providers
    with pytest.raises(SuiteError):
        load_suite(_write_suite(tmp_path, document))


def test_mixed_library_suite_provider_applicability(document, tmp_path):
    document["providers"] = ["scikit-learn", "umap-learn", "hdbscan", "cuml"]
    document["cases"] = [
        _request(
            estimator="UMAP", parameters={}, providers=["umap-learn", "cuml"]
        ),
        _request(
            estimator="HDBSCAN",
            parameters={},
            providers=["scikit-learn", "hdbscan", "cuml"],
        ),
    ]
    path = _write_suite(tmp_path, document)
    plan = load_suite(path)
    assert [[case.estimator for case in run.cases] for run in plan.runs] == [
        ["HDBSCAN"],
        ["UMAP"],
        ["HDBSCAN"],
        ["UMAP", "HDBSCAN"],
    ]
    selected = load_suite(path, providers=["hdbscan", "umap-learn"])
    assert selected.runs == (plan.runs[1], plan.runs[2])
    # Applicability is explicit, not an implicit catalog intersection.
    document["cases"][0].pop("providers")
    with pytest.raises(SuiteError, match="incompatible with 'scikit-learn'"):
        load_suite(_write_suite(tmp_path, document))


@pytest.mark.parametrize(
    "name,module",
    [
        ("scikit-learn", "sklearn"),
        ("umap-learn", "umap"),
        ("hdbscan", "hdbscan"),
        ("cuml", "cuml"),
        ("cuml.accel", "accel"),
        ("cuml.dask", "dask"),
    ],
)
def test_provider_module_owns_catalog(name, module):
    definition = importlib.import_module(f"cuml.benchmark.providers.{module}")
    provider = get_provider(name)
    assert provider is definition.PROVIDER
    assert provider.catalog is definition.CATALOG
    assert all(
        isinstance(spec, EstimatorSpec) for spec in provider.catalog.values()
    )


def test_provider_catalogs_are_distinct():
    sklearn = get_provider("scikit-learn")
    standalone = get_provider("hdbscan")
    assert sklearn.backend is standalone.backend
    assert sklearn.estimator_spec("HDBSCAN").module == "sklearn.cluster"
    assert standalone.estimator_spec("HDBSCAN").module == "hdbscan"
    assert "UMAP" not in sklearn.catalog
    assert (
        get_provider("umap-learn").estimator_spec("UMAP").package
        == "umap-learn"
    )
    assert (
        get_provider("cuml.accel").estimator_spec("HDBSCAN").module
        == "hdbscan"
    )


# Suite coverage and workload definitions. No estimator execution here.


def test_estimator_benchmark_coverage():
    import cuml
    from cuml.internals.base import Base
    from cuml.testing.utils import ClassEnumerator

    public = {
        name
        for name, cls in ClassEnumerator(module=cuml).get_models().items()
        if not name.startswith("_")
        and cls is not Base
        and not inspect.isabstract(cls)
    }
    assert BENCHMARK_EXCLUSIONS.keys() <= public
    assert all(reason.strip() for reason in BENCHMARK_EXCLUSIONS.values())
    manifest = {
        c.estimator
        for c in load_suite_reference("estimators", providers=["cuml"])
        .runs[0]
        .cases
    }
    covered = {
        spec.name
        for key, spec in get_provider("cuml").catalog.items()
        if key in manifest
    }
    missing = public - BENCHMARK_EXCLUSIONS.keys() - covered
    assert not missing, f"Missing benchmark declarations: {sorted(missing)}"

    overrides = SUITES.parents[1] / "accel" / "_overrides"
    exported = set()
    for path in [
        *sorted((overrides / "sklearn").glob("*.py")),
        overrides / "umap.py",
        overrides / "hdbscan.py",
    ]:
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                )
                if any(
                    isinstance(t, ast.Name) and t.id == "__all__"
                    for t in targets
                ):
                    exported.update(ast.literal_eval(node.value))
    assert ACCEL_BENCHMARK_EXCLUSIONS.keys() <= exported
    assert all(
        reason.strip() for reason in ACCEL_BENCHMARK_EXCLUSIONS.values()
    )
    expected = exported - ACCEL_BENCHMARK_EXCLUSIONS.keys()
    assert set(get_provider("cuml.accel").catalog) == expected
    assert {
        c.estimator
        for c in load_suite_reference("estimators", providers=["cuml.accel"])
        .runs[0]
        .cases
    } == expected


def test_packaged_suites_are_valid_and_comparable():
    schema = suite_manifest_json_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    comparable = {}
    for path in sorted(SUITES.glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        validator.validate(document)
        for profile in document["profiles"]:
            plan = load_suite(path, profile)
            assert [run.provider for run in plan.runs] == document["providers"]
            for suite in plan.runs:
                assert suite.cases
                assert len({c.id for c in suite.cases}) == len(suite.cases)
                if path.stem in BUILTIN_SUITES:
                    builtin = load_suite_reference(
                        path.stem, profile, [suite.provider]
                    )
                    assert builtin.runs[0].cases == suite.cases
                    assert builtin.runs[0].path == f"builtin:{path.stem}"
                    assert {c.estimator for c in suite.cases} == set(
                        suite.provider_spec.catalog
                    )
                for case in suite.cases:
                    key = (
                        path.stem,
                        profile,
                        case.estimator,
                        case.operation,
                        case.input_format,
                        case.dtypes["X"],
                    )
                    assert comparable.setdefault(key, case.id) == case.id
                    assert case.id == result_id(case.to_artifact_fields())
    assert {p.stem for p in SUITES.glob("*.yaml")} >= BUILTIN_SUITES


@pytest.mark.parametrize(
    "timeout,expected", [("omitted", 10), (None, None), (3, 3)]
)
@pytest.mark.parametrize(
    "scale,partitions", [(0.01, (33, 1, 32)), (0.5, (76, 26, 50))]
)
def test_python_and_yaml_resolution_agree(
    document, tmp_path, timeout, expected, scale, partitions
):
    request = document["cases"][0]
    request.update(operation="transform", fit_input_selection=["X"])
    request["dataset"].update(dtype={"X": "float64", "y": "int64"})
    request["dataset"]["shape"].update(rows=101, train_rows=51)
    profile = document["profiles"]["standard"]
    profile.update(size_scale=scale, timeout_sec=10)
    if timeout != "omitted":
        request["timeout_sec"] = timeout
    before = copy.deepcopy(document)
    case = resolve_case(request, profile)
    assert load_suite(_write_suite(tmp_path, document)).runs[0].cases == (
        case,
    )
    assert (
        case.generated_rows,
        case.training_rows,
        case.measured_rows,
    ) == partitions
    assert case.features == 8
    assert case.dtypes == {"X": "float64", "y": "int64"}
    assert case.timeout_sec == expected
    assert document == before
    request["parameters"]["n_components"] = 3
    request["dataset"]["dtype"]["y"] = "float32"
    assert case.parameters == {"n_components": 2}
    assert case.dtypes["y"] == "int64"
    assert resolve_case(request, profile).id != case.id


@pytest.mark.parametrize(
    "change,match",
    [
        ("missing", "input_selection"),
        ("unknown", "unknown field"),
        ("boolean", "rows"),
        ("zero", "repetitions"),
        ("nonfinite", "timeout_sec"),
        ("estimator", "incompatible"),
        ("generator", "must not exceed features"),
        ("fit_inputs", "fit_input_selection is required"),
        ("train_rows", "train_rows is required"),
        ("duplicate", "duplicate case identity"),
        ("accel_warmup", "at least one warmup"),
    ],
)
def test_invalid_suite_requests_are_rejected(
    document, tmp_path, change, match
):
    case = document["cases"][0]
    profile = document["profiles"]["standard"]
    if change == "missing":
        del case["input_selection"]
    elif change == "unknown":
        case["dataset"]["shape"]["typo"] = 1
    elif change == "boolean":
        case["dataset"]["shape"]["rows"] = True
    elif change == "zero":
        profile["repetitions"] = 0
    elif change == "nonfinite":
        profile["timeout_sec"] = float("inf")
    elif change == "estimator":
        case["estimator"] = "Unknown"
    elif change == "generator":
        case["dataset"].update(
            kind="classification", parameters={"n_informative": 9}
        )
    elif change in {"fit_inputs", "train_rows"}:
        case["operation"] = "transform"
        if change == "train_rows":
            case["fit_input_selection"] = ["X"]
    elif change == "duplicate":
        duplicate = copy.deepcopy(case)
        duplicate["dataset"].update(
            dtype="float32", format="dense", parameters={}
        )
        document["cases"].append(duplicate)
    else:
        document["providers"] = ["cuml.accel"]
        profile["warmups"] = 0
    with pytest.raises(SuiteError, match=match):
        load_suite(_write_suite(tmp_path, document))
    if change not in {"estimator", "duplicate", "accel_warmup"}:
        with pytest.raises(SuiteError, match=match):
            resolve_case(case, profile)


def test_suite_provider_selection(document, tmp_path):
    document["providers"] = ["cuml", "scikit-learn", "cuml.accel"]
    restricted = _request(estimator="AgglomerativeClustering", parameters={})
    restricted["providers"] = ["cuml", "scikit-learn"]
    document["cases"].append(restricted)
    path = _write_suite(tmp_path, document)
    plan = load_suite(path)
    assert [len(run.cases) for run in plan.runs] == [2, 2, 1]
    assert len({run.cases[0].id for run in plan.runs}) == 1
    selected = load_suite(path, providers=["cuml.accel", "cuml"])
    assert selected.runs == (plan.runs[0], plan.runs[2])
    document["profiles"]["standard"]["warmups"] = 0
    path = _write_suite(tmp_path, document)
    assert load_suite(path, providers=["scikit-learn"])
    with pytest.raises(SuiteError, match="at least one warmup"):
        load_suite(path)


@pytest.mark.parametrize(
    "change,match",
    [
        ("unknown", "invalid provider"),
        ("duplicate", "duplicate providers"),
        ("empty", "length >= 1"),
        ("scalar", "unknown field|missing required field"),
        ("case_unknown", "invalid provider"),
        ("case_duplicate", "duplicate providers"),
        ("case_empty", "length >= 1"),
        ("case_outside", "subset"),
        ("no_cases", "no applicable cases"),
    ],
)
def test_invalid_provider_declarations(document, tmp_path, change, match):
    if change == "scalar":
        document["provider"] = document.pop("providers")[0]
    elif change == "no_cases":
        document["providers"].append("cuml")
        document["cases"][0]["providers"] = ["scikit-learn"]
    else:
        values = {
            "unknown": ["unknown"],
            "duplicate": ["scikit-learn"] * 2,
            "empty": [],
            "outside": ["cuml"],
        }[change.removeprefix("case_")]
        target = (
            document["cases"][0] if change.startswith("case_") else document
        )
        target["providers"] = values
    with pytest.raises(SuiteError, match=match):
        load_suite(_write_suite(tmp_path, document))


@pytest.mark.parametrize(
    "selection", [[], ["cuml"], ["unknown"], ["scikit-learn"] * 2]
)
def test_invalid_provider_selection(document, tmp_path, selection):
    with pytest.raises(SuiteError):
        load_suite(_write_suite(tmp_path, document), providers=selection)


def test_duplicate_workloads_allowed_only_on_disjoint_providers(
    document, tmp_path
):
    document["providers"] = ["cuml", "scikit-learn"]
    document["cases"][0]["providers"] = ["cuml"]
    duplicate = copy.deepcopy(document["cases"][0])
    duplicate["providers"] = ["scikit-learn"]
    document["cases"].append(duplicate)
    assert len(load_suite(_write_suite(tmp_path, document)).runs) == 2
    duplicate["providers"].append("cuml")
    with pytest.raises(SuiteError, match="duplicate case identity"):
        load_suite(_write_suite(tmp_path, document))


def test_workload_identity_contract():
    # This vector is shared with the dashboard, not a snapshot of suite contents.
    golden = {
        "case_label": "display only",
        "algorithm": "kmeans/Δ",
        "dataset": {
            "name": "blobs-雪",
            "kind": "generated",
            "parameters": {
                "clusters": 8,
                "nested": {"scale": 1e-7, "zero": -0.0},
            },
            "generator": "org.example.gen-v1",
            "fingerprint": None,
            "random_seed": 42,
            "legacy_identity": None,
        },
        "operation": {"name": "fit_predict", "lifecycle": "fit"},
        "input": {
            "dimensions": [
                {"name": "rows", "size": 1000},
                {"name": "features", "size": 32},
            ],
            "data_type": "float32",
            "selection": ["X"],
            "attributes": {"ignored": True},
        },
        "parameters": {
            "declared": {
                "tolerance": 1e-6,
                "labels": ["é", "𝄞"],
                "max_iterations": 100,
            },
            "effective": {"ignored": 1},
        },
    }
    assert canonical_json(
        {"numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 1e-27]}
    ) == ('{"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27]}')
    assert (
        result_id(golden)
        == "sha256:8cb1f9fa544012b4b8807e81726987ba331ad2622b2f490817b5f50b628adfcd"
    )
    request = _request(operation="transform", fit_input_selection=["X"])
    request["dataset"]["shape"]["train_rows"] = 128
    case = resolve_case(request, PROFILE)
    explicit = copy.deepcopy(request)
    explicit["dataset"].update(dtype="float32", format="dense", parameters={})
    assert resolve_case(explicit, PROFILE).id == case.id
    for changed in (
        replace(case, estimator="Other"),
        replace(case, parameters={"n_components": 3}),
        replace(case, measured_rows=65),
        replace(case, training_rows=129),
        replace(case, dtypes={"X": "float64", "y": "float32"}),
        replace(case, input_format="csr", dataset_parameters={"density": 0.1}),
        replace(case, input_selection=("y",)),
        replace(case, fit_input_selection=("X", "y")),
    ):
        assert changed.id != case.id
    result = case.to_artifact_fields()
    result.update(
        case_label="display",
        provider={"name": "other"},
        observations=[],
        extensions={},
    )
    result["parameters"]["effective"] = {"n_components": 99}
    assert result_id(result) == case.id
    assert (
        replace(case, warmups=2, repetitions=5, timeout_sec=10).id == case.id
    )


@pytest.mark.parametrize(
    "kind,dtype,input_format,parameters",
    [
        ("matrix", "float32", "dense", {}),
        ("classification", "float64", "dense", {"n_classes": 3}),
        ("regression", "float32", "dense", {}),
        ("blobs", "float64", "dense", {"centers": 3, "cluster_std": 0.5}),
        ("positive", "float32", "dense", {}),
        ("categorical", {"X": "int32", "y": "int64"}, "dense", {}),
        ("classification", "float64", "csr", {"density": 0.25}),
    ],
)
def test_generated_inputs_match_workload(
    kind, dtype, input_format, parameters
):
    request = _request(
        dataset={
            "kind": kind,
            "shape": {"rows": 128, "features": 8},
            "dtype": dtype,
            "format": input_format,
            "parameters": parameters,
        }
    )
    case = resolve_case(request, PROFILE)
    X, y = generate_data(case)
    repeat_X, repeat_y = generate_data(case)
    assert X.shape == (128, 8) and y.shape == (128,)
    assert y.dtype == case.dtypes["y"]
    np.testing.assert_array_equal(y, repeat_y)
    if input_format == "csr":
        assert sparse.isspmatrix_csr(X)
        for name in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(
                getattr(X, name), getattr(repeat_X, name)
            )
        dense_X, dense_y = generate_data(
            replace(
                case,
                input_format="dense",
                dataset_parameters={
                    k: v
                    for k, v in case.dataset_parameters.items()
                    if k != "density"
                },
            )
        )
        np.testing.assert_array_equal(y, dense_y)
        assert 0.15 < X.nnz / np.prod(X.shape) < 0.35
        values, repeat = X.toarray(), repeat_X.toarray()
        np.testing.assert_array_equal(
            values[values != 0], dense_X[values != 0]
        )
        assert sparse.isspmatrix_csr(X[:64])
    else:
        values, repeat = np.asarray(X), np.asarray(repeat_X)
    assert values.dtype == case.dtypes["X"]
    np.testing.assert_array_equal(values, repeat)
    if kind in {"classification", "blobs"}:
        assert len(np.unique(y)) == parameters.get(
            "n_classes", parameters.get("centers", 2)
        )
    elif kind == "positive":
        assert np.all(values >= 0)
    elif kind == "categorical":
        assert np.all((values >= 0) & (values < 8))
        assert set(y) <= {0, 1}


@pytest.mark.parametrize(
    "kind,estimator,other_estimator",
    [
        ("matrix", "PCA", "MultinomialNB"),
        ("categorical", "OneHotEncoder", "LabelEncoder"),
    ],
)
def test_generated_dataset_is_independent_of_estimator(
    kind, estimator, other_estimator
):
    case = resolve_case(
        _request(
            estimator=estimator,
            parameters={},
            dataset={
                "kind": kind,
                "shape": {"rows": 64, "features": 8},
            },
        ),
        PROFILE,
    )
    X, y = generate_data(case)
    other_X, other_y = generate_data(replace(case, estimator=other_estimator))
    expected_type = pd.DataFrame if kind == "categorical" else np.ndarray
    assert type(X) is type(other_X) is expected_type
    assert type(y) is type(other_y) is np.ndarray
    np.testing.assert_array_equal(other_y, y, strict=True)
    if kind == "categorical":
        pd.testing.assert_frame_equal(other_X, X)
    else:
        np.testing.assert_array_equal(other_X, X, strict=True)


# Measurement contracts use fake estimators, not estimator accuracy tests.


@pytest.mark.parametrize(
    "operation,selection,warmups",
    [
        ("fit", ("X", "y"), 1),
        ("fit_transform", ("X",), 1),
        ("predict", ("X",), 1),
        ("predict", ("X",), 0),
        ("transform", ("y",), 1),
    ],
)
def test_estimator_lifecycle_and_inputs(
    monkeypatch, operation, selection, warmups
):
    inference = not operation.startswith("fit")
    request = _request(operation=operation, input_selection=list(selection))
    if inference:
        request["fit_input_selection"] = (
            ["y"] if selection == ("y",) else ["X", "y"]
        )
        request["dataset"]["shape"]["train_rows"] = 32
    case = resolve_case(request, {**PROFILE, "warmups": warmups})
    X = np.arange(case.generated_rows * 8).reshape(-1, 8)
    y = np.arange(case.generated_rows)
    monkeypatch.setattr(harness, "_generate_data", lambda *args: (X, y))
    instances, calls = [], []

    class Estimator:
        def __init__(self, **kwargs):
            instances.append(self)

        def fit(self, *args):
            calls.append((self, "fit", args))
            return self

        def fit_transform(self, *args):
            calls.append((self, "fit_transform", args))
            return args[0]

        def predict(self, *args):
            calls.append((self, operation, args))
            return args[0]

        transform = predict

    monkeypatch.setattr(
        get_backend("cpu"), "load_estimator", lambda spec: Estimator
    )
    result = _benchmark(case)
    assert result["outcome"] == {"status": "success"}
    assert len(instances) == (1 if inference else warmups + 2)
    assert [o["role"] for o in result["observations"]] == [
        "warmup"
    ] * warmups + ["measurement"] * 2
    inputs = {"X": X, "y": y}
    if inference:
        _, method, args = calls.pop(0)
        assert method == "fit"
        assert len(args) == len(case.fit_input_selection)
        for actual, name in zip(args, case.fit_input_selection, strict=True):
            np.testing.assert_array_equal(actual, inputs[name][:32])
    assert len(calls) == warmups + 2
    assert len({id(estimator) for estimator, _, _ in calls}) == len(instances)
    for estimator, method, args in calls:
        assert method == operation
        assert len(args) == len(selection)
        for actual, name in zip(args, selection, strict=True):
            np.testing.assert_array_equal(
                actual, inputs[name][32:] if inference else inputs[name]
            )


@pytest.mark.parametrize("operation", ["fit", "predict"])
def test_only_synchronized_operation_is_timed(monkeypatch, operation):
    events = []

    class Output:
        def __array__(self, *args, **kwargs):
            pytest.fail("must not convert outputs")

        def get(self):
            pytest.fail("must not copy outputs to host")

        def compute(self):
            pytest.fail("CPU backend must not compute outputs")

    output = Output()

    class Estimator:
        def __init__(self, **kwargs):
            events.append("construct")

        def fit(self, *args):
            events.append("fit")
            return output

        def predict(self, *args):
            events.append("predict")
            return output

        def score(self, *args):
            pytest.fail("must not score during benchmarking")

    def generate(case, provider):
        events.append("generate")
        return np.ones((case.generated_rows, case.features)), np.zeros(
            case.generated_rows
        )

    backend = get_backend("cpu")
    monkeypatch.setattr(backend, "load_estimator", lambda spec: Estimator)
    monkeypatch.setattr(harness, "_generate_data", generate)
    monkeypatch.setattr(
        backend,
        "synchronize",
        lambda value=None: events.append(
            "sync-output" if value is output else "sync"
        ),
    )
    ticks = iter([10, 10.25, 20, 20.25, 30, 30.25])
    monkeypatch.setattr(
        harness.time,
        "perf_counter",
        lambda: events.append("clock") or next(ticks),
    )
    request = _request(operation=operation)
    if operation == "predict":
        request.update(fit_input_selection=["X", "y"])
        request["dataset"]["shape"]["train_rows"] = 32
    result = _benchmark(resolve_case(request, PROFILE))
    assert result["outcome"] == {"status": "success"}
    observation = ["sync", "clock", operation, "sync-output", "clock"]
    assert events == (
        ["generate", "construct", "fit", "sync"] + observation * 3
        if operation == "predict"
        else ["generate"] + (["construct"] + observation) * 3
    )
    assert all(
        o["timings"] == [{"name": "wall_time", "value": 0.25, "unit": "s"}]
        and o["metrics"] == []
        for o in result["observations"]
    )


@pytest.mark.parametrize("fallback_at", [None, 0, 1])
def test_accel_warmup_dispatch_verification(monkeypatch, fallback_at):
    backend = get_backend("cuml.accel")
    profiled, executed = [], []
    active = False

    @contextlib.contextmanager
    def profile(quiet=False):
        nonlocal active
        assert quiet
        active = True
        repetition = len(profiled)
        profiled.append(repetition)
        try:
            yield SimpleNamespace(
                method_calls={
                    "PCA.fit": SimpleNamespace(
                        gpu_calls=1,
                        gpu_time=0.01,
                        cpu_calls=int(repetition == fallback_at),
                        cpu_time=0,
                        fallback_reasons={"unsupported"}
                        if repetition == fallback_at
                        else set(),
                    )
                }
            )
        finally:
            active = False

    class Estimator:
        def __init__(self, **kwargs):
            pass

        def fit(self, X):
            executed.append(active)
            return self

    original = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(profile=profile)
        if name == "cuml.accel"
        else original(name),
    )
    monkeypatch.setattr(backend, "load_estimator", lambda spec: Estimator)
    monkeypatch.setattr(backend, "synchronize", lambda value=None: None)
    case = resolve_case(_request(), {**PROFILE, "warmups": 2})
    result = _benchmark(case, "cuml.accel")
    if fallback_at is None:
        assert result["outcome"] == {"status": "success"}
        assert executed == [True, True, False, False]
        assert [
            ACCEL_EXTENSION in o["extensions"] for o in result["observations"]
        ] == executed
    else:
        assert result["outcome"]["error"]["type"] == "CpuFallback"
        assert executed == [True] * (fallback_at + 1)
        assert all(o["role"] == "warmup" for o in result["observations"])
        evidence = result["observations"][-1]["extensions"][ACCEL_EXTENSION]
        assert evidence["cpu_calls"] == 1
        assert evidence["repetition"] == fallback_at
        assert evidence["calls"][0]["fallback_reasons"] == ["unsupported"]
    assert profiled == list(
        range(2 if fallback_at is None else fallback_at + 1)
    )


# One registered backend exercises the worker, artifacts, and resume end to end.


@pytest.fixture
def registered_suite(monkeypatch, tmp_path):
    events = []
    state = {"failure": None, "calls": 0}
    module = ModuleType("benchmark_test_estimator")

    class Estimator:
        def __init__(self, fail=False):
            self.fail = fail

        def get_params(self, deep=False):
            return {"fail": self.fail}

        def fit(self, X):
            events.append(("fit", len(X)))
            if self.fail:
                state["calls"] += 1
                if state["calls"] == 2 and state["failure"] is not None:
                    raise state["failure"]
            return self

    module.Custom = Estimator
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class TestBackend(Backend):
        def bootstrap_process(self):
            events.append("bootstrap")

        def prepare_process(self):
            events.append("prepare")

        @contextlib.contextmanager
        def runtime(self, suite):
            assert events.index("bootstrap") < events.index("prepare")
            events.append("runtime-enter")
            try:
                yield "client"
            finally:
                events.append("runtime-exit")

        def construct_estimator(self, cls, parameters, runtime=None):
            assert runtime == "client"
            return super().construct_estimator(cls, parameters, runtime)

    monkeypatch.setitem(
        PROVIDERS,
        "test",
        Provider(
            TestBackend(),
            {"Custom": EstimatorSpec(module.__name__, "Custom", "custom")},
        ),
    )
    manifest = _write_suite(
        tmp_path,
        {
            "version": 2,
            "name": "test",
            "providers": ["test"],
            "profiles": {"standard": {**PROFILE, "repetitions": 1}},
            "cases": [
                _request(
                    estimator="Custom",
                    parameters={"fail": fail},
                    dataset={
                        "kind": "matrix",
                        "shape": {"rows": rows, "features": 2},
                    },
                )
                for rows, fail in [(32, True), (64, False)]
            ],
        },
    )
    # Keep provenance stable across checkpoint writes and resume attempts.
    monkeypatch.setattr(harness, "_gpu_components", lambda: [])
    return manifest, tmp_path / "artifact.json", events, state


def _run_cli(manifest, output, *, resume=False):
    return cli.main(
        [
            "--_worker",
            "--suite",
            str(manifest),
            "--output",
            str(output),
            "--provider",
            "test",
        ]
        + (["--resume"] if resume else [])
    )


def test_cli_artifact_checkpoint_and_resume(registered_suite, monkeypatch):
    manifest, output, events, state = registered_suite
    checkpoints = []
    write = harness.atomic_write

    def checkpoint(path, artifact, **kwargs):
        write(path, artifact, **kwargs)
        persisted = json.loads(Path(path).read_text())
        _validate_artifact(persisted)
        checkpoints.append(persisted)
        events.append(("checkpoint", len(persisted["results"])))

    monkeypatch.setattr(harness, "atomic_write", checkpoint)
    state["failure"] = RuntimeError("second observation failed")
    assert _run_cli(manifest, output) == 1
    original = json.loads(output.read_text())
    assert [len(a["results"]) for a in checkpoints] == [0, 1, 2, 2]
    assert [r["outcome"]["status"] for r in original["results"]] == [
        "failed",
        "success",
    ]
    failed, success = original["results"]
    assert failed["outcome"]["last_phase"] == "timed_execution"
    assert [o["outcome"]["status"] for o in failed["observations"]] == [
        "success",
        "failed",
    ]
    assert failed["observations"][-1]["timings"] == []
    assert all(
        o["metrics"] == []
        for r in original["results"]
        for o in r["observations"]
    )
    assert events.index(("checkpoint", 0)) < events.index("runtime-enter")
    assert events.index(("checkpoint", 1)) < events.index(("fit", 64))
    assert events[-2:] == ["runtime-exit", ("checkpoint", 2)]
    events.clear()
    state["failure"] = None
    assert _run_cli(manifest, output, resume=True) == 0
    resumed = json.loads(output.read_text())
    assert resumed["run"]["id"] == original["run"]["id"]
    assert resumed["results"][0] == success
    assert {r["id"] for r in resumed["results"]} == {
        r["id"] for r in original["results"]
    }
    assert all(
        r["outcome"] == {"status": "success"} for r in resumed["results"]
    )
    assert ("fit", 64) not in events
    assert events.count(("fit", 32)) == 2
    assert events[-2:] == ["runtime-exit", ("checkpoint", 2)]


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "methodology",
        "plan",
        "provider",
        "software",
        "system",
        "unknown_id",
        "workload",
        "failed_workload",
        "duplicate",
    ],
)
def test_resume_rejects_incompatible_artifact(registered_suite, change):
    manifest, output, events, _ = registered_suite
    assert _run_cli(manifest, output) == 0
    artifact = json.loads(output.read_text())
    if change == "schema":
        artifact["schema_version"] = 1
    elif change in {"methodology", "software", "system"}:
        artifact["run"][change] = {}
    elif change == "plan":
        artifact["run"]["extensions"][harness.EXTENSION]["execution_plan"][0][
            "repetitions"
        ] += 1
    elif change == "provider":
        artifact["run"]["extensions"][harness.EXTENSION][change] = "changed"
    elif change == "unknown_id":
        artifact["results"][0]["id"] = "sha256:unknown"
    elif change in {"workload", "failed_workload"}:
        artifact["results"][0]["parameters"]["declared"]["fail"] = False
        if change == "failed_workload":
            artifact["results"][0]["outcome"] = {"status": "failed"}
    else:
        artifact["results"].append(copy.deepcopy(artifact["results"][0]))
    output.write_text(json.dumps(artifact), encoding="utf-8")
    before = output.read_bytes()
    events.clear()
    with pytest.raises(SystemExit) as error:
        _run_cli(manifest, output, resume=True)
    assert error.value.code == 2
    assert output.read_bytes() == before
    assert "runtime-enter" not in events


@pytest.fixture
def source_checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    module = root / "python/cuml/cuml/benchmark/backends/cuml.py"
    module.parent.mkdir(parents=True)
    module.write_text("# tracked source\n", encoding="utf-8")
    ignore_file = Path(__file__).resolve().parents[3] / ".gitignore"
    (root / ".gitignore").write_bytes(ignore_file.read_bytes())

    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Benchmark test")
    git("config", "user.email", "benchmark@example.invalid")
    git("config", "commit.gpgsign", "false")
    git("add", ".")
    git("commit", "-qm", "Initial source")
    monkeypatch.setattr(cuml_backend, "__file__", str(module))
    monkeypatch.chdir(root)
    return root, module, git


@pytest.mark.parametrize("nested_cwd", [False, True])
@pytest.mark.parametrize("change", ["tracked", "staged", "untracked"])
def test_source_resume_with_ignored_default_output(
    source_checkout, monkeypatch, nested_cwd, change
):
    _, module, git = source_checkout
    if nested_cwd:
        monkeypatch.chdir(module.parent)
    suite = _suite(resolve_case(_request(), PROFILE), provider="cuml")
    output = cli.default_output_path(suite) / "cuml.json"
    original = harness.run_suite(suite, output)
    package = original["run"]["software"]["packages"][0]
    assert package["source"] == {
        "repository": None,
        "revision": git("rev-parse", "HEAD"),
        "dirty": False,
    }
    # Other provider outputs and logs in the default directory are ignored too.
    output.with_name("scikit-learn.json").write_text("{}", encoding="utf-8")
    output.with_name("console.log").write_text("benchmark log\n")
    resumed = harness.run_suite(suite, output, resume=True)
    assert resumed["run"]["id"] == original["run"]["id"]
    assert resumed["results"] == original["results"]
    assert git("status", "--porcelain") == ""

    if change == "untracked":
        module.with_name("new_source.py").write_text("# new source\n")
    else:
        module.write_text("# changed source\n", encoding="utf-8")
        if change == "staged":
            git("add", ".")
    before = output.read_bytes()
    with pytest.raises(
        SuiteError, match="matching suite, software, and system"
    ):
        harness.run_suite(suite, output, resume=True)
    assert output.read_bytes() == before


def test_interrupt_checkpoints_and_cleans_runtime(registered_suite):
    manifest, output, events, state = registered_suite
    state["failure"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        _run_cli(manifest, output)
    assert events[-1] == "runtime-exit"
    artifact = json.loads(output.read_text())
    _validate_artifact(artifact)
    assert "completed_at" in artifact["run"]
    assert len(artifact["results"]) == 1
    assert artifact["results"][0]["outcome"]["last_phase"] == "interrupted"


@pytest.mark.parametrize("kind", ["file", "directory", "dangling_symlink"])
def test_cli_refuses_existing_selected_artifact(
    document, tmp_path, monkeypatch, capsys, kind
):
    document["providers"] = ["cuml", "scikit-learn"]
    path = _write_suite(tmp_path, document)
    output = tmp_path / "results"
    output.mkdir()
    existing = output / "scikit-learn.json"
    if kind == "file":
        existing.write_bytes(b"existing results")
    elif kind == "directory":
        existing.mkdir()
    else:
        existing.symlink_to(tmp_path / "missing")
    monkeypatch.setattr(
        coordinator,
        "_execute_worker",
        lambda *a, **k: pytest.fail("launched worker"),
    )
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "--suite",
                str(path),
                "--provider",
                "cuml",
                "--provider",
                "scikit-learn",
                "--output",
                str(output),
            ]
        )
    assert error.value.code == 2
    assert "use --resume or a new output directory" in capsys.readouterr().err
    assert not (output / "cuml.json").exists()
    if kind == "file":
        assert existing.read_bytes() == b"existing results"


def test_harness_refuses_existing_results(registered_suite):
    manifest, output, events, _ = registered_suite
    output.write_bytes(b"existing results")
    suite = load_suite(manifest).runs[0]
    with pytest.raises(SuiteError, match="results already exist"):
        harness.run_suite(suite, output)
    assert output.read_bytes() == b"existing results"
    assert not events


def test_harness_initial_write_handles_race(registered_suite, monkeypatch):
    manifest, output, events, _ = registered_suite
    suite = load_suite(manifest).runs[0]
    original = harness._run_record

    def record(*args):
        result = original(*args)
        output.write_bytes(b"concurrent results")
        return result

    monkeypatch.setattr(harness, "_run_record", record)
    with pytest.raises(SuiteError, match="results already exist"):
        harness.run_suite(suite, output)
    assert output.read_bytes() == b"concurrent results"
    assert "runtime-enter" not in events
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def _write_worker_artifact(command, *, failed=False):
    provider = command[command.index("--provider") + 1]
    plan = load_suite_reference(
        command[command.index("--suite") + 1],
        command[command.index("--profile") + 1],
        [provider],
    )
    extension = {harness.EXTENSION: {"provider": provider}}
    artifact = {
        "schema_version": 2,
        "run": {"extensions": extension},
        "results": [
            {
                "id": case.id,
                "outcome": {"status": "failed" if failed else "success"},
                "extensions": extension,
            }
            for case in plan.runs[0].cases
        ],
    }
    Path(command[command.index("--output") + 1]).write_text(
        json.dumps(artifact)
    )
    return artifact


def test_cli_allows_new_provider_in_existing_directory(
    document, tmp_path, monkeypatch
):
    path = _write_suite(tmp_path, document)
    output = tmp_path / "results"
    output.mkdir()
    unrelated = output / "cuml.json"
    unrelated.write_bytes(b"unselected results")
    calls = []

    def execute(command, environment):
        calls.append(command)
        _write_worker_artifact(command)

    monkeypatch.setattr(coordinator, "_execute_worker", execute)
    assert (
        cli.main(
            [
                "--suite",
                str(path),
                "--provider",
                "scikit-learn",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert len(calls) == 1
    assert unrelated.read_bytes() == b"unselected results"


def test_cli_output_safety(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--resume"])
    assert error.value.code == 2
    assert "--resume requires --output" in capsys.readouterr().err
    monkeypatch.chdir(tmp_path)
    suite = load_suite_reference("estimators", "smoke", ["scikit-learn"])
    now = dt.datetime(2026, 8, 13, tzinfo=dt.timezone.utc)
    first = cli.default_output_path(suite, now)
    assert first == tmp_path / ".benchmarks/estimators-smoke-20260813T000000Z"
    assert not first.parent.exists()
    first.parent.mkdir()
    first.write_text("existing artifact", encoding="utf-8")
    second = cli.default_output_path(suite, now)
    assert second != first and not second.exists()
    assert first.read_text() == "existing artifact"


@pytest.mark.parametrize("entrypoint", ["python", "cli"])
@pytest.mark.parametrize(
    "inherited,enabled",
    [("1", False), ("true", False), ("TRUE", False), (None, True)],
)
def test_coordinator_rejects_accelerated_caller(
    document, tmp_path, monkeypatch, capsys, entrypoint, inherited, enabled
):
    if inherited is None:
        monkeypatch.delenv("CUML_ACCEL_ENABLED", raising=False)
    else:
        monkeypatch.setenv("CUML_ACCEL_ENABLED", inherited)
    monkeypatch.setitem(
        sys.modules, "cuml.accel", SimpleNamespace(enabled=lambda: enabled)
    )
    monkeypatch.setattr(
        coordinator,
        "_execute_worker",
        lambda *a, **k: pytest.fail("launched worker"),
    )
    path = _write_suite(tmp_path, document)
    output = tmp_path / "results"
    message = "process without cuml.accel enabled"
    if entrypoint == "python":
        with pytest.raises(SuiteError, match=message):
            benchmark.run(path, providers=["scikit-learn"], output=output)
    else:
        with pytest.raises(SystemExit) as error:
            cli.main(
                [
                    "--suite",
                    str(path),
                    "--provider",
                    "scikit-learn",
                    "--output",
                    str(output),
                ]
            )
        assert error.value.code == 2
        assert message in capsys.readouterr().err
    assert not output.exists()
    assert os.environ.get("CUML_ACCEL_ENABLED") == inherited


@pytest.mark.parametrize("inherited", [None, "0", "false"])
def test_coordinator_preserves_worker_environment(
    document, tmp_path, monkeypatch, inherited
):
    if inherited is None:
        monkeypatch.delenv("CUML_ACCEL_ENABLED", raising=False)
    else:
        monkeypatch.setenv("CUML_ACCEL_ENABLED", inherited)
    monkeypatch.delitem(sys.modules, "cuml.accel", raising=False)
    monkeypatch.setenv("BENCHMARK_ENV_TEST", "preserved")
    original = dict(os.environ)
    document["providers"] = ["cuml.accel", "scikit-learn"]
    path = _write_suite(tmp_path, document)
    environments = []

    def execute(command, environment):
        assert environment == original
        assert "cuml.accel" not in sys.modules
        assert all(environment is not previous for previous in environments)
        environments.append(environment)
        environment["BENCHMARK_ENV_TEST"] = "changed"
        _write_worker_artifact(command)

    monkeypatch.setattr(coordinator, "_execute_worker", execute)
    benchmark.run(
        path, providers=document["providers"], output=tmp_path / "results"
    )
    assert len(environments) == 2
    assert dict(os.environ) == original


def test_cli_coordinates_isolated_backends(document, tmp_path, monkeypatch):
    document["providers"] = ["cuml", "scikit-learn", "cuml.accel"]
    path = _write_suite(tmp_path, document)
    output = tmp_path / "results"
    calls = []
    monkeypatch.delenv("CUML_ACCEL_ENABLED", raising=False)
    original = dict(os.environ)

    def execute(command, environment):
        assert "--_worker" in command
        backend = command[command.index("--provider") + 1]
        assert environment == original
        assert "--verbose" in command
        artifact = Path(command[command.index("--output") + 1])
        assert artifact == output / f"{backend}.json"
        calls.append((backend, "--resume" in command))
        _write_worker_artifact(command, failed=backend == "scikit-learn")
        if backend == "scikit-learn":
            raise RuntimeError("worker exited with status 1")

    monkeypatch.setattr(coordinator, "_execute_worker", execute)
    argv = ["--suite", str(path), "--output", str(output), "--verbose"]
    assert cli.main(argv) == 0
    assert calls == [("cuml", False)]
    calls.clear()
    selected = [
        arg for name in document["providers"] for arg in ("--provider", name)
    ]
    (output / "cuml.json").unlink()
    assert cli.main([*argv, *selected]) == 1
    assert calls == [(name, False) for name in document["providers"]]
    assert dict(os.environ) == original
    (output / "cuml.accel.json").unlink()
    calls.clear()
    assert cli.main([*argv, *selected, "--resume"]) == 1
    assert calls == [
        ("cuml", True),
        ("scikit-learn", True),
        ("cuml.accel", False),
    ]
    calls.clear()
    assert cli.main([*argv, "--provider", "cuml", "--resume"]) == 0
    assert calls == [("cuml", True)]
    calls.clear()
    assert cli.main([*argv, "--provider", "scikit-learn", "--resume"]) == 1
    assert calls == [("scikit-learn", True)]


def test_cli_default_requires_cuml_in_suite(
    document, tmp_path, monkeypatch, capsys
):
    path = _write_suite(tmp_path, document)
    monkeypatch.setattr(
        coordinator,
        "_execute_worker",
        lambda *a, **k: pytest.fail("launched worker"),
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(["--suite", str(path), "--output", str(tmp_path / "results")])
    assert exc.value.code == 2
    assert "subset of suite providers" in capsys.readouterr().err


def test_cli_coordinator_output_errors(document, tmp_path, monkeypatch):
    document["providers"] = ["cuml"]
    path = _write_suite(tmp_path, document)
    output = tmp_path / "missing"
    monkeypatch.setattr(
        coordinator,
        "_execute_worker",
        lambda *a, **k: pytest.fail("launched worker"),
    )
    with pytest.raises(SystemExit):
        cli.main(["--suite", str(path), "--output", str(output), "--resume"])
    output.write_text("not a directory")
    with pytest.raises(SystemExit):
        cli.main(["--suite", str(path), "--output", str(output)])
    assert output.read_text() == "not a directory"


@pytest.mark.parametrize(
    "providers",
    [["scikit-learn"], ["cuml.accel", "scikit-learn", "cuml"]],
)
def test_cli_real_workers(document, tmp_path, monkeypatch, providers):
    # A CPU worker must remain unaccelerated even after an accel worker.
    monkeypatch.delenv("CUML_ACCEL_ENABLED", raising=False)
    document["providers"] = providers
    path = _write_suite(tmp_path, document)
    output = tmp_path / "results"
    argv = ["--suite", str(path), "--output", str(output)]
    argv.extend(arg for name in providers for arg in ("--provider", name))
    assert cli.main(argv) == 0
    artifacts = {}
    for provider in providers:
        artifact = json.loads((output / f"{provider}.json").read_text())
        artifacts[provider] = artifact
        _validate_artifact(artifact)
        assert len(artifact["results"]) == 1
        result = artifact["results"][0]
        assert result["outcome"]["status"] == "success"
        assert result["extensions"][harness.EXTENSION]["provider"] == provider
        if provider == "cuml.accel":
            evidence = result["observations"][0]["extensions"][ACCEL_EXTENSION]
            assert evidence["gpu_calls"] > 0 and evidence["cpu_calls"] == 0
    assert len({a["results"][0]["id"] for a in artifacts.values()}) == 1
    assert len({a["run"]["id"] for a in artifacts.values()}) == len(providers)
    assert cli.main([*argv, "--resume"]) == 0
    for provider, artifact in artifacts.items():
        resumed = json.loads((output / f"{provider}.json").read_text())
        assert resumed["run"]["id"] == artifact["run"]["id"]
        assert resumed["results"] == artifact["results"]


@pytest.mark.parametrize(
    "enabled,attempted", [(False, False), (False, True), (True, False)]
)
def test_accel_startup(monkeypatch, enabled, attempted):
    monkeypatch.setattr("cuml.accel.enabled", lambda: enabled)
    if attempted:
        monkeypatch.setenv("CUML_ACCEL_ENABLED", "1")
    else:
        monkeypatch.delenv("CUML_ACCEL_ENABLED", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--suite",
            "estimators",
            "--provider",
            "cuml.accel",
            "--_worker",
        ],
    )
    calls = []
    monkeypatch.setattr(os, "execvpe", lambda *args: calls.append(args))
    backend = get_backend("cuml.accel")
    if attempted:
        with pytest.raises(SuiteError, match="did not activate"):
            backend.bootstrap_process()
        assert not calls
    else:
        backend.bootstrap_process()
        if enabled:
            assert not calls
        else:
            assert len(calls) == 1
            executable, argv, env = calls[0]
            assert executable == sys.executable
            assert argv == [
                sys.executable,
                "-m",
                "cuml.benchmark",
                "--suite",
                "estimators",
                "--provider",
                "cuml.accel",
                "--_worker",
            ]
            assert env["CUML_ACCEL_ENABLED"] == "1"


# Process isolation, real runtimes, and distributed resource safety.


def test_spawned_case_completes(tmp_path, monkeypatch):
    monkeypatch.delenv("CUML_ACCEL_ENABLED", raising=False)
    request = _request(
        operation="transform", fit_input_selection=["X"], timeout_sec=60
    )
    request["dataset"]["shape"]["train_rows"] = 32
    case = resolve_case(request, PROFILE)
    output = tmp_path / "artifact.json"
    artifact = harness.run_suite(_suite(case), output)
    _validate_artifact(artifact)
    assert json.loads(output.read_text()) == artifact
    result = artifact["results"][0]
    assert result["outcome"] == {"status": "success"}
    assert result["id"] == case.id
    assert [o["role"] for o in result["observations"]] == [
        "warmup",
        "measurement",
        "measurement",
    ]


def _worker_failure(crash, *, report_phase):
    report_phase("ready")
    if crash:
        os._exit(7)
    time.sleep(120)


@pytest.mark.parametrize("failure", ["timeout", "exit", "interrupt"])
def test_subprocess_failure_cleans_up_worker(failure):
    before = {c.pid for c in multiprocessing.active_children()}
    phases = []

    def phase(value):
        phases.append(value)
        if failure == "interrupt":
            raise KeyboardInterrupt()

    expected = {
        "timeout": runner.SubprocessTimeout,
        "exit": runner.SubprocessExited,
        "interrupt": KeyboardInterrupt,
    }[failure]
    with pytest.raises(expected) as error:
        runner.run_in_subprocess(
            _worker_failure,
            args=(failure == "exit",),
            timeout=0.01 if failure == "timeout" else 60,
            report_phase=phase,
        )
    if failure == "exit":
        assert error.value.exitcode == 7
    if failure != "timeout":
        assert phases == ["ready"]
    assert {c.pid for c in multiprocessing.active_children()} <= before


@pytest.mark.parametrize(
    "failure,error_type",
    [
        (runner.SubprocessTimeout(), "TimeoutExpired"),
        (runner.SubprocessExited(7), "RuntimeError"),
    ],
)
def test_subprocess_failure_preserves_last_phase(
    monkeypatch, failure, error_type
):
    def fail(*args, report_phase, **kwargs):
        report_phase("warmup")
        raise failure

    monkeypatch.setattr(harness, "run_in_subprocess", fail)
    case = resolve_case(_request(timeout_sec=10), PROFILE)
    result = harness._run_case(_suite(case), case, PACKAGE)
    assert result["outcome"]["status"] == "failed"
    assert result["outcome"]["last_phase"] == "warmup"
    assert result["outcome"]["error"]["type"] == error_type
    assert result["outcome"]["error"]["message"]
    assert result["observations"] == []


def _real_backend_smoke(provider, workload, output, *, report_phase):
    import cuml.accel

    assert cuml.accel.enabled() == (provider == "cuml.accel")
    request = _request()
    if workload == "csr-inference":
        request.update(
            estimator="LogisticRegression",
            operation="predict",
            fit_input_selection=["X", "y"],
            parameters={"max_iter": 100},
        )
        request["dataset"].update(
            kind="classification",
            format="csr",
            dtype="float64",
            parameters={"density": 0.25},
        )
        request["dataset"]["shape"].update(rows=128, train_rows=128)
    case = resolve_case(request, {**PROFILE, "repetitions": 1})
    X, _ = generate_data(case)
    provider_spec = get_provider(provider)
    backend = provider_spec.backend
    cls = backend.load_estimator(provider_spec.estimator_spec(case.estimator))
    assert cuml.accel.is_proxy(cls) == (provider == "cuml.accel")
    invoke = getattr(cls, case.operation)
    calls = []

    def observe(estimator, *args):
        actual = args[0]
        expected = (
            X[case.training_rows :] if case.lifecycle == "inference" else X
        )
        assert actual.dtype == np.dtype(case.dtypes["X"])
        assert sparse.isspmatrix_csr(actual) == (case.input_format == "csr")
        if case.input_format == "csr":
            for name in ("data", "indices", "indptr"):
                np.testing.assert_array_equal(
                    getattr(actual, name), getattr(expected, name)
                )
        else:
            np.testing.assert_array_equal(actual, expected)
        calls.append(actual.shape)
        return invoke(estimator, *args)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cls, case.operation, observe)
        artifact = harness.run_suite(_suite(case, provider=provider), output)
    assert len(calls) == 2
    assert json.loads(output.read_text()) == artifact
    result = artifact["results"][0]
    assert result["outcome"] == {"status": "success"}, result["outcome"]
    assert result["id"] == case.id
    assert result["input"]["data_type"] == case.dtypes["X"]
    if provider == "cuml.accel":
        dispatch = result["observations"][0]["extensions"][ACCEL_EXTENSION]
        assert dispatch["gpu_calls"] >= 1 and dispatch["cpu_calls"] == 0
        assert ACCEL_EXTENSION not in result["observations"][1]["extensions"]
    return artifact


@pytest.mark.parametrize("provider", ["scikit-learn", "cuml", "cuml.accel"])
@pytest.mark.parametrize("workload", ["dense-fit", "csr-inference"])
def test_real_backend_smoke(monkeypatch, tmp_path, provider, workload):
    # Accelerator installation mutates upstream modules; never enable it in pytest.
    if provider == "cuml.accel":
        monkeypatch.setenv("CUML_ACCEL_ENABLED", "1")
    else:
        monkeypatch.delenv("CUML_ACCEL_ENABLED", raising=False)
    artifact = runner.run_in_subprocess(
        _real_backend_smoke,
        args=(provider, workload, tmp_path / "artifact.json"),
        timeout=90,
    )
    _validate_artifact(artifact)


@pytest.mark.parametrize("collection", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_dask_synchronization(monkeypatch, collection, nested):
    events = []

    def wait(value):
        events.append("wait")

    modules = {
        "dask.distributed": SimpleNamespace(wait=wait),
        "cupy": SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(
                    deviceSynchronize=lambda: events.append("gpu")
                )
            )
        ),
    }
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    value = (
        SimpleNamespace(compute=lambda: events.append("compute"))
        if collection
        else object()
    )
    get_backend("cuml.dask").synchronize((value, [value]) if nested else value)
    expected = ["compute", "gpu"] if collection else ["wait", "gpu"]
    assert events == expected * (2 if nested else 1)


@pytest.mark.parametrize("estimator", ["PCA", "LabelEncoder", "MultinomialNB"])
def test_dask_input_conversion(estimator):
    cp = pytest.importorskip("cupy")
    pytest.importorskip("dask_cudf")
    case = resolve_case(_request(estimator=estimator), PROFILE)
    X = np.arange(512, dtype=np.float32).reshape(64, 8)
    y = np.arange(64, dtype=np.int64) % 3
    converted_X, converted_y = get_backend("cuml.dask").convert_data(
        case, X, y
    )
    assert isinstance(converted_X._meta, cp.ndarray)
    cp.testing.assert_array_equal(converted_X.compute(), cp.asarray(X))
    if estimator == "LabelEncoder":
        import cudf

        assert isinstance(converted_y._meta, cudf.Series)
        np.testing.assert_array_equal(converted_y.compute().to_numpy(), y)
    else:
        assert isinstance(converted_y._meta, cp.ndarray)
        cp.testing.assert_array_equal(converted_y.compute(), cp.asarray(y))


@pytest.mark.parametrize("distributed", [False, True])
def test_distributed_effective_parameters_are_an_object(distributed):
    parameters = (
        [{"n_estimators": 2}, {"n_estimators": 2}]
        if distributed
        else {"n_estimators": 2}
    )

    class Estimator:
        def __init__(self, **kwargs):
            pass

        def get_params(self, deep=False):
            assert deep is False
            return parameters

    prepared = harness._PreparedCase(Estimator, (), None)
    case = resolve_case(_request(), PROFILE)
    result = {"parameters": {"effective": {}}}
    harness._construct_estimator(
        get_backend("cuml.dask"), prepared, case, None, result
    )
    expected = parameters[0] if distributed else parameters
    assert result["parameters"]["effective"] == expected


def test_default_backend_effective_parameters():
    parameters = {"n_estimators": 2}
    estimator = SimpleNamespace(get_params=lambda **kwargs: parameters)
    assert Backend().effective_parameters(estimator) is parameters


def test_dask_dbscan_uses_local_input():
    case = resolve_case(_request(estimator="DBSCAN"), PROFILE)
    X = np.zeros((64, 8), dtype=np.float32)
    y = np.zeros(64, dtype=np.int64)
    converted_X, converted_y = get_backend("cuml.dask").convert_data(
        case, X, y
    )
    assert converted_X is X
    assert converted_y is y


@pytest.mark.parametrize("count", [0, 1])
def test_dask_requires_multiple_gpus(monkeypatch, count):
    def import_module(name):
        assert name == "cupy", (
            "must reject before importing cluster dependencies"
        )
        return SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(getDeviceCount=lambda: count)
            )
        )

    monkeypatch.setattr(importlib, "import_module", import_module)
    case = resolve_case(_request(), PROFILE)
    with pytest.raises(
        SuiteError, match=f"at least two visible GPUs; found {count}"
    ):
        with get_backend("cuml.dask").runtime(
            _suite(case, provider="cuml.dask")
        ):
            pytest.fail("must reject before resource creation")


@pytest.mark.parametrize(
    "failure", ["csr", "timeout", "execution", "client-startup"]
)
def test_dask_runtime_safety(monkeypatch, failure):
    case = resolve_case(_request(), PROFILE)
    backend = get_backend("cuml.dask")
    if failure in {"csr", "timeout"}:
        case = (
            replace(case, input_format="csr")
            if failure == "csr"
            else replace(case, timeout_sec=1)
        )
        monkeypatch.setattr(
            importlib,
            "import_module",
            lambda name: pytest.fail(f"runtime import: {name}"),
        )
        with pytest.raises(SuiteError, match="not supported"):
            with backend.runtime(_suite(case, provider="cuml.dask")):
                pytest.fail("must reject before resource creation")
        return
    events = []

    @contextlib.contextmanager
    def cluster(**kwargs):
        events.append("cluster-enter")
        try:
            yield "cluster"
        finally:
            events.append("cluster-exit")

    @contextlib.contextmanager
    def client(cluster):
        if failure == "client-startup":
            raise RuntimeError("startup failed")
        events.append("client-enter")
        try:
            yield "client"
        finally:
            events.append("client-exit")

    modules = {
        "cupy": SimpleNamespace(
            cuda=SimpleNamespace(
                runtime=SimpleNamespace(getDeviceCount=lambda: 2)
            )
        ),
        "dask_cuda": SimpleNamespace(LocalCUDACluster=cluster),
        "dask.distributed": SimpleNamespace(Client=client),
    }
    monkeypatch.setattr(importlib, "import_module", modules.__getitem__)
    with pytest.raises(RuntimeError, match="failed"):
        with backend.runtime(_suite(case, provider="cuml.dask")) as runtime:
            assert runtime == "client"
            raise RuntimeError("execution failed")
    assert events == (
        ["cluster-enter", "cluster-exit"]
        if failure == "client-startup"
        else ["cluster-enter", "client-enter", "client-exit", "cluster-exit"]
    )


# Public runner uses the same provider coordinator as the CLI.


def test_benchmark_results_dataclass():
    artifacts = {"cuml": {"results": []}}
    results = benchmark.BenchmarkResults(artifacts=artifacts)
    assert results.artifacts == artifacts
    with pytest.raises(TypeError):
        benchmark.BenchmarkResults(artifacts)


def test_run_builtin_defaults_and_temporary_cleanup(monkeypatch):
    paths = []

    def execute(command, environment):
        assert command[command.index("--suite") + 1] == "estimators"
        assert command[command.index("--provider") + 1] == "cuml"
        paths.append(Path(command[command.index("--output") + 1]))
        _write_worker_artifact(command)

    monkeypatch.setattr(coordinator, "_execute_worker", execute)
    results = benchmark.run(profile="smoke")
    assert isinstance(results, benchmark.BenchmarkResults)
    assert list(results.artifacts) == ["cuml"]
    assert results.artifacts["cuml"]["results"]
    assert not paths[0].parent.exists()


@pytest.mark.parametrize("reference_type", [str, Path])
def test_run_yaml_persistence_and_resume(
    document, tmp_path, monkeypatch, reference_type
):
    path = _write_suite(tmp_path, document)
    output = tmp_path / "results"
    calls = []

    def execute(command, environment):
        calls.append("--resume" in command)
        _write_worker_artifact(command)

    monkeypatch.setattr(coordinator, "_execute_worker", execute)
    results = benchmark.run(
        reference_type(path), providers=["scikit-learn"], output=output
    )
    artifacts = results.artifacts
    assert (
        json.loads((output / "scikit-learn.json").read_text())
        == artifacts["scikit-learn"]
    )
    assert (
        benchmark.run(
            path, providers=["scikit-learn"], output=output, resume=True
        )
        == results
    )
    assert calls == [False, True]
    with pytest.raises(SuiteError, match="results already exist"):
        benchmark.run(path, providers=["scikit-learn"], output=output)
    assert calls == [False, True]


@pytest.mark.parametrize(
    "settings",
    [
        {"providers": []},
        {"providers": ["missing"]},
        {"profile": "missing"},
        {"resume": True},
        {"resume": True, "output": "nonexistent-output"},
        {"suite": object()},
    ],
)
def test_run_configuration_errors_before_launch(
    monkeypatch, tmp_path, settings
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        coordinator,
        "_execute_worker",
        lambda *a: pytest.fail("launched worker"),
    )
    with pytest.raises(SuiteError):
        benchmark.run(**settings)


@pytest.mark.parametrize(
    "failure",
    [
        "startup",
        "exit",
        "case",
        "missing",
        "malformed",
    ],
)
def test_run_partial_failures_continue_and_clean_output(
    document, tmp_path, monkeypatch, failure
):
    document["providers"] = ["scikit-learn", "cuml"]
    path = _write_suite(tmp_path, document)
    paths = []

    def execute(command, environment):
        provider = command[command.index("--provider") + 1]
        artifact_path = Path(command[command.index("--output") + 1])
        paths.append(artifact_path)
        if provider == "cuml":
            _write_worker_artifact(command)
            return
        if failure == "startup":
            raise OSError("startup unavailable")
        if failure == "missing":
            return
        _write_worker_artifact(command, failed=failure == "case")
        if failure == "exit":
            raise RuntimeError(
                "worker exited with status 7: startup traceback"
            )
        if failure == "malformed":
            artifact_path.write_text("not json")

    monkeypatch.setattr(coordinator, "_execute_worker", execute)
    with pytest.raises(benchmark.BenchmarkRunError) as error:
        benchmark.run(path, providers=document["providers"])
    assert [p.stem for p in paths] == document["providers"]
    assert not paths[0].parent.exists()
    assert list(error.value.failures) == ["scikit-learn"]
    assert (
        error.value.results.artifacts["cuml"]["results"][0]["outcome"][
            "status"
        ]
        == "success"
    )
    assert ("scikit-learn" in error.value.results.artifacts) == (
        failure in {"case", "exit"}
    )
    if failure == "startup":
        assert "startup unavailable" in str(error.value)
    elif failure == "exit":
        assert "status 7: startup traceback" in str(error.value)


@pytest.mark.parametrize("failure", ["provider", "incomplete", "duplicate"])
def test_run_rejects_invalid_worker_artifact(
    document, tmp_path, monkeypatch, failure
):
    path = _write_suite(tmp_path, document)

    def execute(command, environment):
        artifact = _write_worker_artifact(command)
        if failure == "provider":
            artifact["run"]["extensions"][harness.EXTENSION]["provider"] = (
                "wrong"
            )
        else:
            artifact["results"] = (
                [] if failure == "incomplete" else artifact["results"] * 2
            )
        Path(command[command.index("--output") + 1]).write_text(
            json.dumps(artifact)
        )

    monkeypatch.setattr(coordinator, "_execute_worker", execute)
    with pytest.raises(benchmark.BenchmarkRunError) as error:
        benchmark.run(path, providers=["scikit-learn"])
    diagnostic = (
        "invalid artifact envelope"
        if failure == "provider"
        else "artifact does not contain all expected cases"
    )
    assert diagnostic in error.value.failures["scikit-learn"]


@pytest.mark.parametrize("persistent", [False, True])
def test_run_interrupt_output_cleanup(
    document, tmp_path, monkeypatch, persistent
):
    path = _write_suite(tmp_path, document)
    paths = []

    def execute(command, environment):
        _write_worker_artifact(command)
        paths.append(Path(command[command.index("--output") + 1]))
        raise KeyboardInterrupt

    monkeypatch.setattr(coordinator, "_execute_worker", execute)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run(
            path,
            providers=["scikit-learn"],
            output=tmp_path / "results" if persistent else None,
        )
    assert paths[0].exists() == persistent
    assert paths[0].parent.exists() == persistent


@pytest.mark.skipif(
    os.name != "posix", reason="process group cleanup is POSIX-specific"
)
def test_run_interrupt_stops_provider_and_descendant(tmp_path, monkeypatch):
    import psutil

    pid_file = tmp_path / "pids.json"
    processes = []
    original = subprocess.Popen

    class InterruptedProcess(original):
        def communicate(self, **kwargs):
            deadline = time.monotonic() + 30
            while not pid_file.exists():
                assert time.monotonic() < deadline
                time.sleep(0.01)
            processes.extend(
                psutil.Process(pid) for pid in json.loads(pid_file.read_text())
            )
            raise KeyboardInterrupt

    monkeypatch.setattr(coordinator.subprocess, "Popen", InterruptedProcess)
    script = (
        "import json, os, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        f"open({str(pid_file)!r}, 'w').write(json.dumps([os.getpid(), child.pid])); "
        "time.sleep(120)"
    )
    with pytest.raises(KeyboardInterrupt):
        coordinator._execute_worker(
            [sys.executable, "-c", script], dict(os.environ)
        )
    _, alive = psutil.wait_procs(processes, timeout=5)
    assert all(process.status() == psutil.STATUS_ZOMBIE for process in alive)


def test_run_unguarded_script_real_workers(document, tmp_path):
    document["providers"] = ["scikit-learn"]
    document["profiles"]["standard"]["timeout_sec"] = 60
    path = _write_suite(tmp_path, document)
    script = tmp_path / "run.py"
    script.write_text(
        "from cuml import benchmark\nimport json\n"
        f"artifacts = benchmark.run({str(path)!r}, providers={document['providers']!r}, output={str(tmp_path / 'results')!r})\n"
        "print(json.dumps(artifacts.artifacts))\n"
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    artifacts = json.loads(completed.stdout)
    assert list(artifacts) == document["providers"]
    for provider, artifact in artifacts.items():
        _validate_artifact(artifact)
        assert artifact["results"][0]["outcome"] == {"status": "success"}
        assert (
            artifact["run"]["extensions"][harness.EXTENSION]["provider"]
            == provider
        )


@pytest.mark.parametrize(
    "failure", ["stderr", "descendant-stderr", "invalid-stderr"]
)
def test_run_worker_diagnostic_failures_continue(
    document, tmp_path, monkeypatch, capsys, failure
):
    if failure == "descendant-stderr" and os.name != "posix":
        pytest.skip("descendant cleanup is POSIX-specific")
    document["providers"] = ["scikit-learn", "cuml"]
    path = _write_suite(tmp_path, document)
    pid_file = tmp_path / "descendant.pid"
    if failure == "descendant-stderr":
        script = (
            "import subprocess, sys; "
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)']); "
            f"open({str(pid_file)!r}, 'w').write(str(child.pid)); "
            "print('provider crashed', file=sys.stderr); sys.exit(7)"
        )
        diagnostic = "provider crashed"
    elif failure == "stderr":
        script = "import sys; print('startup evidence', file=sys.stderr); sys.exit(7)"
        diagnostic = "startup evidence"
    else:
        script = (
            "import sys; sys.stderr.buffer.write(b'startup \\xff\\n'); "
            "sys.stderr.flush(); sys.exit(7)"
        )
        diagnostic = "startup \ufffd"
    execute_worker = coordinator._execute_worker
    calls = []

    def execute(command, environment):
        provider = command[command.index("--provider") + 1]
        calls.append(provider)
        _write_worker_artifact(command)
        execute_worker(
            [
                sys.executable,
                "-c",
                script if provider == "scikit-learn" else "pass",
            ],
            environment,
        )

    monkeypatch.setattr(coordinator, "_execute_worker", execute)
    started = time.monotonic()
    with pytest.raises(benchmark.BenchmarkRunError) as error:
        benchmark.run(path, providers=document["providers"])
    assert time.monotonic() - started < 2
    assert calls == document["providers"]
    assert isinstance(error.value.results, benchmark.BenchmarkResults)
    assert list(error.value.results.artifacts) == document["providers"]
    assert list(error.value.failures) == ["scikit-learn"]
    assert "status 7" in error.value.failures["scikit-learn"]
    assert diagnostic in error.value.failures["scikit-learn"]
    assert diagnostic in capsys.readouterr().err
    if failure == "descendant-stderr":
        import psutil

        try:
            descendant = psutil.Process(int(pid_file.read_text()))
        except psutil.NoSuchProcess:
            return
        _, alive = psutil.wait_procs([descendant], timeout=1)
        assert all(
            process.status() == psutil.STATUS_ZOMBIE for process in alive
        )
