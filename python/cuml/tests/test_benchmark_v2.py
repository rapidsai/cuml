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
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import jsonschema
import numpy as np
import pytest
import yaml
from scipy import sparse

from cuml.benchmark import _subprocess as runner
from cuml.benchmark import harness
from cuml.benchmark.backends import BACKENDS, get_backend
from cuml.benchmark.backends.accel import ACCEL_EXTENSION
from cuml.benchmark.backends.base import Backend
from cuml.benchmark.datasets import generate_data
from cuml.benchmark.identity import canonical_json, result_id
from cuml.benchmark.registry import ACCEL_REGISTRY, SG_REGISTRY, EstimatorSpec
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


def _suite(*cases, implementation="scikit-learn"):
    return Suite("test", "test", implementation, "standard", tuple(cases))


def _benchmark(case, implementation="scikit-learn", **kwargs):
    return harness._benchmark_case(
        _suite(case, implementation=implementation),
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
        "implementation": "scikit-learn",
        "profiles": {"standard": dict(PROFILE)},
        "cases": [_request()],
    }


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
    assert load_suite(_write_suite(tmp_path, document)).cases == (case,)
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
        document["implementation"] = "cuml.accel"
        profile["warmups"] = 0
    with pytest.raises(SuiteError, match=match):
        load_suite(_write_suite(tmp_path, document))
    if change not in {"estimator", "duplicate", "accel_warmup"}:
        with pytest.raises(SuiteError, match=match):
            resolve_case(case, profile)


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
        implementation={"name": "other"},
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
        get_backend("scikit-learn"), "load_estimator", lambda spec: Estimator
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

    def generate(case, implementation):
        events.append("generate")
        return np.ones((case.generated_rows, case.features)), np.zeros(
            case.generated_rows
        )

    backend = get_backend("scikit-learn")
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


@pytest.mark.parametrize(
    "enabled,attempted", [(False, False), (False, True), (True, False)]
)
def test_accel_startup(monkeypatch, enabled, attempted):
    monkeypatch.setattr("cuml.accel.enabled", lambda: enabled)
    if attempted:
        monkeypatch.setenv("CUML_ACCEL_ENABLED", "1")
    else:
        monkeypatch.delenv("CUML_ACCEL_ENABLED", raising=False)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--suite", "cuml_accel"])
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
                "cuml_accel",
            ]
            assert env["CUML_ACCEL_ENABLED"] == "1"


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


def _real_backend_smoke(implementation, workload, output, *, report_phase):
    import cuml.accel

    assert cuml.accel.enabled() == (implementation == "cuml.accel")
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
    backend = get_backend(implementation)
    cls = backend.load_estimator(backend.estimator_spec(case.estimator))
    assert cuml.accel.is_proxy(cls) == (implementation == "cuml.accel")
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
        artifact = harness.run_suite(
            _suite(case, implementation=implementation), output
        )
    assert len(calls) == 2
    assert json.loads(output.read_text()) == artifact
    result = artifact["results"][0]
    assert result["outcome"] == {"status": "success"}, result["outcome"]
    assert result["id"] == case.id
    assert result["input"]["data_type"] == case.dtypes["X"]
    if implementation == "cuml.accel":
        dispatch = result["observations"][0]["extensions"][ACCEL_EXTENSION]
        assert dispatch["gpu_calls"] >= 1 and dispatch["cpu_calls"] == 0
        assert ACCEL_EXTENSION not in result["observations"][1]["extensions"]
    return artifact


@pytest.mark.parametrize(
    "implementation", ["scikit-learn", "cuml", "cuml.accel"]
)
@pytest.mark.parametrize("workload", ["dense-fit", "csr-inference"])
def test_real_backend_smoke(monkeypatch, tmp_path, implementation, workload):
    # Accelerator installation mutates upstream modules; never enable it in pytest.
    if implementation == "cuml.accel":
        monkeypatch.setenv("CUML_ACCEL_ENABLED", "1")
    else:
        monkeypatch.delenv("CUML_ACCEL_ENABLED", raising=False)
    artifact = runner.run_in_subprocess(
        _real_backend_smoke,
        args=(implementation, workload, tmp_path / "artifact.json"),
        timeout=90,
    )
    _validate_artifact(artifact)


@pytest.mark.parametrize("fallback", [False, True])
def test_dask_synchronization(monkeypatch, fallback):
    events = []

    def wait(value):
        events.append("wait")
        if fallback:
            raise TypeError("not a future")

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
    get_backend("cuml.dask").synchronize(
        SimpleNamespace(compute=lambda: events.append("compute"))
    )
    assert events == (
        ["wait", "compute", "gpu"] if fallback else ["wait", "gpu"]
    )


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
            with backend.runtime(_suite(case, implementation="cuml.dask")):
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
        with backend.runtime(
            _suite(case, implementation="cuml.dask")
        ) as runtime:
            assert runtime == "client"
            raise RuntimeError("execution failed")
    assert events == (
        ["cluster-enter", "cluster-exit"]
        if failure == "client-startup"
        else ["cluster-enter", "client-enter", "client-exit", "cluster-exit"]
    )
