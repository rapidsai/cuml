# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Critical contracts for benchmark workloads, measurements, and recovery."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
import pytest
from cuml.benchmark.identity import canonical_json, result_id
from cuml.benchmark.backends import get_backend

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
