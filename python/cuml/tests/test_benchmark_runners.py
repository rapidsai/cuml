# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
from sklearn.metrics import accuracy_score

from cuml.benchmark.runners import AccuracyComparisonRunner


class _MockCumlModel:
    def predict(self, X):
        return cp.zeros(X.shape[0], dtype=cp.int32)


class _MockAlgoPair:
    name = "MockCumlAlgo"
    accuracy_function = staticmethod(accuracy_score)
    cuml_data_prep_hook = None
    cpu_data_prep_hook = None

    def has_cuml(self):
        return True

    def has_cpu(self):
        return False

    def setup_cuml(self, data, **kwargs):
        return {}

    def run_cuml(self, data, **kwargs):
        return _MockCumlModel()


def test_accuracy_runner_converts_cupy_metric_inputs(monkeypatch):
    runner = AccuracyComparisonRunner(
        [10], [2], dataset_name="classification", input_type="cupy", n_reps=1
    )
    data = (
        cp.zeros((10, 2), dtype=cp.float32),
        cp.zeros(10, dtype=cp.int32),
        cp.zeros((4, 2), dtype=cp.float32),
        cp.zeros(4, dtype=cp.int32),
    )

    monkeypatch.setattr(
        "cuml.benchmark.runners.datagen.gen_data", lambda *args, **kwargs: data
    )
    monkeypatch.setattr(
        "cuml.benchmark.runners.is_gpu_available", lambda: True
    )

    result = runner._run_one_size(
        _MockAlgoPair(),
        n_samples=10,
        n_features=2,
        run_cpu=False,
        run_cuml=True,
    )

    assert result["cuml_acc"] == 1.0
