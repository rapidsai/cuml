# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cuml.benchmark.algorithms import _with_default_cpu_parallelism


class _ParallelEstimator:
    def __init__(self, alpha=1.0, n_jobs=None):
        self.alpha = alpha
        self.n_jobs = n_jobs


class _SerialEstimator:
    def __init__(self, alpha=1.0):
        self.alpha = alpha


def test_default_cpu_parallelism_sets_n_jobs_when_supported():
    assert _with_default_cpu_parallelism(_ParallelEstimator, {"alpha": 2.0}) == {
        "alpha": 2.0,
        "n_jobs": -1,
    }


def test_default_cpu_parallelism_respects_explicit_n_jobs():
    assert _with_default_cpu_parallelism(
        _ParallelEstimator, {"alpha": 2.0, "n_jobs": 4}
    ) == {
        "alpha": 2.0,
        "n_jobs": 4,
    }


def test_default_cpu_parallelism_ignores_estimators_without_n_jobs():
    assert _with_default_cpu_parallelism(_SerialEstimator, {"alpha": 2.0}) == {
        "alpha": 2.0,
    }
