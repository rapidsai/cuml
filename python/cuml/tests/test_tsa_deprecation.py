# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import pytest

import cuml
from cuml.internals import run_in_internal_context
from cuml.tsa import ARIMA, ExponentialSmoothing
from cuml.tsa.auto_arima import AutoARIMA
from cuml.tsa.seasonality import seas_test
from cuml.tsa.stationarity import kpss_test

TSA_DEPRECATION_PATTERN = (
    "along with the entire `cuml\\.tsa` module, was deprecated in version "
    "26\\.08 and will be removed in version 26\\.12"
)


@pytest.mark.parametrize(
    "estimator, args",
    [
        (ARIMA, (np.arange(12, dtype=np.float64),)),
        (AutoARIMA, (np.arange(12, dtype=np.float64),)),
        (ExponentialSmoothing, (np.arange(12, dtype=np.float64),)),
        (cuml.ARIMA, (np.arange(12, dtype=np.float64),)),
        (cuml.AutoARIMA, (np.arange(12, dtype=np.float64),)),
        (cuml.ExponentialSmoothing, (np.arange(12, dtype=np.float64),)),
    ],
)
def test_tsa_estimators_warn_on_construction(estimator, args):
    with pytest.warns(
        FutureWarning,
        match=TSA_DEPRECATION_PATTERN,
    ):
        estimator(*args)


@pytest.mark.parametrize(
    "func, args",
    [
        (
            seas_test,
            (np.arange(12, dtype=np.float64).reshape(-1, 1), 4),
        ),
        (
            kpss_test,
            (np.arange(12, dtype=np.float64).reshape(-1, 1),),
        ),
    ],
)
def test_tsa_functions_warn_on_call(func, args):
    with pytest.warns(FutureWarning, match=TSA_DEPRECATION_PATTERN):
        func(*args)


@run_in_internal_context
def _call_tsa_function(func, args):
    return func(*args)


@pytest.mark.parametrize(
    "func, args",
    [
        (
            seas_test,
            (np.arange(12, dtype=np.float64).reshape(-1, 1), 4),
        ),
        (
            kpss_test,
            (np.arange(12, dtype=np.float64).reshape(-1, 1),),
        ),
    ],
)
def test_tsa_functions_do_not_warn_on_internal_calls(func, args):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning)
        _call_tsa_function(func, args)
