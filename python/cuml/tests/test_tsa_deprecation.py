# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cuml
from cuml.tsa import ARIMA, ExponentialSmoothing
from cuml.tsa.auto_arima import AutoARIMA


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
        match=(
            "was deprecated in version 26\\.08 and will be removed "
            "in version 26\\.12"
        ),
    ):
        estimator(*args)
