# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from cuml.holtwinters.holtwinters import HoltWinters
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pytest
from sklearn.metrics import r2_score

airpassengers = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
                 115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
                 145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
                 171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
                 196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
                 204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
                 242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
                 284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
                 315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
                 340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337]


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


@pytest.mark.parametrize('seasonal', ['ADDITIVE', 'MULTIPLICATIVE'])
@pytest.mark.parametrize('h', [12, 24])
@pytest.mark.parametrize('datatype', [np.float64])
def test_holtwinters(seasonal, h, datatype):
    global airpassengers
    airpassengers = np.asarray(airpassengers, dtype=datatype)
    train = airpassengers[:-h]
    test = airpassengers[-h:]

    cu_hw = HoltWinters(1, 12, seasonal)
    cu_hw.fit(airpassengers)

    sm_hw = ExponentialSmoothing(train, seasonal=seasonal.lower(),
                                 seasonal_periods=12)
    sm_hw = sm_hw.fit()

    cu_pred = cu_hw.predict(1, h)
    sm_pred = sm_hw.forecast(h)

    cu_r2 = r2_score(cu_pred, test)
    sm_r2 = r2_score(sm_pred, test)

    assert (cu_r2 >= sm_r2) or (abs(cu_r2 - sm_r2) < 1e-2)
