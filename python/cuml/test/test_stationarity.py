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
#

import pytest
import numpy as np

from cuml.tsa.stationarity import stationarity


def array_eq(ref, actual):
    success = (sum(1 if ref[i] != actual[i]
                   else 0 for i in range(len(ref))) == 0)
    message = "OK" if success else "Expected: {} ; Got: {}".format(ref, actual)
    return success, message


@pytest.mark.parametrize('precision', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['numpy'])
def test_stationarity(precision, input_type):
    """Test the kpss stationarity check.
    Note: this test is intended to test the Python wrapper.
    Another more exhaustive test is part of the C++ unit tests.
    """
    inc_rates = [-0.7, 0.0, 0.5]
    offsets = [-0.3, 0.5, 0.0]
    d_ref = [1, 0, 1]
    num_samples = 200

    xs = np.linspace(0, 1, num_samples)
    np.random.seed(13)
    noise = np.random.normal(scale=0.1, size=num_samples)

    np_df = np.zeros((num_samples, len(d_ref)), order="F", dtype=precision)
    for i in range(len(d_ref)):
        np_df[:, i] = xs[:] * inc_rates[i] + offsets[i] + noise[:]

    # Numpy is the only tested input type at the moment
    if input_type == 'numpy':
        df = np_df

    d_actual = stationarity(df)

    success, message = array_eq(d_ref, d_actual)

    assert success, message
