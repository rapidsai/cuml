# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

import cuml
import pytest


# Note: this test is not really strict, it only checks that the function
# supports the given parameters and returns an output in the correct form.
# The test doesn't guarantee the quality of the generated series


# Testing parameters
output_type = [
    (None, 100),   # Default is cupy if None is used
    ('numpy', 100),
    ('cupy', 100000),
    ('numba', 100000),
    ('cudf', 100)]

dtype = ['single', 'double']
n_obs = [50, 200]
random_state = [None, 1234]
order = [(3, 0, 0, 0, 0, 0, 0, 1),
         (0, 1, 2, 0, 0, 0, 0, 1),
         (1, 1, 1, 2, 1, 0, 12, 0)]


@pytest.mark.parametrize('dtype', dtype)
@pytest.mark.parametrize(
    'output_type,batch_size', output_type)
@pytest.mark.parametrize('n_obs', n_obs)
@pytest.mark.parametrize('random_state', random_state)
@pytest.mark.parametrize('order', order)
def test_make_arima(dtype, output_type, batch_size, n_obs, random_state,
                    order):
    p, d, q, P, D, Q, s, k = order

    with cuml.using_output_type(output_type):
        out = cuml.make_arima(batch_size, n_obs,
                              (p, d, q), (P, D, Q, s), k,
                              random_state=random_state,
                              dtype=dtype)

    assert out.shape == (n_obs, batch_size), "out shape mismatch"
