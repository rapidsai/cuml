# Copyright (c) 2018, NVIDIA CORPORATION.
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
import cudf
import numpy as np

from numba import cuda
from math import sqrt
from cuml import GMM


def np_to_dataframe(df):
    pdf = cudf.DataFrame()
    for c in range(df.shape[1]):
        pdf[c] = df[:, c]
    return pdf


@pytest.mark.parametrize('precision', ['single', 'double'])
def test_linear_kalman_filter_base(precision):

    f = GMM(nCl=2, nDim=3, nObs=100, n_iter=5, precision=precision)

    if precision == 'single':
        dt = np.float32
    else:
        dt = np.float64

    f.x = cuda.to_device(np.array([[0], [1]], dtype=dt))

    for i in range(n_iter):
        f.step(cuda.to_device(np.array([z], dtype=dt)))
        x = f.x.copy_to_host()
        rmse_x = rmse_x + ((x[0] - i)**2)
        rmse_v = rmse_v + ((x[1] - 1)**2)

    assert sqrt(rmse_x / n) < 0.1
    assert sqrt(rmse_v / n) < 0.1


@pytest.mark.parametrize('dim_x', [2, 10, 25])
@pytest.mark.parametrize('dim_z', [1, 2, 10, 25])
@pytest.mark.parametrize('precision', ['single', 'double'])
@pytest.mark.parametrize('input_type', ['numpy', 'cudf'])
def test_linear_kalman_filter(precision, dim_x, dim_z, input_type):
    f = GMM(dim_x=dim_x, dim_z=dim_z, precision=precision)

    if precision == 'single':
        dt = np.float32
    else:
        dt = np.float64

    if input_type == 'numpy':

        f.x = np.zeros((dim_x, 1), dtype=dt)

        f.F = np.eye(dim_x, dtype=dt)

        h = np.zeros((dim_x, dim_z), dtype=dt)
        h[0] = 1

        f.H = h
        f.P = np.eye(dim_x, dtype=dt) * 1000
        f.R = np.eye(dim_z, dtype=dt) * 5.0

    else:

        f.x = np_to_dataframe(np.zeros((dim_x, 1), dtype=dt))

        tmp = np.eye(dim_x, dtype=dt, order='F')

        f.F = np_to_dataframe(tmp)

        h = np.zeros((dim_x, dim_z), dtype=dt, order='F')
        h[0] = 1

        f.H = np_to_dataframe(h)
        f.P = np_to_dataframe(np.eye(dim_x, dtype=dt, order='F') * 1000)
        f.R = np_to_dataframe(np.eye(dim_z, dtype=dt, order='F') * 5.0)

    rmse_x = 0
    rmse_v = 0

    n = 10

    for i in range(n):
        f.predict()

        z = i * np.ones(dim_z, dtype=dt)

        f.update(cuda.to_device(np.array(z, dtype=dt)))
        x = f.x.copy_to_host()
        rmse_x = rmse_x + ((x[0] - i)**2)
        rmse_v = rmse_v + ((x[1] - 1)**2)

    assert sqrt(rmse_x / n) < 0.1
    assert sqrt(rmse_v / n) == 1.0
