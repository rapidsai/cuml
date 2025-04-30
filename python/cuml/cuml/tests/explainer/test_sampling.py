# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
import cudf
import cupy as cp
import numpy as np
import pandas as pd
import pytest
from cudf.pandas import LOADED as cudf_pandas_active
from numba import cuda

from cuml.explainer.sampling import kmeans_sampling


@pytest.mark.parametrize(
    "input_type",
    ["cudf-df", "cudf-series", "pandas-df", "pandas-series", "cupy", "numpy"],
)
def test_kmeans_input(input_type):
    X = cp.array(
        [[0, 10], [1, 24], [0, 52], [0, 48.0], [0.2, 23], [1, 24], [1, 23]]
    )
    if input_type == "cudf-df":
        X = cudf.DataFrame(X)
    elif input_type == "cudf-series":
        X = cudf.Series(X[:, 1])
    elif input_type == "numba":
        X = cuda.as_cuda_array(X)
    elif input_type == "pandas-df":
        X = pd.DataFrame(cp.asnumpy(X))
    elif input_type == "pandas-series":
        X = pd.Series(cp.asnumpy(X[:, 1]))
    elif input_type == "numpy":
        X = cp.asnumpy(X)

    summary = kmeans_sampling(X, k=2, detailed=True)

    if input_type == "cudf-df":
        cp.testing.assert_array_equal(
            summary[0].values, [[1.0, 23.0], [0.0, 52.0]]
        )
        assert isinstance(summary[0], cudf.DataFrame)
    elif input_type == "pandas-df":
        cp.testing.assert_array_equal(
            summary[0].values, [[1.0, 23.0], [0.0, 52.0]]
        )
        assert isinstance(summary[0], pd.DataFrame)
    elif input_type == "numpy":
        cp.testing.assert_array_equal(summary[0], [[1.0, 23.0], [0.0, 52.0]])
        assert isinstance(summary[0], np.ndarray)
    elif input_type == "cudf-series":
        cp.testing.assert_array_equal(summary[0].values.tolist(), [23.0, 52.0])
        assert isinstance(summary[0], cudf.Series)
    elif input_type == "pandas-series" and not cudf_pandas_active:
        cp.testing.assert_array_equal(
            summary[0].to_numpy().flatten(), [23.0, 52.0]
        )
        assert isinstance(summary[0], pd.Series)
    elif input_type == "numba":
        cp.testing.assert_array_equal(
            cp.array(summary[0]).tolist(), [[1.0, 23.0], [0.0, 52.0]]
        )
        assert isinstance(summary[0], cuda.devicearray.DeviceNDArray)
    elif input_type == "cupy":
        cp.testing.assert_array_equal(
            summary[0].tolist(), [[1.0, 23.0], [0.0, 52.0]]
        )
        assert isinstance(summary[0], cp.ndarray)
