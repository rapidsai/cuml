#
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

from cuml.datasets import (
    make_arima,
    make_blobs,
    make_classification,
    make_regression,
)
import cuml
import pytest
from cuml.internals.safe_imports import cpu_only_import
from cuml.internals.safe_imports import gpu_only_import

cudf = gpu_only_import("cudf")
cp = gpu_only_import("cupy")
numba = gpu_only_import("numba")
np = cpu_only_import("numpy")


TEST_OUTPUT_TYPES = (
    (None, (cp.ndarray, cp.ndarray)),  # Default is cupy if None is used
    ("numpy", (np.ndarray, np.ndarray)),
    ("cupy", (cp.ndarray, cp.ndarray)),
    (
        "numba",
        (
            numba.cuda.devicearray.DeviceNDArrayBase,
            numba.cuda.devicearray.DeviceNDArrayBase,
        ),
    ),
    ("cudf", (cudf.DataFrame, cudf.Series)),
)

GENERATORS = (make_blobs, make_classification, make_regression)


@pytest.mark.parametrize("generator", GENERATORS)
@pytest.mark.parametrize("output_str,output_types", TEST_OUTPUT_TYPES)
def test_xy_output_type(generator, output_str, output_types):

    # Set the output type and ensure data of that type is generated
    with cuml.using_output_type(output_str):
        data = generator(n_samples=10, random_state=0)

    for data, type_ in zip(data, output_types):
        assert isinstance(data, type_)


@pytest.mark.parametrize("output_str,output_types", TEST_OUTPUT_TYPES)
def test_time_series_label_output_type(output_str, output_types):

    # Set the output type and ensure data of that type is generated
    with cuml.using_output_type(output_str):
        data = make_arima(n_obs=10, random_state=0)[0]

    assert isinstance(data, output_types[1])
