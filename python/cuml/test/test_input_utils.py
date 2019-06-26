#
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

import cudf
import numpy as np

from numba import cuda
from copy import deepcopy

from cuml.utils import input_to_dev_array

from cuml.utils.input_utils import convert_dtype

test_dtypes_all = [
    np.float16, np.float32, np.float64,
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64
]

test_dtypes_acceptable = [
    np.float32, np.float64
]

test_input_types = [
    'numpy', 'numba', 'cupy', 'dataframe'
]

test_num_rows = [1, 10, 8000]
test_num_cols = [1, 10, 8000]


@pytest.mark.parametrize('dtype', test_dtypes_acceptable)
@pytest.mark.parametrize('input_type', test_input_types)
@pytest.mark.parametrize('num_rows', test_num_rows)
@pytest.mark.parametrize('num_cols', test_num_cols)
def test_input_to_dev_array(dtype, input_type, num_rows, num_cols):
    input_data, real_data = get_input(input_type, num_rows, num_cols, dtype)

    if input_data is None:
        pytest.skip('cupy not installed')

    X, X_ptr, n_rows, n_cols, dtype = input_to_dev_array(input_data)

    np.testing.assert_equal(X.copy_to_host(), real_data)

    assert n_rows == num_rows
    assert n_cols == num_cols
    assert dtype == dtype

    del input_data
    del real_data


@pytest.mark.parametrize('dtype', test_dtypes_all)
@pytest.mark.parametrize('check_dtype', test_dtypes_all)
@pytest.mark.parametrize('input_type', test_input_types)
def test_dtype_check(dtype, check_dtype, input_type):

    if (dtype == np.float16 or check_dtype == np.float16)\
            and input_type != 'numpy':
        pytest.xfail("float16 not yet supported by numba/cuDF.from_gpu_matrix")

    input_data, real_data = get_input(input_type, 10, 10, dtype)

    if input_data is None:
        pytest.skip('cupy not installed')

    if dtype == check_dtype:
        _, _, _, _, got_dtype = \
            input_to_dev_array(input_data, check_dtype=check_dtype)
        assert got_dtype == check_dtype
    else:
        with pytest.raises(TypeError):
            _, _, _, _, got_dtype = \
                input_to_dev_array(input_data, check_dtype=check_dtype)


@pytest.mark.parametrize('num_rows', [1, 100])
@pytest.mark.parametrize('num_cols', [1, 100])
@pytest.mark.parametrize('to_dtype', test_dtypes_acceptable)
@pytest.mark.parametrize('from_dtype', test_dtypes_all)
@pytest.mark.parametrize('input_type', test_input_types)
def test_convert_inputs(from_dtype, to_dtype, input_type, num_rows, num_cols):

    if from_dtype == np.float16 and input_type != 'numpy':
        pytest.xfail("float16 not yet supported by numba/cuDF.from_gpu_matrix")

    input_data, real_data = get_input(input_type, num_rows, num_cols,
                                      from_dtype, out_dtype=to_dtype)

    if input_data is None:
        pytest.skip('cupy not installed')

    converted_data = convert_dtype(input_data, to_dtype=to_dtype)

    if input_type == 'numpy':
        np.testing.assert_equal(converted_data, real_data)
    elif input_type != 'dataframe':
        np.testing.assert_equal(converted_data.copy_to_host(), real_data)
    else:
        np.testing.assert_equal(converted_data.as_matrix(), real_data)


def get_input(type, nrows, ncols, dtype, out_dtype=False):
    try:
        import cupy as cp
        rand_mat = (cp.random.rand(nrows, ncols)*10).astype(dtype)

        if type == 'numpy':
            result = cp.asnumpy(rand_mat)

        if type == 'cupy':
            result = rand_mat

        if type == 'numba':
            result = cuda.as_cuda_array(rand_mat)

        if type == 'dataframe':
            X_df = cudf.DataFrame()
            result = X_df.from_gpu_matrix(cuda.as_cuda_array(rand_mat))

        if out_dtype:
            return result, cp.asnumpy(rand_mat).astype(out_dtype)
        else:
            return result, cp.asnumpy(rand_mat)

    except ImportError:
        rand_mat = (np.random.rand(nrows, ncols)*10).astype(dtype)

        if type == 'numpy':
            result = deepcopy(rand_mat)

        if type == 'cupy':
            result = None

        if type == 'numba':
            result = cuda.to_device(rand_mat)

        if type == 'dataframe':
            X_df = cudf.DataFrame()
            result = X_df.from_gpu_matrix(cuda.to_device(rand_mat))

        if out_dtype:
            return result, rand_mat.astype(out_dtype)
        else:
            return result, rand_mat
