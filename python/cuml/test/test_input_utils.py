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

from cuml.utils import input_to_dev_array, input_to_host_array, has_cupy
from cuml.utils.input_utils import convert_dtype, check_numba_order

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

test_num_rows = [1, 100]
test_num_cols = [1, 100]


@pytest.mark.parametrize('dtype', test_dtypes_acceptable)
@pytest.mark.parametrize('input_type', test_input_types)
@pytest.mark.parametrize('num_rows', test_num_rows)
@pytest.mark.parametrize('num_cols', test_num_cols)
@pytest.mark.parametrize('order', ['C', 'F'])
def test_input_to_dev_array(dtype, input_type, num_rows, num_cols, order):
    input_data, real_data = get_input(input_type, num_rows, num_cols,
                                      dtype, order=order)

    if input_type == 'cupy' and input_data is None:
        pytest.skip('cupy not installed')

    X, X_ptr, n_rows, n_cols, dtype = input_to_dev_array(input_data,
                                                         order=order)

    np.testing.assert_equal(X.copy_to_host(), real_data)

    assert n_rows == num_rows
    assert n_cols == num_cols
    assert dtype == dtype

    del input_data
    del real_data


@pytest.mark.parametrize('dtype', test_dtypes_acceptable)
@pytest.mark.parametrize('input_type', test_input_types)
@pytest.mark.parametrize('num_rows', test_num_rows)
@pytest.mark.parametrize('num_cols', test_num_cols)
@pytest.mark.parametrize('order', ['C', 'F'])
def test_input_to_host_array(dtype, input_type, num_rows, num_cols, order):
    input_data, real_data = get_input(input_type, num_rows, num_cols, dtype,
                                      order=order)

    if input_type == 'cupy' and input_data is None:
        pytest.skip('cupy not installed')

    X, X_ptr, n_rows, n_cols, dtype = input_to_host_array(input_data,
                                                          order=order)

    np.testing.assert_equal(X, real_data)

    assert n_rows == num_rows
    assert n_cols == num_cols
    assert dtype == dtype

    del input_data
    del real_data


@pytest.mark.parametrize('dtype', test_dtypes_all)
@pytest.mark.parametrize('check_dtype', test_dtypes_all)
@pytest.mark.parametrize('input_type', test_input_types)
@pytest.mark.parametrize('order', ['C', 'F'])
def test_dtype_check(dtype, check_dtype, input_type, order):

    if (dtype == np.float16 or check_dtype == np.float16)\
            and input_type != 'numpy':
        pytest.xfail("float16 not yet supported by numba/cuDF.from_gpu_matrix")

    if dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if input_type == 'dataframe':
            pytest.xfail("unsigned int types not yet supported by \
                         cuDF")

    input_data, real_data = get_input(input_type, 10, 10, dtype, order=order)

    if input_type == 'cupy' and input_data is None:
        pytest.skip('cupy not installed')

    if dtype == check_dtype:
        _, _, _, _, got_dtype = \
            input_to_dev_array(input_data, check_dtype=check_dtype,
                               order=order)
        assert got_dtype == check_dtype
    else:
        with pytest.raises(TypeError):
            _, _, _, _, got_dtype = \
                input_to_dev_array(input_data, check_dtype=check_dtype,
                                   order=order)


@pytest.mark.parametrize('num_rows', test_num_rows)
@pytest.mark.parametrize('num_cols', test_num_cols)
@pytest.mark.parametrize('to_dtype', test_dtypes_acceptable)
@pytest.mark.parametrize('from_dtype', test_dtypes_all)
@pytest.mark.parametrize('input_type', test_input_types)
@pytest.mark.parametrize('order', ['C', 'F'])
def test_convert_input_dtype(from_dtype, to_dtype, input_type, num_rows,
                             num_cols, order):

    if from_dtype == np.float16 and input_type != 'numpy':
        pytest.xfail("float16 not yet supported by numba/cuDF.from_gpu_matrix")

    if from_dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if input_type == 'dataframe':
            pytest.xfail("unsigned int types not yet supported by \
                         cuDF")
        elif not has_cupy():
            pytest.xfail("unsigned int types not yet supported by \
                         cuDF and cuPy is not installed.")

    input_data, real_data = get_input(input_type, num_rows, num_cols,
                                      from_dtype, out_dtype=to_dtype,
                                      order=order)

    if input_type == 'cupy' and input_data is None:
        pytest.skip('cupy not installed')

    converted_data = convert_dtype(input_data, to_dtype=to_dtype)

    if input_type == 'numpy':
        np.testing.assert_equal(converted_data, real_data)
    elif input_type != 'dataframe':
        np.testing.assert_equal(converted_data.copy_to_host(), real_data)
    else:
        np.testing.assert_equal(converted_data.as_matrix(), real_data)


@pytest.mark.parametrize('dtype', test_dtypes_acceptable)
@pytest.mark.parametrize('input_type', ['numba', 'cupy'])
@pytest.mark.parametrize('order', ['C', 'F'])
@pytest.mark.parametrize('order_check', ['C', 'F'])
def test_fail_on_order(dtype, input_type, order, order_check):
    # this is tested only for non cudf dataframe or numpy arrays
    # those are converted form order by their respective libraries
    input_data, real_data = get_input(input_type, 10, 10, dtype, order=order)

    if input_type == 'cupy' and input_data is None:
        pytest.skip('cupy not installed')

    if order == order_check:
        _, _, _, _, _ = \
            input_to_dev_array(input_data, fail_on_order=False, order=order)
    else:
        with pytest.raises(ValueError):
            _, _, _, _, _ = \
                input_to_dev_array(input_data, fail_on_order=True,
                                   order=order_check)


@pytest.mark.parametrize('dtype', test_dtypes_acceptable)
@pytest.mark.parametrize('input_type', test_input_types)
@pytest.mark.parametrize('from_order', ['C', 'F'])
@pytest.mark.parametrize('to_order', ['C', 'F'])
def test_convert_order_dev_array(dtype, input_type, from_order, to_order):
    input_data, real_data = get_input(input_type, 10, 10, dtype,
                                      order=from_order)

    # conv_data = np.array(real_data, order=to_order, copy=True)
    if from_order == to_order:
        conv_data, _, _, _, _ = \
            input_to_dev_array(input_data, fail_on_order=False, order=to_order)
    else:
        # Warning is raised for non cudf dataframe or numpy arrays
        # those are converted form order by their respective libraries
        if input_type in ['cupy', 'numba']:
            with pytest.warns(UserWarning):
                conv_data, _, _, _, _ = \
                    input_to_dev_array(input_data, fail_on_order=False,
                                       order=to_order)
        else:
            conv_data, _, _, _, _ = \
                input_to_dev_array(input_data, fail_on_order=False,
                                   order=to_order)

    assert(check_numba_order(conv_data, to_order))
    np.testing.assert_equal(real_data, conv_data.copy_to_host())


def check_numpy_order(ary, order):
    if order == 'F':
        return ary.flags.f_contiguous
    else:
        return ary.flags.c_contiguous


def get_input(type, nrows, ncols, dtype, order='C', out_dtype=False):
    if has_cupy:
        import cupy as cp
        rand_mat = (cp.random.rand(nrows, ncols)*10)
        rand_mat = cp.array(rand_mat, order=order).astype(dtype)

        if type == 'numpy':
            result = np.array(cp.asnumpy(rand_mat), order=order)

        if type == 'cupy':
            result = rand_mat

        if type == 'numba':
            result = cuda.as_cuda_array(rand_mat)

        if type == 'dataframe':
            X_df = cudf.DataFrame()
            result = X_df.from_gpu_matrix(cuda.as_cuda_array(rand_mat))

        if out_dtype:
            return result, np.array(cp.asnumpy(rand_mat).astype(out_dtype),
                                    order=order)
        else:
            return result, np.array(cp.asnumpy(rand_mat), order=order)

    else:
        rand_mat = (np.random.rand(nrows, ncols)*10)
        rand_mat = np.array(rand_mat, order=order).astype(dtype)

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
