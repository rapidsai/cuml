#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
from numba import cuda as nbcuda
from pandas import Series as pdSeries

from cuml.common import CumlArray, input_to_cuml_array, input_to_host_array
from cuml.internals.input_utils import convert_dtype, input_to_cupy_array
from cuml.manifold import umap

###############################################################################
#                                    Parameters                               #
###############################################################################


test_dtypes_all = [
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

test_dtypes_acceptable = [np.float32, np.float64]

test_input_types = ["numpy", "numba", "cupy", "cudf", "pandas", "cuml"]

test_num_rows = [1, 100]
test_num_cols = [1, 100]


###############################################################################
#                                    Tests                                    #
###############################################################################


@pytest.mark.parametrize("dtype", test_dtypes_acceptable)
@pytest.mark.parametrize("input_type", test_input_types)
@pytest.mark.parametrize("num_rows", test_num_rows)
@pytest.mark.parametrize("num_cols", test_num_cols)
@pytest.mark.parametrize("order", ["C", "F", "K"])
def test_input_to_cuml_array(dtype, input_type, num_rows, num_cols, order):
    input_data, real_data = get_input(
        input_type, num_rows, num_cols, dtype, order=order
    )

    if input_type == "cupy" and input_data is None:
        pytest.skip("cupy not installed")

    X, n_rows, n_cols, res_dtype = input_to_cuml_array(input_data, order=order)

    np.testing.assert_equal(X.to_output("numpy"), real_data)

    assert n_rows == num_rows == X.shape[0] == len(X)
    assert n_cols == num_cols == X.shape[1]
    assert dtype == res_dtype == X.dtype

    del input_data
    del real_data


@pytest.mark.parametrize("dtype", test_dtypes_acceptable)
@pytest.mark.parametrize("input_type", ["numba", "cupy"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("order_check", ["C", "F"])
def test_fail_on_order(dtype, input_type, order, order_check):
    # this is tested only for non cudf dataframe or numpy arrays
    # those are converted form order by their respective libraries
    input_data, real_data = get_input(input_type, 10, 10, dtype, order=order)

    if input_type == "cupy" and input_data is None:
        pytest.skip("cupy not installed")

    if order == order_check:
        input_to_cuml_array(input_data, fail_on_order=False, order=order)
    else:
        with pytest.raises(ValueError):
            input_to_cuml_array(
                input_data, fail_on_order=True, order=order_check
            )


@pytest.mark.parametrize("dtype", test_dtypes_acceptable)
@pytest.mark.parametrize("input_type", test_input_types)
@pytest.mark.parametrize("from_order", ["C", "F"])
@pytest.mark.parametrize("to_order", ["C", "F", "K"])
def test_convert_matrix_order_cuml_array(
    dtype, input_type, from_order, to_order
):
    input_data, real_data = get_input(
        input_type, 10, 10, dtype, order=from_order
    )

    if input_type in ["cudf", "pandas"]:
        from_order = "F"

    # conv_data = np.array(real_data, order=to_order, copy=True)
    if from_order == to_order or to_order == "K":
        conv_data, *_ = input_to_cuml_array(
            input_data, fail_on_order=False, order=to_order
        )
    else:
        # Warning is raised for non cudf dataframe or numpy arrays
        # those are converted form order by their respective libraries
        if input_type in ["numpy", "cupy", "numba"]:
            # with pytest.warns(UserWarning):
            # warning disabled due to using cuml logger, need to
            # adapt tests for that.
            conv_data, *_ = input_to_cuml_array(
                input_data, fail_on_order=False, order=to_order
            )
        else:
            conv_data, *_ = input_to_cuml_array(
                input_data, fail_on_order=False, order=to_order
            )

    if to_order == "K":
        if input_type in ["cudf", "pandas"]:
            assert conv_data.order == "F"
        else:
            assert conv_data.order == from_order
    else:
        assert conv_data.order == to_order
    np.testing.assert_equal(real_data, conv_data.to_output("numpy"))


@pytest.mark.parametrize("dtype", test_dtypes_acceptable)
@pytest.mark.parametrize("input_type", test_input_types)
@pytest.mark.parametrize("shape", [(1, 10), (10, 1)])
@pytest.mark.parametrize("from_order", ["C", "F"])
@pytest.mark.parametrize("to_order", ["C", "F", "K"])
def test_convert_vector_order_cuml_array(
    dtype, input_type, shape, from_order, to_order
):
    input_data, real_data = get_input(
        input_type, shape[0], shape[1], dtype, order=from_order
    )

    # conv_data = np.array(real_data, order=to_order, copy=True)
    conv_data, *_ = input_to_cuml_array(
        input_data, fail_on_order=False, order=to_order
    )

    np.testing.assert_equal(real_data, conv_data.to_output("numpy"))


@pytest.mark.parametrize("dtype", test_dtypes_acceptable)
@pytest.mark.parametrize("input_type", test_input_types)
@pytest.mark.parametrize("num_rows", test_num_rows)
@pytest.mark.parametrize("num_cols", test_num_cols)
@pytest.mark.parametrize("order", ["C", "F"])
def test_input_to_host_array(dtype, input_type, num_rows, num_cols, order):
    input_data, real_data = get_input(
        input_type, num_rows, num_cols, dtype, order=order
    )

    if input_type == "cupy" and input_data is None:
        pytest.skip("cupy not installed")

    X, n_rows, n_cols, out_dtype = input_to_host_array(input_data, order=order)

    np.testing.assert_equal(X, real_data)

    assert n_rows == num_rows
    assert n_cols == num_cols
    assert out_dtype == dtype

    del input_data
    del real_data


@pytest.mark.parametrize("dtype", test_dtypes_acceptable)
@pytest.mark.parametrize("input_type", ["numpy", "cupy"])
@pytest.mark.parametrize("order", ["C", "F", "K"])
def test_non_contiguous_input_to_host_array(dtype, input_type, order):
    input_data, real_data = get_input(input_type, 10, 8, dtype)
    input_data = input_data[:-3]
    real_data = real_data[:-3]

    res = input_to_host_array(input_data, order=order).array
    np.testing.assert_equal(real_data, res)
    if order == "F":
        assert res.flags.f_contiguous
    else:
        assert res.flags.c_contiguous


@pytest.mark.parametrize("dtype", test_dtypes_all)
@pytest.mark.parametrize("check_dtype", test_dtypes_all)
@pytest.mark.parametrize("input_type", test_input_types)
@pytest.mark.parametrize("order", ["C", "F"])
def test_dtype_check(dtype, check_dtype, input_type, order):

    if (
        dtype == np.float16 or check_dtype == np.float16
    ) and input_type != "numpy":
        pytest.xfail("float16 not yet supported by numba/cuDF")

    if dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if input_type in ["cudf", "pandas"]:
            pytest.xfail("unsigned int types not yet supported")

    input_data, real_data = get_input(input_type, 10, 10, dtype, order=order)

    if input_type == "cupy" and input_data is None:
        pytest.skip("cupy not installed")

    if dtype == check_dtype:
        _, _, _, got_dtype = input_to_cuml_array(
            input_data, check_dtype=check_dtype, order=order
        )
        assert got_dtype == check_dtype
    else:
        with pytest.raises(TypeError):
            _, _, _, got_dtype = input_to_cuml_array(
                input_data, check_dtype=check_dtype, order=order
            )


@pytest.mark.parametrize("num_rows", test_num_rows)
@pytest.mark.parametrize("num_cols", test_num_cols)
@pytest.mark.parametrize("to_dtype", test_dtypes_acceptable)
@pytest.mark.parametrize("from_dtype", test_dtypes_all)
@pytest.mark.parametrize("input_type", test_input_types)
@pytest.mark.parametrize("order", ["C", "F"])
def test_convert_input_dtype(
    from_dtype, to_dtype, input_type, num_rows, num_cols, order
):

    if from_dtype == np.float16 and input_type != "numpy":
        pytest.xfail("float16 not yet supported by numba/cuDF")

    if from_dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if input_type == "cudf":
            pytest.xfail(
                "unsigned int types not yet supported by \
                         cuDF"
            )

    input_data, real_data = get_input(
        input_type,
        num_rows,
        num_cols,
        from_dtype,
        out_dtype=to_dtype,
        order=order,
    )

    if input_type == "cupy" and input_data is None:
        pytest.skip("cupy not installed")

    converted_data = convert_dtype(input_data, to_dtype=to_dtype)

    if input_type == "numpy":
        np.testing.assert_equal(converted_data, real_data)
    elif input_type == "cudf":
        np.testing.assert_equal(converted_data.to_numpy(), real_data)
    elif input_type == "pandas":
        np.testing.assert_equal(converted_data.to_numpy(), real_data)
    else:
        np.testing.assert_equal(converted_data.copy_to_host(), real_data)

    # we cannot guarantee that with wrapped dataframes,
    # such as with cudf.pandas the returned pointer is the same
    if not hasattr(input_data, "_fsproxy_slow_type") and not hasattr(
        input_data, "_fsproxy_fast_type"
    ):
        if from_dtype == to_dtype:
            check_ptr(converted_data, input_data, input_type)


@pytest.mark.parametrize("dtype", test_dtypes_acceptable)
@pytest.mark.parametrize("input_type", ["numpy", "cupy"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("force_contiguous", [True, False])
def test_non_contiguous_to_contiguous_input(
    dtype, input_type, order, contiguous, force_contiguous
):
    input_data, real_data = get_input(input_type, 10, 8, dtype, order=order)

    if not contiguous:
        if order == "F":
            data_view = input_data[:-3]
            real_data = real_data[:-3]
        else:
            data_view = input_data[:, :-3]
            real_data = real_data[:, :-3]

    else:
        data_view = input_data

    cumlary, *_ = input_to_cuml_array(
        data_view, force_contiguous=force_contiguous
    )

    if force_contiguous:
        assert cumlary.is_contiguous

    np.testing.assert_equal(real_data, cumlary.to_output("numpy"))


@pytest.mark.parametrize("input_type", ["cudf", "pandas"])
@pytest.mark.parametrize("num_rows", test_num_rows)
@pytest.mark.parametrize("num_cols", test_num_cols)
@pytest.mark.parametrize("order", ["C", "F"])
def test_indexed_inputs(input_type, num_rows, num_cols, order):
    if num_cols == 1:
        input_type += "-series"

    index = np.arange(num_rows, 2 * num_rows)

    input_data, real_data = get_input(
        input_type, num_rows, num_cols, np.float32, index=index
    )

    X, n_rows, n_cols, res_dtype = input_to_cuml_array(input_data, order=order)

    # testing the index in the cuml array
    np.testing.assert_equal(X.index.to_numpy(), index)

    # testing the index in the converted outputs
    cudf_output = X.to_output("cudf")
    np.testing.assert_equal(cudf_output.index.to_numpy(), index)

    pandas_output = X.to_output("pandas")
    np.testing.assert_equal(pandas_output.index.to_numpy(), index)


###############################################################################
#                           Utility Functions                                 #
###############################################################################


def check_numpy_order(ary, order):
    if order == "F":
        return ary.flags.f_contiguous
    else:
        return ary.flags.c_contiguous


def check_ptr(a, b, input_type):
    if input_type == "cudf":
        for col_a, col_b in zip(a._columns, b._columns, strict=True):
            # get_ptr could spill the buffer data, but possibly OK
            # if this is only used for testing
            assert col_a.base_data.get_ptr(
                mode="read"
            ) == col_b.base_data.get_ptr(mode="read")
    else:

        def get_ptr(x):
            try:
                return x.__cuda_array_interface__["data"][0]
            except AttributeError:
                return x.__array_interface__["data"][0]

        if input_type == "pandas":
            a = a.values
            b = b.values

        assert get_ptr(a) == get_ptr(b)


def get_input(
    type, nrows, ncols, dtype, order="C", out_dtype=False, index=None
):
    rand_mat = cp.random.rand(nrows, ncols) * 10
    rand_mat = cp.array(rand_mat, dtype=dtype, order=order)

    if type == "numpy":
        result = np.array(cp.asnumpy(rand_mat), order=order)

    if type == "cupy":
        result = rand_mat

    if type == "numba":
        result = nbcuda.as_cuda_array(rand_mat)

    if type == "cudf":
        result = cudf.DataFrame(rand_mat, index=index)

    if type == "cudf-series":
        result = cudf.Series(rand_mat.reshape(nrows), index=index)

    if type == "pandas":
        result = pd.DataFrame(cp.asnumpy(rand_mat), index=index)

    if type == "pandas-series":
        result = pdSeries(
            cp.asnumpy(rand_mat).reshape(
                nrows,
            ),
            index=index,
        )

    if type == "cuml":
        result = CumlArray(data=rand_mat)

    if out_dtype:
        return result, np.array(
            cp.asnumpy(rand_mat).astype(out_dtype), order=order
        )
    else:
        return result, np.array(cp.asnumpy(rand_mat), order=order)


def test_tocupy_missing_values_handling():
    df = cudf.DataFrame(data=[[7, 2, 3], [4, 5, 6], [10, 5, 9]])
    array, n_rows, n_cols, dtype = input_to_cupy_array(df, fail_on_null=False)
    assert isinstance(array, cp.ndarray)
    assert str(array.dtype) == "int64"

    df = cudf.DataFrame(data=[[7, 2, 3], [4, None, 6], [10, 5, 9]])
    array, n_rows, n_cols, dtype = input_to_cupy_array(df, fail_on_null=False)
    assert isinstance(array, cp.ndarray)
    assert str(array.dtype) == "float64"
    assert cp.isnan(array[1, 1])

    df = cudf.Series(data=[7, None, 3])
    array, n_rows, n_cols, dtype = input_to_cupy_array(df, fail_on_null=False)
    assert str(array.dtype) == "float64"
    assert cp.isnan(array[1])

    # cudf.pandas now mimics pandas better for handling None, so we don't
    # need to fail and raise this error when cudf.pandas is active.
    if not cudf_pandas_active:
        with pytest.raises(ValueError):
            df = cudf.Series(data=[7, None, 3])
            array, n_rows, n_cols, dtype = input_to_cupy_array(
                df, fail_on_null=True
            )


@pytest.mark.cudf_pandas
def test_numpy_output():
    # Check that a Numpy array is used as output when a cudf.pandas wrapped
    # Numpy array is passed in.
    # Non regression test for issue #5784
    df = pd.DataFrame({"a": range(5), "b": range(5)})
    X = df.values

    reducer = umap.UMAP()

    # Check that this is a cudf.pandas wrapped array
    assert hasattr(X, "_fsproxy_fast_type")
    assert isinstance(reducer.fit_transform(X), np.ndarray)
