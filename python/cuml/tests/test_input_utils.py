#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cudf
import cupy as cp
import numba.cuda
import numpy as np
import pandas as pd
import pytest

from cuml.internals.array import CumlArray
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
#                           Utility Functions                                 #
###############################################################################


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
        result = numba.cuda.as_cuda_array(rand_mat)

    if type == "cudf":
        result = cudf.DataFrame(rand_mat, index=index)

    if type == "cudf-series":
        result = cudf.Series(rand_mat.reshape(nrows), index=index)

    if type == "pandas":
        result = pd.DataFrame(cp.asnumpy(rand_mat), index=index)

    if type == "pandas-series":
        result = pd.Series(
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

    X = CumlArray.from_input(input_data, order=order)

    np.testing.assert_equal(X.to_output("numpy"), real_data)

    assert X.shape[0] == num_rows
    assert X.shape[1] == num_cols
    assert X.dtype == dtype


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
        conv_data = CumlArray.from_input(input_data, order=to_order)
    else:
        # Warning is raised for non cudf dataframe or numpy arrays
        # those are converted form order by their respective libraries
        if input_type in ["numpy", "cupy", "numba"]:
            # with pytest.warns(UserWarning):
            # warning disabled due to using cuml logger, need to
            # adapt tests for that.
            conv_data = CumlArray.from_input(input_data, order=to_order)
        else:
            conv_data = CumlArray.from_input(input_data, order=to_order)

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
    conv_data = CumlArray.from_input(input_data, order=to_order)

    np.testing.assert_equal(real_data, conv_data.to_output("numpy"))


@pytest.mark.parametrize("dtype", test_dtypes_all)
@pytest.mark.parametrize("check_dtype", test_dtypes_all)
@pytest.mark.parametrize("input_type", test_input_types)
@pytest.mark.parametrize("order", ["C", "F"])
def test_dtype_check(dtype, check_dtype, input_type, order):
    if (
        dtype == np.float16 or check_dtype == np.float16
    ) and input_type != "numpy":
        pytest.skip("float16 not yet supported by numba/cuDF")

    if dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if input_type in ["cudf", "pandas"]:
            pytest.skip("unsigned int types not yet supported")

    input_data, real_data = get_input(input_type, 10, 10, dtype, order=order)

    if dtype == check_dtype:
        array = CumlArray.from_input(
            input_data, check_dtype=check_dtype, order=order
        )
        assert array.dtype == check_dtype
    else:
        with pytest.raises(TypeError):
            CumlArray.from_input(
                input_data, check_dtype=check_dtype, order=order
            )


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

    cumlary = CumlArray.from_input(
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

    X = CumlArray.from_input(input_data, order=order)

    # testing the index in the cuml array
    np.testing.assert_equal(X.index.to_numpy(), index)

    # testing the index in the converted outputs
    cudf_output = X.to_output("cudf")
    np.testing.assert_equal(cudf_output.index.to_numpy(), index)

    pandas_output = X.to_output("pandas")
    np.testing.assert_equal(pandas_output.index.to_numpy(), index)


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
