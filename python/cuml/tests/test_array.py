#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import gc
import operator
import pickle
from copy import deepcopy

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
from numba import cuda

from cuml.internals.array import (
    CumlArray,
    _determine_memory_order,
    _order_to_strides,
    array_to_memory_order,
)
from cuml.internals.mem_type import MemoryType
from cuml.internals.memory_utils import (
    _get_size_from_shape,
    determine_array_memtype,
)
from cuml.testing.strategies import (
    UNSUPPORTED_CUDF_DTYPES,
    create_cuml_array_input,
    cuml_array_dtypes,
    cuml_array_input_types,
    cuml_array_inputs,
    cuml_array_mem_types,
    cuml_array_orders,
    cuml_array_output_types,
    cuml_array_shapes,
)
from cuml.testing.utils import (
    normalized_shape,
    series_squeezed_shape,
    squeezed_shape,
    to_nparray,
)

test_input_types = ["numpy", "numba", "cupy", "series", None]

test_output_types = (
    "cupy",
    "numpy",
    "cudf",
    "pandas",
    "array",
    "numba",
    "dataframe",
    "series",
    "df_obj",
)

_OUTPUT_TYPES_MAPPING = {
    "cupy": cp.ndarray,
    "numpy": np.ndarray,
    "cudf": (cudf.DataFrame, cudf.Series),
    "pandas": (pd.DataFrame, pd.Series),
    "dataframe": (cudf.DataFrame, pd.DataFrame),
    "series": (cudf.Series, pd.Series),
}


def _multidimensional(shape):
    return len(squeezed_shape(normalized_shape(shape))) > 1


def _get_owner(curr):
    if isinstance(curr, CumlArray):
        return curr._owner
    elif isinstance(curr, cp.ndarray):
        return curr.data.mem._owner
    else:
        return None


def _assert_equal(array_like, cuml_array):
    """Check whether array-like data and cuml array data are equal."""
    assert cp.array_equal(
        cp.asarray(array_like),
        cuml_array.to_output("cupy"),
        equal_nan=True,
    )


def _assert_allclose(array_like, cuml_array, rtol=1e-5, atol=1e-8):
    """Check whether array-like data and cuml array data are approximately equal."""
    a = cp.asarray(array_like)
    b = cuml_array.to_output("cupy")
    assert cp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


@example(
    input_type="numpy",
    dtype=np.float32,
    shape=(10, 10),
    order="C",
    force_gc=False,
)
@given(
    input_type=cuml_array_input_types(),
    dtype=cuml_array_dtypes(),
    shape=cuml_array_shapes(),
    order=cuml_array_orders(),
    force_gc=st.booleans(),
)
@settings(deadline=None)
def test_array_init(input_type, dtype, shape, order, force_gc):
    input_array = create_cuml_array_input(input_type, dtype, shape, order)
    cuml_array = CumlArray(data=input_array)

    # Test basic array properties
    assert cuml_array.dtype == dtype

    if input_type == "series":
        assert cuml_array.shape == series_squeezed_shape(shape)
    else:
        assert cuml_array.shape == normalized_shape(shape)

    # Order is only well-defined (and preserved) for multidimensional arrays.
    md = isinstance(shape, tuple) and len([d for d in shape if d != 1]) > 1
    assert cuml_array.order == order if md else "C"

    # Check input array and array equality.
    _assert_equal(input_array, cuml_array)

    # Check that data is kept in memory even when the input_array reference
    # is deleted.
    input_array_copy = deepcopy(cp.asarray(input_array))
    del input_array
    if force_gc:
        gc.collect()

    _assert_equal(input_array_copy, cuml_array)


@example(
    data_type=bytes,
    dtype=np.float32,
    shape=(10, 10),
    order="C",
    mem_type="device",
)
@given(
    data_type=st.sampled_from([bytes, bytearray, memoryview]),
    dtype=cuml_array_dtypes(),
    shape=cuml_array_shapes(),
    order=cuml_array_orders(),
    mem_type=cuml_array_mem_types(),
)
@settings(deadline=None)
def test_array_init_from_bytes(data_type, dtype, shape, order, mem_type):
    dtype = np.dtype(dtype)
    values = bytes(_get_size_from_shape(shape, dtype)[0])

    # Convert to data_type to be tested if needed.
    if data_type is not bytes:
        values = data_type(values)

    array = CumlArray(
        values, dtype=dtype, shape=shape, order=order, mem_type=mem_type
    )

    assert array.order == order
    assert array.shape in (shape, (shape,))
    assert array.dtype == dtype

    array_copy = cp.zeros(shape, dtype=dtype)

    assert cp.all(cp.asarray(array_copy) == array_copy)


@example(inp=np.array([1, 2, 3]), indices=slice(1, 3))
@given(
    inp=cuml_array_inputs(),
    indices=st.slices(10),  # TODO: should be basic_indices() as shown below
    # indices=basic_indices((10, 10)),
)
@settings(deadline=None)
def test_get_set_item(inp, indices):
    ary = CumlArray(data=inp)

    # Assumption required due to limitation on step size for F-order.
    assume(ary.order != "F" or (indices.step in (None, 1)))

    # Check equality of array views.
    inp_view = inp[indices]

    # Must assume that resulting view must have at least one element to not
    # trigger UnownedMemory exception.
    assume(cp.isscalar(inp_view) or inp_view.size > 0)

    _assert_equal(inp_view, ary[indices])

    # Check equality after assigning to array slice.
    ary[indices] = inp.dtype.type(1.0)
    inp[indices] = inp.dtype.type(1.0)

    # We need to assume that inp is not a cudf.Series here, otherwise
    # ary.to_output("cupy") called by equal() will trigger a
    # CUDARuntimeError: cudaErrorInvalidDevice: invalid device ordinal
    # error.
    assume(not isinstance(inp, cudf.Series))

    _assert_equal(inp, ary)


@example(shape=(10, 10), dtype=np.float32, order="C", mem_type="device")
@given(
    shape=cuml_array_shapes(),
    dtype=cuml_array_dtypes(),
    order=cuml_array_orders(),
    mem_type=cuml_array_mem_types(),
)
@settings(deadline=None)
def test_create_empty(shape, dtype, order, mem_type):
    mem_type = MemoryType.from_str(mem_type)
    ary = CumlArray.empty(
        shape=shape, dtype=dtype, order=order, mem_type=mem_type
    )
    assert isinstance(ary.ptr, int)
    assert ary.shape == normalized_shape(shape)
    assert ary.dtype == np.dtype(dtype)


@example(shape=(10, 10), dtype=np.float32, order="C", mem_type="device")
@given(
    shape=cuml_array_shapes(),
    dtype=cuml_array_dtypes(),
    order=cuml_array_orders(),
    mem_type=cuml_array_mem_types(),
)
@settings(deadline=None)
def test_create_zeros(shape, dtype, order, mem_type):
    mem_type = MemoryType.from_str(mem_type)
    ary = CumlArray.zeros(
        shape=shape, dtype=dtype, order=order, mem_type=mem_type
    )
    test = mem_type.xpy.zeros(shape).astype(dtype)
    assert mem_type.xpy.all(test == mem_type.xpy.asarray(ary))


@example(shape=(10, 10), dtype=np.float32, order="C", mem_type="device")
@given(
    shape=cuml_array_shapes(),
    dtype=cuml_array_dtypes(),
    order=cuml_array_orders(),
    mem_type=cuml_array_mem_types(),
)
@settings(deadline=None)
def test_create_ones(shape, dtype, order, mem_type):
    mem_type = MemoryType.from_str(mem_type)
    ary = CumlArray.ones(
        shape=shape, dtype=dtype, order=order, mem_type=mem_type
    )
    test = mem_type.xpy.ones(shape).astype(dtype)
    assert mem_type.xpy.all(test == mem_type.xpy.asarray(ary))


@example(shape=(10, 10), dtype=np.float32, order="C", mem_type="device")
@given(
    shape=cuml_array_shapes(),
    dtype=cuml_array_dtypes(),
    order=cuml_array_orders(),
    mem_type=cuml_array_mem_types(),
)
@settings(deadline=None)
def test_create_full(shape, dtype, order, mem_type):
    mem_type = MemoryType.from_str(mem_type)
    value = mem_type.xpy.array([mem_type.xpy.random.randint(100)]).astype(
        dtype
    )
    ary = CumlArray.full(
        value=value[0],
        shape=shape,
        dtype=dtype,
        order=order,
        mem_type=mem_type,
    )
    test = mem_type.xpy.zeros(shape).astype(dtype) + value[0]
    assert mem_type.xpy.all(test == mem_type.xpy.asarray(ary))


def cudf_compatible_dtypes(dtype):
    return dtype not in UNSUPPORTED_CUDF_DTYPES


@example(inp=np.array([1, 2, 3]), output_type="cupy")
@given(
    inp=cuml_array_inputs(),
    output_type=cuml_array_output_types(),
)
@settings(deadline=None)
def test_output(inp, output_type):
    # Required assumptions for cudf outputs:
    if output_type in ("cudf", "dataframe", "series"):
        assume(inp.dtype not in UNSUPPORTED_CUDF_DTYPES)
    if output_type == "series":
        assume(not _multidimensional(inp.shape))

    # Generate CumlArray from input and perform conversion.
    arr = CumlArray(inp)
    res = arr.to_output(output_type)

    # Check output type
    if output_type == "numba":  # TODO: is this still needed?
        # using correct numba ndarray check
        assert cuda.devicearray.is_cuda_ndarray(res)
    elif output_type == "cudf":
        assert isinstance(
            res,
            cudf.DataFrame if _multidimensional(inp.shape) else cudf.Series,
        )
    elif output_type == "pandas":
        assert isinstance(
            res,
            pd.DataFrame if _multidimensional(inp.shape) else pd.Series,
        )
    else:
        assert isinstance(res, _OUTPUT_TYPES_MAPPING[output_type])

    def assert_data_equal_(res):
        # Check output data equality
        if isinstance(res, cudf.Series):
            # A simple equality check `assert cudf.Series(inp).equals(res)`
            # does not work for with multi-dimensional data.
            assert cudf.Series(np.ravel(inp)).equals(res)
        elif isinstance(res, pd.Series):
            assert pd.Series(np.ravel(inp)).equals(res)
        elif isinstance(res, cudf.DataFrame):
            # Assumption required because of:
            #   https://github.com/rapidsai/cudf/issues/12266
            assume(not np.isnan(res.to_numpy()).any())

            assert cudf.DataFrame(inp).equals(res)
        elif isinstance(res, pd.DataFrame):
            assert pd.DataFrame(inp).equals(res)
        else:
            assert np.array_equal(
                to_nparray(inp), to_nparray(res), equal_nan=True
            )

    assert_data_equal_(res)


@example(inp=np.array([1, 2, 3]), output_type="cupy")
@given(
    inp=cuml_array_inputs(),
    output_type=cuml_array_output_types(),
)
@settings(deadline=None)
def test_end_to_end_conversion_via_intermediate(inp, output_type):
    # This test requires a lot of assumptions in combination with cuDF
    # intermediates.

    # Assumptions required for cuDF limitations:
    assume(
        # Not all dtypes are supported by cuDF.
        not (
            output_type in ("cudf", "pandas", "dataframe", "series")
            and inp.dtype in UNSUPPORTED_CUDF_DTYPES
        )
    )
    assume(
        # Can't convert multidimensional arrays to a Series.
        not (output_type == "series" and len(inp.shape) > 1)
    )

    assume(
        # Cannot convert from DataFrame to CumlArray wihthout explicitly
        # specifying shape, dtype, and order.
        not (
            output_type == "dataframe"
            or (output_type == "cudf" and len(inp.shape) > 1)
            or (output_type == "pandas" and len(inp.shape) > 1)
        )
    )

    # First conversion:
    array = CumlArray(data=inp)
    _assert_equal(inp, array)

    # Second conversion via intermediate
    intermediate = array.to_output(output_type)

    # Cupy does not support masked arrays.
    cai = getattr(intermediate, "__cuda_array_interface__", dict())
    assume(cai.get("mask") is None)

    array2 = CumlArray(data=intermediate)
    _assert_equal(inp, array2)


@example(
    output_type="cupy",
    shape=(10, 10),
    dtype=np.float32,
    order="C",
    out_dtype=np.float32,
)
@given(
    output_type=cuml_array_output_types(),
    shape=cuml_array_shapes(),
    dtype=cuml_array_dtypes(),
    order=cuml_array_orders(),
    out_dtype=cuml_array_dtypes(),
)
@settings(deadline=None)
def test_output_dtype(output_type, shape, dtype, order, out_dtype):
    # Required assumptions for cudf outputs:
    if output_type in ("cudf", "dataframe", "series"):
        assume(dtype not in UNSUPPORTED_CUDF_DTYPES)
        assume(out_dtype not in UNSUPPORTED_CUDF_DTYPES)
    if output_type == "series":
        assume(not _multidimensional(shape))

    # Perform conversion
    inp = create_cuml_array_input("numpy", dtype, shape, order)
    ary = CumlArray(inp)
    res = ary.to_output(output_type=output_type, output_dtype=out_dtype)

    # Check output dtype
    if isinstance(res, (cudf.DataFrame, pd.DataFrame)):
        res.values.dtype is out_dtype
    else:
        res.dtype is out_dtype


@example(inp=np.array([1, 2, 3]))
@given(inp=cuml_array_inputs())
@settings(deadline=None)
def test_array_interface(inp):
    ary = CumlArray(inp)

    mem_type = determine_array_memtype(inp)
    if isinstance(inp, pd.Series):
        converted_inp = inp.to_numpy()
    elif isinstance(inp, cudf.Series):
        converted_inp = cp.asnumpy(inp.to_cupy())
    else:
        converted_inp = inp

    try:
        inp_ai = converted_inp.__cuda_array_interface__
    except AttributeError:
        inp_ai = converted_inp.__array_interface__

    ary_ai = ary._array_interface

    # Check Array Interface equality.
    assert inp_ai["shape"] == ary_ai["shape"]
    assert inp_ai["typestr"] == ary_ai["typestr"]
    if (
        not isinstance(inp, (pd.Series, cudf.Series))
        and determine_array_memtype(inp) is ary.mem_type
    ):
        assert inp_ai["data"] == ary_ai["data"]
    # Mismatch for one-dimensional arrays:
    if inp_ai.get("strides", None) is not None:
        assert inp_ai["strides"] == ary_ai["strides"]

    # Check equality
    inp_arr = mem_type.xpy.asarray(converted_inp)
    out_arr = mem_type.xpy.asarray(ary)
    assert mem_type.xpy.all(inp_arr == out_arr) or mem_type.xpy.all(
        mem_type.xpy.isnan(inp_arr) == mem_type.xpy.isnan(out_arr)
    )


@example(
    inp=np.array([1, 2, 3]),
    to_serialize_mem_type="device",
    from_serialize_mem_type="device",
)
@example(
    inp=cp.zeros((2, 1), dtype=cp.float16, order="F"),
    to_serialize_mem_type="host",
    from_serialize_mem_type="host",
)
@given(
    inp=cuml_array_inputs(),
    to_serialize_mem_type=cuml_array_mem_types(),
    from_serialize_mem_type=cuml_array_mem_types(),
)
@settings(deadline=None)
def test_serialize(inp, to_serialize_mem_type, from_serialize_mem_type):
    ary = CumlArray(data=inp)
    header, frames = ary.serialize(mem_type=to_serialize_mem_type)

    ary2 = CumlArray.deserialize(
        header, frames, mem_type=from_serialize_mem_type
    )

    _assert_equal(inp, ary2)

    assert ary._array_interface["shape"] == ary2._array_interface["shape"]
    assert ary._array_interface["typestr"] == ary2._array_interface["typestr"]
    assert ary2.mem_type is MemoryType.from_str(from_serialize_mem_type)

    # Assert that if an array was C or F contiguous before, it still is after
    # Note that for arrays that are both C & F contiguous CumlArray will
    # store `order="C"`, but that doesn't mean that it's _not_ F contiguous
    # as well.
    is_c_contig = array_to_memory_order(inp, default="C") == "C"
    is_f_contig = array_to_memory_order(inp, default="F") == "F"
    if is_c_contig and not is_f_contig:
        assert ary2.order == "C"
    elif is_f_contig and not is_c_contig:
        assert ary2.order == "F"
    else:
        assert ary2.order in ("C", "F")

    if isinstance(inp, (cudf.Series, pd.Series)):
        assert ary.order == ary2.order


@pytest.mark.parametrize("protocol", [4, 5])
@example(inp=np.array([1, 2, 3]))
@given(inp=cuml_array_inputs())
@settings(deadline=None)
def test_pickle(protocol, inp):
    # Generate CumlArray
    ary = CumlArray(data=inp)

    # Prepare keyword arguments.
    dumps_kwargs = {"protocol": protocol}
    loads_kwargs = {}
    f = []
    len_f = 0
    if protocol >= 5:
        dumps_kwargs["buffer_callback"] = f.append
        loads_kwargs["buffers"] = f
        len_f = 1

    a = pickle.dumps(ary, **dumps_kwargs)
    b = pickle.loads(a, **loads_kwargs)

    assert ary._array_interface["shape"] == b._array_interface["shape"]
    # Restricting the strides check due to
    # https://github.com/cupy/cupy/issues/5897
    if not (len(ary.shape) > 1 and (ary.shape[0] == 1 or ary.shape[-1] == 1)):
        assert ary._array_interface["strides"] == b._array_interface["strides"]
    assert ary._array_interface["typestr"] == b._array_interface["typestr"]
    # Check equality
    assert len(f) == len_f
    _assert_equal(inp, b)

    if isinstance(inp, (cudf.Series, pd.Series)):
        # skipping one dimensional ary order test
        assert ary.order == b.order


@example(inp=np.array([1, 2, 3]))
@given(inp=cuml_array_inputs())
@settings(deadline=None)
def test_deepcopy(inp):
    # Generate CumlArray
    ary = CumlArray(data=inp)

    # Perform deepcopy
    b = deepcopy(ary)

    # Check equality
    _assert_equal(inp, b)
    assert ary.ptr != b.ptr

    assert ary._array_interface["shape"] == b._array_interface["shape"]
    # Restricting the strides check due to
    # https://github.com/cupy/cupy/issues/5897
    if not (len(ary.shape) > 1 and (ary.shape[0] == 1 or ary.shape[-1] == 1)):
        assert ary._array_interface["strides"] == b._array_interface["strides"]
    assert ary._array_interface["typestr"] == b._array_interface["typestr"]

    if isinstance(inp, (cudf.Series, pd.Series)):
        # skipping one dimensional ary order test
        assert ary.order == b.order


@pytest.mark.parametrize("operation", [operator.add, operator.sub])
@example(a=np.array([1, 2, 3]))
@given(a=cuml_array_inputs())
@settings(deadline=None)
def test_cumlary_binops(operation, a):
    b = deepcopy(a)

    ary_a = CumlArray(a)
    ary_b = CumlArray(b)

    c = operation(a, b)
    ary_c = operation(ary_a, ary_b)

    _assert_allclose(c, ary_c, rtol=1e-5, atol=1e-35)


@pytest.mark.parametrize("order", ["F", "C"])
@example(mem_type="device")
@given(mem_type=cuml_array_mem_types())
@settings(deadline=None)
def test_sliced_array_owner(order, mem_type):
    """
    When slicing a CumlArray, a new object can be created created which
    previously had an incorrect owner. This was due to the requirement by
    `cudf.core.Buffer` that all data be in "u1" form. CumlArray would satisfy
    this requirement by calling
    `cp.asarray(data).ravel(order='A').view('u1')`. If the slice is not
    contiguous, this would create an intermediate object with no references
    that would be cleaned up by GC causing an error when using the memory
    """
    mem_type = MemoryType.from_str(mem_type)
    xpy = mem_type.xpy

    # Create 2 copies of a random array
    random_arr = xpy.array(
        xpy.random.random((500, 4)), dtype=np.float32, order=order
    )
    arr = xpy.array(random_arr, copy=True)
    cuml_array = CumlArray(random_arr)

    # Make sure we have 2 pieces of data
    if cuml_array.mem_type.is_device_accessible:
        assert arr.data.ptr != cuml_array.ptr
    else:
        assert arr.__array_interface__["data"][0] != cuml_array.ptr

    # Since these are C arrays, slice off the first column to ensure they are
    # non-contiguous
    cuml_slice = cuml_array[1:, 1:]
    arr_slice = arr[1:, 1:]

    # Delete the input object just to be sure
    del random_arr

    # Make sure to cleanup any objects. Forces deletion of intermediate owner
    # object
    gc.collect()

    # Calling `to_output` forces use of the pointer. This can fail with a cuda
    # error on cudaErrorInvalidDevice in CUDA > 12.0 (unclear why it changed)
    assert xpy.all(
        cuml_slice.to_output("array", output_mem_type=mem_type) == arr_slice
    )


@example(input_type="numpy", dtype=np.float32, shape=(10, 10), order="C")
@given(
    input_type=cuml_array_input_types(),
    dtype=cuml_array_dtypes(),
    shape=cuml_array_shapes(min_dims=1, max_dims=5),
    order=cuml_array_orders(),
)
@settings(deadline=None)
def test_array_to_memory_order(input_type, dtype, shape, order):
    input_array = create_cuml_array_input(input_type, dtype, shape, order)
    assert array_to_memory_order(input_array, default=order) == order


@example(dtype=np.float32, shape=(5, 5), order="C")
@example(dtype=np.float32, shape=(1, 5), order="C")
@example(dtype=np.float32, shape=(5, 1), order="C")
@example(dtype=np.float32, shape=(5, 5), order="F")
@example(dtype=np.float32, shape=(1, 5), order="F")
@example(dtype=np.float32, shape=(5, 1), order="F")
@given(
    dtype=cuml_array_dtypes(),
    shape=cuml_array_shapes(min_dims=1, max_dims=5),
    order=cuml_array_orders(),
)
def test_determine_memory_order(dtype, shape, order):
    x = create_cuml_array_input("numpy", dtype, shape, order)
    if x.flags["C_CONTIGUOUS"] and x.flags["F_CONTIGUOUS"]:
        assert _determine_memory_order(x.shape, x.strides, x.dtype, "C") == "C"
        assert _determine_memory_order(x.shape, x.strides, x.dtype, "F") == "F"
    else:
        assert (
            _determine_memory_order(x.shape, x.strides, x.dtype, order)
            == order
        )


@example(input_type="numpy", dtype=np.float32, shape=(10, 10), order="C")
@given(
    input_type=st.sampled_from(("cupy", "numpy")),
    dtype=cuml_array_dtypes(),
    shape=cuml_array_shapes(min_dims=1, max_dims=5),
    order=cuml_array_orders(),
)
@settings(deadline=None)
def test_order_to_strides(input_type, dtype, shape, order):
    input_array = create_cuml_array_input(input_type, dtype, shape, order)
    if isinstance(shape, int):
        shape = (shape,)
    assert np.all(
        np.array(_order_to_strides(order, shape, dtype))
        == np.array(input_array.strides)
    )


@pytest.mark.parametrize("kind", ["dataframe", "series"])
def test_output_pandas(kind):
    """Check that `output_type=pandas` matches the user-facing pandas.

    If `cudf.pandas` is enabled we want to output a `cudf.pandas` proxy, not
    the original raw `pandas` object (like `cudf_obj.to_pandas()` does)."""
    if kind == "series":
        shape = 10
        exp_type = pd.Series
    else:
        shape = (10, 3)
        exp_type = pd.DataFrame
    arr = CumlArray.from_input(cp.ones(shape))
    out = arr.to_output("pandas")
    assert isinstance(out, exp_type)


def test_output_cudf_maintain_nan():
    expected = cp.array([[1.1, cp.nan], [cp.nan, 2.2]])
    # to_cupy would raise if NaN was converted to null
    result = CumlArray.from_input(expected).to_output("cudf").to_cupy()
    assert cp.array_equal(result, expected, equal_nan=True)
