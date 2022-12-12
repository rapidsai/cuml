#
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import gc
import operator
import pickle
from copy import deepcopy

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import pytest
from cudf.core.buffer import Buffer
from cuml.common.array import CumlArray
from cuml.common.memory_utils import _get_size_from_shape, _strides_to_order
from cuml.testing.strategies import (UNSUPPORTED_CUDF_DTYPES,
                                     create_cuml_array_input,
                                     cuml_array_dtypes, cuml_array_input_types,
                                     cuml_array_inputs, cuml_array_orders,
                                     cuml_array_output_types,
                                     cuml_array_shapes)
from cuml.testing.utils import (normalized_shape, series_squeezed_shape,
                                squeezed_shape, to_nparray)
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numba import cuda
from rmm import DeviceBuffer

_OUTPUT_TYPES_MAPPING = {
    'cupy': cp.ndarray,
    'numpy': np.ndarray,
    'cudf': (cudf.DataFrame, cudf.Series),
    'dataframe': cudf.DataFrame,
    'series': cudf.Series,
}


def _multidimensional(shape):
    return len(squeezed_shape(normalized_shape(shape))) > 1


def _get_owner(curr):
    if (isinstance(curr, CumlArray)):
        return curr._owner
    elif (isinstance(curr, cp.ndarray)):
        return curr.data.mem._owner
    else:
        return None


def _assert_equal(array_like, cuml_array):
    """Check whether array-like data and cuml array data are equal."""
    assert cp.array_equal(
        cp.asarray(array_like), cuml_array.to_output("cupy"), equal_nan=True,
    )


@given(
    input_type=cuml_array_input_types(),
    dtype=cuml_array_dtypes(),
    shape=cuml_array_shapes(),
    order=cuml_array_orders(),
    force_gc=st.booleans())
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

    # Test ownership
    if input_type == "numpy":
        # Numpy input arrays are expected to be owned by cupy.
        assert isinstance(cuml_array._owner, cp.ndarray)
    else:
        # Check that the original input array is part of the owner chain.
        current_owner = cuml_array
        while (current_owner := _get_owner(current_owner)) is not None:
            if current_owner is input_array:
                break
        else:
            assert False, "Unable to find input array in owner chain."

        # Check that data is kept in memory even when the input_array reference
        # is deleted.
        input_array_copy = deepcopy(cp.asarray(input_array))
        del input_array
        if force_gc:
            gc.collect()

        _assert_equal(input_array_copy, cuml_array)


@given(
    data_type=st.sampled_from([bytes, bytearray, memoryview]),
    dtype=cuml_array_dtypes(),
    shape=cuml_array_shapes(),
    order=cuml_array_orders(),
)
@settings(deadline=None)
def test_array_init_from_bytes(data_type, dtype, shape, order):
    dtype = np.dtype(dtype)
    values = bytes(_get_size_from_shape(shape, dtype)[0])

    # Convert to data_type to be tested if needed.
    if data_type != bytes:
        values = data_type(values)

    array = CumlArray(values, dtype=dtype, shape=shape, order=order)

    assert array.order == order
    assert array.shape in (shape, (shape, ))
    assert array.dtype == dtype

    array_copy = cp.zeros(shape, dtype=dtype)

    assert cp.all(cp.asarray(array_copy) == array_copy)


@given(
    input_type=cuml_array_input_types(),
    dtype=cuml_array_dtypes(),
    shape=cuml_array_shapes(),
    order=cuml_array_orders(),
)
@settings(deadline=None)
def test_array_init_bad(input_type, dtype, shape, order):
    """
    This test ensures that we assert on incorrect combinations of arguments
    when creating CumlArray
    """

    input_array = create_cuml_array_input(input_type, dtype, shape, order)

    # Ensure the array is creatable
    array = CumlArray(input_array)

    with pytest.raises(AssertionError):
        CumlArray(input_array, dtype=array.dtype)

    with pytest.raises(AssertionError):
        CumlArray(input_array, shape=array.shape)

    with pytest.raises(AssertionError):
        CumlArray(
            input_array,
            order=_strides_to_order(
                array.strides, array.shape, array.dtype
            ),
        )

    assert cp.all(cp.asarray(input_array) == cp.asarray(array))


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
    assume(np.isscalar(inp_view) or inp_view.size > 0)

    _assert_equal(inp_view, ary[indices])

    # Check equality after assigning to array slice.
    ary[indices] = 1.0
    inp[indices] = 1.0

    # We need to assume that inp is not a cudf.Series here, otherwise
    # ary.to_output("cupy") called by equal() will trigger a CUDARuntimeError:
    # cudaErrorInvalidDevice: invalid device ordinal error.
    assume(not isinstance(inp, cudf.Series))

    _assert_equal(inp, ary)


@given(
    shape=cuml_array_shapes(),
    dtype=cuml_array_dtypes(),
    order=cuml_array_orders(),
)
@settings(deadline=None)
def test_create_empty(shape, dtype, order):
    ary = CumlArray.empty(shape=shape, dtype=dtype, order=order)
    assert isinstance(ary.ptr, int)
    assert ary.shape == normalized_shape(shape)
    assert ary.dtype == np.dtype(dtype)
    assert isinstance(ary._owner.data.mem._owner, DeviceBuffer)


@given(
    shape=cuml_array_shapes(),
    dtype=cuml_array_dtypes(),
    order=cuml_array_orders(),
)
@settings(deadline=None)
def test_create_zeros(shape, dtype, order):
    ary = CumlArray.zeros(shape=shape, dtype=dtype, order=order)
    test = cp.zeros(shape).astype(dtype)
    assert cp.all(test == cp.asarray(ary))


@given(
    shape=cuml_array_shapes(),
    dtype=cuml_array_dtypes(),
    order=cuml_array_orders(),
)
@settings(deadline=None)
def test_create_ones(shape, dtype, order):
    ary = CumlArray.ones(shape=shape, dtype=dtype, order=order)
    test = cp.ones(shape).astype(dtype)
    assert cp.all(test == cp.asarray(ary))


@given(
    shape=cuml_array_shapes(),
    dtype=cuml_array_dtypes(),
    order=cuml_array_orders(),
)
@settings(deadline=None)
def test_create_full(shape, dtype, order):
    value = cp.array([cp.random.randint(100)]).astype(dtype)
    ary = CumlArray.full(value=value[0], shape=shape, dtype=dtype, order=order)
    test = cp.zeros(shape).astype(dtype) + value[0]
    assert cp.all(test == cp.asarray(ary))


def cudf_compatible_dtypes(dtype):
    return dtype not in UNSUPPORTED_CUDF_DTYPES


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
    res = CumlArray(inp).to_output(output_type)

    # Check output type
    if output_type == 'numba':  # TODO: is this still needed?
        # using correct numba ndarray check
        assert cuda.devicearray.is_cuda_ndarray(res)
    elif output_type == 'cudf':
        assert isinstance(
            res,
            cudf.DataFrame if _multidimensional(inp.shape) else cudf.Series)
    else:
        assert isinstance(res, _OUTPUT_TYPES_MAPPING[output_type])

    def assert_data_equal_(res):
        # Check output data equality
        if isinstance(res, cudf.Series):
            # A simple equality check `assert cudf.Series(inp).equals(res)`
            # does not work for with multi-dimensional data.
            assert cudf.Series(np.ravel(inp)).equals(res)
        elif isinstance(res, cudf.DataFrame):
            # Assumption required because of:
            #   https://github.com/rapidsai/cudf/issues/12266
            assume(not np.isnan(res.to_numpy()).any())

            assert cudf.DataFrame(inp).equals(res)
        else:
            assert np.array_equal(
                to_nparray(inp), to_nparray(res), equal_nan=True)

    assert_data_equal_(res)


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
        not(
            output_type in ("cudf", "dataframe", "series")
            and inp.dtype in UNSUPPORTED_CUDF_DTYPES
        )
    )
    assume(
        # Can't convert multidimensional arrays to a Series.
        not(output_type == "series" and len(inp.shape) > 1)
    )

    # Assumptions required for cuML limitations:
    assume(
        # Cannot convert from DataFrame to CumlArray wihthout explicitly
        # specifying shape, dtype, and order.
        not(
            output_type == "dataframe" or
            (output_type == "cudf" and len(inp.shape) > 1)
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
    if isinstance(res, cudf.DataFrame):
        res.values.dtype is out_dtype
    else:
        res.dtype is out_dtype


@given(cuml_array_inputs(input_types=st.just("cupy")))
@settings(deadline=None)
@pytest.mark.xfail(reason="Fails for version and strides keys.")
def test_cuda_array_interface(inp):
    ary = CumlArray(inp)

    inp_cai = inp.__cuda_array_interface__
    ary_cai = ary.__cuda_array_interface__

    # Check CUDA Array Interface equality.
    assert inp_cai["shape"] == ary_cai["shape"]
    assert inp_cai["typestr"] == ary_cai["typestr"]
    assert inp_cai["data"] == ary_cai["data"]
    assert inp_cai["version"] == ary_cai["version"]  # mismatch
    # Mismatch for one-dimensional arrays:
    assert inp_cai["strides"] == ary_cai["strides"]

    # Check equality
    assert cp.all(inp == cp.asarray(ary))


@given(cuml_array_inputs())
@settings(deadline=None)
def test_serialize(inp):
    ary = CumlArray(data=inp)
    header, frames = ary.serialize()
    ary2 = CumlArray.deserialize(header, frames)

    assert pickle.loads(header['type-serialized']) is CumlArray
    assert all(isinstance(f, Buffer) for f in frames)

    _assert_equal(inp, ary2)

    assert ary.__cuda_array_interface__['shape'] == \
        ary2.__cuda_array_interface__['shape']
    assert ary.__cuda_array_interface__['strides'] == \
        ary2.__cuda_array_interface__['strides']
    assert ary.__cuda_array_interface__['typestr'] == \
        ary2.__cuda_array_interface__['typestr']

    if isinstance(inp, (cudf.Series, pd.Series)):
        assert ary.order == ary2.order


@pytest.mark.parametrize('protocol', [4, 5])
@given(inp=cuml_array_inputs())
@settings(deadline=None)
def test_pickle(protocol, inp):
    if protocol > pickle.HIGHEST_PROTOCOL:
        pytest.skip(
            f"Trying to test with pickle protocol {protocol},"
            f" but highest supported protocol is {pickle.HIGHEST_PROTOCOL}."
        )

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

    # Perform serialization and unserialization.
    a = pickle.dumps(ary, **dumps_kwargs)
    b = pickle.loads(a, **loads_kwargs)

    # Check equality
    assert len(f) == len_f
    _assert_equal(inp, b)

    # Check CUDA Array Interface match.
    assert ary.__cuda_array_interface__['shape'] == \
        b.__cuda_array_interface__['shape']
    assert ary.__cuda_array_interface__['strides'] == \
        b.__cuda_array_interface__['strides']
    assert ary.__cuda_array_interface__['typestr'] == \
        b.__cuda_array_interface__['typestr']

    if isinstance(inp, (cudf.Series, pd.Series)):
        # skipping one dimensional ary order test
        assert ary.order == b.order


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

    # Check CUDA Array Interface match.
    assert ary.__cuda_array_interface__['shape'] == \
        b.__cuda_array_interface__['shape']
    assert ary.__cuda_array_interface__['strides'] == \
        b.__cuda_array_interface__['strides']
    assert ary.__cuda_array_interface__['typestr'] == \
        b.__cuda_array_interface__['typestr']

    if isinstance(inp, (cudf.Series, pd.Series)):
        # skipping one dimensional ary order test
        assert ary.order == b.order


@pytest.mark.parametrize('operation', [operator.add, operator.sub])
@given(a=cuml_array_inputs())
@settings(deadline=None)
def test_cumlary_binops(operation, a):
    b = deepcopy(a)

    ary_a = CumlArray(a)
    ary_b = CumlArray(b)

    c = operation(a, b)
    ary_c = operation(ary_a, ary_b)

    _assert_equal(c, ary_c)


@pytest.mark.parametrize('order', ['F', 'C'])
def test_sliced_array_owner(order):
    """
    When slicing a CumlArray, a new object can be created created which
    previously had an incorrect owner. This was due to the requirement by
    `cudf.core.Buffer` that all data be in "u1" form. CumlArray would satisfy
    this requirement by calling
    `cp.asarray(data).ravel(order='A').view('u1')`. If the slice is not
    contiguous, this would create an intermediate object with no references
    that would be cleaned up by GC causing an error when using the memory
    """

    # Create 2 copies of a random array
    random_cp = cp.array(cp.random.random((500, 4)),
                         dtype=np.float32,
                         order=order)
    cupy_array = cp.array(random_cp, copy=True)
    cuml_array = CumlArray(random_cp)

    # Make sure we have 2 pieces of data
    assert cupy_array.data.ptr != cuml_array.ptr

    # Since these are C arrays, slice off the first column to ensure they are
    # non-contiguous
    cuml_slice = cuml_array[1:, 1:]
    cupy_slice = cupy_array[1:, 1:]

    # Delete the input object just to be sure
    del random_cp

    # Make sure to cleanup any objects. Forces deletion of intermediate owner
    # object
    gc.collect()

    # Calling `to_output` forces use of the pointer. This can fail with a cuda
    # error on `cupy.cuda.runtime.pointerGetAttributes(cuml_slice.ptr)` in CUDA
    # < 11.0 or cudaErrorInvalidDevice in CUDA > 11.0 (unclear why it changed)
    assert (cp.all(cuml_slice.to_output('cupy') == cupy_slice))
