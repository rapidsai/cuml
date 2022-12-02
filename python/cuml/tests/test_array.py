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
import pytest
import sys
from copy import deepcopy
from cuml.internals.array import CumlArray
from cuml import global_settings
from cuml.internals.input_utils import determine_array_memtype
from cuml.internals.mem_type import MemoryType
from cuml.internals.memory_utils import (
    _get_size_from_shape,
    _strides_to_order,
    using_memory_type
)
# Temporarily disabled due to CUDA 11.0 issue
# https://github.com/rapidsai/cuml/issues/4332
# from rmm import DeviceBuffer
from cuml.internals.safe_imports import (
    cpu_only_import,
    cpu_only_import_from,
    gpu_only_import,
    gpu_only_import_from
)
from cuml.testing.strategies import (UNSUPPORTED_CUDF_DTYPES,
                                     create_cuml_array_input,
                                     cuml_array_dtypes, cuml_array_input_types,
                                     cuml_array_inputs, cuml_array_orders,
                                     cuml_array_output_types,
                                     cuml_array_shapes,
                                     cuml_array_mem_types)
from cuml.testing.utils import (normalized_shape, series_squeezed_shape,
                                squeezed_shape, to_nparray)
from hypothesis import assume, given, settings
from hypothesis import strategies as st
cp = gpu_only_import('cupy')
cudf = gpu_only_import('cudf')
np = cpu_only_import('numpy')
pd = cpu_only_import('pandas')

cuda = gpu_only_import_from('numba', 'cuda')
CudfDataFrame = gpu_only_import_from('cudf', 'DataFrame')
CudfSeries = gpu_only_import_from('cudf', 'Series')
PandasSeries = cpu_only_import_from('pandas', 'Series')
PandasDataFrame = cpu_only_import_from('pandas', 'DataFrame')
cp_array = gpu_only_import_from('cupy', 'ndarray')
np_array = gpu_only_import_from('numpy', 'ndarray')
numba_array = gpu_only_import_from('numba.cuda.cudadrv.devicearray',
                                   'DeviceNDArray')

if sys.version_info < (3, 8):
    try:
        import pickle5 as pickle
    except ImportError:
        import pickle
else:
    import pickle


test_input_types = [
    'numpy', 'numba', 'cupy', 'series', None
]

test_output_types = (
    'cupy',
    'numpy',
    'cudf',
    'pandas',
    'array',
    'numba',
    'dataframe',
    'series',
    'df_obj'
)


_OUTPUT_TYPES_MAPPING = {
    'cupy': cp.ndarray,
    'numpy': np.ndarray,
    'cudf': (CudfDataFrame, CudfSeries),
    'pandas': (PandasDataFrame, PandasSeries),
    'dataframe': (CudfDataFrame, PandasDataFrame),
    'series': (CudfSeries, PandasSeries),
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
    mem_type=cuml_array_mem_types(),
    force_gc=st.booleans())
@settings(deadline=None)
def test_array_init(input_type, dtype, shape, order, mem_type, force_gc):
    input_array = create_cuml_array_input(input_type, dtype, shape, order)
    with using_memory_type(mem_type):
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


@given(
    data_type=st.sampled_from([bytes, bytearray, memoryview]),
    dtype=cuml_array_dtypes(),
    shape=cuml_array_shapes(),
    order=cuml_array_orders(),
    mem_type=cuml_array_mem_types(),
)
def test_array_init_from_bytes(data_type, dtype, shape, order, mem_type):
    dtype = np.dtype(dtype)
    values = bytes(_get_size_from_shape(shape, dtype)[0])

    # Convert to data_type to be tested if needed.
    if data_type != bytes:
        values = data_type(values)

    array = CumlArray(
        values, dtype=dtype, shape=shape, order=order, mem_type=mem_type
    )

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
    mem_type=cuml_array_mem_types(),
)
@settings(deadline=None)
def test_array_init_bad(input_type, dtype, shape, order, mem_type):
    """
    This test ensures that we assert on incorrect combinations of arguments
    when creating CumlArray
    """
    mem_type = MemoryType.from_str(mem_type)

    with using_memory_type(mem_type):
        input_array = create_cuml_array_input(input_type, dtype, shape, order)

        # Ensure the array is creatable
        array = CumlArray(input_array)

        with pytest.raises(ValueError):
            bad_dtype = np.float16 if dtype != np.float16 else np.float32
            CumlArray(input_array, dtype=bad_dtype)

        with pytest.raises(ValueError):
            CumlArray(input_array, shape=(*array.shape, 1))

        input_mem_type = determine_array_memtype(input_array)
        if input_mem_type.is_device_accessible:
            joint_mem_type = input_mem_type
        else:
            joint_mem_type = mem_type


        assert (joint_mem_type.xpy.all(
            joint_mem_type.xpy.asarray(input_array) ==
            joint_mem_type.xpy.asarray(array)
        ))


@given(
    inp=cuml_array_inputs(),
    indices=st.slices(10),  # TODO: should be basic_indices() as shown below
    # indices=basic_indices((10, 10)),
    mem_type=cuml_array_mem_types()
)
@settings(deadline=None)
def test_get_set_item(inp, indices, mem_type):
    mem_type = MemoryType.from_str(mem_type)
    with using_memory_type(mem_type):
        ary = CumlArray(data=inp)

        # Assumption required due to limitation on step size for F-order.
        assume(ary.order != "F" or (indices.step in (None, 1)))

        # Check equality of array views.
        inp_view = inp[indices]

        # Must assume that resulting view must have at least one element to not
        # trigger UnownedMemory exception.
        assume(mem_type.xpy.isscalar(inp_view) or inp_view.size > 0)

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
    mem_type=cuml_array_mem_types(),
)
@settings(deadline=None)
def test_create_empty(shape, dtype, order, mem_type):
    with using_memory_type(mem_type):
        ary = CumlArray.empty(shape=shape, dtype=dtype, order=order)
        assert isinstance(ary.ptr, int)
        assert ary.shape == normalized_shape(shape)
        assert ary.dtype == np.dtype(dtype)
        assert isinstance(ary._owner.data.mem._owner, DeviceBuffer)


@given(
    shape=cuml_array_shapes(),
    dtype=cuml_array_dtypes(),
    order=cuml_array_orders(),
    mem_type=cuml_array_mem_types(),
)
@settings(deadline=None)
def test_create_zeros(shape, dtype, order, mem_type):
    mem_type = MemoryType.from_str(mem_type)
    with using_memory_type(mem_type):
        ary = CumlArray.zeros(shape=shape, dtype=dtype, order=order)
        test = mem_type.xpy.zeros(shape).astype(dtype)
        assert mem_type.xpy.all(test == mem_type.xpy.asarray(ary))


@given(
    shape=cuml_array_shapes(),
    dtype=cuml_array_dtypes(),
    order=cuml_array_orders(),
    mem_type=cuml_array_mem_types(),
)
@settings(deadline=None)
def test_create_ones(shape, dtype, order, mem_type):
    mem_type = MemoryType.from_str(mem_type)
    with using_memory_type(mem_type):
        ary = CumlArray.ones(shape=shape, dtype=dtype, order=order)
        test = mem_type.xpy.ones(shape).astype(dtype)
        assert mem_type.xpy.all(test == mem_type.xpy.asarray(ary))


@given(
    shape=cuml_array_shapes(),
    dtype=cuml_array_dtypes(),
    order=cuml_array_orders(),
    mem_type=cuml_array_mem_types(),
)
@settings(deadline=None)
def test_create_full(shape, dtype, order, mem_type):
    mem_type = MemoryType.from_str(mem_type)
    with using_memory_type(mem_type):
        value = mem_type.xpy.array([cp.random.randint(100)]).astype(dtype)
        ary = CumlArray.full(value=value[0], shape=shape, dtype=dtype, order=order)
        test = mem_type.xpy.zeros(shape).astype(dtype) + value[0]
        assert mem_type.xpy.all(test == mem_type.xpy.asarray(ary))


def cudf_compatible_dtypes(dtype):
    return dtype not in UNSUPPORTED_CUDF_DTYPES


@given(
    inp=cuml_array_inputs(),
    input_mem_type=cuml_array_mem_types(),
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
    with using_memory_type(input_mem_type):
        arr = CumlArray(inp)
    res = arr.to_output(output_type)

    # Check output type
    if output_type == 'numba':  # TODO: is this still needed?
        # using correct numba ndarray check
        assert cuda.devicearray.is_cuda_ndarray(res)
    elif output_type == 'cudf':
        assert isinstance(
            res,
            CudfDataFrame if _multidimensional(inp.shape) else CudfSeries)
    elif output_type == 'pandas':
        assert isinstance(
            res,
            PandasDataFrame if _multidimensional(inp.shape) else PandasSeries)
    else:
        assert isinstance(res, _OUTPUT_TYPES_MAPPING[output_type])

    def assert_data_equal_(res):
        # Check output data equality
        if isinstance(res, CudfSeries):
            # A simple equality check `assert cudf.Series(inp).equals(res)`
            # does not work for with multi-dimensional data.
            assert cudf.Series(np.ravel(inp)).equals(res)
        elif isinstance(res, CudfDataFrame):
            # Assumption required because of:
            #   https://github.com/rapidsai/cudf/issues/12266
            assume(not np.isnan(res.to_numpy()).any())

            assert CudfDataFrame(inp).equals(res)
        else:
            assert np.array_equal(
                to_nparray(inp), to_nparray(res), equal_nan=True)

    assert_data_equal_(res)


@given(
    inp=cuml_array_inputs(),
    output_type=cuml_array_output_types(),
    mem_type=cuml_array_mem_types(),
)
@settings(deadline=None)
def test_end_to_end_conversion_via_intermediate(inp,
                                                output_type,
                                                mem_type):
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
    # TODO(wphicks): This might no longer be true?
    assume(
        # Cannot convert from DataFrame to CumlArray wihthout explicitly
        # specifying shape, dtype, and order.
        not(
            output_type == "dataframe" or
            (output_type == "cudf" and len(inp.shape) > 1)
        )
    )

    with using_memory_type(mem_type):
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
    mem_type=cuml_array_mem_types(),
)
@settings(deadline=None)
def test_output_dtype(
        output_type, shape, dtype, order, out_dtype, mem_type):

    with using_memory_type(mem_type):
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
        if isinstance(res, (CudfDataFrame, PandasDataFrame)):
            res.values.dtype is out_dtype
        else:
            res.dtype is out_dtype


@given(cuml_array_inputs())
@settings(deadline=None)
def test_array_interface(inp):
    ary = CumlArray(inp)

    inp_ai = getattr(
        inp,
        '__cuda_array_interface__',
        inp.__array_interface__
    )
    ary_ai = ary._array_interface

    # Check CUDA Array Interface equality.
    assert inp_ai["shape"] == ary_ai["shape"]
    assert inp_ai["typestr"] == ary_ai["typestr"]
    assert inp_ai["data"] == ary_ai["data"]
    assert inp_ai["version"] == ary_ai["version"]  # mismatch
    # Mismatch for one-dimensional arrays:
    assert inp_ai["strides"] == ary_ai["strides"]

    # Check equality
    assert cp.all(cp.asarray(inp) == cp.asarray(ary))


@given(
    inp=cuml_array_inputs(),
    to_serialize_mem_type=cuml_array_mem_types(),
    from_serialize_mem_type=cuml_array_mem_types(),
)
@settings(deadline=None)
def test_serialize(inp, to_serialize_mem_type, from_serialize_mem_type):
    with using_memory_type(to_serialize_mem_type):
        ary = CumlArray(data=inp)
        header, frames = ary.serialize()
    with using_memory_type(from_serialize_mem_type):
        ary2 = CumlArray.deserialize(header, frames)

    assert pickle.loads(header['type-serialized']) is CumlArray

    _assert_equal(inp, ary2)

    assert ary._array_interface['shape'] == \
        ary2._array_interface['shape']
    assert ary._array_interface['strides'] == \
        ary2._array_interface['strides']
    assert ary._array_interface['typestr'] == \
        ary2._array_interface['typestr']
    assert ary2.mem_type is global_settings.memory_type

    if isinstance(inp, (cudf.Series, pd.Series)):
        assert ary.order == ary2.order


@pytest.mark.parametrize('protocol', [4, 5])
@given(
    inp=cuml_array_inputs(),
    to_serialize_mem_type=cuml_array_mem_types(),
    from_serialize_mem_type=cuml_array_mem_types(),
)
@settings(deadline=None)
def test_pickle(
        protocol, inp, to_serialize_mem_type, from_serialize_mem_type):
    with using_memory_type(to_serialize_mem_type):
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

        a = pickle.dumps(ary, **dumps_kwargs)
    with using_memory_type(from_serialize_mem_type):
        b = pickle.loads(a, **loads_kwargs)

    assert ary._array_interface['shape'] == \
        b._array_interface['shape']
    assert ary._array_interface['strides'] == \
        b._array_interface['strides']
    assert ary._array_interface['typestr'] == \
        b._array_interface['typestr']
    # Check equality
    assert len(f) == len_f
    _assert_equal(inp, b)

    if isinstance(inp, (cudf.Series, pd.Series)):
        # skipping one dimensional ary order test
        assert ary.order == b.order


@given(
    inp=cuml_array_inputs(),
    mem_type=cuml_array_mem_types()
)
@settings(deadline=None)
def test_deepcopy(inp, mem_type):
    with using_memory_type(mem_type):
        # Generate CumlArray
        ary = CumlArray(data=inp)

        # Perform deepcopy
        b = deepcopy(ary)

        # Check equality
        _assert_equal(inp, b)
        assert ary.ptr != b.ptr

        assert ary._array_interface['shape'] == \
            b._array_interface['shape']
        assert ary._array_interface['strides'] == \
            b._array_interface['strides']
        assert ary._array_interface['typestr'] == \
            b._array_interface['typestr']

        if isinstance(inp, (cudf.Series, pd.Series)):
            # skipping one dimensional ary order test
            assert ary.order == b.order


@pytest.mark.parametrize('operation', [operator.add, operator.sub])
@given(
    a=cuml_array_inputs(),
    mem_type=cuml_array_mem_types(),
)
@settings(deadline=None)
def test_cumlary_binops(operation, a, mem_type):
    with using_memory_type(mem_type):
        b = deepcopy(a)

        ary_a = CumlArray(a)
        ary_b = CumlArray(b)

        c = operation(a, b)
        ary_c = operation(ary_a, ary_b)

        _assert_equal(c, ary_c)


@pytest.mark.parametrize('order', ['F', 'C'])
@given(
    mem_type=cuml_array_mem_types()
)
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

    # Create 2 copies of a random array
    random_cp = cp.array(cp.random.random((500, 4)),
                         dtype=np.float32,
                         order=order)
    cupy_array = cp.array(random_cp, copy=True)
    with using_memory_type(mem_type):
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
