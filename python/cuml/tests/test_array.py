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

import sys
import gc

import pytest

import operator

from copy import deepcopy
from cuml.internals.array import CumlArray
from cuml.internals.global_settings import global_settings
from cuml.internals.mem_type import MemoryType
from cuml.internals.memory_utils import _get_size_from_shape
from cuml.internals.memory_utils import _strides_to_order
# Temporarily disabled due to CUDA 11.0 issue
# https://github.com/rapidsai/cuml/issues/4332
# from rmm import DeviceBuffer
from cuml.internals.safe_imports import (
    cpu_only_import,
    cpu_only_import_from,
    gpu_only_import,
    gpu_only_import_from
)
cp = gpu_only_import('cupy')
cudf = gpu_only_import('cudf')
np = cpu_only_import('numpy')

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

test_dtypes_all = [
    np.float16, np.float32, np.float64,
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64
]

test_dtypes_output = [
    np.float16, np.float32, np.float64,
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64
]

test_shapes = [10, (10,), (10, 1), (10, 5), (1, 10)]

test_slices = [0, 5, 'left', 'right', 'both', 'bool_op']

unsupported_cudf_dtypes = [np.uint8, np.uint16, np.uint32, np.uint64,
                           np.float16]


@pytest.mark.parametrize('input_type', test_input_types)
@pytest.mark.parametrize('dtype', test_dtypes_all)
@pytest.mark.parametrize('shape', test_shapes)
@pytest.mark.parametrize('order', ['F', 'C'])
def test_array_init(input_type, dtype, shape, order):
    if input_type == 'series':
        if dtype in unsupported_cudf_dtypes or \
                shape in [(10, 5), (1, 10)]:
            pytest.skip("Unsupported cuDF Series parameter")

    inp, ary, ptr = create_ary_init_tests(input_type, dtype, shape, order)

    if shape == (10, 5):
        assert ary.order == order

    if shape == 10:
        assert ary.shape == (10,)
        assert len(ary) == 10
    elif input_type == 'series':
        # cudf Series make their shape (10,) from (10, 1)
        if shape == (10, 1):
            assert ary.shape == (10,)
    else:
        assert ary.shape == shape

    assert ary.dtype == np.dtype(dtype)

    if (input_type == "numpy"):
        assert isinstance(ary._owner, np.ndarray)

        truth = cp.asnumpy(inp)
        del inp

        assert ary.ptr == ptr
        data = ary.to_output('numpy')

        assert np.array_equal(truth, data)
    else:
        helper_test_ownership(ary, inp, False)


@pytest.mark.parametrize('input_type', test_input_types)
def test_ownership_with_gc(input_type):
    # garbage collection slows down the test suite significantly, we only
    # need to test for each input type, not for shapes/dtypes/etc.
    if input_type == 'numpy':
        pytest.skip("test not valid for numpy input")

    inp, ary, ptr = create_ary_init_tests(input_type, np.float32, (10, 10),
                                          'F')

    helper_test_ownership(ary, inp, True)


def create_ary_init_tests(ary_type, dtype, shape, order):
    if ary_type is not None:
        inp = create_input(ary_type, dtype, shape, order)
        ary = CumlArray(data=inp)
        ptr = ary.ptr
    else:
        inp = create_input('cupy', dtype, shape, order)
        ptr = inp.__cuda_array_interface__['data'][0]
        ary = CumlArray(data=ptr, owner=inp, dtype=inp.dtype, shape=inp.shape,
                        order=order, mem_type=MemoryType.device)

    return (inp, ary, ptr)


def get_owner(curr):
    if (isinstance(curr, CumlArray)):
        return curr._owner
    elif (isinstance(curr, cp.ndarray)):
        return curr.data.mem._owner
    else:
        return None


def helper_test_ownership(ary, inp, garbage_collect):
    found_owner = False
    # Make sure the input array is in the ownership chain
    curr_owner = ary

    while (curr_owner is not None):
        if (curr_owner is inp):
            found_owner = True
            break

        curr_owner = get_owner(curr_owner)

    assert found_owner, "GPU input arrays must be in the owner chain"

    inp_copy = deepcopy(cp.asarray(inp))

    # testing owner reference keeps data of ary alive
    del inp

    if garbage_collect:
        # Force GC just in case it lingers
        gc.collect()

    assert cp.all(cp.asarray(ary._owner) == cp.asarray(inp_copy))


@pytest.mark.parametrize('data_type', [bytes, bytearray, memoryview])
@pytest.mark.parametrize('dtype', test_dtypes_all)
@pytest.mark.parametrize('shape', test_shapes)
@pytest.mark.parametrize('order', ['F', 'C'])
@pytest.mark.parametrize('mem_type', [MemoryType.host, MemoryType.device])
def test_array_init_from_bytes(data_type, dtype, shape, order, mem_type):
    dtype = np.dtype(dtype)
    bts = bytes(_get_size_from_shape(shape, dtype)[0])

    if data_type != bytes:
        bts = data_type(bts)

    ary = CumlArray(bts, dtype=dtype, shape=shape, order=order,
                    mem_type=mem_type)

    if shape == (10, 5):
        assert ary.order == order

    if shape == 10:
        assert ary.shape == (10,)
    else:
        assert ary.shape == shape

    assert ary.dtype == dtype

    cp_ary = cp.zeros(shape, dtype=dtype)

    assert cp.all(cp.asarray(cp_ary) == cp_ary)


@pytest.mark.parametrize('input_type', test_input_types)
@pytest.mark.parametrize('dtype', test_dtypes_all)
@pytest.mark.parametrize('shape', test_shapes)
@pytest.mark.parametrize('order', ['F', 'C'])
def test_array_init_bad(input_type, dtype, shape, order):
    """
    This test ensures that we assert on incorrect combinations of arguments
    when creating CumlArray
    """
    if input_type == 'series':
        if dtype == np.float16:
            pytest.skip("Skipping due to cuDF issue #9065")
        inp = create_input(input_type, dtype, shape, 'C')
    else:
        inp = create_input(input_type, dtype, shape, order)

    # Ensure the array is creatable
    cuml_ary = CumlArray(inp)

    with pytest.raises(ValueError):
        bad_dtype = np.float16 if dtype != np.float16 else np.float32
        CumlArray(inp, dtype=bad_dtype)

    with pytest.raises(ValueError):
        CumlArray(inp, shape=(*cuml_ary.shape, 1))

    assert cp.all(cp.asarray(inp) == cp.asarray(cuml_ary))


@pytest.mark.parametrize('slice', test_slices)
@pytest.mark.parametrize('order', ['C', 'F'])
def test_get_set_item(slice, order):
    if order == 'F' and slice != 'both':
        pytest.skip("See issue https://github.com/rapidsai/cuml/issues/2412")

    inp = create_input('numpy', 'float32', (10, 10), order)
    ary = CumlArray(data=inp)

    if isinstance(slice, int):
        assert np.array_equal(inp[slice], ary[slice].to_output('numpy'))
        inp[slice] = 1.0
        ary[slice] = 1.0

    elif slice == 'left':
        assert np.array_equal(inp[5:], ary[5:].to_output('numpy'))
        inp[5:] = 1.0
        ary[5:] = 1.0

    elif slice == 'right':
        assert np.array_equal(inp[:5], ary[:5].to_output('numpy'))
        inp[:5] = 1.0
        ary[:5] = 1.0

    elif slice == 'both':
        assert np.array_equal(inp[:], ary[:].to_output('numpy'))
        inp[:] = 1.0
        ary[:] = 1.0

    else:
        pytest.skip("not implemented logical indexing, unless we need it")

    assert np.array_equal(inp, ary.to_output('numpy'))


@pytest.mark.parametrize('shape', test_shapes)
@pytest.mark.parametrize('dtype', test_dtypes_all)
@pytest.mark.parametrize('order', ['C', 'F'])
def test_create_empty(shape, dtype, order):
    ary = CumlArray.empty(shape=shape, dtype=dtype, order=order)
    assert isinstance(ary.ptr, int)
    if shape == 10:
        assert ary.shape == (shape,)
    else:
        assert ary.shape == shape
    assert ary.dtype == np.dtype(dtype)
    # Temporarily disabled due to CUDA 11.0 issue
    # https://github.com/rapidsai/cuml/issues/4332
    # assert isinstance(ary._owner.data.mem._owner, DeviceBuffer)


@pytest.mark.parametrize('shape', test_shapes)
@pytest.mark.parametrize('dtype', test_dtypes_all)
@pytest.mark.parametrize('order', ['F', 'C'])
def test_create_zeros(shape, dtype, order):
    ary = CumlArray.zeros(shape=shape, dtype=dtype, order=order)
    test = cp.zeros(shape).astype(dtype)
    assert cp.all(test == cp.asarray(ary))


@pytest.mark.parametrize('shape', test_shapes)
@pytest.mark.parametrize('dtype', test_dtypes_all)
@pytest.mark.parametrize('order', ['F', 'C'])
def test_create_ones(shape, dtype, order):
    ary = CumlArray.ones(shape=shape, dtype=dtype, order=order)
    test = cp.ones(shape).astype(dtype)
    assert cp.all(test == cp.asarray(ary))


@pytest.mark.parametrize('shape', test_shapes)
@pytest.mark.parametrize('dtype', test_dtypes_all)
@pytest.mark.parametrize('order', ['F', 'C'])
def test_create_full(shape, dtype, order):
    value = cp.array([cp.random.randint(100)]).astype(dtype)
    ary = CumlArray.full(value=value[0], shape=shape, dtype=dtype, order=order)
    test = cp.zeros(shape).astype(dtype) + value[0]
    assert cp.all(test == cp.asarray(ary))


@pytest.mark.parametrize('output_type', test_output_types)
@pytest.mark.parametrize('dtype', test_dtypes_output)
@pytest.mark.parametrize('output_dtype', test_dtypes_output)
@pytest.mark.parametrize('order', ['F', 'C'])
@pytest.mark.parametrize('shape', test_shapes)
@pytest.mark.parametrize('mem_type', [MemoryType.host, MemoryType.device])
def test_output(output_type, dtype, output_dtype, order, shape, mem_type):
    inp = create_input('numpy', dtype, shape, order)
    ary = CumlArray(inp)

    try:
        res = ary.to_output(
            output_type,
            output_dtype,
            mem_type
        )
    except ValueError as exc:
        if (
            output_dtype == mem_type.xpy.float16 and (
                output_type == 'cudf'
                or (
                    mem_type.is_device_accessible
                    and output_type in ('series', 'dataframe', 'df_obj')
                )
            )
        ):
            # CUDF does not currently support half-precision floats
            return
        try:
            if (
                output_type == 'series'
                and len(shape) > 1
                and shape[1] != 1
            ):
                return
            else:
                raise
        except TypeError:
            raise exc

    if output_type in ('cupy', 'cudf', 'numba'):
        mem_type = MemoryType.device
    elif output_type in ('numpy', 'pandas'):
        mem_type = MemoryType.host

    if output_type in ('cupy', 'numpy'):
        output_type = 'array'
    elif output_type in ('cudf', 'pandas'):
        output_type = 'df_obj'

    if mem_type == MemoryType.device:
        assert isinstance(
            res,
            (CudfDataFrame, CudfSeries, cp_array, numba_array)
        )
    elif mem_type == MemoryType.host:
        assert isinstance(
            res,
            (PandasDataFrame, PandasSeries, np_array)
        )
    if output_type == 'array':
        assert isinstance(
            res,
            (cp_array, np_array)
        )
        mem_type.xpy.all(
            mem_type.xpy.asarray(inp) == res
        )
    elif output_type == 'df_obj':
        assert isinstance(
            res,
            (CudfDataFrame, CudfSeries, PandasDataFrame, PandasSeries)
        )
    elif output_type == 'series':
        assert isinstance(
            res,
            (CudfSeries, PandasSeries)
        )
    elif output_type == 'dataframe':
        assert isinstance(
            res,
            (CudfDataFrame, PandasDataFrame)
        )
    elif output_type == 'numba':
        assert isinstance(res, numba_array)
        assert cp.all(
            cp.asarray(cuda.to_device(inp)) == cp.asarray(res)
        )

    if isinstance(res, (CudfSeries, PandasSeries)):
        orig = mem_type.xdf.Series(mem_type.xpy.ravel(
            mem_type.xpy.asarray(inp)
        ))
        assert (orig == res).all()
    elif isinstance(res, (CudfDataFrame, PandasDataFrame)):
        if len(inp.shape) == 1:
            inp = inp.reshape(inp.shape[0], 1)
        orig = mem_type.xdf.DataFrame(inp)
        assert (orig == res).all().all()

    try:
        res_dtype = res.dtype
    except AttributeError:
        res_dtype = res.values.dtype

    if not (
        output_dtype == mem_type.xpy.float16
        and isinstance(res, (CudfSeries, CudfDataFrame))
    ):
        assert res_dtype == output_dtype


@pytest.mark.parametrize('dtype', test_dtypes_all)
@pytest.mark.parametrize('shape', test_shapes)
@pytest.mark.parametrize('order', ['F', 'C'])
def test_cuda_array_interface(dtype, shape, order):
    inp = create_input('numba', dtype, shape, 'F')
    ary = CumlArray(inp)

    if isinstance(shape, tuple):
        assert ary.__cuda_array_interface__['shape'] == shape
    else:
        assert ary.__cuda_array_interface__['shape'] == (shape,)

    input_strides = inp.__cuda_array_interface__['strides']
    if input_strides is None:
        input_strides = inp.strides
    assert (
        ary.__cuda_array_interface__['strides'] ==
        input_strides
    )
    assert ary.__cuda_array_interface__['typestr'] == inp.dtype.str
    assert ary.__cuda_array_interface__['data'] == \
        (inp.device_ctypes_pointer.value, False)
    assert ary.__cuda_array_interface__['version'] == 3

    # since our test array is small, its faster to transfer it to numpy to
    # square rather than a numba cuda kernel

    truth = np.sqrt(inp.copy_to_host())
    result = cp.sqrt(ary)

    assert np.all(truth == cp.asnumpy(result))


@pytest.mark.parametrize('input_type', test_input_types)
def test_serialize(input_type):
    if input_type == 'series':
        inp = create_input(input_type, np.float32, (10, 1), 'C')
    else:
        inp = create_input(input_type, np.float32, (10, 5), 'F')
    ary = CumlArray(data=inp)
    header, frames = ary.serialize()
    ary2 = CumlArray.deserialize(header, frames)

    assert pickle.loads(header['type-serialized']) is CumlArray

    if input_type == 'numpy':
        assert np.all(inp == ary2.to_output('numpy'))
    elif input_type == 'series':
        assert np.all(inp == ary2.to_output('series'))
    else:
        assert cp.all(cp.asarray(inp) == cp.asarray(ary2))

    assert ary._array_interface['shape'] == \
        ary2._array_interface['shape']
    assert ary._array_interface['strides'] == \
        ary2._array_interface['strides']
    assert ary._array_interface['typestr'] == \
        ary2._array_interface['typestr']
    assert ary2.mem_type is global_settings.memory_type

    if input_type != 'series':
        # skipping one dimensional ary order test
        assert ary.order == ary2.order


@pytest.mark.parametrize('input_type', test_input_types)
@pytest.mark.parametrize('protocol', [4, 5])
def test_pickle(input_type, protocol):
    if protocol > pickle.HIGHEST_PROTOCOL:
        pytest.skip(
            f"Trying to test with pickle protocol {protocol},"
            f" but highest supported protocol is {pickle.HIGHEST_PROTOCOL}."
        )
    if input_type == 'series':
        inp = create_input(input_type, np.float32, (10, 1), 'C')
    else:
        inp = create_input(input_type, np.float32, (10, 5), 'F')
    ary = CumlArray(data=inp)
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
    assert len(f) == len_f
    if input_type == 'numpy':
        assert np.all(inp == b.to_output('numpy'))
    elif input_type == 'series':
        assert np.all(inp == b.to_output('series'))
    else:
        assert cp.all(cp.asarray(inp) == cp.asarray(b))

    assert ary._array_interface['shape'] == \
        b._array_interface['shape']
    assert ary._array_interface['strides'] == \
        b._array_interface['strides']
    assert ary._array_interface['typestr'] == \
        b._array_interface['typestr']

    if input_type != 'series':
        # skipping one dimensional ary order test
        assert ary.order == b.order


@pytest.mark.parametrize('input_type', test_input_types)
def test_deepcopy(input_type):
    if input_type == 'series':
        inp = create_input(input_type, np.float32, (10, 1), 'C')
    else:
        inp = create_input(input_type, np.float32, (10, 5), 'F')
    ary = CumlArray(data=inp)
    b = deepcopy(ary)
    if input_type == 'numpy':
        assert np.all(inp == b.to_output('numpy'))
    elif input_type == 'series':
        assert np.all(inp == b.to_output('series'))
    else:
        assert cp.all(cp.asarray(inp) == cp.asarray(b))

    assert ary.ptr != b.ptr

    assert ary._array_interface['shape'] == \
        b._array_interface['shape']
    assert ary._array_interface['strides'] == \
        b._array_interface['strides']
    assert ary._array_interface['typestr'] == \
        b._array_interface['typestr']

    if input_type != 'series':
        # skipping one dimensional ary order test
        assert ary.order == b.order


@pytest.mark.parametrize('operation', [operator.add, operator.sub])
def test_cumlary_binops(operation):
    a = cp.arange(5)
    b = cp.arange(5)

    ary_a = CumlArray(a)
    ary_b = CumlArray(b)

    c = operation(a, b)
    ary_c = operation(ary_a, ary_b)

    assert(cp.all(ary_c.to_output('cupy') == c))


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


def create_input(input_type, dtype, shape, order):
    rand_ary = cp.ones(shape, dtype=dtype, order=order)

    if input_type == 'numpy':
        return np.array(cp.asnumpy(rand_ary), dtype=dtype, order=order)

    elif input_type == 'numba':
        return cuda.as_cuda_array(rand_ary)

    elif input_type == 'series':
        return cudf.Series(rand_ary)

    else:
        return rand_ary
