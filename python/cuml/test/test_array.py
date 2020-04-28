#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

import cupy as cp
import numpy as np
import cudf
import pickle

from copy import deepcopy
from numba import cuda
from cudf.core.buffer import Buffer
from cuml.common.array import CumlArray
from cuml.utils.memory_utils import _get_size_from_shape
from rmm import DeviceBuffer

test_input_types = [
    'numpy', 'numba', 'cupy', 'series', None
]

test_output_types = {
    'numpy': np.ndarray,
    'cupy': cp.ndarray,
    'numba': None,
    'series': cudf.Series,
    'dataframe': cudf.DataFrame,
    'cudf': None
}

test_dtypes_all = [
    np.float16, np.float32, np.float64,
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    "float", "float32", "double", "float64",
    "int8", "short", "int16", "int", "int32", "long", "int64",
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

    if input_type is not None:
        inp = create_input(input_type, dtype, shape, order)
        ary = CumlArray(data=inp)
    else:
        inp = create_input('cupy', dtype, shape, order)
        ptr = inp.__cuda_array_interface__['data'][0]
        ary = CumlArray(data=ptr, owner=inp, dtype=inp.dtype, shape=inp.shape,
                        order=order)

    if shape == (10, 5):
        assert ary.order == order

    if shape == 10:
        assert ary.shape == (10,)
        len(ary) == 10
    elif input_type == 'series':
        # cudf Series make their shape (10,) from (10, 1)
        if shape == (10, 1):
            assert ary.shape == (10,)
    else:
        assert ary.shape == shape

    assert ary.dtype == np.dtype(dtype)

    if input_type == 'numpy':
        assert isinstance(ary._owner, DeviceBuffer)
    elif input_type in ['cupy', 'numba', 'series']:
        assert ary._owner is inp
        inp_copy = deepcopy(cp.asarray(inp))

        # testing owner reference keeps data of ary alive
        del inp
        assert cp.all(cp.asarray(ary._owner) == cp.asarray(inp_copy))

    else:
        assert isinstance(ary._owner, cp.ndarray)

        truth = cp.asnumpy(inp)
        del inp

        assert ary.ptr == ptr
        data = ary.to_output('numpy')

        assert np.array_equal(truth, data)

    return True


@pytest.mark.parametrize('data_type', [bytes, bytearray, memoryview])
@pytest.mark.parametrize('dtype', test_dtypes_all)
@pytest.mark.parametrize('shape', test_shapes)
@pytest.mark.parametrize('order', ['F', 'C'])
def test_array_init_from_bytes(data_type, dtype, shape, order):
    dtype = np.dtype(dtype)
    bts = bytes(_get_size_from_shape(shape, dtype)[0])

    if data_type != bytes:
        bts = data_type(bts)

    ary = CumlArray(bts, dtype=dtype, shape=shape, order=order)

    if shape == (10, 5):
        assert ary.order == order

    if shape == 10:
        assert ary.shape == (10,)
    else:
        assert ary.shape == shape

    assert ary.dtype == dtype

    cp_ary = cp.zeros(shape, dtype=dtype)

    assert cp.all(cp.asarray(cp_ary) == cp_ary)


@pytest.mark.parametrize('slice', test_slices)
@pytest.mark.parametrize('order', ['C', 'F'])
def test_get_set_item(slice, order):
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
    assert isinstance(ary._owner, DeviceBuffer)


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
@pytest.mark.parametrize('order', ['F', 'C'])
@pytest.mark.parametrize('shape', test_shapes)
def test_output(output_type, dtype, order, shape):
    inp = create_input('numpy', dtype, shape, order)
    ary = CumlArray(inp)

    if dtype in unsupported_cudf_dtypes and \
            output_type in ['series', 'dataframe', 'cudf']:
        with pytest.raises(ValueError):
            res = ary.to_output(output_type)
    elif shape in [(10, 5), (1, 10)] and output_type == 'series':
        with pytest.raises(ValueError):
            res = ary.to_output(output_type)
    else:
        res = ary.to_output(output_type)

        # using correct numba ndarray check
        if output_type == 'numba':
            assert cuda.devicearray.is_cuda_ndarray(res)
        elif output_type == 'cudf':
            if shape in [(10, 5), (1, 10)]:
                assert isinstance(res, cudf.DataFrame)
            else:
                assert isinstance(res, cudf.Series)
        else:
            assert isinstance(res, test_output_types[output_type])

        if output_type == 'numpy':
            assert np.all(inp == ary.to_output('numpy'))

        elif output_type == 'cupy':
            assert cp.all(cp.asarray(inp) == ary.to_output('cupy'))

        elif output_type == 'numba':
            assert cp.all(cp.asarray(cuda.to_device(inp)) == cp.asarray(res))

        elif output_type == 'series':
            comp = cudf.Series(np.ravel(inp)) == res
            assert np.all(comp.to_array())

        elif output_type == 'dataframe':
            mat = cuda.to_device(inp)
            if len(mat.shape) == 1:
                mat = mat.reshape(mat.shape[0], 1)
            comp = cudf.DataFrame.from_gpu_matrix(mat)
            comp = comp == res
            assert np.all(comp.as_gpu_matrix().copy_to_host())

        # check for e2e cartesian product:
        if output_type not in ['dataframe', 'cudf']:
            res2 = CumlArray(res)
            res2 = res2.to_output('numpy')
            if output_type == 'series' and shape == (10, 1):
                assert np.all(inp.reshape((1, 10)) == res2)
            else:
                assert np.all(inp == res2)


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

    assert ary.__cuda_array_interface__['strides'] == inp.strides
    assert ary.__cuda_array_interface__['typestr'] == inp.dtype.str
    assert ary.__cuda_array_interface__['data'] == \
        (inp.device_ctypes_pointer.value, False)
    assert ary.__cuda_array_interface__['version'] == 2

    # since our test array is small, its faster to transfer it to numpy to
    # square rather than a numba cuda kernel

    truth = np.sqrt(inp.copy_to_host())
    result = cp.sqrt(ary)

    assert np.all(truth == cp.asnumpy(result))

    return True


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
    assert all(isinstance(f, Buffer) for f in frames)

    if input_type == 'numpy':
        assert np.all(inp == ary2.to_output('numpy'))
    elif input_type == 'series':
        assert np.all(inp == ary2.to_output('series'))
    else:
        assert cp.all(inp == cp.asarray(ary2))

    assert ary.__cuda_array_interface__['shape'] == \
        ary2.__cuda_array_interface__['shape']
    assert ary.__cuda_array_interface__['strides'] == \
        ary2.__cuda_array_interface__['strides']
    assert ary.__cuda_array_interface__['typestr'] == \
        ary2.__cuda_array_interface__['typestr']

    if input_type != 'series':
        # skipping one dimensional ary order test
        assert ary.order == ary2.order


@pytest.mark.parametrize('input_type', test_input_types)
def test_pickle(input_type):
    if input_type == 'series':
        inp = create_input(input_type, np.float32, (10, 1), 'C')
    else:
        inp = create_input(input_type, np.float32, (10, 5), 'F')
    ary = CumlArray(data=inp)
    a = pickle.dumps(ary)
    b = pickle.loads(a)
    if input_type == 'numpy':
        assert np.all(inp == b.to_output('numpy'))
    elif input_type == 'series':
        assert np.all(inp == b.to_output('series'))
    else:
        assert cp.all(inp == cp.asarray(b))

    assert ary.__cuda_array_interface__['shape'] == \
        b.__cuda_array_interface__['shape']
    assert ary.__cuda_array_interface__['strides'] == \
        b.__cuda_array_interface__['strides']
    assert ary.__cuda_array_interface__['typestr'] == \
        b.__cuda_array_interface__['typestr']

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
        assert cp.all(inp == cp.asarray(b))

    assert ary.ptr != b.ptr

    assert ary.__cuda_array_interface__['shape'] == \
        b.__cuda_array_interface__['shape']
    assert ary.__cuda_array_interface__['strides'] == \
        b.__cuda_array_interface__['strides']
    assert ary.__cuda_array_interface__['typestr'] == \
        b.__cuda_array_interface__['typestr']

    if input_type != 'series':
        # skipping one dimensional ary order test
        assert ary.order == b.order


def create_input(input_type, dtype, shape, order):
    float_dtypes = [np.float16, np.float32, np.float64]
    if dtype in float_dtypes:
        rand_ary = np.random.random(shape)
    else:
        rand_ary = cp.random.randint(100, size=shape)

    rand_ary = cp.array(rand_ary, dtype=dtype, order=order)

    if input_type == 'numpy':
        return np.array(cp.asnumpy(rand_ary), dtype=dtype, order=order)

    elif input_type == 'numba':
        return cuda.as_cuda_array(rand_ary)

    elif input_type == 'series':
        return cudf.Series(cuda.as_cuda_array(rand_ary))

    else:
        return rand_ary
