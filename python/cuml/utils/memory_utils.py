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

import cupy as cp
import numpy as np
import rmm

from cuml.utils.import_utils import check_min_cupy_version


def rmm_cupy_ary(cupy_fn, *args, **kwargs):
    """

    Function to call CuPy functions with RMM memory management


    Parameters
    ----------
    cupy_fn : cupy function,
        CuPy function to execute, for example cp.array

    *args :
        Non keyword arguments to pass to the CuPy function

    **kwargs :
        Keyword named arguments to pass to the CuPy function


    Note: this function should be used if the result of cupy_fn creates
    a new array. Functions to create a new CuPy array by reference to
    existing device array (through __cuda_array_interface__) can be used
    directly.

    Examples
    ---------

    .. code-block:: python

        from cuml.utils import rmm_cupy_ary
        import cupy as cp

        # Get a new array filled with 0, column major
        a = rmm_cupy_ary(cp.zeros, 5, order='F')


    """

    # using_allocator was introduced in CuPy 7. Once 7+ is required,
    # this check can be removed alongside the else code path.
    if check_min_cupy_version("7.0"):
        with cp.cuda.memory.using_allocator(rmm.rmm_cupy_allocator):
            result = cupy_fn(*args, **kwargs)

    else:
        temp_res = cupy_fn(*args, **kwargs)
        result = \
            _rmm_cupy6_array_like(temp_res,
                                  order=_strides_to_order(temp_res.strides,
                                                          temp_res.dtype))
        cp.copyto(result, temp_res)

    return result


def _rmm_cupy6_array_like(ary, order):
    nbytes = np.ndarray(ary.shape,
                        dtype=ary.dtype,
                        strides=ary.strides,
                        order=order).nbytes
    memptr = cp.cuda.MemoryPointer(rmm.rmm.RMMCuPyMemory(nbytes), 0)
    arr = cp.ndarray(ary.shape,
                     dtype=ary.dtype,
                     memptr=memptr,
                     strides=ary.strides,
                     order=order)
    return arr


def _strides_to_order(strides, dtype):
    if strides[0] == dtype.itemsize:
        return 'F'
    elif strides[1] == dtype.itemsize:
        return 'C'
    else:
        raise ValueError("Invalid strides value for dtype")
