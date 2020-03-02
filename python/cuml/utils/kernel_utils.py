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

import cupy as cp
import numpy as np

import functools

from uuid import uuid1

# Mapping of common PyData dtypes to their corresponding C-primitive
dtype_str_map = {cp.dtype("float32"): "float",
                 cp.dtype("float64"): "double",
                 cp.dtype("int32"): "int",
                 cp.dtype("int64"): "long long int",
                 np.dtype("float32"): "float",
                 np.dtype("float64"): "double",
                 np.dtype("int32"): "int",
                 np.dtype("int64"): "long long int",
                 "float32": "float",
                 "float64": "double",
                 "int32": "int",
                 "int64": "long long int",
                 }

extern_prefix = r'extern "C" __global__'


def get_dtype_str(dtype):
    if dtype not in dtype_str_map:
        raise ValueError(f'{dtype} is not a valid type for this kernel.')

    return dtype_str_map[dtype]


def get_dtype_strs(dtypes): return list(map(get_dtype_str, dtypes))


@functools.lru_cache(maxsize=5000)
def cuda_kernel_factory(nvrtc_kernel_str, dtypes, kernel_name=None,
                        verbose=False):
    """
    A factory wrapper function to perform some of the boiler-plate involved in
    making cuPy RawKernels type-agnostic.

    Until a better method is created, either by RAPIDS or cuPy, this function
    will perform a string search and replace of c-based datatype primitives
    in ``nvrtc_kernel_str`` using a numerical placeholder (eg. {0}, {1}) for
    the dtype in the corresponding index of tuple ``dtypes``.

    Note that the extern, function scope, and function name should not be
    included in the kernel string. These will be added by this function and
    the function name will be made unique, based on the given dtypes.

    Example
    -------

        The following kernel string with dtypes = [float, double, int]

        ({0} *a, {1} *b, {2} *c) {}

        Will become

        (float *a, double *b, int *c) {}

    Parameters
    ----------

    nvrtc_kernel_str : string valid nvrtc kernel string without extern, scope,
                       or function name.
    dtypes : tuple of dtypes to search and replace.
    kernel_name : string prefix and function name to use. Note that when
                  this not set (or is set to None), a UUID will
                  be used, which will stop this function from
                  being memoized.

    Returns
    -------

    kernel_name : string unique function name created for kernel,
    raw_kernel : cupy.RawKernel object ready for use
    """

    dtype_strs = get_dtype_strs(dtypes)

    for idx, dtype in enumerate(dtypes):
        nvrtc_kernel_str = nvrtc_kernel_str.replace("{%d}" % idx,
                                                    dtype_strs[idx])

    kernel_name = f'''{uuid1()
                      if kernel_name is None
                      else kernel_name}_{
                        "".join(dtype_strs).replace(" ", "_")
                    }'''

    nvrtc_kernel_str = "%s\nvoid %s%s" % \
                       (extern_prefix, kernel_name, nvrtc_kernel_str)

    if verbose:
        print(str(nvrtc_kernel_str))

    return cp.RawKernel(nvrtc_kernel_str, kernel_name)
