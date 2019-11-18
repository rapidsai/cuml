# Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

import numpy as np
import warnings

from numba import cuda

from cuml.utils.import_utils import check_min_numba_version, \
    check_min_cupy_version, has_cupy
from cuml.utils.numba_utils import PatchedNumbaDeviceArray


def test_numba_cupy_version_conflict(X):
    """
    Function to test whether cuda_array_interface version conflict
    may cause issues with array X. True if CuPy < 7.0 and Numba >= 0.46.
    """
    if cuda.devicearray.is_cuda_ndarray(X) and \
            check_min_numba_version("0.46") and \
            not check_min_cupy_version("7.0"):
        return True

    else:
        return False


def checked_cupy_fn(cupy_fn, *argv):
    """
    Function to call cupy functions with "patched" numba arrays if needed
    due to version conflict from test_numba_cupy_version_conflict
    """

    cp_argv = []

    for arg in argv:
        if test_numba_cupy_version_conflict(arg):
            cp_argv.append(PatchedNumbaDeviceArray(arg))
        else:
            cp_argv.append(arg)

    result = cupy_fn(*cp_argv)

    return result


def checked_cupy_unique(x):
    """
    Returns the unique elements from X as an array, using either cupy (if
    installed) or numpy (fallback).
    """

    if has_cupy():
        import cupy as cp  # noqa: E402
        unique = checked_cupy_fn(cp.unique, x)
    else:
        warnings.warn("Using NumPy for number of class detection,"
                      "install CuPy for faster processing.")
        if isinstance(x, np.ndarray):
            unique = np.unique(x)
        else:
            unique = np.unique(x.copy_to_host())

    return unique
