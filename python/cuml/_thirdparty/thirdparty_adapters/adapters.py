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

from cuml.common import input_to_cuml_array
from cupy import sparse
import numpy as np
import cupy as cp

from numpy import ndarray as numpyArray
from cupy import ndarray as cupyArray
from cudf.core import Series as cuSeries
from cudf.core import DataFrame as cuDataFrame
from pandas import Series as pdSeries
from pandas import DataFrame as pdDataFrame
from numba.cuda import devicearray as numbaArray


numeric_types = [
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.intp, np.uintp,
    np.float32, np.float64,
    np.complex64, np.complex128
]


def check_array(array, accept_sparse=False, accept_large_sparse=True,
                dtype=numeric_types, order=None, copy=False,
                force_all_finite=True, ensure_2d=True, allow_nd=False,
                ensure_min_samples=1, ensure_min_features=1,
                warn_on_dtype=None, estimator=None):
    # TODO: build this out with input_utils for real checks
    if sparse.issparse(array):
        raise NotImplementedError("Sparse matrix support not "
                                  "available yet in cuML check_array")

    # fp16 is not supported, so remove from the list of dtypes if present
    if isinstance(dtype, (list, tuple)):
        dtype = [d for d in dtype if d != np.float16]
    elif dtype == np.float16:
        raise NotImplementedError("Float16 not supported by cuML")

    X, n_rows, n_cols, dtype = input_to_cuml_array(array,
                                                   deepcopy=copy,
                                                   check_dtype=dtype)

    X = X.to_output('cupy')

    # TODO: implement other checks
    if force_all_finite is True and not cp.all(cp.isfinite(X)):
        raise NotImplementedError("Non-finite value encountered in array")

    return X


_input_type_to_str = {
    numpyArray: 'numpy',
    cupyArray: 'cupy',
    cuSeries: 'cudf',
    cuDataFrame: 'cudf',
    pdSeries: 'numpy',
    pdDataFrame: 'numpy'
}


def get_input_type(input):
    # function to access _input_to_str, while still using the correct
    # numba check for a numba device_array
    if type(input) in _input_type_to_str.keys():
        return _input_type_to_str[type(input)]
    elif numbaArray.is_cuda_ndarray(input):
        return 'numba'
    else:
        return 'cupy'


def to_output_type(array, output_type, order='F'):
    cuml_array = input_to_cuml_array(array, order=order)[0]
    return cuml_array.to_output(output_type)
