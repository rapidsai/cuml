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

import cuml.utils.numba_utils

import cudf
import numpy as np

from numba import cuda

from librmm_cffi import librmm as rmm


def get_dev_array_ptr(ary):
    """
    Returns ctype pointer of a numba style device array
    """
    return ary.device_ctypes_pointer.value


def get_cudf_column_ptr(col):
    """
    Returns ctype pointer of a cudf column
    """
    return cudf.bindings.cudf_cpp.get_column_data_ptr(col._column)


def get_dtype(X):
    """
    Returns dtype of obj as a Numpy style dtype (like np.float32)
    """
    if isinstance(X, cudf.DataFrame):
        dtype = np.dtype(X[X.columns[0]]._column.dtype)
    elif (isinstance(X, cudf.Series)):
        dtype = np.dtype(X._column.dtype)
    elif isinstance(X, np.ndarray):
        dtype = X.dtype
    elif cuda.is_cuda_array(X):
        dtype = X.dtype
    elif cuda.devicearray.is_cuda_ndarray(X):
        dtype = X.dtype

    return dtype


def input_to_array(X, order='F', deepcopy=False,
                   check_dtype=False, convert_to_dtype=False,
                   check_cols=False, check_rows=False):
    """
    Convert input X to device array suitable for C++ methods
    Acceptable input formats:
    * cuDF Dataframe - returns a deep copy always
    * cuDF Series - returns by reference or a deep copy depending on
        `deepcopy`
    * Numpy array - returns a copy in device always
    * cuda array interface compliant array (like Cupy) - returns a
        reference unless deepcopy=True
    * numba device array - returns a reference unless deepcopy=True

    Returns a new device array if the input was not a numba device array.
    Returns a reference to the input X if its a numba device array or cuda
        array interface compliant (like cupy)
    """

    if convert_to_dtype:
        X = convert_dtype(X, to_dtype=convert_to_dtype)
        check_dtype = False

    if isinstance(X, cudf.DataFrame):
        dtype = np.dtype(X[X.columns[0]]._column.dtype)
        if order == 'F':
            X_m = X.as_gpu_matrix(order='F')
        elif order == 'C':
            X_m = cuml.utils.numba_utils.row_matrix(X)

    elif (isinstance(X, cudf.Series)):
        if deepcopy:
            X_m = X.to_gpu_array()
        else:
            X_m = X._column._data.mem

    elif isinstance(X, np.ndarray):
        dtype = X.dtype
        X_m = rmm.to_device(np.array(X, order=order, copy=False))

    elif cuda.is_cuda_array(X):
        # Use cuda array interface to create a device array by reference
        X_m = cuda.as_cuda_array(X)

    elif cuda.devicearray.is_cuda_ndarray(X):
        X_m = X

    else:
        msg = "X matrix format " + str(X.__class__) + " not supported"
        raise TypeError(msg)

    dtype = X_m.dtype

    if check_dtype:
        if dtype != check_dtype:
            del X_m
            raise TypeError("ba")

    n_rows = X_m.shape[0]
    if len(X_m.shape) > 1:
        n_cols = X_m.shape[1]
    else:
        n_cols = 1

    if check_cols:
        if n_cols != check_cols:
            raise ValueError("ba")

    if check_rows:
        if n_rows != check_rows:
            raise ValueError("ba")

    X_ptr = get_dev_array_ptr(X_m)

    # todo: add check of alignment and nans

    return X_m, X_ptr, n_rows, n_cols, dtype


def convert_dtype(X, to_dtype=np.float32):
    """
    Convert X to be of dtype `dtype`
    """

    # Using cuDF for converting numba and device array interface inputs

    if cuda.devicearray.is_cuda_ndarray(X):
        X_df = cudf.Dataframe()
        X = X_df.from_gpu_matrix(X)

    if cuda.is_cuda_array(X):
        X_df = cudf.Dataframe()
        X = X_df.from_gpu_matrix(cuda.as_cuda_array(X))

    if isinstance(X, cudf.DataFrame):
        dtype = np.dtype(X[X.columns[0]]._column.dtype)
        if dtype != to_dtype:
            new_cols = [(col, X._cols[col].astype(to_dtype))
                        for col in X._cols]
            overflowed = sum([len(colval[colval >= np.inf])
                              for colname, colval in new_cols])

            if overflowed > 0:
                raise TypeError("Data type conversion resulted"
                                "in data loss.")

            return cudf.DataFrame(new_cols)

    elif isinstance(X, np.ndarray):
        dtype = X.dtype
        if dtype != to_dtype:
            X_m = np.ascontiguousarray(X.astype(to_dtype))
            if len(X[X == np.inf]) > 0:
                raise TypeError("Data type conversion resulted"
                                "in data loss.")
            return X_m

    else:
        raise TypeError("Received unsupported input type " % type(X))

    return X
