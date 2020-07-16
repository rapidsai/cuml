#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import copy
import cudf
import cupy as cp
import numpy as np
import pandas as pd

from collections import namedtuple
from cuml.common import CumlArray
from cuml.common.logger import warn
from cuml.common.memory_utils import with_cupy_rmm
from cuml.common.memory_utils import _check_array_contiguity
from numba import cuda


cuml_array = namedtuple('cuml_array', 'array n_rows n_cols dtype')

# inp_array is deprecated and will be dropped once cuml array is adopted
# in all algos. Github issue #1716
inp_array = namedtuple('inp_array', 'array pointer n_rows n_cols dtype')


def get_dev_array_ptr(ary):
    """
    Returns ctype pointer of a numba style device array

    Deprecated: will be removed once all codebase uses cuml Array
    See Github issue #1716
    """
    return ary.device_ctypes_pointer.value


def get_cudf_column_ptr(col):
    """
    Returns pointer of a cudf Series

    Deprecated: will be removed once all codebase uses cuml Array
    See Github issue #1716
    """
    return col.__cuda_array_interface__['data'][0]


@with_cupy_rmm
def input_to_cuml_array(X, order='F', deepcopy=False,
                        check_dtype=False, convert_to_dtype=False,
                        check_cols=False, check_rows=False,
                        fail_on_order=False, force_contiguous=True):
    """
    Convert input X to CumlArray.

    Acceptable input formats:

    * cuDF Dataframe - returns a deep copy always.
    * cuDF Series - returns by reference or a deep copy depending on
        `deepcopy`.
    * Numpy array - returns a copy in device always
    * cuda array interface compliant array (like Cupy) - returns a
        reference unless `deepcopy`=True.
    * numba device array - returns a reference unless deepcopy=True

    Parameters
    ----------

    X : cuDF.DataFrame, cuDF.Series, NumPy array, Pandas DataFrame, Pandas
        Series or any cuda_array_interface (CAI) compliant array like CuPy,
        Numba or pytorch.

    order: 'F', 'C' or 'K' (default: 'F')
        Whether to return a F-major ('F'),  C-major ('C') array or Keep ('K')
        the order of X. Used to check the order of the input. If
        fail_on_order=True, the method will raise ValueError,
        otherwise it will convert X to be of order `order` if needed.

    deepcopy: boolean (default: False)
        Set to True to always return a deep copy of X.

    check_dtype: np.dtype (default: False)
        Set to a np.dtype to throw an error if X is not of dtype `check_dtype`.

    convert_to_dtype: np.dtype (default: False)
        Set to a dtype if you want X to be converted to that dtype if it is
        not that dtype already.

    check_cols: int (default: False)
        Set to an int `i` to check that input X has `i` columns. Set to False
        (default) to not check at all.

    check_rows: boolean (default: False)
        Set to an int `i` to check that input X has `i` columns. Set to False
        (default) to not check at all.

    fail_on_order: boolean (default: False)
        Set to True if you want the method to raise a ValueError if X is not
        of order `order`.

    force_contiguous: boolean (default: True)
        Set to True to force CumlArray produced to be contiguous. If `X` is
        non contiguous then a contiguous copy will be done.
        If False, and `X` doesn't need to be converted and is not contiguous,
        the underlying memory underneath the CumlArray will be non contiguous.
        Only affects CAI inputs. Only affects CuPy and Numba device array
        views, all other input methods produce contiguous CumlArrays.

    Returns
    -------
    `cuml_array`: namedtuple('cuml_array', 'array n_rows n_cols dtype')

        A new CumlArray and associated data.

    """

    # dtype conversion

    # force_contiguous set to True always for now
    # upcoming CumlArray improvements will affect this
    # https://github.com/rapidsai/cuml/issues/2412
    force_contiguous = True

    if convert_to_dtype:
        X = convert_dtype(X, to_dtype=convert_to_dtype)
        check_dtype = False

    # format conversion

    if (isinstance(X, cudf.Series)):
        if X.null_count != 0:
            raise ValueError("Error: cuDF Series has missing/null values, \
                             which are not supported by cuML.")

    # converting pandas to numpy before sending it to CumlArray
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        # pandas doesn't support custom order in to_numpy
        X = cp.asarray(X.to_numpy(copy=False), order=order)

    if isinstance(X, cudf.DataFrame):
        if order == 'K':
            X_m = CumlArray(data=X.as_gpu_matrix(order='F'))
        else:
            X_m = CumlArray(data=X.as_gpu_matrix(order=order))

    elif isinstance(X, CumlArray):
        X_m = X

    elif hasattr(X, "__array_interface__") or \
            hasattr(X, "__cuda_array_interface__"):

        if force_contiguous or hasattr(X, "__array_interface__"):
            if not _check_array_contiguity(X):
                warn("Non contiguous array or view detected, a \
                     contiguous copy of the data will be done. ")
                X = cp.array(X, order=order, copy=True)

        X_m = CumlArray(data=X)

        if deepcopy:
            X_m = copy.deepcopy(X_m)

    else:
        msg = "X matrix format " + str(X.__class__) + " not supported"
        raise TypeError(msg)

    if check_dtype:
        if not isinstance(check_dtype, list):
            check_dtype = [check_dtype]

        check_dtype = [np.dtype(dtype) for dtype in check_dtype]

        if X_m.dtype not in check_dtype:
            type_str = X_m.dtype
            del X_m
            raise TypeError("Expected input to be of type in " +
                            str(check_dtype) + " but got " + str(type_str))

    # Checks based on parameters

    n_rows = X_m.shape[0]

    if len(X_m.shape) > 1:
        n_cols = X_m.shape[1]
    else:
        n_cols = 1

    if n_cols == 1 or n_rows == 1:
        order = 'K'

    if check_cols:
        if n_cols != check_cols:
            raise ValueError("Expected " + str(check_cols) +
                             " columns but got " + str(n_cols) +
                             " columns.")

    if check_rows:
        if n_rows != check_rows:
            raise ValueError("Expected " + str(check_rows) +
                             " rows but got " + str(n_rows) +
                             " rows.")

    if order != 'K' and X_m.order != order:
        if fail_on_order:
            raise ValueError("Expected " + order_to_str(order) +
                             " major order, but got the opposite.")
        else:
            warn("Expected " + order_to_str(order) + " major order, "
                 "but got the opposite. Converting data, this will "
                 "result in additional memory utilization.")
            X_m = cp.array(X_m, copy=False, order=order)
            X_m = CumlArray(data=X_m)

    return cuml_array(array=X_m, n_rows=n_rows, n_cols=n_cols, dtype=X_m.dtype)


def input_to_host_array(X, order='F', deepcopy=False,
                        check_dtype=False, convert_to_dtype=False,
                        check_cols=False, check_rows=False,
                        fail_on_order=False):
    """
    Convert input X to host array (NumPy) suitable for C++ methods that accept
    host arrays.

    Acceptable input formats:

    * Numpy array - returns a pointer to the original input

    * cuDF Dataframe - returns a deep copy always

    * cuDF Series - returns by reference or a deep copy depending on `deepcopy`

    * cuda array interface compliant array (like Cupy) - returns a \
        reference unless deepcopy=True

    * numba device array - returns a reference unless deepcopy=True

    Parameters
        ----------

    X:
        cuDF.DataFrame, cuDF.Series, numba array, NumPy array or any
        cuda_array_interface compliant array like CuPy or pytorch.

    order: string (default: 'F')
        Whether to return a F-major or C-major array. Used to check the order
        of the input. If fail_on_order=True method will raise ValueError,
        otherwise it will convert X to be of order `order`.

    deepcopy: boolean (default: False)
        Set to True to always return a deep copy of X.

    check_dtype: np.dtype (default: False)
        Set to a np.dtype to throw an error if X is not of dtype `check_dtype`.

    convert_to_dtype: np.dtype (default: False)
        Set to a dtype if you want X to be converted to that dtype if it is
        not that dtype already.

    check_cols: int (default: False)
        Set to an int `i` to check that input X has `i` columns. Set to False
        (default) to not check at all.

    check_rows: boolean (default: False)
        Set to an int `i` to check that input X has `i` columns. Set to False
        (default) to not check at all.

    fail_on_order: boolean (default: False)
        Set to True if you want the method to raise a ValueError if X is not
        of order `order`.


    Returns
    -------
    `inp_array`: namedtuple('inp_array', 'array pointer n_rows n_cols dtype')

    `inp_array` is a new device array if the input was not a NumPy device
        array. It is a reference to the input X if it was a NumPy host array
    """

    ary_tuple = input_to_cuml_array(X,
                                    order=order,
                                    deepcopy=deepcopy,
                                    check_dtype=check_dtype,
                                    convert_to_dtype=convert_to_dtype,
                                    check_cols=check_cols,
                                    check_rows=check_rows,
                                    fail_on_order=fail_on_order)

    X_m = ary_tuple.array.to_output('numpy')

    return inp_array(array=X_m,
                     pointer=X_m.__array_interface__['data'][0],
                     n_rows=ary_tuple.n_rows,
                     n_cols=ary_tuple.n_cols,
                     dtype=ary_tuple.dtype)


def input_to_dev_array(X, order='F', deepcopy=False,
                       check_dtype=False, convert_to_dtype=False,
                       check_cols=False, check_rows=False,
                       fail_on_order=False):
    """
    *** Deprecated, used in classes that have not migrated to use cuML Array
    yet. Please use input_to_cuml_array instead for cuml Array.
    See Github issue #1716 ***

    Convert input X to device array suitable for C++ methods.

    Acceptable input formats:

    * cuDF Dataframe - returns a deep copy always.
    * cuDF Series - returns by reference or a deep copy depending on
        `deepcopy`.
    * Numpy array - returns a copy in device always
    * cuda array interface compliant array (like Cupy) - returns a
        reference unless `deepcopy`=True.
    * numba device array - returns a reference unless deepcopy=True

    Parameters
    ----------

    X : cuDF.DataFrame, cuDF.Series, numba array, NumPy array or any
        cuda_array_interface compliant array like CuPy or pytorch.

    order: string (default: 'F')
        Whether to return a F-major or C-major array. Used to check the order
        of the input. If fail_on_order=True method will raise ValueError,
        otherwise it will convert X to be of order `order`.

    deepcopy: boolean (default: False)
        Set to True to always return a deep copy of X.

    check_dtype: np.dtype (default: False)
        Set to a np.dtype to throw an error if X is not of dtype `check_dtype`.

    convert_to_dtype: np.dtype (default: False)
        Set to a dtype if you want X to be converted to that dtype if it is
        not that dtype already.

    check_cols: int (default: False)
        Set to an int `i` to check that input X has `i` columns. Set to False
        (default) to not check at all.

    check_rows: boolean (default: False)
        Set to an int `i` to check that input X has `i` columns. Set to False
        (default) to not check at all.

    fail_on_order: boolean (default: False)
        Set to True if you want the method to raise a ValueError if X is not
        of order `order`.

    Returns
    -------
    `inp_array`: namedtuple('inp_array', 'array pointer n_rows n_cols dtype')

        A new device array if the input was not a numba device
        array. It is a reference to the input X if it was a numba device array
        or cuda array interface compliant (like cupy)

    """

    ary_tuple = input_to_cuml_array(X,
                                    order=order,
                                    deepcopy=deepcopy,
                                    check_dtype=check_dtype,
                                    convert_to_dtype=convert_to_dtype,
                                    check_cols=check_cols,
                                    check_rows=check_rows,
                                    fail_on_order=fail_on_order)

    return inp_array(array=cuda.as_cuda_array(ary_tuple.array),
                     pointer=ary_tuple.array.ptr,
                     n_rows=ary_tuple.n_rows,
                     n_cols=ary_tuple.n_cols,
                     dtype=ary_tuple.dtype)


@with_cupy_rmm
def convert_dtype(X, to_dtype=np.float32, legacy=True):
    """
    Convert X to be of dtype `dtype`, raising a TypeError
    if the conversion would lose information.
    """
    would_lose_info = _typecast_will_lose_information(X, to_dtype)
    if would_lose_info:
        raise TypeError("Data type conversion would lose information.")

    if isinstance(X, np.ndarray):
        dtype = X.dtype
        if dtype != to_dtype:
            X_m = X.astype(to_dtype)
            return X_m

    elif isinstance(X, (cudf.Series, cudf.DataFrame, pd.Series, pd.DataFrame)):
        return X.astype(to_dtype, copy=False)

    elif cuda.is_cuda_array(X):
        X_m = cp.asarray(X)
        X_m = X_m.astype(to_dtype, copy=False)

        if legacy:
            return cuda.as_cuda_array(X_m)
        else:
            return CumlArray(data=X_m)

    else:
        raise TypeError("Received unsupported input type: %s" % type(X))

    return X


def _typecast_will_lose_information(X, target_dtype):
    """
    Returns True if typecast will cause information loss, else False.
    Handles float/float, float/int, and int/int typecasts.
    """
    target_dtype = np.dtype(target_dtype).type

    if target_dtype in (np.int8, np.int16, np.int32, np.int64):
        target_dtype_range = np.iinfo(target_dtype)
    else:
        target_dtype_range = np.finfo(target_dtype)

    if isinstance(X, (np.ndarray, cp.ndarray, pd.Series, cudf.Series)):
        if X.dtype.type == target_dtype:
            return False

        return (
            (X < target_dtype_range.min) |
            (X > target_dtype_range.max)
        ).any()

    elif isinstance(X, (pd.DataFrame, cudf.DataFrame)):
        X_m = X.values
        return _typecast_will_lose_information(X_m, target_dtype)

    elif cuda.is_cuda_array(X):
        X_m = cp.asarray(X)
        return _typecast_will_lose_information(X_m, target_dtype)

    else:
        raise TypeError("Received unsupported input type: %s" % type(X))


def order_to_str(order):
    if order == 'F':
        return 'column (\'F\')'
    elif order == 'C':
        return 'row (\'C\')'


def sparse_scipy_to_cp(sp, dtype):
    """
    Convert object of scipy.sparse to
    cupy.sparse.coo_matrix
    """

    coo = sp.tocoo()
    values = coo.data

    r = cp.asarray(coo.row)
    c = cp.asarray(coo.col)
    v = cp.asarray(values, dtype=dtype)

    return cp.sparse.coo_matrix((v, (r, c)), sp.shape)
