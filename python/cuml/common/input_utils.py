#
# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
from collections import namedtuple
import nvtx

import cudf
import cupy as cp
import cupyx
import numba.cuda
import numpy as np
import pandas as pd
import cuml.internals
import cuml.common.array
from cuml.common.array import CumlArray
from cuml.common.array_sparse import SparseCumlArray
from cuml.common.import_utils import has_scipy, has_dask_cudf
from cuml.common.logger import debug
from cuml.common.memory_utils import ArrayInfo
from cuml.common.memory_utils import _check_array_contiguity

if has_scipy():
    import scipy.sparse

if has_dask_cudf():
    import dask_cudf

cuml_array = namedtuple('cuml_array', 'array n_rows n_cols dtype')

# inp_array is deprecated and will be dropped once cuml array is adopted
# in all algos. Github issue #1716
inp_array = namedtuple('inp_array', 'array pointer n_rows n_cols dtype')

unsupported_cudf_dtypes = [
    np.uint8, np.uint16, np.uint32, np.uint64, np.float16
]

_input_type_to_str = {
    CumlArray: "cuml",
    SparseCumlArray: "cuml",
    np.ndarray: "numpy",
    cp.ndarray: "cupy",
    cudf.Series: "cudf",
    cudf.DataFrame: "cudf",
    pd.Series: "numpy",
    pd.DataFrame: "numpy",
    numba.cuda.devicearray.DeviceNDArrayBase: "numba",
    cupyx.scipy.sparse.spmatrix: "cupy",
}

_sparse_types = [
    SparseCumlArray,
    cupyx.scipy.sparse.spmatrix,
]

if has_scipy():
    _input_type_to_str.update({
        scipy.sparse.spmatrix: "numpy",
    })

    _sparse_types.append(scipy.sparse.spmatrix)


def get_supported_input_type(X):
    """
    Determines if the input object is a supported input array-like object or
    not. If supported, the type is returned. Otherwise, `None` is returned.

    Parameters
    ----------
    X : object
        Input object to test

    Notes
    -----
    To closely match the functionality of
    :func:`~cuml.common.input_utils.input_to_cuml_array`, this method will
    return `cupy.ndarray` for any object supporting
    `__cuda_array_interface__` and `numpy.ndarray` for any object supporting
    `__array_interface__`.

    Returns
    -------
    array-like type or None
        If the array-like object is supported, the type is returned.
        Otherwise, `None` is returned.
    """
    # Check CumlArray first to shorten search time
    if isinstance(X, CumlArray):
        return CumlArray

    if isinstance(X, SparseCumlArray):
        return SparseCumlArray

    if (isinstance(X, cudf.Series)):
        if X.null_count != 0:
            return None
        else:
            return cudf.Series

    # converting pandas to numpy before sending it to CumlArray
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame

    if isinstance(X, pd.Series):
        return pd.Series

    if isinstance(X, cudf.DataFrame):
        return cudf.DataFrame

    if numba.cuda.devicearray.is_cuda_ndarray(X):
        return numba.cuda.devicearray.DeviceNDArrayBase

    if hasattr(X, "__cuda_array_interface__"):
        return cp.ndarray

    if hasattr(X, "__array_interface__"):
        # For some reason, numpy scalar types also implement
        # `__array_interface__`. See numpy.generic.__doc__. Exclude those types
        # as well as np.dtypes
        if (not isinstance(X, np.generic) and not isinstance(X, type)):
            return np.ndarray

    if cupyx.scipy.sparse.isspmatrix(X):
        return cupyx.scipy.sparse.spmatrix

    if has_scipy():
        if (scipy.sparse.isspmatrix(X)):
            return scipy.sparse.spmatrix

    # Return None if this type isnt supported
    return None


def determine_array_type(X):
    if (X is None):
        return None

    # Get the generic type
    gen_type = get_supported_input_type(X)

    return None if gen_type is None else _input_type_to_str[gen_type]


def determine_array_dtype(X):

    if (X is None):
        return None

    canonical_input_types = tuple(_input_type_to_str.keys())

    if isinstance(X, (cudf.DataFrame, pd.DataFrame)):
        # Assume single-label target
        dtype = X[X.columns[0]].dtype
    elif isinstance(X, canonical_input_types):
        dtype = X.dtype
    else:
        dtype = None

    return dtype


def determine_array_type_full(X):
    """
    Returns a tuple of the array type, and a boolean if it is sparse

    Parameters
    ----------
    X : array-like
        Input array to test

    Returns
    -------
    (string, bool) Returns a tuple of the array type string and a boolean if it
        is a sparse array.
    """
    if (X is None):
        return None, None

    # Get the generic type
    gen_type = get_supported_input_type(X)

    if (gen_type is None):
        return None, None

    return _input_type_to_str[gen_type], gen_type in _sparse_types


def is_array_like(X):
    return determine_array_type(X) is not None


@nvtx.annotate(message="common.input_utils.input_to_cuml_array",
               category="utils", domain="cuml_python")
@cuml.internals.api_return_any()
def input_to_cuml_array(X,
                        order='F',
                        deepcopy=False,
                        check_dtype=False,
                        convert_to_dtype=False,
                        safe_dtype_conversion=True,
                        check_cols=False,
                        check_rows=False,
                        fail_on_order=False,
                        force_contiguous=True):
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

    safe_convert_to_dtype: bool (default: True)
        Set to True to check whether a typecasting performed when
        convert_to_dtype is True will cause information loss. This has a
        performance implication that might be significant for very fast
        methods like FIL and linear models inference.

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
    def check_order(arr_order):
        if order != 'K' and arr_order != order:
            if fail_on_order:
                raise ValueError("Expected " + order_to_str(order) +
                                 " major order, but got the opposite.")
            else:
                debug("Expected " + order_to_str(order) + " major order, "
                      "but got the opposite. Converting data, this will "
                      "result in additional memory utilization.")
                return True
        return False

    # dtype conversion

    # force_contiguous set to True always for now
    # upcoming CumlArray improvements will affect this
    # https://github.com/rapidsai/cuml/issues/2412
    force_contiguous = True

    if convert_to_dtype:
        X = convert_dtype(X,
                          to_dtype=convert_to_dtype,
                          safe_dtype=safe_dtype_conversion)
        check_dtype = False

    index = getattr(X, 'index', None)

    # format conversion

    if isinstance(X, (dask_cudf.core.Series, dask_cudf.core.DataFrame)):
        # TODO: Warn, but not when using dask_sql
        X = X.compute()

    if (isinstance(X, cudf.Series)):
        if X.null_count != 0:
            raise ValueError("Error: cuDF Series has missing/null values, "
                             "which are not supported by cuML.")

    # converting pandas to numpy before sending it to CumlArray
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        # pandas doesn't support custom order in to_numpy
        X = cp.asarray(X.to_numpy(copy=False), order=order)

    if isinstance(X, cudf.DataFrame):
        if order == 'K':
            X_m = CumlArray(data=X.to_cupy(), index=index)
        else:
            X_m = CumlArray(data=cp.array(X.to_cupy(), order=order),
                            index=index)

    elif isinstance(X, CumlArray):
        X_m = X

    elif hasattr(X, "__array_interface__") or \
            hasattr(X, "__cuda_array_interface__"):

        host_array = hasattr(X, "__array_interface__")

        # Since we create the array with the correct order here, do the order
        # check now if necessary
        interface = getattr(X, "__array_interface__", None) or getattr(
            X, "__cuda_array_interface__", None)

        arr_info = ArrayInfo.from_interface(interface)

        check_order(arr_info.order)

        make_copy = False

        if force_contiguous or hasattr(X, "__array_interface__"):
            if not _check_array_contiguity(X):
                debug("Non contiguous array or view detected, a "
                      "contiguous copy of the data will be done.")
                make_copy = True

        # If we have a host array, we copy it first before changing order
        # to transpose using the GPU
        if host_array:
            X = cp.array(X)

        cp_arr = cp.array(X, copy=make_copy, order=order)

        X_m = CumlArray(data=cp_arr,
                        index=index)

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
                             " columns but got " + str(n_cols) + " columns.")

    if check_rows:
        if n_rows != check_rows:
            raise ValueError("Expected " + str(check_rows) + " rows but got " +
                             str(n_rows) + " rows.")

    if (check_order(X_m.order)):
        X_m = cp.array(X_m, copy=False, order=order)
        X_m = CumlArray(data=X_m,
                        index=index)

    return cuml_array(array=X_m,
                      n_rows=n_rows,
                      n_cols=n_cols,
                      dtype=X_m.dtype)


@nvtx.annotate(message="common.input_utils.input_to_cupy_array",
               category="utils", domain="cuml_python")
def input_to_cupy_array(X,
                        order='F',
                        deepcopy=False,
                        check_dtype=False,
                        convert_to_dtype=False,
                        check_cols=False,
                        check_rows=False,
                        fail_on_order=False,
                        force_contiguous=True,
                        fail_on_null=True) -> cuml_array:
    """
    Identical to input_to_cuml_array but it returns a cupy array instead of
    CumlArray
    """
    if not fail_on_null:
        if isinstance(X, (cudf.DataFrame, cudf.Series)):
            try:
                X = X.values
            except ValueError:
                X = X.astype('float64', copy=False)
                X.fillna(cp.nan, inplace=True)
                X = X.values

    out_data = input_to_cuml_array(X,
                                   order=order,
                                   deepcopy=deepcopy,
                                   check_dtype=check_dtype,
                                   convert_to_dtype=convert_to_dtype,
                                   check_cols=check_cols,
                                   check_rows=check_rows,
                                   fail_on_order=fail_on_order,
                                   force_contiguous=force_contiguous)

    return out_data._replace(array=out_data.array.to_output("cupy"))


@nvtx.annotate(message="common.input_utils.input_to_host_array",
               category="utils", domain="cuml_python")
def input_to_host_array(X,
                        order='F',
                        deepcopy=False,
                        check_dtype=False,
                        convert_to_dtype=False,
                        check_cols=False,
                        check_rows=False,
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

    if isinstance(X, np.ndarray):
        if len(X.shape) > 1:
            n_cols = X.shape[1]
        else:
            n_cols = 1
        return inp_array(array=X,
                         pointer=X.__array_interface__['data'][0],
                         n_rows=X.shape[0],
                         n_cols=n_cols,
                         dtype=X.dtype)

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


@cuml.internals.api_return_any()
def convert_dtype(X,
                  to_dtype=np.float32,
                  legacy=True,
                  safe_dtype=True):
    """
    Convert X to be of dtype `dtype`, raising a TypeError
    if the conversion would lose information.
    """

    if isinstance(X, (dask_cudf.core.Series, dask_cudf.core.DataFrame)):
        # TODO: Warn, but not when using dask_sql
        X = X.compute()

    if safe_dtype:
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

    elif numba.cuda.is_cuda_array(X):
        X_m = cp.asarray(X)
        X_m = X_m.astype(to_dtype, copy=False)

        if legacy:
            return numba.cuda.as_cuda_array(X_m)
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

        # if we are casting to a bigger data type
        if np.dtype(X.dtype) <= np.dtype(target_dtype):
            return False

        return ((X < target_dtype_range.min) |
                (X > target_dtype_range.max)).any()

    elif isinstance(X, (pd.DataFrame, cudf.DataFrame)):
        X_m = X.values
        return _typecast_will_lose_information(X_m, target_dtype)

    elif numba.cuda.is_cuda_array(X):
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
    cupyx.scipy.sparse.coo_matrix
    """

    coo = sp.tocoo()
    values = coo.data

    r = cp.asarray(coo.row)
    c = cp.asarray(coo.col)
    v = cp.asarray(values, dtype=dtype)

    return cupyx.scipy.sparse.coo_matrix((v, (r, c)), sp.shape)
