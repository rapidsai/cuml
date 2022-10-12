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

import cuml.internals
import cuml.internals.array
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.logger import debug
from cuml.internals.mem_type import MemoryType
from cuml.internals.safe_imports import (
    cpu_only_import,
    cpu_only_import_from,
    gpu_only_import,
    gpu_only_import_from,
    safe_import,
    null_decorator,
    UnavailableError
)

cudf = gpu_only_import('cudf')
cp = gpu_only_import('cupy')
cupyx = gpu_only_import('cupyx')
dask_cudf = safe_import(
    'dask_cudf',
    msg='Optional dependency dask_cudf is not installed'
)
global_settings = GlobalSettings()
numba_cuda = gpu_only_import('numba.cuda')
np = cpu_only_import('numpy')
pd = cpu_only_import('pandas')
scipy_sparse = safe_import(
    'scipy.sparse',
    msg='Optional dependency scipy is not installed'
)

cp_ndarray = gpu_only_import_from('cupy', 'ndarray')
CudfSeries = gpu_only_import_from('cudf', 'Series')
CudfDataFrame = gpu_only_import_from('cudf', 'DataFrame')
DaskCudfSeries = gpu_only_import_from('dask_cudf.core', 'Series')
DaskCudfDataFrame = gpu_only_import_from('dask_cudf.core', 'DataFrame')
DaskDataFrame = gpu_only_import_from('dask', 'DataFrame')
DaskSeries = gpu_only_import_from('dask', 'Series')
np_ndarray = cpu_only_import_from('numpy', 'ndarray')
NumbaDeviceNDArrayBase = gpu_only_import_from(
    'numba.cuda.devicearray', 'DeviceNDArrayBase'
)
nvtx_annotate = gpu_only_import_from(
    'nvtx',
    'annotate',
    alt=null_decorator
)
PandasSeries = cpu_only_import_from('pandas', 'Series')
PandasDataFrame = cpu_only_import_from('pandas', 'DataFrame')

cuml_array = namedtuple('cuml_array', 'array n_rows n_cols dtype')

_input_type_to_str = {
    CumlArray: "cuml",
    SparseCumlArray: "cuml",
    np_ndarray: "numpy",
    cp_ndarray: "cupy",
    CudfSeries: "cudf",
    CudfDataFrame: "cudf",
    PandasSeries: "numpy",
    PandasDataFrame: "numpy",
    NumbaDeviceNDArrayBase: "numba"
}

_sparse_types = [SparseCumlArray]

try:
    _input_type_to_str[cupyx.scipy.sparse.spmatrix] = 'cupy'
    _sparse_types.append(cupyx.scipy.sparase.spmatrix)
except UnavailableError:
    pass

try:
    _input_type_to_str[scipy_sparse.spmatrix] = 'numpy'
    _sparse_types.append(scipy_sparse.spmatrix)
except UnavailableError:
    pass


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
    :func:`~cuml.internals.input_utils.input_to_cuml_array`, this method will
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

    if (isinstance(X, CudfSeries)):
        if X.null_count != 0:
            return None
        else:
            return CudfSeries

    if isinstance(X, PandasDataFrame):
        return PandasDataFrame

    if isinstance(X, PandasSeries):
        return PandasSeries

    if isinstance(X, CudfDataFrame):
        return CudfDataFrame

    try:
        if numba_cuda.devicearray.is_cuda_ndarray(X):
            return numba_cuda.devicearray.DeviceNDArrayBase
    except UnavailableError:
        pass

    if hasattr(X, "__cuda_array_interface__"):
        return cp.ndarray

    if hasattr(X, "__array_interface__"):
        # For some reason, numpy scalar types also implement
        # `__array_interface__`. See numpy.generic.__doc__. Exclude those types
        # as well as np.dtypes
        if (not isinstance(X, np.generic) and not isinstance(X, type)):
            return np.ndarray

    try:
        if cupyx.scipy.sparse.isspmatrix(X):
            return cupyx.scipy.sparse.spmatrix
    except UnavailableError:
        pass

    try:
        if (scipy_sparse.isspmatrix(X)):
            return scipy_sparse.spmatrix
    except UnavailableError:
        pass

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

    if isinstance(X, (cudf.DataFrame, pd.DataFrame)):
        # Assume single-label target
        dtype = X[X.columns[0]].dtype
    else:
        try:
            dtype = X.dtype
        except AttributeError:
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
    if isinstance(X, CumlArray):
        # Redundant with below, but we try to short-circuit on CumlArray for
        # speed
        return True
    try:
        return (
            hasattr(X, '__cuda_array_interface__')
            or hasattr(X, '__array_interface__')
            or isinstance(X, (
                SparseCumlArray, CudfSeries, PandasSeries, CudfDataFrame,
                PandasDataFrame))
            or numba_cuda.devicearray.is_cuda_ndarray(X)
        ) and not (
            isinstance(X, global_settings.xpy.generic)
            or isinstance(X, type)
        )
    except UnavailableError:
        return False


@nvtx_annotate(message="common.input_utils.input_to_cuml_array",
               category="utils", domain="cuml_python")
@cuml.internals.api_return_any()
def input_to_cuml_array(X,
                        order='F',
                        deepcopy=False,
                        check_dtype=False,
                        convert_to_dtype=False,
                        check_mem_type=False,
                        convert_to_mem_type=None,
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
    if convert_to_mem_type is None:
        convert_to_mem_type = global_settings.memory_type

    if isinstance(
        X,
        (DaskCudfSeries, DaskCudfDataFrame, DaskSeries, DaskDataFrame)
    ):
        # TODO: Warn, but not when using dask_sql
        X = X.compute()

    index = getattr(X, 'index', None)

    if (isinstance(X, CudfSeries)):
        if X.null_count != 0:
            raise ValueError("Error: cuDF Series has missing/null values, "
                             "which are not supported by cuML.")

    if isinstance(X, (PandasSeries, PandasDataFrame)):
        X = X.to_numpy(copy=False)
    if isinstance(X, (CudfSeries, CudfDataFrame)):
        X = X.to_cupy(copy=False)

    arr = CumlArray(X, index=index)
    if deepcopy:
        arr = copy.deepcopy(arr)

    if convert_to_mem_type == MemoryType.mirror:
        convert_to_mem_type = arr.mem_type

    conversion_required = (
        (convert_to_dtype and (convert_to_dtype != arr.dtype))
        or (
            convert_to_mem_type
            and (convert_to_mem_type != arr.mem_type)
        )
    )

    make_copy = False
    if conversion_required:
        convert_to_dtype = convert_to_dtype or None
        convert_to_mem_type = convert_to_mem_type or None
        if (
            safe_dtype_conversion
            and convert_to_dtype is not None
            and not arr.mem_type.xpy.can_cast(
                arr.dtype, convert_to_dtype, casting='safe'
            )
        ):
            raise TypeError('Data type conversion would lose information.')
        arr = CumlArray(
            arr.to_output(
                output_dtype=convert_to_dtype,
                output_mem_type=convert_to_mem_type
            )
        )

    make_copy = force_contiguous and not arr.is_contiguous

    if (order != arr.order and order != 'K') or make_copy:
        arr = CumlArray(arr.mem_type.xpy.array(
            arr.to_output('array'),
            order=order,
            copy=make_copy
        ))

    n_rows = arr.shape[0]

    if len(arr.shape) > 1:
        n_cols = arr.shape[1]
    else:
        n_cols = 1

    if (n_cols == 1 or n_rows == 1) and len(arr.shape) == 2:
        order = 'K'

    if order != 'K' and arr.order != order:
        order_str = order_to_str(order)
        if fail_on_order:
            raise ValueError(
                f"Expected {order_str} major order but got something else."
            )
        else:
            debug(
                f"Expected {order_str} major order but got something else."
                " Converting data; this will result in additional memory"
                " utilization."
            )

    if check_dtype:
        try:
            check_dtype = [
                arr.mem_type.xpy.dtype(dtype) for dtype in check_dtype
            ]
        except TypeError:
            check_dtype = [arr.mem_type.xpy.dtype(check_dtype)]

        if arr.dtype not in check_dtype:
            raise TypeError(
                f"Expected input to be of type in {check_dtype} but got"
                f" {arr.dtype}"
            )

    if check_cols:
        if n_cols != check_cols:
            raise ValueError("Expected " + str(check_cols) +
                             " columns but got " + str(n_cols) + " columns.")

    if check_rows:
        if n_rows != check_rows:
            raise ValueError("Expected " + str(check_rows) + " rows but got " +
                             str(n_rows) + " rows.")

    return cuml_array(array=arr,
                      n_rows=n_rows,
                      n_cols=n_cols,
                      dtype=arr.dtype)


@nvtx_annotate(message="common.input_utils.input_to_cupy_array",
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
        if isinstance(X, (CudfDataFrame, CudfSeries)):
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
                                   force_contiguous=force_contiguous,
                                   convert_to_mem_type=MemoryType.device)

    return out_data._replace(array=out_data.array.to_output("array"))


@nvtx_annotate(message="common.input_utils.input_to_host_array",
               category="utils", domain="cuml_python")
def input_to_host_array(X,
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
    Identical to input_to_cuml_array but it returns a host (NumPy array instead
    of CumlArray
    """
    if not fail_on_null:
        if isinstance(X, (CudfDataFrame, CudfSeries)):
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
                                   force_contiguous=force_contiguous,
                                   convert_to_mem_type=MemoryType.host)

    return out_data._replace(array=out_data.array.to_output("array"))


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
