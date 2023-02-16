# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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


from cuml.common import rmm_cupy_ary, has_scipy
from cuml.dask.common.part_utils import _extract_partitions
from dask.distributed import default_client
from cuml.dask.common.dask_df_utils import to_dask_cudf as df_to_dask_cudf
from cuml.internals.memory_utils import with_cupy_rmm
import dask.dataframe as dd
import dask
from cuml.internals.safe_imports import gpu_only_import
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")
cupyx = gpu_only_import("cupyx")
cudf = gpu_only_import("cudf")


def validate_dask_array(darray, client=None):
    if len(darray.chunks) > 2:
        raise ValueError("Input array cannot have more than two dimensions")
    elif len(darray.chunks) == 2 and len(darray.chunks[1]) > 1:
        raise ValueError("Input array cannot be chunked along axis 1")


def _conv_df_to_sparse(x):
    cupy_ary = rmm_cupy_ary(cp.asarray, x.to_cupy(), dtype=x.dtypes[0])

    return cupyx.scipy.sparse.csr_matrix(cupy_ary)


def _conv_array_to_sparse(arr):
    """
    Converts an array (or cudf.DataFrame) to a sparse array
    :param arr: scipy or cupy sparse matrix, cudf DataFrame,
                dense numpy or cupy array
    :return: cupy sparse CSR matrix
    """
    if has_scipy():
        from scipy.sparse import isspmatrix as scipy_sparse_isspmatrix
    else:
        from cuml.internals.import_utils import (
            dummy_function_always_false as scipy_sparse_isspmatrix,
        )
    if scipy_sparse_isspmatrix(arr):
        ret = cupyx.scipy.sparse.csr_matrix(arr.tocsr())
    elif cupyx.scipy.sparse.isspmatrix(arr):
        ret = arr
    elif isinstance(arr, cudf.DataFrame):
        ret = _conv_df_to_sparse(arr)
    elif isinstance(arr, np.ndarray):
        cupy_ary = rmm_cupy_ary(cp.asarray, arr, dtype=arr.dtype)
        ret = cupyx.scipy.sparse.csr_matrix(cupy_ary)

    elif isinstance(arr, cp.ndarray):
        ret = cupyx.scipy.sparse.csr_matrix(arr)
    else:
        raise ValueError("Unexpected input type %s" % type(arr))
    return ret


@with_cupy_rmm
def to_sparse_dask_array(cudf_or_array, client=None):
    """
    Converts an array or cuDF to a sparse Dask array backed by sparse CuPy.
    CSR matrices. Unfortunately, due to current limitations in Dask, there is
    no direct path to convert a cupyx.scipy.sparse.spmatrix into a CuPy backed
    dask.Array without copying to host.


    NOTE: Until https://github.com/cupy/cupy/issues/2655 and
    https://github.com/dask/dask/issues/5604 are implemented, compute()
    will not be able to be called on a Dask.array that is backed with
    sparse CuPy arrays because they lack the necessary functionality
    to be stacked into a single array. The array returned from this
    utility will, however, still be able to be passed into functions
    that can make use of sparse CuPy-backed Dask.Array (eg. Distributed
    Naive Bayes).

    Relevant cuML issue: https://github.com/rapidsai/cuml/issues/1387

    Parameters
    ----------
    cudf_or_array : cuDF Dataframe, array-like sparse / dense array, or
                    Dask DataFrame/Array
    client : dask.distributed.Client (optional) Dask client

    dtype : output dtype

    Returns
    -------
    dask_array : dask.Array backed by cupyx.scipy.sparse.csr_matrix
    """
    client = default_client() if client is None else client

    # Makes sure the MatDescriptor workaround for CuPy sparse arrays
    # is loaded (since Dask lazy-loaded serialization in cuML is only
    # executed when object from the cuML package needs serialization.
    # This can go away once the MatDescriptor pickling bug is fixed
    # in CuPy.
    # Ref: https://github.com/cupy/cupy/issues/3061
    from cuml.comm import serialize  # NOQA

    shape = cudf_or_array.shape

    meta = cupyx.scipy.sparse.csr_matrix(rmm_cupy_ary(cp.zeros, 1))

    ret = cudf_or_array

    # If we have a Dask array, convert it to a Dask DataFrame
    if isinstance(ret, dask.array.Array):
        # At the time of developing this, using map_blocks will not work
        # to convert a Dask.Array to CuPy sparse arrays underneath.

        def _conv_np_to_df(x):
            cupy_ary = rmm_cupy_ary(cp.asarray, x, dtype=x.dtype)
            return cudf.DataFrame(cupy_ary)

        parts = client.sync(_extract_partitions, ret)
        futures = [
            client.submit(_conv_np_to_df, part, workers=[w], pure=False)
            for w, part in parts
        ]

        ret = df_to_dask_cudf(futures)

    # If we have a Dask Dataframe, use `map_partitions` to convert it
    # to a Sparse Cupy-backed Dask Array. This will also convert the dense
    # Dask array above to a Sparse Cupy-backed Dask Array, since we cannot
    # use map_blocks on the array, but we can use `map_partitions` on the
    # Dataframe.
    if isinstance(ret, dask.dataframe.DataFrame):
        ret = ret.map_partitions(
            _conv_df_to_sparse, meta=dask.array.from_array(meta)
        )

        # This will also handle the input of dask.array.Array
        return ret

    else:

        ret = _conv_array_to_sparse(ret)

        # Push to worker
        final_result = client.scatter(ret)

        return dask.array.from_delayed(final_result, shape=shape, meta=meta)


def _get_meta(df):
    ret = df.iloc[:0]
    return ret


@dask.delayed
def _to_cudf(arr):
    if arr.ndim == 2:
        return cudf.DataFrame(arr)
    elif arr.ndim == 1:
        return cudf.Series(arr)


def to_dask_cudf(dask_arr, client=None):
    client = default_client() if client is None else client

    elms = [_to_cudf(dp) for dp in dask_arr.to_delayed().flatten()]
    dfs = client.compute(elms)

    meta = client.submit(_get_meta, dfs[0])
    meta_local = meta.result()

    return dd.from_delayed(dfs, meta=meta_local)
