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

from collections.abc import Iterable

import scipy.sparse
import numpy as np
import cupy as cp
import cupyx
import cudf
import dask

from cuml.utils.memory_utils import with_cupy_rmm

from cuml.dask.common.dask_df_utils import to_dask_cudf
from tornado import gen
from dask.distributed import default_client
from toolz import first

from cuml.dask.common.part_utils import _extract_partitions

from cuml.utils import rmm_cupy_ary

from dask.distributed import wait
from dask import delayed


def validate_dask_array(darray, client=None):
    if len(darray.chunks) > 2:
        raise ValueError("Input array cannot have more than two dimensions")
    elif len(darray.chunks) == 2 and len(darray.chunks[1]) > 1:
        raise ValueError("Input array cannot be chunked along axis 1")


@gen.coroutine
def extract_arr_partitions(darray, client=None):

    # TODO: This will go away once ridge is consolidated to use the mixin

    """
    Given a Dask Array, return an array of tuples mapping each
    worker to their list of futures.

    :param darray: Dask.array split array partitions into a list of
               futures.
    :param client: dask.distributed.Client Optional client to use
    """
    client = default_client() if client is None else client

    if not isinstance(darray, Iterable):
        dist_arr = darray.to_delayed().ravel()
        to_map = dist_arr
    else:
        parts = [arr.to_delayed().ravel() for arr in darray]
        to_map = zip(*parts)

    parts = list(map(delayed, to_map))
    parts = client.compute(parts)

    yield wait(parts)

    who_has = yield client.who_has(parts)

    key_to_part_dict = dict([(str(part.key), part) for part in parts])

    worker_map = {}  # Map from part -> worker
    for key, workers in who_has.items():
        worker = first(workers)
        worker_map[key_to_part_dict[key]] = worker

    worker_to_parts = []
    for part in parts:
        worker = worker_map[part]
        worker_to_parts.append((worker, part))

    yield wait(worker_to_parts)
    raise gen.Return(worker_to_parts)


def _conv_np_to_df(x):
    cupy_ary = rmm_cupy_ary(cp.asarray,
                            x,
                            dtype=x.dtype)
    return cudf.DataFrame.from_gpu_matrix(cupy_ary)


def _conv_df_to_sp(x):
    cupy_ary = rmm_cupy_ary(cp.asarray,
                            x.as_gpu_matrix(),
                            dtype=x.dtypes[0])

    return cp.sparse.csr_matrix(cupy_ary)


@with_cupy_rmm
def to_sp_dask_array(cudf_or_array, client=None):
    """
    Converts an array or cuDF to a sparse Dask array backed by sparse CuPy.
    CSR matrices. Unfortunately, due to current limitations in Dask, there is
    no direct path to convert a cupy.sparse.spmatrix into a CuPy backed
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
    dask_array : dask.Array backed by cupy.sparse.csr_matrix
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
    if isinstance(cudf_or_array, dask.dataframe.DataFrame) or \
       isinstance(cudf_or_array, cudf.DataFrame):
        dtypes = np.unique(cudf_or_array.dtypes)

        if len(dtypes) > 1:
            raise ValueError("DataFrame should contain only a single dtype")

        dtype = dtypes[0]
    else:
        dtype = cudf_or_array.dtype

    meta = cupyx.scipy.sparse.csr_matrix(rmm_cupy_ary(cp.zeros, 1))

    if isinstance(cudf_or_array, dask.array.Array):
        # At the time of developing this, using map_blocks will not work
        # to convert a Dask.Array to CuPy sparse arrays underneath.

        parts = client.sync(_extract_partitions, cudf_or_array)
        cudf_or_array = [client.submit(_conv_np_to_df, part, workers=[w])
                         for w, part in parts]

        cudf_or_array = to_dask_cudf(cudf_or_array)

    if isinstance(cudf_or_array, dask.dataframe.DataFrame):
        """
        Dask.Dataframe needs special attention since it has multiple dtypes.
        Just use the first (and assume all the rest are the same)
        """
        cudf_or_array = cudf_or_array.map_partitions(
            _conv_df_to_sp, meta=dask.array.from_array(meta))

        # This will also handle the input of dask.array.Array
        return cudf_or_array

    else:
        if scipy.sparse.isspmatrix(cudf_or_array):
            cudf_or_array = \
                cupyx.scipy.sparse.csr_matrix(cudf_or_array.tocsr())
        elif cupyx.scipy.sparse.isspmatrix(cudf_or_array):
            pass
        elif isinstance(cudf_or_array, cudf.DataFrame):
            cupy_ary = cp.asarray(cudf_or_array.as_gpu_matrix(), dtype)
            cudf_or_array = cupyx.scipy.sparse.csr_matrix(cupy_ary)
        elif isinstance(cudf_or_array, np.ndarray):
            cupy_ary = rmm_cupy_ary(cp.asarray,
                                    cudf_or_array,
                                    dtype=cudf_or_array.dtype)
            cudf_or_array = cupyx.scipy.sparse.csr_matrix(cupy_ary)

        elif isinstance(cudf_or_array, cp.core.core.ndarray):
            cudf_or_array = cupyx.scipy.sparse.csr_matrix(cudf_or_array)
        else:
            raise ValueError("Unexpected input type %s" % type(cudf_or_array))

        # Push to worker
        cudf_or_array = client.scatter(cudf_or_array)

    return dask.array.from_delayed(cudf_or_array, shape=shape,
                                   meta=meta)
