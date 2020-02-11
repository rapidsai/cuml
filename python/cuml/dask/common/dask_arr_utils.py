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
import cudf
import dask_cudf

import copyreg

import rmm
import dask

from tornado import gen
from dask.distributed import default_client
from toolz import first

from cuml.utils import rmm_cupy_ary

from dask.distributed import wait
from dask import delayed


@gen.coroutine
def extract_arr_partitions(darray, client=None):
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


def _conv_to_sp(dtype):
    def _conv(x):
        data = x.data
        indices = x.indices
        indptr = x.indptr

        data_cp = cp.asarray(data)
        indices_cp = cp.asarray(indices)
        indptr_cp = cp.asarray(indptr)

        ret = cp.sparse.csr_matrix((data_cp, indices_cp, indptr_cp),
                                    dtype=dtype)

        cp.cuda.Stream.null.synchronize()

        return ret

    return _conv


def x_p(x):
    return x


def to_sp_dask_array(cudf_or_array, client=None):
    """
    Converts an array or cuDF to a sparse Dask array backed by sparse CuPy.
    CSR matrices. Unfortunately, due to current limitations in Dask, there is
    no direct path to convert a cupy.sparse.spmatrix into a CuPy backed
    dask.Array without copying to host.


    Parameters
    ----------
    cudf_or_array : cuDF or array-like sparse or dense array
    client : dask.distributed.Client (optional) Dask client

    Returns
    -------
    dask_array : dask.Array backed by cupy.sparse.csr_matrix
    """
    client = default_client() if client is None else client

    if scipy.sparse.isspmatrix(cudf_or_array):
        cudf_or_array = _conv_to_sp(cudf_or_array.dtype)(
            cudf_or_array.tocsr())

    if cp.sparse.isspmatrix(cudf_or_array):

        dtype = cudf_or_array.dtype
        shape = cudf_or_array.shape

        f = client.submit(x_p, cudf_or_array)

        meta = cp.sparse.csr_matrix(cp.zeros(1), dtype=dtype)

        ret = dask.array.from_delayed(f, shape=shape,
                                      meta=meta).persist()

        return ret

    elif isinstance(cudf_or_array, cudf.DataFrame):
        df = cudf_or_array

    elif isinstance(cudf_or_array, np.ndarray):
        df = cudf.DataFrame.from_gpu_matrix(cp.asarray(cudf_or_array))

    elif isinstance(cudf_or_array, cp.ndarray):
        df = cudf.DataFrame.from_gpu_matrix(cudf_or_array)

    else:
        raise ValueError("Unexpected input type.")

    sp = cp.sparse.csr_matrix(cp.asarray(df.as_gpu_matrix(), dtype=df.dtypes[0]))

    f = client.submit(x_p, sp)

    meta = cp.sparse.csr_matrix(cp.zeros(1), dtype=df.dtypes[0])

    return dask.array.from_delayed(f, shape=df.shape,
                                  meta=meta).persist()
