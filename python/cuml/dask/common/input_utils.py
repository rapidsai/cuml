#
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


import cudf
import cupy as cp
import dask.array as da

from cuml.dask.common.dask_df_utils import to_dask_cudf
from cuml.utils.memory_utils import with_cupy_rmm

from collections import OrderedDict
from cudf.core import DataFrame
from dask_cudf.core import DataFrame as dcDataFrame

from dask import delayed
from dask.distributed import wait
from dask.distributed import default_client
from tornado import gen
from toolz import first
from collections.abc import Iterable
from dask.array.core import Array as daskArray


class MGData:

    def __init__(self, gpu_futures=None, worker_to_parts=None, workers=None,
                 datatype=None):
        self.gpu_futures = gpu_futures
        self.worker_to_parts = worker_to_parts
        self.workers = workers
        self.datatype = datatype

    @classmethod
    def single(cls, data, client=None):
        client = default_client() if client is None else client

        gpu_futures = client.sync(_extract_partitions, data, client)

        worker_to_parts = _workers_to_parts(gpu_futures)

        workers = list(map(lambda x: x[0], worker_to_parts.items()))

        datatype = 'cudf' if isinstance(data, dcDataFrame) else 'cupy'

        return MGData(gpu_futures=gpu_futures, worker_to_parts=worker_to_parts,
                      workers=workers, datatype=datatype)

    @classmethod
    def collocated(cls, data, client=None):
        client = default_client() if client is None else client
        print(client)


@with_cupy_rmm
def concatenate(objs, axis=0):
    if isinstance(objs[0], DataFrame):
        if len(objs) == 1:
            return objs[0]
        else:
            return cudf.concat(objs)

    elif isinstance(objs[0], cp.ndarray):
        return cp.concatenate(objs, axis=axis)


@gen.coroutine
def _extract_partitions(dask_obj, client=None):

    client = default_client() if client is None else client

    if isinstance(dask_obj, dcDataFrame):
        delayed_ddf = dask_obj.to_delayed()
        parts = client.compute(delayed_ddf)

    elif isinstance(dask_obj, daskArray):
        dist_arr = dask_obj.to_delayed().ravel()
        to_map = dist_arr
        parts = list(map(delayed, to_map))

    elif isinstance(dask_obj, Iterable):
        parts = [arr.to_delayed().ravel() for arr in dask_obj]
        to_map = zip(*parts)
        parts = list(map(delayed, to_map))

    else:
        raise TypeError("Unsupported dask_obj type: " + type(dask_obj))

    parts = client.compute(parts)
    yield wait(parts)

    key_to_part_dict = dict([(str(part.key), part) for part in parts])
    who_has = yield client.who_has(parts)

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


def _workers_to_parts(futures):
    """
    Builds an ordered dict mapping each worker to their list
    of parts
    :param futures: list of (worker, part) tuples
    :return:
    """
    w_to_p_map = OrderedDict()
    for w, p in futures:
        if w not in w_to_p_map:
            w_to_p_map[w] = []
        w_to_p_map[w].append(p)
    return w_to_p_map


def get_ary_meta(ary):
    return ary.shape, ary.dtype


def to_dask_cupy(futures, dtype=None, shapes=None, client=None, verbose=False):

    c = default_client() if client is None else client
    meta = [c.submit(get_ary_meta, future) for future in futures]

    objs = []
    for i in range(len(futures)):
        if not isinstance(futures[i].type, type(None)):
            met = meta[i].result()
            obj = da.from_delayed(futures[i], shape=met[0],
                                  dtype=met[1])
            objs.append(obj)

    return da.concatenate(objs, axis=0)


def to_output(futures, type, client=None, verbose=False):
    if type == 'cupy':
        return to_dask_cupy(futures, client=client, verbose=verbose)
    else:
        return to_dask_cudf(futures, client=client, verbose=verbose)
