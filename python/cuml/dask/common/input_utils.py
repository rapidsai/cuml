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
from uuid import uuid1

from collections import defaultdict
from functools import reduce

import dask.dataframe as dd


class MGData:

    def __init__(self, gpu_futures=None, worker_to_parts=None, workers=None,
                 datatype=None, multiple=False, client=None):
        self.client = default_client() if client is None else client
        self.gpu_futures = gpu_futures
        self.worker_to_parts = worker_to_parts
        self.workers = workers
        self.datatype = datatype
        self.multiple = multiple
        self.worker_info = None
        self.total_rows = None
        self.ranks = None
        self.parts_to_sizes = None
        self.total_rows = None

    """ Class methods for initalization """

    @classmethod
    def single(cls, data, client=None):
        gpu_futures = client.sync(_extract_partitions, data, client)

        worker_to_parts = _workers_to_parts(gpu_futures)

        workers = list(map(lambda x: x[0], worker_to_parts.items()))

        datatype = 'cudf' if isinstance(data, dcDataFrame) else 'cupy'

        return MGData(gpu_futures=gpu_futures, worker_to_parts=worker_to_parts,
                      workers=workers, datatype=datatype, multiple=False,
                      client=client)

    @classmethod
    def colocated(cls, data, force=False, client=None):
        gpu_futures = client.sync(_extract_colocated_partitions,
                                  data[0], data[1], client)

        workers = list(gpu_futures.keys())

        datatype = 'cudf' if isinstance(data[0], dcDataFrame) else 'cupy'

        return MGData(gpu_futures=gpu_futures, workers=workers,
                      datatype=datatype, multiple=True, client=client)

    """ Methods to calculate further attributes """

    def calculate_worker_and_rank_info(self, comms):
        self.worker_info = comms.worker_info(comms.worker_addresses)
        self.ranks = dict()
        for w, futures in self.gpu_futures.items():
            self.ranks[w] = self.worker_info[w]["r"]

    def calculate_parts_to_sizes(self, comms=None, ranks=None):
        if self.worker_info is None and comms is not None:
            self.calculate_worker_and_rank_info(comms)

        ranks = self.ranks if ranks is None else ranks

        self.total_rows = 0

        self.parts_to_sizes = dict()

        # func = _get_rows_from_colocated_tuple if self.multiple is True else \
        #     _get_rows_from_single_obj

        key = uuid1()

        if self.multiple:
            for w, futures in self.gpu_futures.items():
                parts = [(self.client.submit(
                    _get_rows_from_colocated_tuple,
                    future,
                    workers=[w],
                    key="%s-%s" % (key, idx)).result())
                    for idx, future in enumerate(futures)]

                self.parts_to_sizes[self.worker_info[w]["r"]] = parts
                for p in parts:
                    self.total_rows = self.total_rows + p

        else:
            self.parts_to_sizes = [(ranks[wf[0]], self.client.submit(
                                   _get_rows_from_single_obj,
                                   wf[1],
                                   workers=[wf[0]],
                                   key="%s-%s" % (key, idx)).result())
                                   for idx,
                                   wf in enumerate(self.gpu_futures)]

            self.total_rows = reduce(lambda a, b:
                                     a + b, map(lambda x: x[1],
                                                self.parts_to_sizes))


@with_cupy_rmm
def concatenate(objs, axis=0):
    if isinstance(objs[0], DataFrame):
        if len(objs) == 1:
            return objs[0]
        else:
            return cudf.concat(objs)

    elif isinstance(objs[0], cp.ndarray):
        return cp.concatenate(objs, axis=axis)


def to_output(futures, type, client=None, verbose=False):
    if type == 'cupy':
        return to_dask_cupy(futures, client=client, verbose=verbose)
    else:
        return to_dask_cudf(futures, client=client, verbose=verbose)


def _get_meta(df):
    """
    Return the metadata from a single dataframe
    :param df: cudf.dataframe
    :return: Row data from the first row of the dataframe
    """
    ret = df[0].iloc[:0]
    return ret


def _to_dask_cudf(futures, client=None, verbose=False):
    """
    Convert a list of futures containing cudf Dataframes into a Dask.Dataframe
    :param futures: list[cudf.Dataframe] list of futures containing dataframes
    :param client: dask.distributed.Client Optional client to use
    :return: dask.Dataframe a dask.Dataframe
    """
    c = default_client() if client is None else client
    # Convert a list of futures containing dfs back into a dask_cudf
    dfs = [d for d in futures if d.type != type(None)]  # NOQA
    if verbose:
        print("to_dask_cudf dfs=%s" % str(dfs))
    meta = c.submit(_get_meta, dfs[0]).result()
    return dd.from_delayed(dfs, meta=meta)


""" Internal methods, API subject to change """


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


@gen.coroutine
def _extract_colocated_partitions(X_ddf, y_ddf, client=None):
    """
    Given Dask cuDF input X and y, return an OrderedDict mapping
    'worker -> [list of futures] of X and y' with the enforced
     co-locality for each partition in ddf.

    :param X_ddf: Dask.dataframe
    :param y_ddf: Dask.dataframe
    :param client: dask.distributed.Client
    """

    client = default_client() if client is None else client

    if isinstance(X_ddf, dcDataFrame):
        data_parts = X_ddf.to_delayed()
        label_parts = y_ddf.to_delayed()

    else:
        data_arr = X_ddf.to_delayed().ravel()
        to_map = data_arr
        data_parts = list(map(delayed, to_map))
        label_arr = y_ddf.to_delayed().ravel()
        to_map2 = label_arr
        label_parts = list(map(delayed, to_map2))
        # data_parts = X_ddf.to_delayed().ravel()
        # label_parts = y_ddf.to_delayed().ravel()

    parts = list(map(delayed, zip(data_parts, label_parts)))
    parts = client.compute(parts)
    yield wait(parts)

    key_to_part_dict = dict([(part.key, part) for part in parts])
    who_has = yield client.scheduler.who_has(
        keys=[part.key for part in parts]
    )

    worker_map = defaultdict(list)
    for key, workers in who_has.items():
        worker_map[first(workers)].append(key_to_part_dict[key])

    return worker_map


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


def _get_ary_meta(ary):
    return ary.shape, ary.dtype


def _get_rows_from_colocated_tuple(obj):
    return obj[0].shape[0]


def _get_rows_from_single_obj(obj):
    return obj.shape[0]


def to_dask_cupy(futures, dtype=None, shapes=None, client=None, verbose=False):

    wait(futures)

    c = default_client() if client is None else client
    meta = [c.submit(_get_ary_meta, future) for future in futures]

    objs = []
    for i in range(len(futures)):
        if not isinstance(futures[i].type, type(None)):
            met = meta[i].result()
            obj = da.from_delayed(futures[i], shape=met[0],
                                  dtype=met[1])
            objs.append(obj)

    return da.concatenate(objs, axis=0)
