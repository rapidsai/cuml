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

from collections.abc import Sequence

from cuml.utils.memory_utils import with_cupy_rmm

from collections import OrderedDict
from cudf.core import DataFrame
from dask_cudf.core import DataFrame as dcDataFrame

from dask import delayed

from cuml.dask.common.utils import get_client
from cuml.dask.common.dask_df_utils import to_dask_cudf
from cuml.dask.common.dask_arr_utils import validate_dask_array
from cuml.dask.common.part_utils import _extract_partitions
from dask.distributed import wait
from dask.distributed import default_client
from tornado import gen
from toolz import first

from functools import reduce

import dask.dataframe as dd


class DistributedDataHandler:
    """
    Class to centralize distributed data management. Functionalities include:
    - Data colocation
    - Worker information extraction
    - GPU futures extraction,

    Additional functionality can be added as needed. This class **does not**
    contain the actual data, just the metadata necessary to handle it,
    including common pieces of code that need to be performed to call
    Dask functions.

    The constructor is not meant to be used directly, but through the factory
    method DistributedDataHandler.create

    """

    def __init__(self, gpu_futures=None, workers=None,
                 datatype=None, multiple=False, client=None):
        self.client = get_client(client)
        self.gpu_futures = gpu_futures
        self.worker_to_parts = _workers_to_parts(gpu_futures)
        self.workers = workers
        self.datatype = datatype
        self.multiple = multiple
        self.worker_info = None
        self.total_rows = None
        self.ranks = None
        self.parts_to_sizes = None
        self.total_rows = None

    @classmethod
    def get_client(cls, client=None):
        return default_client() if client is None else client

    """ Class methods for initalization """

    @classmethod
    def create(cls, data, client=None):
        """
        Creates a distributed data handler instance with the given
        distributed data set(s).

        Parameters
        ----------

        data : dask.array, dask.dataframe, or unbounded Sequence of
               dask.array or dask.dataframe.

        client : dask.distributedClient
        """

        client = cls.get_client(client)

        multiple = isinstance(data, Sequence)

        datatype = 'cudf' if isinstance(first(data) if multiple else data,
                                        dcDataFrame) else 'cupy'

        if datatype == 'cupy':
            if multiple:
                for d in data:
                    validate_dask_array(d)
            else:
                validate_dask_array(data)

        gpu_futures = client.sync(_extract_partitions, data, client)
        workers = tuple(set(map(lambda x: x[0], gpu_futures)))

        return DistributedDataHandler(gpu_futures=gpu_futures, workers=workers,
                                      datatype=datatype, multiple=multiple,
                                      client=client)

    # TODO: Remove the following two functions
    #  (just here to keep from breaking everythign)
    @classmethod
    def single(cls, data, client=None):
        return cls.create(data, client)

    @classmethod
    def colocated(cls, data, force=False, client=None):
        return cls.create(data, client)

    """ Methods to calculate further attributes """

    def calculate_worker_and_rank_info(self, comms):

        self.worker_info = comms.worker_info(comms.worker_addresses)
        self.ranks = dict()

        for w, futures in self.worker_to_parts.items():
            self.ranks[w] = self.worker_info[w]["r"]

    def calculate_parts_to_sizes(self, comms=None, ranks=None):

        if self.worker_info is None and comms is not None:
            self.calculate_worker_and_rank_info(comms)

        self.total_rows = 0

        self.parts_to_sizes = dict()

        parts = [(wf[0], self.client.submit(
            _get_rows,
            wf[1],
            self.multiple,
            workers=[wf[0]],
            pure=False))
            for idx, wf in enumerate(self.worker_to_parts.items())]

        sizes = self.client.compute(parts, sync=True)

        for w, sizes_parts in sizes:
            sizes, total = sizes_parts
            self.parts_to_sizes[self.worker_info[w]["r"]] = \
                sizes

            self.total_rows += total


@with_cupy_rmm
def concatenate(objs, axis=0):
    if isinstance(objs[0], DataFrame):
        if len(objs) == 1:
            return objs[0]
        else:
            return cudf.concat(objs)

    elif isinstance(objs[0], cp.ndarray):
        return cp.concatenate(objs, axis=axis)


# TODO: This should be delayed.
def to_output(futures, type, client=None, verbose=False):
    if type == 'cupy':
        return to_dask_cupy(futures, client=client)
    else:
        return to_dask_cudf(futures, client=client)


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
    meta_future = c.submit(_get_meta, dfs[0], pure=False)
    meta = meta_future.result()
    return dd.from_delayed(dfs, meta=meta)


""" Internal methods, API subject to change """

# TODO: This can go away once all remaining estimators are updated
#  to use _extract_partitions
@gen.coroutine
def _extract_colocated_partitions(data, client=None):

    # TODO: This should go away once Ridge is fully consolidated
    """
    Given Dask cuDF input X and y, return an OrderedDict mapping
    'worker -> [list of futures] of X and y' with the enforced
     co-locality for each partition in ddf.

    :param X_ddf: Dask.dataframe
    :param y_ddf: Dask.dataframe
    :param client: dask.distributed.Client
    """

    client = default_client() if client is None else client

    X_ddf, y_ddf = data

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

    parts = [delayed(x, pure=False) for x in zip(data_parts, label_parts)]
    parts = client.compute(parts)
    yield wait(parts)

    key_to_part = [(part.key, part) for part in parts]
    who_has = yield client.scheduler.who_has(
        keys=[part.key for part in parts]
    )

    raise gen.Return([(first(who_has[key]), part)
                      for key, part in key_to_part])


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

    if isinstance(ary, cp.ndarray):
        return ary.shape, ary.dtype
    elif isinstance(ary, cudf.DataFrame):
        return ary.shape, first(set(ary.dtypes))
    else:
        raise ValueError("Expected dask.Dataframe "
                         "or dask.Array, received %s" % type(ary))


def _get_rows(objs, multiple):
    def get_obj(x): return x[0] if multiple else x
    total = list(map(lambda x: get_obj(x).shape[0], objs))
    return total, reduce(lambda a, b: a + b, total)


def to_dask_cupy(futures, dtype=None, shapes=None, client=None):

    wait(futures)

    c = default_client() if client is None else client
    meta = [c.submit(_get_ary_meta, future, pure=False)
            for future in futures]

    objs = []
    for i in range(len(futures)):
        if not isinstance(futures[i].type, type(None)):
            met_future = meta[i]
            met = met_future.result()
            obj = da.from_delayed(futures[i], shape=met[0],
                                  dtype=met[1])
            objs.append(obj)

    return da.concatenate(objs, axis=0)
