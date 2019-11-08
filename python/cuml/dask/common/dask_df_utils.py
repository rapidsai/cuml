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

from tornado import gen
from dask.distributed import default_client
from toolz import first
from uuid import uuid1
import dask.dataframe as dd
from collections import OrderedDict

from dask.distributed import wait

from functools import reduce


def workers_to_parts(futures):
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


def _func_get_size(df):
    return df.shape[0]


def parts_to_ranks(client, worker_info, part_futures):
    """
    Builds a list of (rank, size) tuples of partitions
    :param worker_info: dict of {worker, {"r": rank }}. Note: \
        This usually comes from the underlying communicator
    :param part_futures: list of (worker, future) tuples
    :return:
    """
    key = uuid1()
    futures = [(worker_info[wf[0]]["r"],
                client.submit(_func_get_size,
                              wf[1],
                              workers=[wf[0]],
                              key="%s-%s" % (key, idx)))
               for idx, wf in enumerate(part_futures)]

    sizes = client.compute(list(map(lambda x: x[1], futures)), sync=True)
    total = reduce(lambda a, b: a + b, sizes)

    return [(futures[idx][0], size) for idx, size in enumerate(sizes)], total


def _default_part_getter(f, idx): return f[idx]


def flatten_grouped_results(client, parts_to_ranks,
                            worker_results_map,
                            getter_func=_default_part_getter):
    """
    Given a grouped map of 
    :param client:
    :param parts_to_ranks:
    :param worker_results_map:
    :param getter_func:
    :return:
    """
    futures = []
    completed_part_map = {}
    for rank, size in parts_to_ranks:
        if rank not in completed_part_map:
            completed_part_map[rank] = 0

        f = worker_results_map[rank]

        futures.append(client.submit(
            getter_func, f, completed_part_map[rank]))

        completed_part_map[rank] += 1

    return futures


@gen.coroutine
def extract_ddf_partitions(ddf, client=None):
    """
    Given a Dask cuDF, return an OrderedDict mapping
    'worker -> [list of futures]' for each partition in ddf.

    :param ddf: Dask.dataframe split dataframe partitions into a list of
               futures.
    :param client: dask.distributed.Client Optional client to use
    """
    client = default_client() if client is None else client

    delayed_ddf = ddf.to_delayed()
    parts = client.compute(delayed_ddf)
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


def get_meta(df):
    """
    Return the metadata from a single dataframe
    :param df: cudf.dataframe
    :return: Row data from the first row of the dataframe
    """
    ret = df.iloc[:0]
    return ret


def to_dask_cudf(futures, client=None):
    """
    Convert a list of futures containing cudf Dataframes into a Dask.Dataframe
    :param futures: list[cudf.Dataframe] list of futures containing dataframes
    :param client: dask.distributed.Client Optional client to use
    :return: dask.Dataframe a dask.Dataframe
    """
    c = default_client() if client is None else client
    # Convert a list of futures containing dfs back into a dask_cudf
    dfs = [d for d in futures if d.type != type(None)]  # NOQA
    meta = c.submit(get_meta, dfs[0]).result()
    return dd.from_delayed(dfs, meta=meta)


def to_dask_df(dask_cudf, client=None):
    """
    Convert a Dask-cuDF into a Pandas-backed Dask Dataframe.
    :param dask_cudf : dask_cudf.DataFrame
    :param client: dask.distributed.Client Optional client to use
    :return : dask.DataFrame
    """

    def to_pandas(df):
        return df.to_pandas()

    c = default_client() if client is None else client
    delayed_ddf = dask_cudf.to_delayed()
    gpu_futures = c.compute(delayed_ddf)

    key = uuid1()
    dfs = [c.submit(
        to_pandas,
        f,
        key="%s-%s" % (key, idx)) for idx, f in enumerate(gpu_futures)]

    meta = c.submit(get_meta, dfs[0]).result()

    return dd.from_delayed(dfs, meta=meta)
