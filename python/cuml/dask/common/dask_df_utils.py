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
from dask import delayed
from toolz import first
from uuid import uuid1
import dask.dataframe as dd
import numpy as np
from collections import defaultdict

from dask.distributed import wait


@gen.coroutine
def extract_ddf_partitions(ddf, client=None, agg=True):
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


@gen.coroutine
def extract_colocated_ddf_partitions(X_ddf, y_ddf, client=None):
    """
    Given Dask cuDF input X and y, return an OrderedDict mapping
    'worker -> [list of futures] of X and y' with the enforced
     co-locality for each partition in ddf.

    :param X_ddf: Dask.dataframe
    :param y_ddf: Dask.dataframe
    :param client: dask.distributed.Client
    """
    client = default_client() if client is None else client
    data_parts = X_ddf.to_delayed()
    label_parts = y_ddf.to_delayed()

    if isinstance(data_parts, np.ndarray):
        assert data_parts.shape[1] == 1
        data_parts = data_parts.flatten().tolist()

    if isinstance(label_parts, np.ndarray):
        assert label_parts.ndim == 1 or label_parts.shape[1] == 1
        label_parts = label_parts.flatten().tolist()

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
