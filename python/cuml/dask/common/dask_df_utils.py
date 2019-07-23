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
import dask.dataframe as dd
from dask_cudf import from_delayed

from cuml.dask.common.utils import parse_host_port

from dask.distributed import wait


@gen.coroutine
def extract_ddf_partitions(ddf):
    """
    Given a Dask cuDF, return a tuple with (worker, future) for each partition
    :param ddf: Dask.dataframe split dataframe partitions into a list of
               futures.
    """
    client = default_client()

    delayed_ddf = ddf.to_delayed()
    parts = client.compute(delayed_ddf)
    yield wait(parts)

    key_to_part_dict = dict([(str(part.key), part) for part in parts])
    who_has = yield client.who_has(parts)

    worker_map = []
    for key, workers in who_has.items():
        worker = parse_host_port(first(workers))
        worker_map.append((worker, key_to_part_dict[key]))

    gpu_data = [(worker, part) for worker, part in worker_map]

    yield wait(gpu_data)

    raise gen.Return(gpu_data)


def get_meta(df):
    """
    Return the metadata from a single dataframe
    :param df: cudf.dataframe
    :return: Row data from the first row of the dataframe
    """
    ret = df.iloc[:0]
    return ret


def to_dask_cudf(futures):
    """
    Convert a list of futures containing cudf Dataframes into a Dask.Dataframe
    :param futures: list[cudf.Dataframe] list of futures containing dataframes
    :return: dask.Dataframe a dask.Dataframe
    """
    c = default_client()
    # Convert a list of futures containing dfs back into a dask_cudf
    dfs = [d for d in futures if d.type != type(None)]  # NOQA
    meta = c.submit(get_meta, dfs[0]).result()
    return from_delayed(dfs, meta=meta)
