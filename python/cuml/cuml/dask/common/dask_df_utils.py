# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import cuml.internals.logger as logger
import dask.dataframe as dd

from dask.distributed import default_client


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
    if logger.should_log_for(logger.level_enum.debug):
        logger.debug("to_dask_cudf dfs=%s" % str(dfs))
    meta = c.submit(get_meta, dfs[0])
    meta_local = meta.result()
    return dd.from_delayed(dfs, meta=meta_local)


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

    dfs = [
        c.submit(to_pandas, f, pure=False) for idx, f in enumerate(gpu_futures)
    ]

    meta = c.submit(get_meta, dfs[0])

    # Using new variable for local result to stop race-condition in scheduler
    # Ref: https://github.com/dask/dask/issues/6027
    meta_local = meta.result()

    return dd.from_delayed(dfs, meta=meta_local)
