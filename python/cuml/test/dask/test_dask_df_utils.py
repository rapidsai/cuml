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

import numpy as np

import dask.dataframe

from cuml.dask.common.dask_df_utils import to_dask_cudf


def test_to_dask_cudf(client):

    def _create_cudf(size):
        c = cudf.DataFrame()
        c["0"] = np.arange(size)
        return c

    workers = list(client.scheduler_info()["workers"].keys())

    preds = [client.submit(_create_cudf, worker) for worker in workers]

    df = to_dask_cudf(preds, client)

    assert isinstance(df, dask.dataframe)
    assert df.npartitions == len(workers)

    assert "0" in df.columns
    assert df.compute().shape[0]




