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

import dask.dataframe as dd
import pandas as pd

import cudf

import dask_cudf

from dask.distributed import default_client, wait

from sklearn.datasets import make_blobs

import numpy as np

import random
import math


def create_df(f, m, n, centers, cluster_std, random_state, r):
    X, y = make_blobs(m, n, centers=centers, cluster_std=cluster_std,
                      random_state=random_state)
    ret = pd.DataFrame(X)
    return ret


def get_meta(df):
    ret = df.iloc[:0]
    return ret


def to_cudf(df, r):
    return cudf.from_pandas(df)


def dask_make_blobs(nrows, ncols, n_centers=8, cluster_std=1.0,
                    center_box=(-10, 10), random_state=None, verbose=False):

    client = default_client()

    workers = client.has_what().keys()

    centers = np.random.uniform(center_box[0], center_box[1],
                                size=(n_centers, ncols)).astype(np.float32)

    if verbose:
        print("Generating %d samples on %d workers (total=%d samples)" %
              (math.ceil(nrows/len(workers)), len(workers), nrows))

    # Create dfs on each worker (gpu)
    dfs = [client.submit(create_df, n, math.ceil(nrows/len(workers)), ncols,
                         centers, cluster_std,
                         random_state, random.random(), workers=[worker])
           for worker, n in list(zip(workers, list(range(len(workers)))))]
    # Wait for completion
    wait(dfs)

    ddfs = [client.submit(to_cudf, df, random.random()) for df in dfs]
    # Wait for completion
    wait(ddfs)

    meta_ddf = client.submit(get_meta, dfs[0]).result()
    meta_cudf = client.submit(get_meta, ddfs[0]).result()

    d_df = dd.from_delayed(dfs, meta=meta_ddf)
    d_cudf = dask_cudf.from_delayed(ddfs, meta=meta_cudf)

    return d_df, d_cudf
