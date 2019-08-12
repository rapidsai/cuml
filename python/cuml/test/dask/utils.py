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

from dask.distributed import default_client
from dask.distributed import wait as dask_wait

from cuml.dask.common import extract_ddf_partitions

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


def dask_make_blobs(nrows, ncols, n_centers=8, n_parts=None, cluster_std=1.0,
                    center_box=(-10, 10), random_state=None, verbose=False):

    """
    Makes unlabeled dask.Dataframe and dask_cudf.Dataframes containing blobs
    for a randomly generated set of centroids.

    This function calls `make_blobs` from Scikitlearn on each Dask worker
    and aggregates them into a single Dask Dataframe.

    For more information on Scikit-learn's `make_blobs:
    <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html>`_.

    :param nrows: number of rows
    :param ncols: number of features
    :param n_centers: number of centers to generate
    :param n_parts: number of partitions to generate (this can be greater
    than the number of workers)
    :param cluster_std: how far can each generated point deviate from its
    closest centroid?
    :param center_box: the bounding box which constrains all the centroids
    :param random_state: sets random seed
    :param verbose: enables / disables verbose printing.
    :return: dask.Dataframe & dask_cudf.Dataframe
    """

    client = default_client()

    workers = list(client.has_what().keys())

    n_parts = n_parts if n_parts is not None else len(workers)

    parts = (workers * math.ceil(n_parts / len(workers)))[:n_parts]

    centers = np.random.uniform(center_box[0], center_box[1],
                                size=(n_centers, ncols)).astype(np.float32)

    if verbose:
        print("Generating %d samples across %d partitions on "
              "%d workers (total=%d samples)" %
              (math.ceil(nrows/len(workers)), len(parts), len(workers), nrows))

    # Create dfs on each worker (gpu)
    dfs = [client.submit(create_df, n, math.ceil(nrows/len(workers)), ncols,
                         centers, cluster_std,
                         random_state, random.random(), workers=[worker])
           for worker, n in list(zip(parts, list(range(len(workers)))))]
    # Wait for completion
    dask_wait(dfs)

    ddfs = [client.submit(to_cudf, df, random.random()) for df in dfs]
    # Wait for completion
    dask_wait(ddfs)

    meta_ddf = client.submit(get_meta, dfs[0]).result()
    meta_cudf = client.submit(get_meta, ddfs[0]).result()

    d_df = dd.from_delayed(dfs, meta=meta_ddf)
    d_cudf = dask_cudf.from_delayed(ddfs, meta=meta_cudf)

    return d_df, d_cudf
