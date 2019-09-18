#
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


from dask.dataframe import from_delayed
import pandas as pd

import cudf

from dask.distributed import default_client
from dask.distributed import wait as dask_wait

from sklearn.datasets import make_blobs as skl_make_blobs

import numpy as np

import random
import math


def create_df(f, m, n, centers, cluster_std, random_state, dtype, r):
    """
    Returns Dask Dataframes on device for X and y.
    """
    X, y = skl_make_blobs(m, n, centers=centers, cluster_std=cluster_std,
                         random_state=random_state)
    X = cudf.DataFrame.from_pandas(pd.DataFrame(X.astype(dtype)))
    y = cudf.DataFrame.from_pandas(pd.DataFrame(y))
    return X, y


def get_meta(df):
    ret = df.iloc[:0]
    return ret


def get_X(t, r):
    return t[0]


def get_labels(t, r):
    return t[1]


def make_blobs(nrows, ncols, centers=8, n_parts=None, cluster_std=1.0,
               center_box=(-10, 10), random_state=None, verbose=False,
               dtype=np.float32):

    """
    Makes unlabeled dask.Dataframe and dask_cudf.Dataframes containing blobs
    for a randomly generated set of centroids.

    This function calls `make_blobs` from Scikitlearn on each Dask worker
    and aggregates them into a single Dask Dataframe.

    For more information on Scikit-learn's `make_blobs:
    <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html>`_.

    :param nrows : number of rows
    :param ncols : number of features
    :param n_centers : number of centers to generate
    :param n_parts : number of partitions to generate (this can be greater
    than the number of workers)
    :param cluster_std : how far can each generated point deviate from its
    closest centroid?
    :param center_box : the bounding box which constrains all the centroids
    :param random_state : sets random seed
    :param verbose : enables / disables verbose printing.
    :param dtype : (default = np.float32) datatype to generate

    :return: (dask.Dataframe for X, dask.Series for labels)
    """

    client = default_client()

    workers = list(client.has_what().keys())

    n_parts = n_parts if n_parts is not None else len(workers)

    parts = (workers * math.ceil(n_parts / len(workers)))[:n_parts]

    if not isinstance(centers, np.ndarray):
        centers = np.random.uniform(center_box[0], center_box[1],
                                    size=(centers, ncols)).astype(np.float32)

    if verbose:
        print("Generating %d samples across %d partitions on "
              "%d workers (total=%d samples)" %
              (math.ceil(nrows/len(workers)), len(parts), len(workers), nrows))

    # Create dfs on each worker (gpu)
    dfs = [client.submit(create_df, n, math.ceil(nrows/n_parts), ncols,
                         centers, cluster_std, random_state, dtype,
                         random.random(),
                         workers=[worker])
           for worker, n in list(zip(parts, range(n_parts)))]

    # Wait for completion
    dask_wait(dfs)

    X = [client.submit(get_X, f, random.random()) for f in dfs]
    y = [client.submit(get_labels, f, random.random()) for f in dfs]

    meta_X = client.submit(get_meta, X[0]).result()
    X_cudf = from_delayed(X, meta=meta_X)

    meta_y = client.submit(get_meta, y[0]).result()
    y_cudf = from_delayed(y, meta=meta_y)

    return X_cudf, y_cudf
