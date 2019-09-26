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

from sklearn.datasets import make_blobs as skl_make_blobs

import numpy as np

from uuid import uuid1
import math


def create_df(m, n, centers, cluster_std, random_state, dtype):
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


def get_X(t):
    return t[0]


def get_labels(t):
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
    parts_workers = (workers * n_parts)[:n_parts]
    rows_per_part = math.ceil(nrows/n_parts)

    if not isinstance(centers, np.ndarray):
        centers = np.random.uniform(center_box[0], center_box[1],
                                    size=(centers, ncols)).astype(np.float32)

    if verbose:
        print("Generating %d samples across %d partitions on "
              "%d workers (total=%d samples)" %
              (math.ceil(nrows/len(workers)), n_parts, len(workers), nrows))

    key = str(uuid1())
    # Create dfs on each worker (gpu)
    dfs = []
    for idx, worker in enumerate(parts_workers):
        worker_rows = rows_per_part \
            if rows_per_part*(idx+1) <= nrows \
            else nrows - (rows_per_part*(idx+1))

        dfs.append(client.submit(create_df, worker_rows, ncols,
                                 centers, cluster_std, random_state, dtype,
                                 key="%s-%s" % (key, idx),
                                 workers=[worker]))

    x_key = str(uuid1())
    y_key = str(uuid1())
    X = [client.submit(get_X, f, key="%s-%s" % (x_key, idx))
         for idx, f in enumerate(dfs)]
    y = [client.submit(get_labels, f, key="%s-%s" % (y_key, idx))
         for idx, f in enumerate(dfs)]

    meta_X = client.submit(get_meta, X[0]).result()
    X_cudf = from_delayed(X, meta=meta_X)

    meta_y = client.submit(get_meta, y[0]).result()
    y_cudf = from_delayed(y, meta=meta_y)

    return X_cudf, y_cudf
