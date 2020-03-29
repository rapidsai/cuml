#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
import dask
import dask.array as da
import math
import numpy as np
import pandas as pd

from cuml.utils import rmm_cupy_ary
from dask.dataframe import from_delayed
from dask.distributed import default_client

from sklearn.datasets import make_blobs as skl_make_blobs


def create_local_data(m, n, centers, cluster_std, random_state,
                      dtype, type, order='F', shuffle=False):
    X, y = skl_make_blobs(m, n, centers=centers, cluster_std=cluster_std,
                          random_state=random_state, shuffle=shuffle)

    if type == 'array':
        X = rmm_cupy_ary(cp.array, X.astype(dtype), order=order)
        y = rmm_cupy_ary(cp.array, y.astype(dtype),
                         order=order).reshape(m)

    elif type == 'dataframe':
        X = cudf.DataFrame.from_pandas(pd.DataFrame(X.astype(dtype)))
        y = cudf.DataFrame.from_pandas(pd.DataFrame(y))

    else:
        raise ValueError('type must be array or dataframe')

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
               dtype=np.float32, output='dataframe',
               order='F', shuffle=False):
    """
    Makes labeled dask.Dataframe and dask_cudf.Dataframes containing blobs
    for a randomly generated set of centroids.

    This function calls `make_blobs` from Scikitlearn on each Dask worker
    and aggregates them into a single Dask Dataframe.

    For more information on Scikit-learn's `make_blobs:
    <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html>`_.

    Parameters
    ----------

    nrows : int
        number of rows
    ncols : int
        number of features
    n_centers : int (default = 8)
        number of centers to generate
    n_parts : int (default = None)
        number of partitions to generate (this can be greater
        than the number of workers)
    cluster_std : float (default = 1.0)
         standard deviation of points around centroid
    center_box : tuple (int, int) (default = (-10, 10))
         the bounding box which constrains all the centroids
    random_state : int (default = None)
         sets random seed (or use None to reinitialize each time)
    verbose : bool (default = False)
         enables / disables verbose printing.
    dtype : dtype (default = np.float32)
         datatype to generate
    output : str (default = 'dataframe')
         whether to generate dask array or
         dask dataframe output. Default will be array in the future.

    Returns
    -------
         (dask.Dataframe for X, dask.Series for labels)
    """

    client = default_client()

    workers = list(client.has_what().keys())

    n_parts = n_parts if n_parts is not None else len(workers)
    parts_workers = (workers * n_parts)[:n_parts]
    rows_per_part = math.ceil(nrows / n_parts)

    if not isinstance(centers, np.ndarray):
        centers = np.random.uniform(center_box[0], center_box[1],
                                    size=(centers, ncols)).astype(np.float32)

    if verbose:
        print("Generating %d samples across %d partitions on "
              "%d workers (total=%d samples)" %
              (math.ceil(nrows / len(workers)), n_parts, len(workers), nrows))

    # Create dfs on each worker (gpu)
    parts = []
    worker_rows = []
    rows_so_far = 0
    for idx, worker in enumerate(parts_workers):
        if rows_so_far + rows_per_part <= nrows:
            rows_so_far += rows_per_part
            worker_rows.append(rows_per_part)
        else:
            worker_rows.append((int(nrows) - rows_so_far))

        parts.append(client.submit(create_local_data, worker_rows[idx], ncols,
                                   centers, cluster_std, random_state, dtype,
                                   output, order, shuffle,
                                   pure=False,
                                   workers=[worker]))

    X = [client.submit(get_X, f, pure=False)
         for idx, f in enumerate(parts)]
    Y = [client.submit(get_labels, f, pure=False)
         for idx, f in enumerate(parts)]

    if output == 'dataframe':

        meta_X = client.submit(get_meta, X[0])
        meta_X_local = meta_X.result()
        X_final = from_delayed([dask.delayed(x, pure=False)
                                for x in X], meta=meta_X_local)

        meta_y = client.submit(get_meta, Y[0])
        meta_y_local = meta_y.result()
        Y_final = from_delayed([dask.delayed(y, pure=False)
                                for y in Y], meta=meta_y_local)

    elif output == 'array':

        X_del = [da.from_delayed(dask.delayed(chunk, pure=False),
                                 shape=(worker_rows[idx], ncols),
                                 dtype=dtype,
                                 meta=cp.zeros((1)))
                 for idx, chunk in enumerate(X)]
        Y_del = [da.from_delayed(dask.delayed(chunk, pure=False),
                                 shape=(worker_rows[idx],),
                                 dtype=dtype,
                                 meta=cp.zeros((1)))
                 for idx, chunk in enumerate(Y)]

        X_final = da.concatenate(X_del, axis=0)
        Y_final = da.concatenate(Y_del, axis=0)

    return X_final, Y_final
