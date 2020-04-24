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

# from sklearn.datasets import make_blobs as skl_make_blobs
from cuml.datasets.blobs import _get_centers
from cuml.datasets.blobs import make_blobs as sg_make_blobs
from cuml.utils import with_cupy_rmm


def create_local_data(m, n, centers, cluster_std, shuffle, random_state,
                      order, dtype):

    X, y = sg_make_blobs(m, n, centers=centers,
                         cluster_std=cluster_std,
                         random_state=random_state,
                         shuffle=shuffle,
                         order=order,
                         dtype=dtype)

    return X, y


def get_X(t):
    return t[0]


def get_labels(t):
    return t[1]


@with_cupy_rmm
def make_blobs(n_samples=100, n_features=2, centers=None, cluster_std=1.0,
               n_parts=None, center_box=(-10, 10), shuffle=True,
               random_state=None, return_centers=False, verbose=False,
               order='F', dtype='float32', client=None):
    """
    Makes labeled Dask-Cupy arrays containing blobs
    for a randomly generated set of centroids.

    This function calls `make_blobs` from `cuml.datasets` on each Dask worker
    and aggregates them into a single Dask Dataframe.

    For more information on Scikit-learn's `make_blobs:
    <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html>`_.

    Parameters
    ----------

    n_samples : int
        number of rows
    n_features : int
        number of features
    centers : int or array of shape [n_centers, n_features],
        optional (default=None) The number of centers to generate, or the fixed
        center locations. If n_samples is an int and centers is None, 3 centers
        are generated. If n_samples is array-like, centers must be either None
        or an array of length equal to the length of n_samples.
    cluster_std : float (default = 1.0)
         standard deviation of points around centroid
    n_parts : int (default = None)
        number of partitions to generate (this can be greater
        than the number of workers)
    center_box : tuple (int, int) (default = (-10, 10))
         the bounding box which constrains all the centroids
    random_state : int (default = None)
         sets random seed (or use None to reinitialize each time)
    verbose : bool (default = False)
         enables / disables verbose printing.
    dtype : dtype (default = np.float32)
         datatype to generate
    output : str { 'dataframe', 'array' } (default = 'dataframe')
         whether to generate dask array or
         dask dataframe output. Default will be array in the future.
    shuffle : bool (default=False)
              Shuffles the samples on each worker.
    client : dask.distributed.Client (optional)
             Dask client to use

    Returns
    -------
    X : Dask-CuPy array of shape [n_samples, n_features]
        The input samples.
    y : Dask-CuPy array of shape [n_samples]
        The output values.
    centers : Dask-CuPy array of shape [n_centers, n_features], optional
        The centers of the underlying blobs. It is returned only if
        return_centers is True.
    """

    client = default_client() if client is None else client

    generator = cp.random.RandomState()

    workers = list(client.has_what().keys())

    n_parts = n_parts if n_parts is not None else len(workers)
    parts_workers = (workers * n_parts)[:n_parts]
    rows_per_part = math.ceil(n_samples / n_parts)

    centers, n_centers = _get_centers(generator, centers, center_box,
                                      n_samples, n_features,
                                      dtype)

    # random_state = np.random.randint(0, 100) \
    #     if random_state is None else random_state

    if verbose:
        print("Generating %d samples across %d partitions on "
              "%d workers (total=%d samples)" %
              (math.ceil(n_samples / len(workers)),
               n_parts, len(workers), n_samples))

    # Create dfs on each worker (gpu)
    parts = []
    worker_rows = []
    rows_so_far = 0
    for idx, worker in enumerate(parts_workers):
        if rows_so_far + rows_per_part <= n_samples:
            rows_so_far += rows_per_part
            worker_rows.append(rows_per_part)
        else:
            worker_rows.append((int(n_samples) - rows_so_far))

    seeds = generator.randint(n_samples, size=len(parts_workers))
    parts = [client.submit(create_local_data,
                                   part_rows,
                                   n_features,
                                   centers,
                                   cluster_std,
                                   shuffle,
                                   int(seeds[idx]),
                                   order,
                                   dtype,
                                   pure=False,
                                   workers=[parts_workers[idx]])
             for idx, part_rows in enumerate(worker_rows)]

    X = [client.submit(get_X, f, pure=False)
         for idx, f in enumerate(parts)]
    Y = [client.submit(get_labels, f, pure=False)
         for idx, f in enumerate(parts)]

    X_del = [da.from_delayed(dask.delayed(chunk, pure=False),
                                shape=(worker_rows[idx], n_features),
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

    if return_centers:
        return X_final, Y_final, centers
    else:
        return X_final, Y_final
