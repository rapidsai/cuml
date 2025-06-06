#
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


import math

import dask.array as da

import cuml.internals.logger as logger
from cuml.common import with_cupy_rmm
from cuml.dask._compat import DASK_2025_4_0
from cuml.dask.common.utils import get_client
from cuml.dask.datasets.utils import _create_delayed, _get_labels, _get_X
from cuml.datasets.blobs import _get_centers
from cuml.datasets.blobs import make_blobs as sg_make_blobs
from cuml.datasets.utils import _create_rs_generator


def _create_local_data(
    m, n, centers, cluster_std, shuffle, random_state, order, dtype
):
    X, y = sg_make_blobs(
        m,
        n,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
        shuffle=shuffle,
        order=order,
        dtype=dtype,
    )

    return X, y


@with_cupy_rmm
def make_blobs(
    n_samples=100,
    n_features=2,
    centers=None,
    cluster_std=1.0,
    n_parts=None,
    center_box=(-10, 10),
    shuffle=True,
    random_state=None,
    return_centers=False,
    verbose=False,
    order="F",
    dtype="float32",
    client=None,
    workers=None,
):
    """
    Makes labeled Dask-Cupy arrays containing blobs
    for a randomly generated set of centroids.

    This function calls `make_blobs` from `cuml.datasets` on each Dask worker
    and aggregates them into a single Dask Dataframe.

    For more information on Scikit-learn's `make_blobs
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
    return_centers : bool, optional (default=False)
        If True, then return the centers of each cluster
    verbose : int or boolean (default = False)
         Logging level.
    shuffle : bool (default=False)
              Shuffles the samples on each worker.
    order: str, optional (default='F')
        The order of the generated samples
    dtype : str, optional (default='float32')
        Dtype of the generated samples
    client : dask.distributed.Client (optional)
             Dask client to use
    workers : optional, list of strings
        Dask addresses of workers to use for computation.
        If None, all available Dask workers will be used.
        (e.g. : `workers = list(client.scheduler_info()['workers'].keys())`)

    Returns
    -------
    X : dask.array backed by CuPy array of shape [n_samples, n_features]
        The input samples.
    y : dask.array backed by CuPy array of shape [n_samples]
        The output values.
    centers : dask.array backed by CuPy array of shape
        [n_centers, n_features], optional
        The centers of the underlying blobs. It is returned only if
        return_centers is True.

    Examples
    --------
    .. code-block:: python

        >>> from dask_cuda import LocalCUDACluster
        >>> from dask.distributed import Client
        >>> from cuml.dask.datasets import make_blobs

        >>> cluster = LocalCUDACluster(threads_per_worker=1)
        >>> client = Client(cluster)

        >>> workers = list(client.scheduler_info()['workers'].keys())
        >>> X, y = make_blobs(1000, 10, centers=42, cluster_std=0.1,
        ...                   workers=workers)

        >>> client.close()
        >>> cluster.close()
    """

    client = get_client(client=client)

    generator = _create_rs_generator(random_state=random_state)

    if workers is None:
        kwargs = {"n_workers": -1} if DASK_2025_4_0() else {}
        workers = list(client.scheduler_info(**kwargs)["workers"].keys())

    n_parts = n_parts if n_parts is not None else len(workers)
    parts_workers = (workers * n_parts)[:n_parts]

    centers, n_centers = _get_centers(
        generator, centers, center_box, n_samples, n_features, dtype
    )

    rows_per_part = max(1, int(n_samples / n_parts))

    worker_rows = [rows_per_part] * n_parts

    worker_rows[-1] += n_samples % n_parts

    worker_rows = tuple(worker_rows)

    logger.debug(
        "Generating %d samples across %d partitions on "
        "%d workers (total=%d samples)"
        % (
            math.ceil(n_samples / len(workers)),
            n_parts,
            len(workers),
            n_samples,
        )
    )

    seeds = generator.randint(n_samples, size=len(parts_workers))
    parts = [
        client.submit(
            _create_local_data,
            part_rows,
            n_features,
            centers,
            cluster_std,
            shuffle,
            int(seeds[idx]),
            order,
            dtype,
            pure=False,
            workers=[parts_workers[idx]],
        )
        for idx, part_rows in enumerate(worker_rows)
    ]

    X = [client.submit(_get_X, f, pure=False) for idx, f in enumerate(parts)]
    y = [
        client.submit(_get_labels, f, pure=False)
        for idx, f in enumerate(parts)
    ]

    X_del = _create_delayed(X, dtype, worker_rows, n_features)
    y_del = _create_delayed(y, dtype, worker_rows)

    X_final = da.concatenate(X_del, axis=0)
    y_final = da.concatenate(y_del, axis=0)

    if return_centers:
        return X_final, y_final, centers
    else:
        return X_final, y_final
