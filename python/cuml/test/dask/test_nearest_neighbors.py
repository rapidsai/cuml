# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

import pytest

import cudf
import dask_cudf

import pandas as pd

import numpy as np

from cuml.common import has_scipy

from cuml.dask.common import utils as dask_utils

from cuml.test.utils import unit_param, quality_param, stress_param

from sklearn.neighbors import KNeighborsClassifier

from cuml.test.utils import array_equal


def predict(neigh_ind, _y, n_neighbors):
    if has_scipy():
        import scipy.stats as stats
    else:
        raise RuntimeError('Scipy is needed to run predict()')

    neigh_ind = neigh_ind.astype(np.int64)

    ypred, count = stats.mode(_y[neigh_ind], axis=1)
    return ypred.ravel(), count.ravel() * 1.0 / n_neighbors


def _prep_training_data(c, X_train, partitions_per_worker,
                        reverse_order=False):
    workers = c.has_what().keys()

    if reverse_order:
        workers = list(workers)[::-1]

    n_partitions = partitions_per_worker * len(workers)

    X_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))

    X_train_df = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)
    X_train_df, = dask_utils.persist_across_workers(c,
                                                    [X_train_df],
                                                    workers=list(workers))

    return X_train_df


def _scale_rows(client, nrows):
    workers = list(client.scheduler_info()['workers'].keys())
    n_workers = len(workers)
    return n_workers * nrows


@pytest.mark.parametrize("nrows", [unit_param(100),
                                   unit_param(1e3),
                                   quality_param(1e6),
                                   stress_param(5e8)])
@pytest.mark.parametrize("ncols", [10, 30])
@pytest.mark.parametrize("nclusters", [unit_param(5), quality_param(10),
                                       stress_param(15)])
@pytest.mark.parametrize("n_neighbors", [unit_param(10), quality_param(4),
                                         stress_param(100)])
@pytest.mark.parametrize("n_parts", [unit_param(1), unit_param(5),
                                     quality_param(7), stress_param(50)])
@pytest.mark.parametrize("streams_per_handle", [5, 10])
@pytest.mark.parametrize("reverse_worker_order", [True, False])
def test_compare_skl(nrows, ncols, nclusters, n_parts, n_neighbors,
                     streams_per_handle, reverse_worker_order, client):

    from cuml.dask.neighbors import NearestNeighbors as daskNN

    from sklearn.datasets import make_blobs

    nrows = _scale_rows(client, nrows)

    X, y = make_blobs(n_samples=int(nrows),
                      n_features=ncols,
                      centers=nclusters,
                      random_state=0)
    X = X.astype(np.float32)

    X_cudf = _prep_training_data(client, X, n_parts, reverse_worker_order)

    from dask.distributed import wait

    wait(X_cudf)

    dist = np.array([len(v) for v in client.has_what().values()])

    assert np.all(dist == dist[0])

    cumlModel = daskNN(n_neighbors=n_neighbors,
                       streams_per_handle=streams_per_handle)
    cumlModel.fit(X_cudf)

    out_d, out_i = cumlModel.kneighbors(X_cudf)

    local_i = np.array(out_i.compute().as_gpu_matrix(), dtype="int64")

    sklModel = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    skl_y_hat = sklModel.predict(X)
    y_hat, _ = predict(local_i, y, n_neighbors)

    sk_d, sk_i = sklModel.kneighbors(X)

    sk_i = sk_i.astype("int64")

    assert array_equal(local_i[:, 0], np.arange(nrows))

    diff = sk_i-local_i
    n_diff = len(diff[diff > 0])

    perc_diff = n_diff / (nrows * n_neighbors)

    assert perc_diff <= 3e-3

    assert array_equal(y_hat, skl_y_hat)


@pytest.mark.parametrize("nrows", [unit_param(1000), stress_param(1e5)])
@pytest.mark.parametrize("ncols", [unit_param(10), stress_param(500)])
@pytest.mark.parametrize("n_parts", [unit_param(10), stress_param(100)])
@pytest.mark.parametrize("batch_size", [unit_param(100), stress_param(1e3)])
def test_batch_size(nrows, ncols, n_parts,
                    batch_size, client):

    n_neighbors = 10
    n_clusters = 5
    from cuml.dask.neighbors import NearestNeighbors as daskNN

    from sklearn.datasets import make_blobs

    nrows = _scale_rows(client, nrows)

    X, y = make_blobs(n_samples=int(nrows),
                      n_features=ncols,
                      centers=n_clusters,
                      random_state=0)

    X = X.astype(np.float32)

    X_cudf = _prep_training_data(client, X, n_parts)

    cumlModel = daskNN(n_neighbors=n_neighbors,
                       batch_size=batch_size,
                       streams_per_handle=5)

    cumlModel.fit(X_cudf)

    out_d, out_i = cumlModel.kneighbors(X_cudf)

    local_i = np.array(out_i.compute().as_gpu_matrix())

    y_hat, _ = predict(local_i, y, n_neighbors)

    assert array_equal(y_hat, y)


def test_return_distance(client):

    n_samples = 50
    n_feats = 50
    k = 5

    from cuml.dask.neighbors import NearestNeighbors as daskNN

    from sklearn.datasets import make_blobs

    n_samples = _scale_rows(client, n_samples)

    X, y = make_blobs(n_samples=n_samples,
                      n_features=n_feats, random_state=0)

    X = X.astype(np.float32)

    X_cudf = _prep_training_data(client, X, 1)

    cumlModel = daskNN(streams_per_handle=5)
    cumlModel.fit(X_cudf)

    ret = cumlModel.kneighbors(X_cudf, k, return_distance=False)
    assert not isinstance(ret, tuple)
    ret = ret.compute()
    assert ret.shape == (n_samples, k)

    ret = cumlModel.kneighbors(X_cudf, k, return_distance=True)
    assert isinstance(ret, tuple)
    assert len(ret) == 2


def test_default_n_neighbors(client):

    n_samples = 50
    n_feats = 50
    k = 15

    from cuml.dask.neighbors import NearestNeighbors as daskNN
    from cuml.neighbors.nearest_neighbors_mg import \
        NearestNeighborsMG as cumlNN

    from sklearn.datasets import make_blobs

    n_samples = _scale_rows(client, n_samples)

    X, y = make_blobs(n_samples=n_samples,
                      n_features=n_feats, random_state=0)

    X = X.astype(np.float32)

    X_cudf = _prep_training_data(client, X, 1)

    cumlModel = daskNN(streams_per_handle=5)
    cumlModel.fit(X_cudf)

    ret = cumlModel.kneighbors(X_cudf, return_distance=False)

    assert ret.shape[1] == cumlNN().n_neighbors

    cumlModel = daskNN(n_neighbors=k)
    cumlModel.fit(X_cudf)

    ret = cumlModel.kneighbors(X_cudf, k, return_distance=False)

    assert ret.shape[1] == k


def test_one_query_partition(client):
    from cuml.dask.neighbors import NearestNeighbors as daskNN
    from cuml.dask.datasets import make_blobs

    X_train, _ = make_blobs(n_samples=4000,
                            n_features=16,
                            n_parts=8)

    X_test, _ = make_blobs(n_samples=200,
                           n_features=16,
                           n_parts=1)

    cumlModel = daskNN(n_neighbors=4)
    cumlModel.fit(X_train)
    cumlModel.kneighbors(X_test)
