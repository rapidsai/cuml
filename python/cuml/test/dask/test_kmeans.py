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

import pytest
from dask_cuda import LocalCUDACluster

from dask.distributed import Client, wait


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [1e3, 1e5, 5e5])
@pytest.mark.parametrize("ncols", [10, 30])
@pytest.mark.parametrize("nclusters", [5, 10])
@pytest.mark.parametrize("n_parts", [None, 50])
def test_end_to_end(nrows, ncols, nclusters, n_parts, client=None):

    owns_cluster = False
    if client is None:
        owns_cluster = True
        cluster = LocalCUDACluster(threads_per_worker=1)
        client = Client(cluster)

    from cuml.dask.cluster import KMeans as cumlKMeans
    from dask_ml.cluster import KMeans as dmlKMeans

    from cuml.dask.datasets import make_blobs

    from cuml.dask.common import to_dask_df

    X_cudf, _ = make_blobs(nrows, ncols, nclusters, n_parts,
                           cluster_std=0.1, verbose=True,
                           random_state=10)

    X_df = to_dask_df(X_cudf)

    wait(X_df)

    cumlModel = cumlKMeans(verbose=0, init="k-means||", n_clusters=nclusters,
                           random_state=10)

    daskmlModel = dmlKMeans(init="k-means||", n_clusters=nclusters,
                            random_state=10)

    cumlModel.fit(X_cudf)
    daskmlModel.fit(X_df)

    cumlLabels = cumlModel.predict(X_cudf)
    daskmlLabels = daskmlModel.predict(X_df)

    from sklearn.metrics import adjusted_rand_score

    cumlPred = cumlLabels.compute().to_pandas().values
    daskmlPred = daskmlLabels.compute()

    score = adjusted_rand_score(cumlPred, daskmlPred)

    if owns_cluster:
        client.close()
        cluster.close()

    assert 1.0 == score
