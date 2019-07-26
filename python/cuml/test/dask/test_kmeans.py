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

from dask.distributed import Client


@pytest.mark.mg
def test_end_to_end(nrows, ncols, nclusters, client=None):

    owns_cluster = False
    if client is None:
        owns_cluster = True
        cluster = LocalCUDACluster(threads_per_worker=1)
        client = Client(cluster)

    from cuml.dask.cluster.kmeans import KMeans as cumlKMeans
    from dask_ml.cluster import KMeans as dmlKMeans

    from cuml.test.dask.utils import dask_make_blobs

    print("Building dask df")

    X_df, X_cudf = dask_make_blobs(nrows, ncols, nclusters,
                                   cluster_std=0.1, verbose=True)

    X_df = X_df.persist()
    X_cudf = X_cudf.persist()

    print("Building model")
    cumlModel = cumlKMeans(verbose=0, init="k-means||", n_clusters=nclusters)
    daskmlModel1 = dmlKMeans(init="k-means||", n_clusters=nclusters)

    print("Fitting model")
    import time

    cumlStart = time.time()
    cumlModel.fit(X_cudf)
    cumlDuration = time.time() - cumlStart
    print(str(cumlDuration))

    daskStart = time.time()
    daskmlModel1.fit(X_df)
    daskDuration = time.time()-daskStart

    print(str(daskDuration))

    print("Speedup=" + str(daskDuration/cumlDuration))

    print("Predicting model")

    cumlLabels = cumlModel.predict(X_cudf)
    daskmlLabels1 = daskmlModel1.predict(X_df)

    print("SCORE: " + str(cumlModel.score(X_cudf)))

    from sklearn.metrics import adjusted_rand_score

    cumlPred = cumlLabels.compute().to_pandas().values

    daskmlPred1 = daskmlLabels1.compute()

    print(str(cumlPred))
    print(str(daskmlPred1))

    score = adjusted_rand_score(cumlPred, daskmlPred1)

    if owns_cluster:
        client.close()
        cluster.close()

    assert 1.0 == score
