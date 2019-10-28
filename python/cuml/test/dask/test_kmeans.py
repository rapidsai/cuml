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

import numpy as np

from cuml.test.utils import unit_param, quality_param, stress_param


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [unit_param(5e3), quality_param(1e5),
                                   stress_param(1e6)])
@pytest.mark.parametrize("ncols", [unit_param(10), quality_param(30),
                                   stress_param(50)])
@pytest.mark.parametrize("nclusters", [1, 10, 30])
@pytest.mark.parametrize("n_parts", [unit_param(None), quality_param(7),
                                     stress_param(50)])
def test_transform(nrows, ncols, nclusters, n_parts, cluster):

    client = Client(cluster)

    try:

        from cuml.dask.cluster import KMeans as cumlKMeans

        from cuml.dask.datasets import make_blobs

        X_cudf, y = make_blobs(nrows, ncols, nclusters, n_parts,
                               cluster_std=0.01, verbose=True,
                               random_state=10)

        wait(X_cudf)

        cumlModel = cumlKMeans(verbose=0, init="k-means||",
                               n_clusters=nclusters,
                               random_state=10)

        cumlModel.fit(X_cudf)

        labels = y.compute().to_pandas().values
        labels = labels.reshape(labels.shape[0])

        xformed = cumlModel.transform(X_cudf).compute()

        assert xformed.shape == (nrows, nclusters)

        # The argmin of the transformed values should be equal to the labels
        xformed_labels = np.argmin(xformed.to_pandas().to_numpy(), axis=1)

        from sklearn.metrics import adjusted_rand_score
        assert adjusted_rand_score(labels, xformed_labels)

    finally:
        client.close()


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [unit_param(10e3), quality_param(1e5),
                                   stress_param(5e6)])
@pytest.mark.parametrize("ncols", [10, 30])
@pytest.mark.parametrize("nclusters", [unit_param(5), quality_param(10),
                                       stress_param(50)])
@pytest.mark.parametrize("n_parts", [unit_param(None), quality_param(7),
                                     stress_param(50)])
def test_end_to_end(nrows, ncols, nclusters, n_parts, cluster):

    client = Client(cluster)

    try:
        from cuml.dask.cluster import KMeans as cumlKMeans

        from cuml.dask.datasets import make_blobs

        X_cudf, y = make_blobs(nrows, ncols, nclusters, n_parts,
                               cluster_std=0.01, verbose=True,
                               random_state=10)

        wait(X_cudf)

        cumlModel = cumlKMeans(verbose=0, init="k-means||",
                               n_clusters=nclusters,
                               random_state=10)

        cumlModel.fit(X_cudf)

        cumlLabels = cumlModel.predict(X_cudf)

        from sklearn.metrics import adjusted_rand_score

        cumlPred = cumlLabels.compute().to_pandas().values

        assert cumlPred.shape[0] == nrows
        assert np.max(cumlPred) == nclusters-1
        assert np.min(cumlPred) == 0

        labels = y.compute().to_pandas().values
        labels = labels.reshape(labels.shape[0])

        score = adjusted_rand_score(labels, cumlPred)

        assert 1.0 == score

    finally:
        client.close()
