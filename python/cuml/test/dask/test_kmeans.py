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

import cupy as cp
import numpy as np
import pytest

from cuml.test.utils import unit_param
from cuml.test.utils import quality_param
from cuml.test.utils import stress_param

from dask.distributed import Client, wait

from sklearn.metrics import adjusted_rand_score

SCORE_EPS = 0.06


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [unit_param(1e3), quality_param(1e5),
                                   stress_param(5e6)])
@pytest.mark.parametrize("ncols", [10, 30])
@pytest.mark.parametrize("nclusters", [unit_param(5), quality_param(10),
                                       stress_param(50)])
@pytest.mark.parametrize("n_parts", [unit_param(None), quality_param(7),
                                     stress_param(50)])
@pytest.mark.parametrize("delayed_predict", [True, False])
def test_end_to_end(nrows, ncols, nclusters, n_parts,
                    delayed_predict, cluster):

    client = None

    try:

        client = Client(cluster)
        from cuml.dask.cluster import KMeans as cumlKMeans

        from cuml.dask.datasets import make_blobs

        X_cudf, y = make_blobs(nrows, ncols, nclusters, n_parts,
                               cluster_std=0.01, verbose=False,
                               random_state=10)

        wait(X_cudf)

        cumlModel = cumlKMeans(verbose=0, init="k-means||",
                               n_clusters=nclusters,
                               random_state=10)

        cumlModel.fit(X_cudf)
        cumlLabels = cumlModel.predict(X_cudf, delayed_predict)

        n_workers = len(list(client.has_what().keys()))

        # Verifying we are grouping partitions. This should be changed soon.
        if n_parts is not None and n_parts < n_workers:
            assert cumlLabels.npartitions == n_parts
        else:
            assert cumlLabels.npartitions == n_workers

        cumlPred = cp.array(cumlLabels.compute())

        assert cumlPred.shape[0] == nrows
        assert np.max(cumlPred) == nclusters - 1
        assert np.min(cumlPred) == 0

        labels = np.squeeze(y.compute().to_pandas().values)

        score = adjusted_rand_score(labels, cp.squeeze(cumlPred.get()))

        print(str(score))

        assert 1.0 == score

    finally:
        client.close()


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [unit_param(5e3), quality_param(1e5),
                                   stress_param(1e6)])
@pytest.mark.parametrize("ncols", [unit_param(10), quality_param(30),
                                   stress_param(50)])
@pytest.mark.parametrize("nclusters", [1, 10, 30])
@pytest.mark.parametrize("n_parts", [unit_param(None), quality_param(7),
                                     stress_param(50)])
def test_transform(nrows, ncols, nclusters, n_parts, cluster):

    client = None

    try:

        client = Client(cluster)

        from cuml.dask.cluster import KMeans as cumlKMeans

        from cuml.dask.datasets import make_blobs

        X_cudf, y = make_blobs(nrows, ncols, nclusters, n_parts,
                               cluster_std=0.01, verbose=False,
                               shuffle=False,
                               random_state=10)

        wait(X_cudf)

        cumlModel = cumlKMeans(verbose=0, init="k-means||",
                               n_clusters=nclusters,
                               random_state=10)

        cumlModel.fit(X_cudf)

        labels = np.squeeze(y.compute().to_pandas().values)

        xformed = cumlModel.transform(X_cudf).compute()

        if nclusters == 1:
            # series shape is (nrows,) not (nrows, 1) but both are valid
            # and equivalent for this test
            assert xformed.shape in [(nrows, nclusters), (nrows,)]
        else:
            assert xformed.shape == (nrows, nclusters)

        xformed = cp.array(xformed
                           if len(xformed.shape) == 1
                           else xformed.as_gpu_matrix())

        # The argmin of the transformed values should be equal to the labels
        # reshape is a quick manner of dealing with (nrows,) is not (nrows, 1)
        xformed_labels = cp.argmin(xformed.reshape((int(nrows),
                                                    int(nclusters))), axis=1)

        assert adjusted_rand_score(labels, cp.squeeze(xformed_labels.get()))

    finally:
        client.close()


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [unit_param(1e3), quality_param(1e5),
                                   stress_param(5e6)])
@pytest.mark.parametrize("ncols", [10, 30])
@pytest.mark.parametrize("nclusters", [unit_param(5), quality_param(10),
                                       stress_param(50)])
@pytest.mark.parametrize("n_parts", [unit_param(None), quality_param(7),
                                     stress_param(50)])
def test_score(nrows, ncols, nclusters, n_parts, cluster):

    client = None

    try:

        client = Client(cluster)
        from cuml.dask.cluster import KMeans as cumlKMeans

        from cuml.dask.datasets import make_blobs

        X_cudf, y = make_blobs(nrows, ncols, nclusters, n_parts,
                               cluster_std=0.01, verbose=False,
                               shuffle=False,
                               random_state=10)

        wait(X_cudf)

        cumlModel = cumlKMeans(verbose=0, init="k-means||",
                               n_clusters=nclusters,
                               random_state=10)

        cumlModel.fit(X_cudf)

        actual_score = cumlModel.score(X_cudf)

        X = cp.array(X_cudf.compute().as_gpu_matrix())

        predictions = cumlModel.predict(X_cudf).compute()
        predictions = cp.array(predictions)

        centers = cp.array(cumlModel.cluster_centers_.as_gpu_matrix())

        expected_score = 0
        for idx, label in enumerate(predictions):

            x = X[idx]
            y = centers[label]

            dist = np.sqrt(np.sum((x - y)**2))
            expected_score += dist**2

        assert actual_score + SCORE_EPS \
            >= (-1 * expected_score) \
            >= actual_score - SCORE_EPS

    finally:
        client.close()
