# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
import pytest

from cuml.test.utils import unit_param
from cuml.test.utils import quality_param
from cuml.test.utils import stress_param

from cuml.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_rand_score as sk_adjusted_rand_score

from cuml.dask.common.dask_arr_utils import to_dask_cudf


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [unit_param(1e3), quality_param(1e5),
                                   stress_param(5e6)])
@pytest.mark.parametrize("ncols", [10, 30])
@pytest.mark.parametrize("nclusters", [unit_param(5), quality_param(10),
                                       stress_param(50)])
@pytest.mark.parametrize("n_parts", [unit_param(None), quality_param(7),
                                     stress_param(50)])
@pytest.mark.parametrize("delayed_predict", [True, False])
@pytest.mark.parametrize("input_type", ["dataframe", "array"])
def test_end_to_end(nrows, ncols, nclusters, n_parts,
                    delayed_predict, input_type, client):

    from cuml.dask.cluster import KMeans as cumlKMeans

    from cuml.dask.datasets import make_blobs

    X, y = make_blobs(n_samples=int(nrows),
                      n_features=ncols,
                      centers=nclusters,
                      n_parts=n_parts,
                      cluster_std=0.01,
                      random_state=10)

    if input_type == "dataframe":
        X_train = to_dask_cudf(X)
        y_train = to_dask_cudf(y)
    elif input_type == "array":
        X_train, y_train = X, y

    cumlModel = cumlKMeans(init="k-means||",
                           n_clusters=nclusters,
                           random_state=10)

    cumlModel.fit(X_train)
    cumlLabels = cumlModel.predict(X_train, delayed_predict)

    n_workers = len(list(client.has_what().keys()))

    # Verifying we are grouping partitions. This should be changed soon.
    if n_parts is not None:
        parts_len = n_parts
    else:
        parts_len = n_workers

    if input_type == "dataframe":
        assert cumlLabels.npartitions == parts_len
        cumlPred = cumlLabels.compute().values
        labels = y_train.compute().values
    elif input_type == "array":
        assert len(cumlLabels.chunks[0]) == parts_len
        cumlPred = cp.array(cumlLabels.compute())
        labels = cp.squeeze(y_train.compute())

    assert cumlPred.shape[0] == nrows
    assert cp.max(cumlPred) == nclusters - 1
    assert cp.min(cumlPred) == 0

    score = adjusted_rand_score(labels, cumlPred)

    print(str(score))

    assert 1.0 == score


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [unit_param(5e3), quality_param(1e5),
                                   stress_param(1e6)])
@pytest.mark.parametrize("ncols", [unit_param(10), quality_param(30),
                                   stress_param(50)])
@pytest.mark.parametrize("nclusters", [1, 10, 30])
@pytest.mark.parametrize("n_parts", [unit_param(None), quality_param(7),
                                     stress_param(50)])
@pytest.mark.parametrize("input_type", ["dataframe", "array"])
def test_transform(nrows, ncols, nclusters, n_parts, input_type, client):

    from cuml.dask.cluster import KMeans as cumlKMeans

    from cuml.dask.datasets import make_blobs

    X, y = make_blobs(n_samples=int(nrows),
                      n_features=ncols,
                      centers=nclusters,
                      n_parts=n_parts,
                      cluster_std=0.01,
                      shuffle=False,
                      random_state=10)
    y = y.astype('int64')

    if input_type == "dataframe":
        X_train = to_dask_cudf(X)
        y_train = to_dask_cudf(y)
        labels = y_train.compute().values
    elif input_type == "array":
        X_train, y_train = X, y
        labels = cp.squeeze(y_train.compute())

    cumlModel = cumlKMeans(init="k-means||",
                           n_clusters=nclusters,
                           random_state=10)

    cumlModel.fit(X_train)

    xformed = cumlModel.transform(X_train).compute()
    if input_type == "dataframe":
        xformed = cp.array(xformed
                           if len(xformed.shape) == 1
                           else xformed.as_gpu_matrix())

    if nclusters == 1:
        # series shape is (nrows,) not (nrows, 1) but both are valid
        # and equivalent for this test
        assert xformed.shape in [(nrows, nclusters), (nrows,)]
    else:
        assert xformed.shape == (nrows, nclusters)

    # The argmin of the transformed values should be equal to the labels
    # reshape is a quick manner of dealing with (nrows,) is not (nrows, 1)
    xformed_labels = cp.argmin(xformed.reshape((int(nrows),
                                                int(nclusters))), axis=1)

    assert sk_adjusted_rand_score(cp.asnumpy(labels),
                                  cp.asnumpy(xformed_labels))


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [unit_param(1e3), quality_param(1e5),
                                   stress_param(5e6)])
@pytest.mark.parametrize("ncols", [10, 30])
@pytest.mark.parametrize("nclusters", [unit_param(5), quality_param(10),
                                       stress_param(50)])
@pytest.mark.parametrize("n_parts", [unit_param(None), quality_param(7),
                                     stress_param(50)])
@pytest.mark.parametrize("input_type", ["dataframe", "array"])
def test_score(nrows, ncols, nclusters, n_parts,
               input_type, client):

    from cuml.dask.cluster import KMeans as cumlKMeans

    from cuml.dask.datasets import make_blobs

    X, y = make_blobs(n_samples=int(nrows),
                      n_features=ncols,
                      centers=nclusters,
                      n_parts=n_parts,
                      cluster_std=0.01,
                      shuffle=False,
                      random_state=10)

    if input_type == "dataframe":
        X_train = to_dask_cudf(X)
        y_train = to_dask_cudf(y)
        y = y_train
    elif input_type == "array":
        X_train, y_train = X, y

    cumlModel = cumlKMeans(init="k-means||",
                           n_clusters=nclusters,
                           random_state=10)

    cumlModel.fit(X_train)

    actual_score = cumlModel.score(X_train)

    local_model = cumlModel.get_combined_model()
    expected_score = local_model.score(X_train.compute())

    assert abs(actual_score - expected_score) < 1e-3
