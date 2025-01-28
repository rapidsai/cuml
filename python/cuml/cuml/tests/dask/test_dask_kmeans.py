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

from cuml.dask.common.dask_arr_utils import to_dask_cudf
from sklearn.metrics import adjusted_rand_score as sk_adjusted_rand_score
from cuml.metrics import adjusted_rand_score
import dask.array as da
from cuml.testing.utils import stress_param
from cuml.testing.utils import quality_param
from cuml.testing.utils import unit_param
import pytest
from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")


@pytest.mark.mg
@pytest.mark.parametrize(
    "nrows", [unit_param(1e3), quality_param(1e5), stress_param(5e6)]
)
@pytest.mark.parametrize("ncols", [10, 30])
@pytest.mark.parametrize(
    "nclusters", [unit_param(5), quality_param(10), stress_param(50)]
)
@pytest.mark.parametrize(
    "n_parts", [unit_param(None), quality_param(7), stress_param(50)]
)
@pytest.mark.parametrize("delayed_predict", [True, False])
@pytest.mark.parametrize("input_type", ["dataframe", "array"])
def test_end_to_end(
    nrows, ncols, nclusters, n_parts, delayed_predict, input_type, client
):

    from cuml.dask.cluster import KMeans as cumlKMeans

    from cuml.dask.datasets import make_blobs

    X, y = make_blobs(
        n_samples=int(nrows),
        n_features=ncols,
        centers=nclusters,
        n_parts=n_parts,
        cluster_std=0.01,
        random_state=10,
    )

    if input_type == "dataframe":
        X_train = to_dask_cudf(X)
        y_train = to_dask_cudf(y)
    elif input_type == "array":
        X_train, y_train = X, y

    cumlModel = cumlKMeans(
        init="k-means||",
        n_clusters=nclusters,
        random_state=10,
        n_init="auto",
    )

    cumlModel.fit(X_train)
    cumlLabels = cumlModel.predict(X_train, delayed=delayed_predict)

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

    assert 1.0 == score


@pytest.mark.mg
@pytest.mark.parametrize("nrows_per_part", [quality_param(1e7)])
@pytest.mark.parametrize("ncols", [quality_param(256)])
@pytest.mark.parametrize("nclusters", [quality_param(5)])
def test_large_data_no_overflow(nrows_per_part, ncols, nclusters, client):

    from cuml.dask.cluster import KMeans as cumlKMeans
    from cuml.dask.datasets import make_blobs

    n_parts = len(list(client.has_what().keys()))

    X, y = make_blobs(
        n_samples=nrows_per_part * n_parts,
        n_features=ncols,
        centers=nclusters,
        n_parts=n_parts,
        cluster_std=0.01,
        random_state=10,
    )

    X_train, y_train = X, y

    X.compute_chunk_sizes().persist()

    cumlModel = cumlKMeans(
        init="k-means||", n_clusters=nclusters, random_state=10, n_init="auto"
    )

    cumlModel.fit(X_train)
    n_predict = int(X_train.shape[0] / 4)
    cumlLabels = cumlModel.predict(X_train[:n_predict, :], delayed=False)

    cumlPred = cp.array(cumlLabels.compute())
    labels = cp.squeeze(y_train.compute()[:n_predict])

    print(str(cumlPred))
    print(str(labels))

    assert 1.0 == adjusted_rand_score(labels, cumlPred)


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [500])
@pytest.mark.parametrize("ncols", [5])
@pytest.mark.parametrize("nclusters", [3, 10])
@pytest.mark.parametrize("n_parts", [1, 5])
def test_weighted_kmeans(nrows, ncols, nclusters, n_parts, client):
    cluster_std = 10000.0
    from cuml.dask.cluster import KMeans as cumlKMeans

    from cuml.dask.datasets import make_blobs

    # Using fairly high variance between points in clusters
    wt = cp.array([0.00001 for j in range(nrows)])

    bound = nclusters * 100000

    # Open the space really large
    centers = cp.random.uniform(-bound, bound, size=(nclusters, ncols))

    X_cudf, y = make_blobs(
        n_samples=nrows,
        n_features=ncols,
        centers=centers,
        n_parts=n_parts,
        cluster_std=cluster_std,
        shuffle=False,
        verbose=False,
        random_state=10,
    )

    # Choose one sample from each label and increase its weight
    for i in range(nclusters):
        wt[cp.argmax(cp.array(y.compute()) == i).item()] = 5000.0

    cumlModel = cumlKMeans(
        verbose=0,
        init="k-means||",
        n_clusters=nclusters,
        random_state=10,
        n_init="auto",
    )

    chunk_parts = int(nrows / n_parts)
    sample_weights = da.from_array(wt, chunks=(chunk_parts,))
    cumlModel.fit(X_cudf, sample_weight=sample_weights)

    X = X_cudf.compute()

    labels_ = cumlModel.predict(X_cudf).compute()
    cluster_centers_ = cumlModel.cluster_centers_

    for i in range(nrows):

        label = labels_[i]
        actual_center = cluster_centers_[label]

        diff = sum(abs(X[i] - actual_center))

        # The large weight should be the centroid
        if wt[i] > 1.0:
            assert diff < 1.0

        # Otherwise it should be pretty far away
        else:
            assert diff > 1000.0


@pytest.mark.mg
@pytest.mark.parametrize(
    "nrows", [unit_param(5e3), quality_param(1e5), stress_param(1e6)]
)
@pytest.mark.parametrize(
    "ncols", [unit_param(10), quality_param(30), stress_param(50)]
)
@pytest.mark.parametrize("nclusters", [1, 10, 30])
@pytest.mark.parametrize(
    "n_parts", [unit_param(None), quality_param(7), stress_param(50)]
)
@pytest.mark.parametrize("input_type", ["dataframe", "array"])
def test_transform(nrows, ncols, nclusters, n_parts, input_type, client):

    from cuml.dask.cluster import KMeans as cumlKMeans

    from cuml.dask.datasets import make_blobs

    X, y = make_blobs(
        n_samples=int(nrows),
        n_features=ncols,
        centers=nclusters,
        n_parts=n_parts,
        cluster_std=0.01,
        shuffle=False,
        random_state=10,
    )
    y = y.astype("int64")

    if input_type == "dataframe":
        X_train = to_dask_cudf(X)
        y_train = to_dask_cudf(y)
        labels = y_train.compute().values
    elif input_type == "array":
        X_train, y_train = X, y
        labels = cp.squeeze(y_train.compute())

    cumlModel = cumlKMeans(
        init="k-means||",
        n_clusters=nclusters,
        random_state=10,
        n_init="auto",
    )

    cumlModel.fit(X_train)

    xformed = cumlModel.transform(X_train).compute()
    if input_type == "dataframe":
        xformed = cp.array(
            xformed if len(xformed.shape) == 1 else xformed.to_cupy()
        )

    if nclusters == 1:
        # series shape is (nrows,) not (nrows, 1) but both are valid
        # and equivalent for this test
        assert xformed.shape in [(nrows, nclusters), (nrows,)]
    else:
        assert xformed.shape == (nrows, nclusters)

    # The argmin of the transformed values should be equal to the labels
    # reshape is a quick manner of dealing with (nrows,) is not (nrows, 1)
    xformed_labels = cp.argmin(
        xformed.reshape((int(nrows), int(nclusters))), axis=1
    )

    assert sk_adjusted_rand_score(
        cp.asnumpy(labels), cp.asnumpy(xformed_labels)
    )


@pytest.mark.mg
@pytest.mark.parametrize(
    "nrows", [unit_param(1e3), quality_param(1e5), stress_param(5e6)]
)
@pytest.mark.parametrize("ncols", [10, 30])
@pytest.mark.parametrize(
    "nclusters", [unit_param(5), quality_param(10), stress_param(50)]
)
@pytest.mark.parametrize(
    "n_parts", [unit_param(None), quality_param(7), stress_param(50)]
)
@pytest.mark.parametrize("input_type", ["dataframe", "array"])
def test_score(nrows, ncols, nclusters, n_parts, input_type, client):

    from cuml.dask.cluster import KMeans as cumlKMeans

    from cuml.dask.datasets import make_blobs

    X, y = make_blobs(
        n_samples=int(nrows),
        n_features=ncols,
        centers=nclusters,
        n_parts=n_parts,
        cluster_std=0.01,
        shuffle=False,
        random_state=10,
    )

    if input_type == "dataframe":
        X_train = to_dask_cudf(X)
        y_train = to_dask_cudf(y)
        y = y_train
    elif input_type == "array":
        X_train, y_train = X, y

    cumlModel = cumlKMeans(
        init="k-means||",
        n_clusters=nclusters,
        random_state=10,
        n_init="auto",
    )

    cumlModel.fit(X_train)

    actual_score = cumlModel.score(X_train)

    local_model = cumlModel.get_combined_model()
    expected_score = local_model.score(X_train.compute())

    assert abs(actual_score - expected_score) < 9e-3
