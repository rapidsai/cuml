# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
import dask.array as da
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score as sk_adjusted_rand_score

from cuml.dask.common.dask_arr_utils import to_dask_cudf
from cuml.metrics import adjusted_rand_score
from cuml.testing.utils import quality_param, stress_param, unit_param


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
    from cuml.dask.cluster import KMeans
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

    model = KMeans(
        init="k-means||",
        n_clusters=nclusters,
        random_state=10,
        n_init="auto",
    )

    dask_fit_predict_labels = model.fit_predict(X_train)
    dask_predict_labels = model.predict(X_train, delayed=delayed_predict)

    n_workers = len(list(client.has_what().keys()))

    # Verifying we are grouping partitions. This should be changed soon.
    if n_parts is not None:
        parts_len = n_parts
    else:
        parts_len = n_workers

    if input_type == "dataframe":
        assert dask_predict_labels.npartitions == parts_len
        pred_labels = dask_predict_labels.compute().values
        fit_pred_labels = dask_fit_predict_labels.compute().values
        labels = y_train.compute().values
    elif input_type == "array":
        assert len(dask_predict_labels.chunks[0]) == parts_len
        pred_labels = cp.array(dask_predict_labels.compute())
        fit_pred_labels = cp.array(dask_fit_predict_labels.compute())
        labels = cp.squeeze(y_train.compute())

    assert pred_labels.shape[0] == nrows
    assert cp.max(pred_labels) == nclusters - 1
    assert cp.min(pred_labels) == 0

    # Assert fit_predict(X) and fit(X).predict(X) have same result
    cp.testing.assert_array_equal(pred_labels, fit_pred_labels)

    score = adjusted_rand_score(labels, pred_labels)

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

    np.testing.assert_allclose(actual_score, expected_score, atol=9e-3)
    # The score is -1 * inertia. Scoring the training data should result in
    # -1 * model.inertia_
    np.testing.assert_allclose(actual_score, -cumlModel.inertia_, atol=9e-3)


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [2000])
@pytest.mark.parametrize("ncols", [16])
@pytest.mark.parametrize("nclusters", [5])
@pytest.mark.parametrize("n_parts", [2, 4, 8])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("with_weights", [False, True])
def test_parts_fit_matches_concatenated(
    nrows, ncols, nclusters, n_parts, dtype, with_weights, client
):
    from cuml.cluster import KMeans as cumlKMeans
    from cuml.dask.cluster import KMeans as daskKMeans
    from cuml.dask.datasets import make_blobs

    X_dask, _ = make_blobs(
        n_samples=nrows,
        n_features=ncols,
        centers=nclusters,
        n_parts=n_parts,
        cluster_std=0.5,
        random_state=42,
        dtype=dtype,
    )

    X_local = cp.asarray(X_dask.compute(), dtype=dtype)

    if with_weights:
        rng = cp.random.RandomState(0)
        weights_local = rng.uniform(0.5, 1.5, size=nrows).astype(dtype)
        chunk_size = int(nrows / n_parts)
        weights_dask = da.from_array(weights_local, chunks=(chunk_size,))
    else:
        weights_local = None
        weights_dask = None

    rng = cp.random.RandomState(123)
    init_centers = rng.choice(nrows, size=nclusters, replace=False)
    init = cp.asarray(X_local[init_centers], dtype=dtype)

    dask_model = daskKMeans(
        n_clusters=nclusters,
        init=cp.asnumpy(init),
        n_init=1,
        max_iter=300,
        tol=1e-6,
        random_state=10,
    )
    dask_model.fit(X_dask, sample_weight=weights_dask)

    local_model = cumlKMeans(
        n_clusters=nclusters,
        init=cp.asnumpy(init),
        n_init=1,
        max_iter=300,
        tol=1e-6,
        random_state=10,
    )
    local_model.fit(X_local, sample_weight=weights_local)

    rtol = 1e-4 if dtype == "float32" else 1e-6
    atol = 1e-3 if dtype == "float32" else 1e-6

    dask_centers = cp.asarray(dask_model.cluster_centers_)
    local_centers = cp.asarray(local_model.cluster_centers_)
    dask_order = cp.argsort(dask_centers[:, 0]).get()
    local_order = cp.argsort(local_centers[:, 0]).get()

    cp.testing.assert_allclose(
        dask_centers[dask_order],
        local_centers[local_order],
        rtol=rtol,
        atol=atol,
    )

    assert dask_model.n_iter_ == local_model.n_iter_
    np.testing.assert_allclose(
        float(dask_model.inertia_),
        float(local_model.inertia_),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.mg
def test_nclusters_exceeds_n_samples(client):
    """Test that n_clusters > n_samples raises a clear ValueError."""
    from cuml.dask.cluster import KMeans
    from cuml.dask.datasets import make_blobs

    # Use fewer data points than clusters
    n_clusters = 11
    n_samples = 10

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=5,
        centers=5,
        n_parts=2,
        random_state=10,
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=10,
    )

    with pytest.raises(
        ValueError, match="n_samples=10 should be >= n_clusters=11"
    ):
        model.fit(X)
