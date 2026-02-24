# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import warnings

import cudf
import cupy
import dask_cudf
import numpy as np
import pandas as pd
import pytest
from dask_ml.wrappers import ParallelPostFit
from numpy.testing import assert_equal
from sklearn.linear_model import LogisticRegression as skLR
from sklearn.model_selection import train_test_split

import cuml
from cuml.dask.cluster import KMeans
from cuml.dask.common import utils as dask_utils
from cuml.dask.common.input_utils import DistributedDataHandler
from cuml.dask.datasets import make_blobs, make_regression
from cuml.dask.linear_model import LinearRegression
from cuml.dask.linear_model import LogisticRegression as cumlLR_dask
from cuml.dask.naive_bayes.naive_bayes import MultinomialNB
from cuml.testing.dask.utils import load_text_corpus


def make_dataset(datatype, nrows, ncols, n_info):
    X, y = make_regression(
        n_samples=nrows, n_features=ncols, n_informative=n_info, random_state=0
    )
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    return X_train, y_train, X_test


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("keys", [cuml.dask.linear_model.LinearRegression])
@pytest.mark.parametrize("data_size", [[500, 20, 10]])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_get_combined_model(datatype, keys, data_size, fit_intercept, client):
    nrows, ncols, n_info = data_size
    X_train, y_train, X_test = make_dataset(datatype, nrows, ncols, n_info)
    model = LinearRegression(
        fit_intercept=fit_intercept, client=client, verbose=True
    )
    model.fit(X_train, y_train)
    print("Fit done")

    combined_model = model.get_combined_model()
    assert combined_model.coef_ is not None
    assert combined_model.intercept_ is not None

    y_hat = combined_model.predict(X_train.compute())

    np.testing.assert_allclose(
        y_hat.get(), y_train.compute().get(), atol=1e-3, rtol=1e-3
    )


def test_check_internal_model_failures(client):
    # Test model not trained yet
    model = LinearRegression(client=client)
    assert model.get_combined_model() is None

    # Test single Int future fails
    int_future = client.submit(lambda: 1)
    with pytest.raises(ValueError):
        model._set_internal_model(int_future)

    # Test list Int future fails
    with pytest.raises(ValueError):
        model._set_internal_model([int_future])

    # Test directly setting Int fails
    with pytest.raises(ValueError):
        model._set_internal_model(1)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("keys", [cuml.dask.linear_model.LinearRegression])
@pytest.mark.parametrize("data_size", [[500, 20, 10]])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_regressor_mg_train_sg_predict(
    datatype, keys, data_size, fit_intercept, client
):
    nrows, ncols, n_info = data_size
    X_train, y_train, X_test = make_dataset(datatype, nrows, ncols, n_info)

    X_test_local = X_test.compute()

    dist_model = LinearRegression(fit_intercept=fit_intercept, client=client)
    dist_model.fit(X_train, y_train)

    expected = dist_model.predict(X_test).compute()

    local_model = dist_model.get_combined_model()
    actual = local_model.predict(X_test_local)

    assert_equal(expected.get(), actual.get())


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("keys", [cuml.linear_model.LinearRegression])
@pytest.mark.parametrize("data_size", [[500, 20, 10]])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_regressor_sg_train_mg_predict(
    datatype, keys, data_size, fit_intercept, client
):
    # Just testing for basic compatibility w/ dask-ml's ParallelPostFit.
    # Refer to test_pickle.py for more extensive testing of single-GPU
    # model serialization.

    nrows, ncols, n_info = data_size
    X_train, y_train, _ = make_dataset(datatype, nrows, ncols, n_info)

    X_train_local = X_train.compute()
    y_train_local = y_train.compute()

    local_model = cuml.linear_model.LinearRegression(
        fit_intercept=fit_intercept
    )
    local_model.fit(X_train_local, y_train_local)

    dist_model = ParallelPostFit(estimator=local_model)

    predictions = dist_model.predict(X_train).compute()

    assert isinstance(predictions, cupy.ndarray)

    # Dataset should be fairly linear already so the predictions should
    # be very close to the training data.
    np.testing.assert_allclose(
        predictions.get(), y_train.compute().get(), atol=1e-3, rtol=1e-3
    )


def test_getattr(client):
    # Test getattr on local param
    kmeans_model = KMeans(client=client)

    # Test AttributeError
    with pytest.raises(AttributeError):
        kmeans_model.cluster_centers_

    assert kmeans_model.client is not None

    # Test getattr on local_model param with a non-distributed model
    X, y = make_blobs(
        n_samples=20,
        n_features=5,
        centers=8,
        n_parts=2,
        cluster_std=0.01,
        random_state=10,
    )
    kmeans_model.fit(X)

    assert kmeans_model.cluster_centers_ is not None
    assert isinstance(kmeans_model.cluster_centers_, cupy.ndarray)

    # Test getattr on trained distributed model

    X, y = load_text_corpus(client)

    nb_model = MultinomialNB(client=client)
    nb_model.fit(X, y)

    assert nb_model.feature_count_ is not None
    assert isinstance(nb_model.feature_count_, cupy.ndarray)


def _make_ddh_with_empty_worker(client):
    """Build a DDH where one worker holds real data and another holds an
    empty (0-row) partition.  Requires at least 2 workers."""
    workers = list(client.scheduler_info()["workers"].keys())
    if len(workers) < 2:
        pytest.skip(
            "Need at least 2 workers to test empty-partition filtering"
        )

    real = cupy.random.randn(100, 5).astype(np.float32)
    empty = cupy.empty((0, 5), dtype=np.float32)

    real_f = client.scatter(real, workers=[workers[0]])
    empty_f = client.scatter(empty, workers=[workers[1]])

    gpu_futures = [(workers[0], real_f), (workers[1], empty_f)]
    return DistributedDataHandler(
        gpu_futures=gpu_futures,
        workers=tuple(workers[:2]),
        datatype="cupy",
        multiple=False,
        client=client,
    )


def test_ddh_warns_on_empty_partitions(client):
    """_fetch_worker_sizes should warn when a worker has zero rows."""
    ddh = _make_ddh_with_empty_worker(client)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ddh._fetch_worker_sizes()
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "no data" in str(user_warnings[0].message)


def test_ddh_filters_empty_workers(client):
    """After _fetch_worker_sizes, workers with 0 rows should be removed."""
    ddh = _make_ddh_with_empty_worker(client)
    ddh._fetch_worker_sizes()

    assert len(ddh.workers) == 1
    assert len(ddh.worker_to_parts) == 1
    assert len(ddh.gpu_futures) == 1
    for _, (_, total) in ddh._worker_sizes.items():
        assert total > 0


def test_ddh_total_rows_after_filtering(client):
    """Total rows should reflect only the non-empty worker's data."""
    ddh = _make_ddh_with_empty_worker(client)
    ddh._fetch_worker_sizes()

    total = sum(t for _, (_, t) in ddh._worker_sizes.items())
    assert total == 100


def test_logistic_regression_with_empty_partitions(client):
    """LogisticRegression should train and predict correctly when some
    partitions are empty (more partitions than data rows)."""
    workers = list(client.scheduler_info()["workers"].keys())
    if len(workers) < 2:
        pytest.skip(
            "Need at least 2 workers to test empty-partition filtering"
        )

    X = np.array([(1, 2), (1, 3), (2, 1), (3, 1)], np.float32)
    y = np.array([1.0, 1.0, 0.0, 0.0], np.float32)

    n_partitions = len(workers) * 2  # guarantees empty partitions with 4 rows

    X_df = dask_cudf.from_cudf(
        cudf.DataFrame(pd.DataFrame(X)), npartitions=n_partitions
    )
    y_series = dask_cudf.from_cudf(cudf.Series(y), npartitions=n_partitions)
    X_df, y_series = dask_utils.persist_across_workers(
        client, [X_df, y_series], workers=workers
    )

    mg = cumlLR_dask(client=client)
    mg.fit(X_df, y_series)
    mg_preds = mg.predict(X_df).compute()

    cpu_lr = skLR()
    cpu_lr.fit(X, y)
    cpu_preds = cpu_lr.predict(X)

    np.testing.assert_array_equal(
        np.array(
            mg_preds.values_host
            if hasattr(mg_preds, "values_host")
            else mg_preds.get()
        ),
        cpu_preds,
    )
