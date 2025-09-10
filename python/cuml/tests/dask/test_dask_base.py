# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import cupy
import numpy as np
import pytest
from dask_ml.wrappers import ParallelPostFit
from numpy.testing import assert_equal
from sklearn.model_selection import train_test_split

import cuml
from cuml.dask.cluster import KMeans
from cuml.dask.datasets import make_blobs, make_regression
from cuml.dask.linear_model import LinearRegression
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
