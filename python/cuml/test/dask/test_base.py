# Copyright (c) 2020, NVIDIA CORPORATION.
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

from cuml.dask.cluster import KMeans
from cuml.dask.naive_bayes.naive_bayes import MultinomialNB
from cuml.test.dask.utils import load_text_corpus

from cuml.dask.datasets import make_blobs

import cuml
import numpy as np
import pytest

from numpy.testing import assert_equal

from cuml.dask.linear_model import LinearRegression

from cuml.dask.datasets import make_regression
from sklearn.model_selection import train_test_split


def make_dataset(datatype, nrows, ncols, n_info):
    X, y = make_regression(n_samples=nrows, n_features=ncols,
                           n_informative=n_info, random_state=0)
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    return X_train, y_train, X_test


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('keys', [cuml.dask.linear_model.LinearRegression])
@pytest.mark.parametrize('data_size', [[500, 20, 10]])
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_get_combined_model(datatype, keys, data_size, fit_intercept,
                            client):

    nrows, ncols, n_info = data_size
    X_train, y_train, X_test = make_dataset(datatype, nrows,
                                            ncols, n_info)
    model = LinearRegression(fit_intercept=fit_intercept,
                             client=client)
    model.fit(X_train, y_train)

    combined_model = model.get_combined_model()
    assert combined_model.coef_ is not None
    assert combined_model.intercept_ is not None


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('keys', [cuml.dask.linear_model.LinearRegression])
@pytest.mark.parametrize('data_size', [[500, 20, 10]])
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_regressor_mg_train_sg_predict(datatype, keys, data_size,
                                       fit_intercept, client):

    nrows, ncols, n_info = data_size
    X_train, y_train, X_test = make_dataset(datatype, nrows, ncols, n_info)

    X_test_local = X_test.compute()

    dist_model = LinearRegression(fit_intercept=fit_intercept, client=client)
    dist_model.fit(X_train, y_train)

    expected = dist_model.predict(X_test).compute()

    local_model = dist_model.get_combined_model()
    actual = local_model.predict(X_test_local)

    assert_equal(expected.get(), actual.get())


def test_getattr(client):

    # Test getattr on local param
    kmeans_model = KMeans(client=client)

    assert kmeans_model.client is not None

    # Test getattr on local_model param with a non-distributed model

    X, y = make_blobs(n_samples=5,
                      n_features=5,
                      centers=2,
                      n_parts=2,
                      cluster_std=0.01,
                      random_state=10)

    kmeans_model.fit(X)

    assert kmeans_model.cluster_centers_ is not None
    assert isinstance(kmeans_model.cluster_centers_, cupy.core.ndarray)

    # Test getattr on trained distributed model

    X, y = load_text_corpus(client)

    nb_model = MultinomialNB(client=client)
    nb_model.fit(X, y)

    assert nb_model.feature_count_ is not None
    assert isinstance(nb_model.feature_count_, cupy.core.ndarray)
