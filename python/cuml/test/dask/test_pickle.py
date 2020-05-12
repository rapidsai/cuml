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

import cuml
import numpy as np
import pickle
import pytest

from numpy.testing import assert_equal

from dask.distributed import Client

from cuml.dask.linear_model import LinearRegression

from cuml.dask.datasets import make_regression
from sklearn.model_selection import train_test_split


def pickle_save_load(tmpdir, func_create_model, func_assert):
    model, X_test = func_create_model()
    pickle_file = tmpdir.join('cu_model.pickle')

    try:
        with open(pickle_file, 'wb') as pf:
            pickle.dump(model, pf)
    except (TypeError, ValueError) as e:
        pf.close()
        pytest.fail(e)

    del model

    with open(pickle_file, 'rb') as pf:
        cu_after_pickle_model = pickle.load(pf)

    func_assert(cu_after_pickle_model, X_test)


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
def test_regressor_pickle(tmpdir, datatype, keys, data_size, fit_intercept,
                          cluster):

    client = Client(cluster)
    result = {}

    def create_mod():
        nrows, ncols, n_info = data_size
        X_train, y_train, X_test = make_dataset(datatype, nrows,
                                                ncols, n_info)
        model = LinearRegression(fit_intercept=fit_intercept,
                                 client=client)
        model.fit(X_train, y_train)
        result["regressor"] = model.predict(X_test)
        return model, X_test

    def assert_model(pickled_model, X_test):
        expected = result["regressor"].compute()
        actual = pickled_model.predict(X_test).compute()

        assert_equal(expected.get(), actual.get())

    pickle_save_load(tmpdir, create_mod, assert_model)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('keys', [cuml.dask.linear_model.LinearRegression])
@pytest.mark.parametrize('data_size', [[500, 20, 10]])
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_regressor_sg_train_mg_predict(datatype, keys, data_size, fit_intercept, cluster):

    client = Client(cluster)

    from cuml.linear_model import LinearRegression as sgLR

    nrows, ncols, n_info = data_size
    X_train, y_train, X_test = make_dataset(datatype, nrows, ncols, n_info)

    X_train = X_train.compute()
    y_train = y_train.compute()

    X_test_local = X_test.compute()

    local_model = sgLR(fit_intercept=fit_intercept)
    local_model.fit(X_train, y_train)

    expected = local_model.predict(X_test_local)

    dist_model = LinearRegression(model=local_model)
    actual = dist_model.predict(X_test).compute()

    assert_equal(expected.get(), actual.get())


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('keys', [cuml.dask.linear_model.LinearRegression])
@pytest.mark.parametrize('data_size', [[500, 20, 10]])
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_regressor_mg_train_sg_predict(datatype, keys, data_size, fit_intercept, cluster):

    client = Client(cluster)

    from cuml.linear_model import LinearRegression as sgLR

    nrows, ncols, n_info = data_size
    X_train, y_train, X_test = make_dataset(datatype, nrows, ncols, n_info)

    X_test_local = X_test.compute()

    dist_model = LinearRegression(fit_intercept=fit_intercept)
    dist_model.fit(X_train, y_train)

    expected = dist_model.predict(X_test).compute()

    local_model = dist_model.get_model()
    actual = local_model.predict(X_test_local)

    assert_equal(expected.get(), actual.get())

