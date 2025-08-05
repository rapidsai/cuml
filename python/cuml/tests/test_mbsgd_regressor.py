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

import cupy as cp
import numpy as np
import pytest
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split

from cuml.datasets import make_regression
from cuml.linear_model import MBSGDRegressor as cumlMBSGRegressor
from cuml.metrics import r2_score
from cuml.testing.utils import quality_param, stress_param, unit_param


@pytest.fixture(
    scope="module",
    params=[
        unit_param([500, 20, 10, np.float32]),
        unit_param([500, 20, 10, np.float64]),
        quality_param([5000, 100, 50, np.float32]),
        quality_param([5000, 100, 50, np.float64]),
        stress_param([500000, 1000, 500, np.float32]),
        stress_param([500000, 1000, 500, np.float64]),
    ],
    ids=[
        "500-20-10-f32",
        "500-20-10-f64",
        "5000-100-50-f32",
        "5000-100-50-f64",
        "500000-1000-500-f32",
        "500000-1000-500-f64",
    ],
)
def make_dataset(request):
    nrows, ncols, n_info, datatype = request.param
    # Assume at least 4GB memory
    max_gpu_memory = pytest.max_gpu_memory or 4

    if nrows == 500000 and datatype == np.float64 and max_gpu_memory < 32:
        if pytest.adapt_stress_test:
            nrows = nrows * max_gpu_memory // 32
        else:
            pytest.skip(
                "Insufficient GPU memory for this test."
                "Re-run with 'CUML_ADAPT_STRESS_TESTS=True'"
            )
    X, y = make_regression(
        n_samples=nrows, n_informative=n_info, n_features=ncols, random_state=0
    )
    X = cp.array(X).astype(datatype)
    y = cp.array(y).astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=10
    )

    return nrows, datatype, X_train, X_test, y_train, y_test


@pytest.mark.parametrize(
    # Grouped those tests to reduce the total number of individual tests
    # while still keeping good coverage of the different features of MBSGD
    ("lrate", "penalty"),
    [
        ("constant", None),
        ("invscaling", "l1"),
        ("adaptive", "l2"),
        ("constant", "elasticnet"),
    ],
)
@pytest.mark.filterwarnings("ignore:Maximum::sklearn[.*]")
def test_mbsgd_regressor_vs_skl(lrate, penalty, make_dataset):
    nrows, datatype, X_train, X_test, y_train, y_test = make_dataset

    if nrows < 500000:

        cu_mbsgd_regressor = cumlMBSGRegressor(
            learning_rate=lrate,
            eta0=0.005,
            epochs=100,
            fit_intercept=True,
            batch_size=2,
            tol=0.0,
            penalty=penalty,
        )

        cu_mbsgd_regressor.fit(X_train, y_train)
        cu_pred = cu_mbsgd_regressor.predict(X_test)
        cu_r2 = r2_score(cu_pred, y_test)

        skl_sgd_regressor = SGDRegressor(
            learning_rate=lrate,
            eta0=0.005,
            max_iter=100,
            fit_intercept=True,
            tol=0.0,
            penalty=penalty,
            random_state=0,
        )

        skl_sgd_regressor.fit(cp.asnumpy(X_train), cp.asnumpy(y_train).ravel())
        skl_pred = skl_sgd_regressor.predict(cp.asnumpy(X_test))
        skl_r2 = r2_score(skl_pred, y_test)
        assert abs(cu_r2 - skl_r2) <= 0.021


@pytest.mark.parametrize(
    # Grouped those tests to reduce the total number of individual tests
    # while still keeping good coverage of the different features of MBSGD
    ("lrate", "penalty"),
    [
        ("constant", None),
        ("invscaling", "l1"),
        ("adaptive", "l2"),
        ("constant", "elasticnet"),
    ],
)
def test_mbsgd_regressor(lrate, penalty, make_dataset):
    nrows, datatype, X_train, X_test, y_train, y_test = make_dataset

    model = cumlMBSGRegressor(
        learning_rate=lrate,
        eta0=0.005,
        epochs=100,
        fit_intercept=True,
        batch_size=nrows / 100,
        tol=0.0,
        penalty=penalty,
    )
    # Fitted attributes don't exist before fit
    assert not hasattr(model, "coef_")
    assert not hasattr(model, "intercept_")

    model.fit(X_train, y_train)

    # Fitted attributes exist and have correct types after fit
    assert isinstance(model.coef_, type(X_train))
    assert isinstance(model.intercept_, float)

    cu_pred = model.predict(X_test)
    cu_r2 = r2_score(cu_pred, y_test)

    assert cu_r2 >= 0.88


def test_mbsgd_regressor_default(make_dataset):
    nrows, datatype, X_train, X_test, y_train, y_test = make_dataset

    cu_mbsgd_regressor = cumlMBSGRegressor(batch_size=nrows / 100)
    cu_mbsgd_regressor.fit(X_train, y_train)
    cu_pred = cu_mbsgd_regressor.predict(X_test)
    cu_r2 = r2_score(cu_pred, y_test)

    assert cu_r2 > 0.9


def test_mbsgd_regressor_set_params():
    x = np.linspace(0, 1, 50)
    y = x * 2

    model = cumlMBSGRegressor()
    model.fit(x, y)
    coef_before = model.coef_

    model = cumlMBSGRegressor(eta0=0.1, fit_intercept=False)
    model.fit(x, y)
    coef_after = model.coef_

    model = cumlMBSGRegressor()
    model.set_params(**{"eta0": 0.1, "fit_intercept": False})
    model.fit(x, y)
    coef_test = model.coef_

    assert coef_before != coef_after
    assert coef_after == coef_test
