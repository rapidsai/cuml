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

import numpy as np
import pytest

from cuml.dask.datasets import make_regression
from cuml.dask.linear_model import ElasticNet, Lasso
from cuml.metrics import r2_score
from cuml.testing.utils import quality_param, stress_param, unit_param


@pytest.mark.mg
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("alpha", [0.001])
@pytest.mark.parametrize("algorithm", ["cyclic", "random"])
@pytest.mark.parametrize(
    "nrows", [unit_param(50), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500]),
    ],
)
@pytest.mark.parametrize(
    "n_parts", [unit_param(4), quality_param(32), stress_param(64)]
)
@pytest.mark.parametrize("delayed", [True, False])
def test_lasso(
    dtype, alpha, algorithm, nrows, column_info, n_parts, delayed, client
):
    ncols, n_info = column_info

    X, y = make_regression(
        n_samples=nrows,
        n_features=ncols,
        n_informative=n_info,
        n_parts=n_parts,
        client=client,
        dtype=dtype,
    )

    lasso = Lasso(
        alpha=np.array([alpha]),
        fit_intercept=True,
        normalize=False,
        max_iter=1000,
        selection=algorithm,
        tol=1e-10,
        client=client,
    )

    lasso.fit(X, y)

    y_hat = lasso.predict(X, delayed=delayed)

    assert r2_score(y.compute(), y_hat.compute()) >= 0.99


@pytest.mark.mg
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "nrows", [unit_param(50), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500]),
    ],
)
@pytest.mark.parametrize(
    "n_parts", [unit_param(16), quality_param(32), stress_param(64)]
)
def test_lasso_default(dtype, nrows, column_info, n_parts, client):

    ncols, n_info = column_info

    X, y = make_regression(
        n_samples=nrows,
        n_features=ncols,
        n_informative=n_info,
        client=client,
        dtype=dtype,
    )

    lasso = Lasso(client=client)

    lasso.fit(X, y)

    y_hat = lasso.predict(X)

    assert r2_score(y.compute(), y_hat.compute()) >= 0.99


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("alpha", [0.5])
@pytest.mark.parametrize("algorithm", ["cyclic", "random"])
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500]),
    ],
)
@pytest.mark.parametrize(
    "n_parts", [unit_param(16), quality_param(32), stress_param(64)]
)
@pytest.mark.parametrize("delayed", [True, False])
def test_elastic_net(
    dtype, alpha, algorithm, nrows, column_info, n_parts, client, delayed
):
    ncols, n_info = column_info

    X, y = make_regression(
        n_samples=nrows,
        n_features=ncols,
        n_informative=n_info,
        n_parts=n_parts,
        client=client,
        dtype=dtype,
    )

    elasticnet = ElasticNet(
        alpha=np.array([alpha]),
        fit_intercept=True,
        normalize=False,
        max_iter=1000,
        selection=algorithm,
        tol=1e-10,
        client=client,
    )

    elasticnet.fit(X, y)

    y_hat = elasticnet.predict(X, delayed=delayed)

    # based on differences with scikit-learn 0.22
    if alpha == 0.2:
        assert r2_score(y.compute(), y_hat.compute()) >= 0.96

    else:
        assert r2_score(y.compute(), y_hat.compute()) >= 0.80


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([20, 10]),
        quality_param([100, 50]),
        stress_param([1000, 500]),
    ],
)
@pytest.mark.parametrize(
    "n_parts", [unit_param(16), quality_param(32), stress_param(64)]
)
def test_elastic_net_default(dtype, nrows, column_info, n_parts, client):
    ncols, n_info = column_info

    X, y = make_regression(
        n_samples=nrows,
        n_features=ncols,
        n_informative=n_info,
        n_parts=n_parts,
        client=client,
        dtype=dtype,
    )

    elasticnet = ElasticNet(client=client)

    elasticnet.fit(X, y)

    y_hat = elasticnet.predict(X)

    assert r2_score(y.compute(), y_hat.compute()) >= 0.96
