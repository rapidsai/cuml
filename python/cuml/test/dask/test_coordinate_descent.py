# Copyright (c) 2019, NVIDIA CORPORATION.
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

import pytest
from dask_cuda import LocalCUDACluster

from dask.distributed import Client
from cuml.dask.common import utils as dask_utils
from cuml.metrics import r2_score
from cuml.test.utils import unit_param, quality_param, stress_param

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import dask_cudf
import cudf


def _prep_training_data(c, X_train, y_train, partitions_per_worker):
    workers = c.has_what().keys()
    n_partitions = partitions_per_worker * len(workers)
    X_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X_train))
    X_train_df = dask_cudf.from_cudf(X_cudf, npartitions=n_partitions)

    y_cudf = np.array(pd.DataFrame(y_train).values)
    y_cudf = y_cudf[:, 0]
    y_cudf = cudf.Series(y_cudf)
    y_train_df = \
        dask_cudf.from_cudf(y_cudf, npartitions=n_partitions)

    X_train_df, \
        y_train_df = dask_utils.persist_across_workers(c,
                                                       [X_train_df,
                                                        y_train_df],
                                                       workers=workers)
    return X_train_df, y_train_df


def make_regression_dataset(datatype, nrows, ncols, n_info):
    X, y = make_regression(n_samples=nrows, n_features=ncols,
                           n_informative=5, random_state=0)
    X = X.astype(datatype)
    y = y.astype(datatype)

    return X, y


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('X_type', ['ndarray'])
@pytest.mark.parametrize('alpha', [0.1, 0.001])
@pytest.mark.parametrize('algorithm', ['cyclic', 'random'])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('column_info', [unit_param([20, 10]),
                         quality_param([100, 50]),
                         stress_param([1000, 500])])
@pytest.mark.parametrize('n_parts', [unit_param(16), quality_param(32),
                         stress_param(64)])
def test_lasso(datatype, X_type, alpha, algorithm,
               nrows, column_info, n_parts, client=None):
    ncols, n_info = column_info
    
    ncols, n_info = column_info
    if client is None:
        cluster = LocalCUDACluster()
        client = Client(cluster)

    try:
        from cuml.dask.linear_model import Lasso as cuLasso

        nrows = np.int(nrows)
        ncols = np.int(ncols)
        X, y = make_regression_dataset(datatype, nrows, ncols, n_info)

        X_df, y_df = _prep_training_data(client, X, y, n_parts)

        cu_lasso = cuLasso(alpha=np.array([alpha]), fit_intercept=True,
                       normalize=False, max_iter=1000,
                       selection=algorithm, tol=1e-10)

        cu_lasso.fit(X_df, y_df)
        cu_predict = cu_lasso.predict(X_df)
        cu_r2 = r2_score(y, cu_predict.compute().to_pandas().values)

        if nrows < 500000:
            sk_lasso = Lasso(alpha=np.array([alpha]), fit_intercept=True,
                         normalize=False, max_iter=1000,
                         selection=algorithm, tol=1e-10)
            sk_lasso.fit(X, y)
            sk_predict = sk_lasso.predict(X)
            sk_r2 = r2_score(y, sk_predict)
            assert cu_r2 >= sk_r2 - 0.07

    finally:
        client.close()
        cluster.close()


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('column_info', [unit_param([20, 10]),
                         quality_param([100, 50]),
                         stress_param([1000, 500])])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_parts', [unit_param(16), quality_param(32),
                         stress_param(110)])
def test_lasso_default(datatype, nrows, column_info, n_parts, client=None):

    ncols, n_info = column_info
    if client is None:
        cluster = LocalCUDACluster()
        client = Client(cluster)

    try:
        from cuml.dask.linear_model import Lasso as cuLasso

        nrows = np.int(nrows)
        ncols = np.int(ncols)
        X, y = make_regression_dataset(datatype, nrows, ncols, n_info)

        X_df, y_df = _prep_training_data(client, X, y, n_parts)

        cu_lasso = cuLasso()

        cu_lasso.fit(X_df, y_df)
        cu_predict = cu_lasso.predict(X_df)
        cu_r2 = r2_score(y, cu_predict.compute().to_pandas().values)

        sk_lasso = Lasso()
        sk_lasso.fit(X, y)
        sk_predict = sk_lasso.predict(X)
        sk_r2 = r2_score(y, sk_predict)
        assert cu_r2 >= sk_r2 - 0.07

    finally:
        client.close()
        cluster.close()


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('X_type', ['ndarray'])
@pytest.mark.parametrize('alpha', [0.2, 0.7])
@pytest.mark.parametrize('algorithm', ['cyclic', 'random'])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('column_info', [unit_param([20, 10]),
                         quality_param([100, 50]),
                         stress_param([1000, 500])])
@pytest.mark.parametrize('n_parts', [unit_param(16), quality_param(32),
                         stress_param(64)])
def test_elastic_net(datatype, X_type, alpha, algorithm,
               nrows, column_info, n_parts, client=None):
    ncols, n_info = column_info
    
    ncols, n_info = column_info
    if client is None:
        cluster = LocalCUDACluster()
        client = Client(cluster)

    try:
        from cuml.dask.linear_model import ElasticNet as cuElasticNet

        nrows = np.int(nrows)
        ncols = np.int(ncols)
        X, y = make_regression_dataset(datatype, nrows, ncols, n_info)

        X_df, y_df = _prep_training_data(client, X, y, n_parts)

        elastic_cu = cuElasticNet(alpha=np.array([alpha]), fit_intercept=True,
                              normalize=False, max_iter=1000,
                              selection=algorithm, tol=1e-10)

        elastic_cu.fit(X_df, y_df)
        cu_predict = elastic_cu.predict(X_df)
        cu_r2 = r2_score(y, cu_predict.compute().to_pandas().values)

        if nrows < 500000:
            sk_elasticnet = ElasticNet(alpha=np.array([alpha]), fit_intercept=True,
                         normalize=False, max_iter=1000,
                         selection=algorithm, tol=1e-10)
            sk_elasticnet.fit(X, y)
            sk_predict = sk_elasticnet.predict(X)
            sk_r2 = r2_score(y, sk_predict)
            assert cu_r2 >= sk_r2 - 0.07

    finally:
        client.close()
        cluster.close()


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('column_info', [unit_param([20, 10]),
                         quality_param([100, 50]),
                         stress_param([1000, 500])])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_parts', [unit_param(16), quality_param(32),
                         stress_param(110)])
def test_elastic_net_default(datatype, nrows, column_info, n_parts, client=None):

    ncols, n_info = column_info
    if client is None:
        cluster = LocalCUDACluster()
        client = Client(cluster)

    try:
        from cuml.dask.linear_model import ElasticNet as cuElasticNet

        nrows = np.int(nrows)
        ncols = np.int(ncols)
        X, y = make_regression_dataset(datatype, nrows, ncols, n_info)

        X_df, y_df = _prep_training_data(client, X, y, n_parts)

        elastic_cu = cuElasticNet()

        elastic_cu.fit(X_df, y_df)
        cu_predict = elastic_cu.predict(X_df)
        cu_r2 = r2_score(y, cu_predict.compute().to_pandas().values)

        sk_elasticnet = ElasticNet()
        sk_elasticnet.fit(X, y)
        sk_predict = sk_elasticnet.predict(X)
        sk_r2 = r2_score(y, sk_predict)
        assert cu_r2 >= sk_r2 - 0.07

    finally:
        client.close()
        cluster.close()
