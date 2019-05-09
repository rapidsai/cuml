# Copyright (c) 2018, NVIDIA CORPORATION.
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
from cuml import LinearRegression as cuLinearRegression
from cuml import Ridge as cuRidge
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.linear_model import Ridge as skRidge
from cuml.test.utils import array_equal
import cudf
import numpy as np


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('X_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('y_type', ['series', 'ndarray'])
@pytest.mark.parametrize('algorithm', ['eig', 'svd'])
def test_ols(datatype, X_type, y_type, algorithm):

    X = np.array([[2.0, 5.0], [6.0, 9.0], [2.0, 2.0], [2.0, 3.0]],
                 dtype=datatype)
    y = np.dot(X, np.array([5.0, 10.0]).astype(datatype))

    pred_data = np.array([[3.0, 5.0], [2.0, 5.0]]).astype(datatype)

    skols = skLinearRegression(fit_intercept=True,
                               normalize=False)
    skols.fit(X, y)

    cuols = cuLinearRegression(fit_intercept=True,
                               normalize=False,
                               algorithm=algorithm)

    if X_type == 'dataframe':
        gdf = cudf.DataFrame()
        gdf['0'] = np.asarray([2, 6, 2, 2], dtype=datatype)
        gdf['1'] = np.asarray([5, 9, 2, 3], dtype=datatype)
        cuols.fit(gdf, y)

    elif X_type == 'ndarray':
        cuols.fit(X, y)

    sk_predict = skols.predict(pred_data)
    cu_predict = cuols.predict(pred_data).to_array()

    print(sk_predict)
    print(cu_predict)

    # print(skols.coef_)
    print(cuols.gdf_datatype)
    print(y.dtype)

    assert array_equal(sk_predict, cu_predict, 1e-3, with_sign=True)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('X_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('y_type', ['series', 'ndarray'])
@pytest.mark.parametrize('algorithm', ['eig', 'svd'])
def test_ridge(datatype, X_type, y_type, algorithm):

    X = np.array([[2.0, 5.0], [6.0, 9.0], [2.0, 2.0], [2.0, 3.0]],
                 dtype=datatype)
    y = np.dot(X, np.array([5.0, 10.0]).astype(datatype))

    pred_data = np.array([[3.0, 5.0], [2.0, 5.0]]).astype(datatype)

    skridge = skRidge(fit_intercept=False,
                      normalize=False)
    skridge.fit(X, y)

    curidge = cuRidge(fit_intercept=False,
                      normalize=False,
                      solver=algorithm)

    if X_type == 'dataframe':
        gdf = cudf.DataFrame()
        gdf['0'] = np.asarray([2, 6, 2, 2], dtype=datatype)
        gdf['1'] = np.asarray([5, 9, 2, 3], dtype=datatype)
        curidge.fit(gdf, y)

    elif X_type == 'ndarray':
        curidge.fit(X, y)

    sk_predict = skridge.predict(pred_data)
    cu_predict = curidge.predict(pred_data).to_array()

    assert array_equal(sk_predict, cu_predict, 1e-3, with_sign=True)
