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
import pandas as pd
from sklearn.datasets import make_regression


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('X_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('y_type', ['series', 'ndarray'])
@pytest.mark.parametrize('algorithm', ['eig', 'svd'])
def test_ols(datatype, X_type, y_type, algorithm,run_stress,run_correctness_test):
    #pdb.set_trace()
    nrows = 5000
    ncols = 1000
    n_info = 500
    if run_stress==True:
        train_rows = np.int32(nrows*80)
        X,y = make_regression(n_samples=(nrows*100),n_features=ncols,n_informative=n_info, random_state=0) 
        X_test = np.array(X[train_rows:,0:]).astype(datatype)
        X_train = np.array(X[0:train_rows,:]).astype(datatype)
        y_train = np.array(y[0:train_rows,]).astype(datatype)

    if run_correctness_test==True:
        train_rows = np.int32(nrows*0.8)
        X,y = make_regression(n_samples=nrows,n_features=int(ncols/2),n_informative=int(n_info/2), random_state=0) 
        X_test = np.array(X[train_rows:,0:]).astype(datatype)
        X_train = np.array(X[0:train_rows,:]).astype(datatype)
        y_train = np.array(y[0:train_rows,]).astype(datatype)


    else:
        X_train = np.array([[2.0, 5.0], [6.0, 9.0], [2.0, 2.0], [2.0, 3.0]],
                 dtype=datatype)
        train_rows = X_train.shape[0]
        y_train = np.dot(X_train, np.array([5.0, 10.0]).astype(datatype))
        X_test = np.array([[3.0, 5.0], [2.0, 5.0]]).astype(datatype)

    skols = skLinearRegression(fit_intercept=True,
                               normalize=False)
    skols.fit(X_train, y_train)

    cuols = cuLinearRegression(fit_intercept=True,
                               normalize=False,
                               algorithm=algorithm)

    if X_type == 'dataframe':
        y_train = pd.DataFrame({'fea0':y_train[0:,]})
        X_train = pd.DataFrame({'fea%d'%i:X_train[0:,i] for i in range(X_train.shape[1])})
        X_test = pd.DataFrame({'fea%d'%i:X_test[0:,i] for i in range(X_test.shape[1])})
        X_cudf = cudf.DataFrame.from_pandas(X_train) 
        X_cudf_test = cudf.DataFrame.from_pandas(X_test)
        y_cudf = y_train.values
        y_cudf = y_cudf[:,0] 
        y_cudf = cudf.Series(y_cudf)
        cuols.fit(X_cudf,y_cudf)
        cu_predict = cuols.predict(X_cudf_test).to_array()

    elif X_type == 'ndarray':

        cuols.fit(X_train, y_train)
        cu_predict = cuols.predict(X_test).to_array()

    sk_predict = skols.predict(X_test)
    assert array_equal(sk_predict, cu_predict, 1e-1, with_sign=True)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('X_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('y_type', ['series', 'ndarray'])
@pytest.mark.parametrize('algorithm', ['eig', 'svd'])
def test_ridge(datatype, X_type, y_type, algorithm,run_stress,run_correctness_test):

    nrows = 5000
    ncols = 1000
    n_info = 500
    if run_stress==True:
        train_rows = np.int32(nrows*80)
        X,y = make_regression(n_samples=(nrows*100),n_features=ncols,n_informative=n_info, random_state=0) 
        X_test = np.asarray(X[train_rows:,0:]).astype(datatype)
        X_train = np.asarray(X[0:train_rows,:]).astype(datatype)
        y_train = np.asarray(y[0:train_rows,]).astype(datatype)

    if run_correctness_test == True:
        train_rows = np.int32(nrows*0.8)
        X,y = make_regression(n_samples=nrows,n_features=ncols,n_informative=n_info, random_state=0) 
        X_test = np.asarray(X[train_rows:,0:]).astype(datatype)
        X_train = np.asarray(X[0:train_rows,:]).astype(datatype)
        y_train = np.asarray(y[0:train_rows,]).astype(datatype)


    else:
        X_train = np.array([[2.0, 5.0], [6.0, 9.0], [2.0, 2.0], [2.0, 3.0]],
                 dtype=datatype)
        train_rows = X_train.shape[0]
        y_train = np.dot(X_train, np.array([5.0, 10.0]).astype(datatype))
        X_test = np.array([[3.0, 5.0], [2.0, 5.0]]).astype(datatype)

    skridge = skRidge(fit_intercept=False,
                      normalize=False)
    skridge.fit(X_train, y_train)

    curidge = cuRidge(fit_intercept=False,
                      normalize=False,
                      solver=algorithm)

    if X_type == 'dataframe':
        y_train = pd.DataFrame({'fea0':y_train[0:,]})
        X_train = pd.DataFrame({'fea%d'%i:X_train[0:,i] for i in range(X_train.shape[1])})
        X_test = pd.DataFrame({'fea%d'%i:X_test[0:,i] for i in range(X_test.shape[1])})
        X_cudf = cudf.DataFrame.from_pandas(X_train) 
        X_cudf_test = cudf.DataFrame.from_pandas(X_test)
        y_cudf = y_train.values
        y_cudf = y_cudf[:,0] 
        y_cudf = cudf.Series(y_cudf)
        curidge.fit(X_cudf,y_cudf)
        cu_predict = curidge.predict(X_cudf_test).to_array()

    elif X_type == 'ndarray':

        curidge.fit(X_train, y_train)
        cu_predict = curidge.predict(X_test).to_array()

    sk_predict = skridge.predict(X_test)

    assert array_equal(sk_predict, cu_predict, 1e-1, with_sign=True)
