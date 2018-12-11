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
from cuml import KNN as cumlKNN
from sklearn.neighbors import KDTree as skKNN
import pandas as pd
import cudf
import numpy as np


from sklearn.metrics import mean_squared_error
def array_equal(a,b,threshold=1e-2,with_sign=True,metric='mse'):
    a = to_nparray(a)
    b = to_nparray(b)
    if with_sign == False:
        a,b = np.abs(a),np.abs(b)
    if metric=='mse':
        error = mean_squared_error(a,b)
    else:
        error = np.sum(a!=b)/(a.shape[0]*a.shape[1])
    res = error<threshold
    return res

def to_nparray(x):
    if isinstance(x,np.ndarray) or isinstance(x,pd.DataFrame):
        return np.array(x)
    elif isinstance(x,np.float64):
        return np.array([x])
    elif isinstance(x,cudf.DataFrame) or isinstance(x,cudf.Series):
        return x.to_pandas().values
    return x     

@pytest.mark.parametrize('datatype', [np.float32, np.float32])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
def test_knn_predict(datatype, input_type):

    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]],
                 dtype=datatype)
    skknn = skKNN(X)
    sk_d, sk_i = skknn.query(X, 2)


    if input_type == 'dataframe':
        gdf = cudf.DataFrame()
        gdf['0'] = np.asarray([1, 2, 2, 8, 8, 25], dtype=datatype)
        gdf['1'] = np.asarray([2, 2, 3, 7, 8, 80], dtype=datatype)
        cumlknn = cumlKNN()
        cumlknn.fit(gdf)

        cuml_d, cuml_i = cumlknn.query(gdf, 2)
    else:
        cumlknn = cumlKNN()
        cumlknn.fit(X)
        cuml_d, cuml_i = cumlknn.query(X, 2)

    assert array_equal(cuml_i, sk_i)


