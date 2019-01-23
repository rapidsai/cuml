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
from cuml import KNN as cuKNN
from sklearn.neighbors import KDTree as skKNN
import cudf
import pandas as pd
import numpy as np


@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
def test_knn_search(input_type):

    X = np.array([[1.0], [50.0], [51.0]], dtype=np.float32) # For now, FAISS only seems to support single precision

    knn_sk = skKNN(X, metric = "l2")
    D_sk, I_sk = knn_sk.query(X, len(X))

    knn_cu = cuKNN()
    if input_type == 'dataframe':
        X = cudf.DataFrame.from_pandas(pd.DataFrame(X))
        knn_cu.fit(X)
        D_cuml, I_cuml = knn_cu.query(X, len(X))

        assert type(D_cuml) == cudf.DataFrame
        assert type(I_cuml) == cudf.DataFrame

        # FAISS does not perform sqrt on L2 because it's expensive

        D_cuml_arr = np.asarray(D_cuml.as_gpu_matrix(order="C"))
        I_cuml_arr = np.asarray(I_cuml.as_gpu_matrix(order="C"))

    else:
        knn_cu.fit(X)
        D_cuml, I_cuml = knn_cu.query(X, len(X))

        assert type(D_cuml) == np.ndarray
        assert type(I_cuml) == np.ndarray

        D_cuml_arr = D_cuml
        I_cuml_arr = I_cuml


    assert np.array_equal(D_cuml_arr, np.square(D_sk))
    assert np.array_equal(I_cuml_arr, I_sk)

@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
def test_knn_downcast_fails(input_type):

    X = np.array([[1.0], [50.0], [51.0]], dtype=np.float64)

    # Test fit() fails with double precision when should_downcast set to False
    knn_cu = cuKNN()
    if input_type == 'dataframe':
        X = cudf.DataFrame.from_pandas(pd.DataFrame(X))

    with pytest.raises(Exception):
        knn_cu.fit(X, should_downcast = False)

    # Test fit() fails when downcast corrupted data
    X = np.array([[np.finfo(np.float32).max]], dtype=np.float64)

    knn_cu = cuKNN()
    if input_type == 'dataframe':
        X = cudf.DataFrame.from_pandas(pd.DataFrame(X))

    with pytest.raises(Exception):
        knn_cu.fit(X, should_downcast = True)


