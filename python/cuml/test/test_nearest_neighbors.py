
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
from cuml.neighbors import NearestNeighbors as cuKNN
from sklearn.neighbors import NearestNeighbors as skKNN
from sklearn.datasets.samples_generator import make_blobs
import cudf
import pandas as pd
import numpy as np
from cuml.test.utils import array_equal


@pytest.mark.parametrize('should_downcast', [True])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('nrows', [pytest.param(20,
                                                marks=pytest.mark.unit),
                                   pytest.param(500,
                                                marks=pytest.mark.stress),
                                   pytest.param(5000,
                                                marks=pytest.mark.quality)])
@pytest.mark.parametrize('num_feats', [pytest.param(10,
                                                    marks=pytest.mark.unit),
                                       pytest.param(1000,
                                                    marks=pytest.mark.stress),
                                       pytest.param(500,
                                                    marks=pytest.mark.quality
                                                    )])
@pytest.mark.parametrize('k', [pytest.param(3,
                                            marks=pytest.mark.unit),
                               pytest.param(30,
                                            marks=pytest.mark.stress),
                               pytest.param(50,
                                            marks=pytest.mark.quality
                                            )])
def test_knn(input_type, should_downcast, nrows, num_feats, k):
    n_samples = nrows
    n_feats = num_feats
    X, y = make_blobs(n_samples=n_samples,
                      n_features=n_feats, random_state=0)

    knn_sk = skKNN(metric="l2")
    knn_sk.fit(X)
    D_sk, I_sk = knn_sk.kneighbors(X, k)
    knn_cu = cuKNN(should_downcast=should_downcast)

    if input_type == 'dataframe':
        X_pd = pd.DataFrame({'fea%d' % i: X[0:, i] for i in range(X.shape[1])})
        X_cudf = cudf.DataFrame.from_pandas(X_pd)
        knn_cu.fit(X_cudf)
        D_cuml, I_cuml = knn_cu.kneighbors(X_cudf, k)

        assert type(D_cuml) == cudf.DataFrame
        assert type(I_cuml) == cudf.DataFrame

        # FAISS does not perform sqrt on L2 because it's expensive

        D_cuml_arr = np.asarray(D_cuml.as_gpu_matrix(order="C"))
        I_cuml_arr = np.asarray(I_cuml.as_gpu_matrix(order="C"))

    elif input_type == 'ndarray':

        knn_cu.fit(X)
        D_cuml, I_cuml = knn_cu.kneighbors(X, k)
        assert type(D_cuml) == np.ndarray
        assert type(I_cuml) == np.ndarray

        D_cuml_arr = D_cuml
        I_cuml_arr = I_cuml

    assert array_equal(D_cuml_arr, np.square(D_sk), 1e-2, with_sign=True)
    assert I_cuml_arr.all() == I_sk.all()


@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('nrows', [pytest.param(20,
                                                marks=pytest.mark.unit),
                                   pytest.param(500000,
                                                marks=pytest.mark.stress),
                                   pytest.param(5000,
                                                marks=pytest.mark.quality)])
@pytest.mark.parametrize('num_feats', [pytest.param(10,
                                                    marks=pytest.mark.unit),
                                       pytest.param(1000,
                                                    marks=pytest.mark.stress),
                                       pytest.param(500,
                                                    marks=pytest.mark.quality
                                                    )])
def test_nn_downcast_fails(input_type, nrows, num_feats):
    n_feats = num_feats
    X, y = make_blobs(n_samples=nrows,
                      n_features=n_feats, random_state=0)

    knn_cu = cuKNN()
    if input_type == 'dataframe':
        X_pd = pd.DataFrame({'fea%d' % i: X[0:, i] for i in range(X.shape[1])})
        X_cudf = cudf.DataFrame.from_pandas(X_pd)
        knn_cu.fit(X_cudf)

    with pytest.raises(Exception):
        knn_cu.fit(X, should_downcast=False)

    # Test fit() fails when downcast corrupted data
    X = np.array([[np.finfo(np.float32).max]], dtype=np.float64)

    knn_cu = cuKNN()
    if input_type == 'dataframe':
        X = cudf.DataFrame.from_pandas(pd.DataFrame(X))

    with pytest.raises(Exception):
        knn_cu.fit(X, should_downcast=True)
