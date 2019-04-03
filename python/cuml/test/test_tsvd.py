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

import pytest
from cuml import TruncatedSVD as cuTSVD
from sklearn.decomposition import TruncatedSVD as skTSVD
from cuml.test.utils import array_equal
import cudf
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.datasets.samples_generator import make_blobs


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
def test_tsvd_fit(datatype, input_type, run_stress, run_correctness_test):

    n_samples = 10000
    n_feats = 50
    if run_stress==True:
        X,y = make_blobs(n_samples=n_samples*50,n_features=n_feats,random_state=0) 

    elif run_correctness_test==True:
        shape = n_samples, n_feats
        rng = check_random_state(42)
        X = rng.randint(-100, 20, np.product(shape)).reshape(shape) 

    else:
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
                 dtype=datatype)

    sktsvd = skTSVD(n_components=1)
    sktsvd.fit(X)

    cutsvd = cuTSVD(n_components=1)

    if input_type == 'dataframe':
        X = pd.DataFrame({'fea%d'%i:X[0:,i] for i in range(X.shape[1])})
        X_cudf = cudf.DataFrame.from_pandas(X) 
        cutsvd.fit(X_cudf)

    else:
        cutsvd.fit(X)

    for attr in ['singular_values_', 'components_',
                 'explained_variance_ratio_']:
        with_sign = False if attr in ['components_'] else True
        assert array_equal(getattr(cutsvd, attr), getattr(sktsvd, attr),
                           0.4, with_sign=with_sign)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
def test_tsvd_fit_transform(datatype, input_type, run_stress, run_correctness_test):
    n_samples = 10000
    n_feats = 50
    if run_stress==True:
        X,y = make_blobs(n_samples=n_samples*50,n_features=n_feats,random_state=0) 

    elif run_correctness_test==True:
        shape = n_samples, n_feats
        rng = check_random_state(42)
        X = rng.randint(-100, 20, np.product(shape)).reshape(shape) 

    else:
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
                 dtype=datatype)

    skpca = skTSVD(n_components=1)
    Xsktsvd = skpca.fit_transform(X)

    cutsvd = cuTSVD(n_components=1)

    if input_type == 'dataframe':
        X = pd.DataFrame({'fea%d'%i:X[0:,i] for i in range(X.shape[1])})
        X_cudf = cudf.DataFrame.from_pandas(X) 
        Xcutsvd = cutsvd.fit_transform(X_cudf)

    else:
        Xcutsvd = cutsvd.fit_transform(X)

    assert array_equal(Xcutsvd, Xsktsvd, 1e-3, with_sign=True)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
def test_tsvd_inverse_transform(datatype, input_type, run_stress, run_correctness_test):

    n_samples = 10000
    n_feats = 50
    if run_stress==True:
        X,y = make_blobs(n_samples=n_samples*50,n_features=n_feats,random_state=0) 

    elif run_correctness_test==True:
        shape = n_samples, n_feats
        rng = check_random_state(42)
        X = rng.randint(-100, 20, np.product(shape)).reshape(shape) 

    else:
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
                 dtype=datatype)

    X_pd = pd.DataFrame({'fea%d'%i:X[0:,i] for i in range(X.shape[1])})
    X_cudf = cudf.DataFrame.from_pandas(X_pd) 
    cutsvd = cuTSVD(n_components=1)

    if input_type == 'dataframe':
        Xcutsvd = cutsvd.fit_transform(X_cudf)

    else:
        Xcutsvd = cutsvd.fit_transform(X)

    input_gdf = cutsvd.inverse_transform(Xcutsvd)
    assert array_equal(input_gdf, X_cudf, 0.4, with_sign=True)
