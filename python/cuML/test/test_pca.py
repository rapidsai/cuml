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
from cuML import PCA as cuPCA
from sklearn.decomposition import PCA as skPCA
from test_utils import array_equal
import cudf
import numpy as np

@pytest.mark.parametrize('datatype', [np.float32, np.float64])

def test_pca_fit(datatype):
    gdf = cudf.DataFrame()
    gdf['0']=np.asarray([-1,-2,-3,1,2,3],dtype=datatype)
    gdf['1']=np.asarray([-1,-1,-2,1,1,2],dtype=datatype)

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype = datatype)

    print("Calling fit")
    cupca = cuPCA(n_components = 2)
    cupca.fit(gdf)
    skpca = skPCA(n_components = 2)
    skpca.fit(X)

    for attr in ['singular_values_','components_','explained_variance_','explained_variance_ratio_','noise_variance_']:
        with_sign = False if attr in ['components_'] else True
        assert array_equal(getattr(cupca,attr),getattr(skpca,attr),
            1e-3,with_sign=with_sign)

@pytest.mark.parametrize('datatype', [np.float32, np.float64])

def test_pca_fit_transform(datatype):
    gdf = cudf.DataFrame()
    gdf['0']=np.asarray([-1,-2,-3,1,2,3],dtype=datatype)
    gdf['1']=np.asarray([-1,-1,-2,1,1,2],dtype=datatype)

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype = datatype)

    print("Calling fit_transform")
    cupca = cuPCA(n_components = 2)
    Xcupca = cupca.fit_transform(gdf)
    skpca = skPCA(n_components = 2)
    Xskpca = skpca.fit_transform(X)

    assert array_equal(Xcupca, Xskpca,
            1e-3,with_sign=False)

@pytest.mark.parametrize('datatype', [np.float32, np.float64])

def test_pca_inverse_transform(datatype):
    gdf = cudf.DataFrame()
    gdf['0']=np.asarray([-1,-2,-3,1,2,3],dtype=datatype)
    gdf['1']=np.asarray([-1,-1,-2,1,1,2],dtype=datatype)

    cupca = cuPCA(n_components = 2)
    Xcupca = cupca.fit_transform(gdf)

    print("Calling inverse_transform")
    input_gdf = cupca.inverse_transform(Xcupca)

    assert array_equal(input_gdf, gdf,
            1e-3,with_sign=True)
