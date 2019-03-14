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
from cuml import PCA as cuPCA
from cuml.test.utils import get_handle
from sklearn.decomposition import PCA as skPCA
from cuml.test.utils import array_equal
import cudf
import numpy as np


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('use_handle', [True, False])
def test_pca_fit(datatype, input_type, use_handle):

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
                 dtype=datatype)
    skpca = skPCA(n_components=2)
    skpca.fit(X)

    cupca = cuPCA(n_components=2, handle=get_handle(use_handle))

    if input_type == 'dataframe':
        gdf = cudf.DataFrame()
        gdf['0'] = np.asarray([-1, -2, -3, 1, 2, 3], dtype=datatype)
        gdf['1'] = np.asarray([-1, -1, -2, 1, 1, 2], dtype=datatype)
        cupca.fit(gdf)

    else:
        cupca.fit(X)

    for attr in ['singular_values_', 'components_', 'explained_variance_',
                 'explained_variance_ratio_', 'noise_variance_']:
        with_sign = False if attr in ['components_'] else True
        print(attr)
        print(getattr(cupca, attr))
        print(getattr(skpca, attr))
        cuml_res = (getattr(cupca, attr))
        if isinstance(cuml_res, cudf.Series):
            cuml_res = cuml_res.to_array()
        else:
            cuml_res = cuml_res.as_matrix()
        skl_res = getattr(skpca, attr)
        assert array_equal(cuml_res, skl_res, 1e-3, with_sign=with_sign)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('use_handle', [True, False])
def test_pca_fit_transform(datatype, input_type, use_handle):
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
                 dtype=datatype)
    skpca = skPCA(n_components=2)
    Xskpca = skpca.fit_transform(X)

    cupca = cuPCA(n_components=2, handle=get_handle(use_handle))

    if input_type == 'dataframe':
        gdf = cudf.DataFrame()
        gdf['0'] = np.asarray([-1, -2, -3, 1, 2, 3], dtype=datatype)
        gdf['1'] = np.asarray([-1, -1, -2, 1, 1, 2], dtype=datatype)
        Xcupca = cupca.fit_transform(gdf)

    else:
        Xcupca = cupca.fit_transform(X)

    assert array_equal(Xcupca, Xskpca, 1e-3, with_sign=True)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('use_handle', [True, False])
def test_pca_inverse_transform(datatype, input_type, use_handle):
    gdf = cudf.DataFrame()
    gdf['0'] = np.asarray([-1, -2, -3, 1, 2, 3], dtype=datatype)
    gdf['1'] = np.asarray([-1, -1, -2, 1, 1, 2], dtype=datatype)
    cupca = cuPCA(n_components=2, handle=get_handle(use_handle))

    if input_type == 'dataframe':
        Xcupca = cupca.fit_transform(gdf)

    else:
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
                     dtype=datatype)
        Xcupca = cupca.fit_transform(X)

    input_gdf = cupca.inverse_transform(Xcupca)

    assert array_equal(input_gdf, gdf,
                       1e-3, with_sign=True)
