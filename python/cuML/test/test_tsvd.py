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
from test_utils import array_equal
import cudf
import numpy as np


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
def test_tsvd_fit(datatype, input_type):
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
                 dtype=datatype)
    sktsvd = skTSVD(n_components=1)
    sktsvd.fit(X)

    cutsvd = cuTSVD(n_components=1)

    if input_type == 'dataframe':
        gdf = cudf.DataFrame()
        gdf['0'] = np.asarray([-1, -2, -3, 1, 2, 3], dtype=datatype)
        gdf['1'] = np.asarray([-1, -1, -2, 1, 1, 2], dtype=datatype)
        cutsvd.fit(gdf)

    else:
        cutsvd.fit(X)

    for attr in ['singular_values_', 'components_',
                 'explained_variance_ratio_']:
        with_sign = False if attr in ['components_'] else True
        assert array_equal(getattr(cutsvd, attr), getattr(sktsvd, attr),
                           0.4, with_sign=with_sign)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
def test_tsvd_fit_transform(datatype, input_type):
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
                 dtype=datatype)
    skpca = skTSVD(n_components=1)
    Xsktsvd = skpca.fit_transform(X)

    cutsvd = cuTSVD(n_components=1)

    if input_type == 'dataframe':
        gdf = cudf.DataFrame()
        gdf['0'] = np.asarray([-1, -2, -3, 1, 2, 3], dtype=datatype)
        gdf['1'] = np.asarray([-1, -1, -2, 1, 1, 2], dtype=datatype)
        Xcutsvd = cutsvd.fit_transform(gdf)

    else:
        Xcutsvd = cutsvd.fit_transform(X)

    assert array_equal(Xcutsvd, Xsktsvd, 1e-3, with_sign=True)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
def test_tsvd_inverse_transform(datatype, input_type):
    gdf = cudf.DataFrame()
    gdf['0'] = np.asarray([-1, -2, -3, 1, 2, 3], dtype=datatype)
    gdf['1'] = np.asarray([-1, -1, -2, 1, 1, 2], dtype=datatype)
    cutsvd = cuTSVD(n_components=1)

    if input_type == 'dataframe':
        Xcutsvd = cutsvd.fit_transform(gdf)

    else:
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
                     dtype=datatype)
        Xcutsvd = cutsvd.fit_transform(X)

    input_gdf = cutsvd.inverse_transform(Xcutsvd)
    assert array_equal(input_gdf, gdf, 0.4, with_sign=True)
