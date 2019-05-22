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
from cuml import PCA as cuPCA
from cuml.test.utils import get_handle
from sklearn.decomposition import PCA as skPCA
from cuml.test.utils import array_equal
import cudf
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets.samples_generator import make_blobs


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('name', [unit_param(None), quality_param('iris'),
                         stress_param('blobs')])
def test_pca_fit(datatype, input_type, name, use_handle):

    if name == 'blobs':
        pytest.skip('fails when using blobs dataset')
        X, y = make_blobs(n_samples=500000,
                          n_features=1000, random_state=0)

    elif name == 'iris':
        iris = datasets.load_iris()
        X = iris.data

    else:
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
                     dtype=datatype)

    skpca = skPCA(n_components=2)
    skpca.fit(X)

    handle, stream = get_handle(use_handle)
    cupca = cuPCA(n_components=2, handle=handle)

    if input_type == 'dataframe':
        X = pd.DataFrame({'fea%d' % i: X[0:, i] for i in range(X.shape[1])})
        X_cudf = cudf.DataFrame.from_pandas(X)
        cupca.fit(X_cudf)

    else:
        cupca.fit(X)

    cupca.handle.sync()

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
        assert array_equal(cuml_res, skl_res, 1e-1, with_sign=with_sign)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('name', [unit_param(None), quality_param('iris'),
                         stress_param('blobs')])
def test_pca_fit_transform(datatype, input_type,
                           name, use_handle):
    if name == 'blobs':
        X, y = make_blobs(n_samples=500000,
                          n_features=1000, random_state=0)

    elif name == 'iris':
        iris = datasets.load_iris()
        X = iris.data

    else:
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
                     dtype=datatype)

    if name != 'blobs':
        skpca = skPCA(n_components=2)
        Xskpca = skpca.fit_transform(X)

    handle, stream = get_handle(use_handle)
    cupca = cuPCA(n_components=2, handle=handle)

    if input_type == 'dataframe':
        X = pd.DataFrame(
                        {'fea%d' % i: X[0:, i] for i in range(X.shape[1])})
        X_cudf = cudf.DataFrame.from_pandas(X)
        X_cupca = cupca.fit_transform(X_cudf)

    else:
        X_cupca = cupca.fit_transform(X)
    cupca.handle.sync()

    if name != 'blobs':
        assert array_equal(X_cupca, Xskpca, 1e-3, with_sign=True)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('name', [unit_param(None), quality_param('iris'),
                         stress_param('blobs')])
def test_pca_inverse_transform(datatype, input_type,
                               name, use_handle):
    if name == 'blobs':
        pytest.skip('fails when using blobs dataset')
        X, y = make_blobs(n_samples=500000,
                          n_features=1000, random_state=0)

    elif name == 'iris':
        iris = datasets.load_iris()
        X = iris.data

    else:
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
                     dtype=datatype)

    X_pd = pd.DataFrame(
                       {'fea%d' % i: X[0:, i] for i in range(X.shape[1])})
    X_cudf = cudf.DataFrame.from_pandas(X_pd)

    handle, stream = get_handle(use_handle)
    cupca = cuPCA(n_components=2, handle=handle)

    if input_type == 'dataframe':
        X_cupca = cupca.fit_transform(X_cudf)

    else:
        X_cupca = cupca.fit_transform(X)

    input_gdf = cupca.inverse_transform(X_cupca)
    cupca.handle.sync()

    assert array_equal(input_gdf, X,
                       1e-0, with_sign=True)
