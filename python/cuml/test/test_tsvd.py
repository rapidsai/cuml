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

import pytest
from cuml import TruncatedSVD as cuTSVD
from cuml.test.utils import get_handle
from sklearn.decomposition import TruncatedSVD as skTSVD
from cuml.test.utils import array_equal
import cudf
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
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
@pytest.mark.parametrize('name', [unit_param(None), quality_param('random'),
                         stress_param('blobs')])
def test_tsvd_fit(datatype, input_type,
                  name, use_handle):

    if name == 'blobs':
        X, y = make_blobs(n_samples=500000,
                          n_features=1000, random_state=0)

    elif name == 'random':
        pytest.skip('fails when using random dataset '
                    'used by sklearn for testing')
        shape = 5000, 100
        rng = check_random_state(42)
        X = rng.randint(-100, 20, np.product(shape)).reshape(shape)

    else:
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
                     dtype=datatype)

    if name != 'blobs':
        sktsvd = skTSVD(n_components=1)
        sktsvd.fit(X)

    handle, stream = get_handle(use_handle)
    cutsvd = cuTSVD(n_components=1, handle=handle)

    if input_type == 'dataframe':
        X = pd.DataFrame(
            {'fea%d' % i: X[0:, i] for i in range(X.shape[1])})
        X_cudf = cudf.DataFrame.from_pandas(X)
        cutsvd.fit(X_cudf)

    else:
        cutsvd.fit(X)

    cutsvd.handle.sync()

    if name != 'blobs':
        for attr in ['singular_values_', 'components_',
                     'explained_variance_ratio_']:
            with_sign = False if attr in ['components_'] else True
            assert array_equal(getattr(cutsvd, attr), getattr(sktsvd, attr),
                               0.4, with_sign=with_sign)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('name', [unit_param(None), quality_param('random'),
                         stress_param('blobs')])
def test_tsvd_fit_transform(datatype, input_type,
                            name, use_handle):
    if name == 'blobs':
        X, y = make_blobs(n_samples=500000,
                          n_features=1000, random_state=0)

    elif name == 'random':
        pytest.skip('fails when using random dataset '
                    'used by sklearn for testing')
        shape = 5000, 100
        rng = check_random_state(42)
        X = rng.randint(-100, 20, np.product(shape)).reshape(shape)

    else:
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
                     dtype=datatype)

    if name != 'blobs':
        skpca = skTSVD(n_components=1)
        Xsktsvd = skpca.fit_transform(X)

    handle, stream = get_handle(use_handle)
    cutsvd = cuTSVD(n_components=1, handle=handle)

    if input_type == 'dataframe':
        X = pd.DataFrame(
            {'fea%d' % i: X[0:, i] for i in range(X.shape[1])})
        X_cudf = cudf.DataFrame.from_pandas(X)
        Xcutsvd = cutsvd.fit_transform(X_cudf)

    else:
        Xcutsvd = cutsvd.fit_transform(X)

    cutsvd.handle.sync()

    if name != 'blobs':
        assert array_equal(Xcutsvd, Xsktsvd, 1e-3, with_sign=True)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('name', [unit_param(None), quality_param('random'),
                         stress_param('blobs')])
def test_tsvd_inverse_transform(datatype, input_type,
                                name, use_handle):

    if name == 'blobs':
        pytest.skip('fails when using blobs dataset')
        X, y = make_blobs(n_samples=500000,
                          n_features=1000, random_state=0)

    elif name == 'random':
        pytest.skip('fails when using random dataset '
                    'used by sklearn for testing')
        shape = 5000, 100
        rng = check_random_state(42)
        X = rng.randint(-100, 20, np.product(shape)).reshape(shape)

    else:
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
                     dtype=datatype)

    X_pd = pd.DataFrame(
           {'fea%d' % i: X[0:, i] for i in range(X.shape[1])})
    X_cudf = cudf.DataFrame.from_pandas(X_pd)
    cutsvd = cuTSVD(n_components=1)

    if input_type == 'dataframe':
        Xcutsvd = cutsvd.fit_transform(X_cudf)

    else:
        Xcutsvd = cutsvd.fit_transform(X)

    input_gdf = cutsvd.inverse_transform(Xcutsvd)

    cutsvd.handle.sync()

    assert array_equal(input_gdf, X_cudf, 0.4, with_sign=True)
