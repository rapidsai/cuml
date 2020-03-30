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

import numpy as np
import pytest

from cuml import PCA as cuPCA
from cuml.test.utils import get_handle, array_equal, unit_param, \
    quality_param, stress_param

from sklearn import datasets
from sklearn.datasets import make_multilabel_classification
from sklearn.decomposition import PCA as skPCA
from sklearn.datasets.samples_generator import make_blobs


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['ndarray'])
@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('name', [unit_param(None), quality_param('digits'),
                         stress_param('blobs')])
def test_pca_fit(datatype, input_type, name, use_handle):

    if name == 'blobs':
        pytest.skip('fails when using blobs dataset')
        X, y = make_blobs(n_samples=500000,
                          n_features=1000, random_state=0)

    elif name == 'digits':
        X, _ = datasets.load_digits(return_X_y=True)

    else:
        X, Y = make_multilabel_classification(n_samples=500,
                                              n_classes=2,
                                              n_labels=1,
                                              allow_unlabeled=False,
                                              random_state=1)

    skpca = skPCA(n_components=2)
    skpca.fit(X)

    handle, stream = get_handle(use_handle)
    cupca = cuPCA(n_components=2, handle=handle)
    cupca.fit(X)
    cupca.handle.sync()

    for attr in ['singular_values_', 'components_', 'explained_variance_',
                 'explained_variance_ratio_']:
        with_sign = False if attr in ['components_'] else True
        print(attr)
        print(getattr(cupca, attr))
        print(getattr(skpca, attr))
        cuml_res = (getattr(cupca, attr))

        skl_res = getattr(skpca, attr)
        assert array_equal(cuml_res, skl_res, 1e-3, with_sign=with_sign)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['ndarray'])
@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('name', [unit_param(None), quality_param('iris'),
                         stress_param('blobs')])
def test_pca_fit_then_transform(datatype, input_type,
                                name, use_handle):

    if name == 'blobs':
        X, y = make_blobs(n_samples=500000,
                          n_features=1000, random_state=0)

    elif name == 'iris':
        iris = datasets.load_iris()
        X = iris.data

    else:
        X, Y = make_multilabel_classification(n_samples=500,
                                              n_classes=2,
                                              n_labels=1,
                                              allow_unlabeled=False,
                                              random_state=1)

    if name != 'blobs':
        skpca = skPCA(n_components=2)
        Xskpca = skpca.fit_transform(X)

    handle, stream = get_handle(use_handle)
    cupca = cuPCA(n_components=2, handle=handle)

    X_cupca = cupca.fit_transform(X)
    cupca.handle.sync()

    if name != 'blobs':
        assert array_equal(X_cupca, Xskpca, 1e-3, with_sign=True)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['ndarray'])
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
        X, Y = make_multilabel_classification(n_samples=500,
                                              n_classes=2,
                                              n_labels=1,
                                              allow_unlabeled=False,
                                              random_state=1)

    if name != 'blobs':
        skpca = skPCA(n_components=2)
        Xskpca = skpca.fit_transform(X)

    handle, stream = get_handle(use_handle)
    cupca = cuPCA(n_components=2, handle=handle)

    X_cupca = cupca.fit_transform(X)
    cupca.handle.sync()

    if name != 'blobs':
        assert array_equal(X_cupca, Xskpca, 1e-3, with_sign=True)


@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('input_type', ['ndarray'])
@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('name', [unit_param(None), quality_param('quality'),
                         stress_param('blobs')])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000)])
def test_pca_inverse_transform(datatype, input_type,
                               name, use_handle, nrows):
    if name == 'blobs':
        pytest.skip('fails when using blobs dataset')
        X, y = make_blobs(n_samples=500000,
                          n_features=1000, random_state=0)

    else:
        rng = np.random.RandomState(0)
        n, p = nrows, 3
        X = rng.randn(n, p)  # spherical data
        X[:, 1] *= .00001  # make middle component relatively small
        X += [3, 4, 2]  # make a large mean

    handle, stream = get_handle(use_handle)
    cupca = cuPCA(n_components=2, handle=handle)

    X_cupca = cupca.fit_transform(X)

    input_gdf = cupca.inverse_transform(X_cupca)
    cupca.handle.sync()
    assert array_equal(input_gdf, X,
                       5e-5, with_sign=True)
