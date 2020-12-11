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

from cuml.manifold import TSNE
from cuml.test.utils import stress_param

from sklearn.datasets import make_blobs
from sklearn.manifold.t_sne import trustworthiness
from sklearn import datasets

import cuml.common.logger as logger


dataset_names = ['digits', 'boston', 'iris', 'breast_cancer',
                 'diabetes']


@pytest.mark.parametrize('name', dataset_names)
def test_tsne(name):
    """
    This tests how TSNE handles a lot of input data across time.
    (1) Numpy arrays are passed in
    (2) Params are changed in the TSNE class
    (3) The class gets re-used across time
    (4) Trustworthiness is checked
    (5) Tests NAN in TSNE output for learning rate explosions
    (6) Tests verbosity
    """
    datasets
    X = eval("datasets.load_{}".format(name))().data

    for i in range(3):
        print("iteration = ", i)

        tsne = TSNE(2, random_state=i, verbose=False,
                    learning_rate=2+i)

        # Reuse
        Y = tsne.fit_transform(X)
        nans = np.sum(np.isnan(Y))
        trust = trustworthiness(X, Y)
        print("Trust = ", trust)
        assert trust > 0.76
        assert nans == 0
        del Y

        # Again
        tsne = TSNE(2, random_state=i+2, verbose=logger.level_debug,
                    learning_rate=2+i+2)

        # Reuse
        Y = tsne.fit_transform(X)
        nans = np.sum(np.isnan(Y))
        trust = trustworthiness(X, Y)
        print("Trust = ", trust)
        assert trust > 0.76
        assert nans == 0
        del Y


@pytest.mark.parametrize('name', dataset_names)
def test_tsne_default(name):

    datasets
    X = eval("datasets.load_{}".format(name))().data

    for i in range(3):
        print("iteration = ", i)

        tsne = TSNE()
        Y = tsne.fit_transform(X)
        nans = np.sum(np.isnan(Y))
        trust = trustworthiness(X, Y)
        print("Trust = ", trust)
        assert trust > 0.76
        assert nans == 0
        del Y


@pytest.mark.parametrize('nrows', [stress_param(2400000)])
@pytest.mark.parametrize('ncols', [stress_param(250)])
def test_tsne_large(nrows, ncols):
    """
    This tests how TSNE handles large input
    """
    X, y = make_blobs(n_samples=nrows, centers=8,
                      n_features=ncols, random_state=0)

    X = X.astype(np.float32)

    tsne = TSNE(random_state=0, exaggeration_iter=1, n_iter=2)
    Y = tsne.fit_transform(X)
    nans = np.sum(np.isnan(Y))
    assert nans == 0


def test_components_exception():
    with pytest.raises(ValueError):
        TSNE(n_components=3)
