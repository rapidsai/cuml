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


# Please install UMAP before running the code
# use 'conda install -c conda-forge umap-learn' command to install it

import numpy as np
import pytest
import umap

from cuml.manifold.umap import UMAP as cuUMAP
from cuml.test.utils import array_equal, unit_param, \
    quality_param, stress_param

import joblib

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.manifold.t_sne import trustworthiness
from sklearn.metrics import adjusted_rand_score

dataset_names = ['iris', 'digits', 'wine', 'blobs']


@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_feats', [unit_param(20), quality_param(100),
                         stress_param(1000)])
def test_blobs_cluster(nrows, n_feats):
    data, labels = datasets.make_blobs(
        n_samples=nrows, n_features=n_feats, centers=5, random_state=0)
    embedding = cuUMAP(verbose=False).fit_transform(data, convert_dtype=True)

    if nrows < 500000:
        score = adjusted_rand_score(labels,
                                    KMeans(5).fit_predict(embedding))
        assert score == 1.0


@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_feats', [unit_param(10), quality_param(100),
                         stress_param(1000)])
def test_umap_fit_transform_score(nrows, n_feats):

    n_samples = nrows
    n_features = n_feats

    data, labels = make_blobs(n_samples=n_samples, n_features=n_features,
                              centers=10, random_state=42)

    model = umap.UMAP(n_neighbors=10, min_dist=0.1)
    cuml_model = cuUMAP(n_neighbors=10, min_dist=0.01, verbose=False)

    embedding = model.fit_transform(data)
    cuml_embedding = cuml_model.fit_transform(data, convert_dtype=True)

    if nrows < 500000:
        cuml_score = adjusted_rand_score(labels,
                                         KMeans(10).fit_predict(
                                             cuml_embedding))
        score = adjusted_rand_score(labels,
                                    KMeans(10).fit_predict(embedding))

        assert array_equal(score, cuml_score, 1e-2, with_sign=True)


# Allow slight deviation from expected trust due to numerical error
TRUST_TOLERANCE_THRESH = 0.005


def test_supervised_umap_trustworthiness_on_iris():
    iris = datasets.load_iris()
    data = iris.data
    embedding = cuUMAP(n_neighbors=10, min_dist=0.01,
                       verbose=False).fit_transform(data, iris.target,
                                                    convert_dtype=True)
    trust = trustworthiness(iris.data, embedding, 10)
    assert trust >= 0.97 - TRUST_TOLERANCE_THRESH


def test_semisupervised_umap_trustworthiness_on_iris():
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target.copy()
    target[25:75] = -1
    embedding = cuUMAP(n_neighbors=10, min_dist=0.01,
                       verbose=False).fit_transform(data, target,
                                                    convert_dtype=True)

    trust = trustworthiness(iris.data, embedding, 10)
    assert trust >= 0.97 - TRUST_TOLERANCE_THRESH


def test_umap_trustworthiness_on_iris():
    iris = datasets.load_iris()
    data = iris.data
    embedding = cuUMAP(n_neighbors=10, min_dist=0.01,
                       verbose=False).fit_transform(data, convert_dtype=True)
    trust = trustworthiness(iris.data, embedding, 10)

    # We are doing a spectral embedding but not a
    # multi-component layout (which is marked experimental).
    # As a result, our score drops by 0.006.
    assert trust >= 0.964 - TRUST_TOLERANCE_THRESH


def test_umap_transform_on_iris():

    iris = datasets.load_iris()
    iris_selection = np.random.RandomState(42).choice(
        [True, False], 150, replace=True, p=[0.75, 0.25])
    data = iris.data[iris_selection]

    fitter = cuUMAP(n_neighbors=10, min_dist=0.01, verbose=False)
    fitter.fit(data, convert_dtype=True)
    new_data = iris.data[~iris_selection]
    embedding = fitter.transform(new_data, convert_dtype=True)

    trust = trustworthiness(new_data, embedding, 10)

    assert trust >= 0.89


@pytest.mark.parametrize('name', dataset_names)
def test_umap_fit_transform_trust(name):

    if name == 'iris':
        iris = datasets.load_iris()
        data = iris.data
        labels = iris.target

    elif name == 'digits':
        digits = datasets.load_digits(n_class=5)
        data = digits.data
        labels = digits.target

    elif name == 'wine':
        wine = datasets.load_wine()
        data = wine.data
        labels = wine.target
    else:
        data, labels = make_blobs(n_samples=5000, n_features=10,
                                  centers=10, random_state=42)

    model = umap.UMAP(n_neighbors=10, min_dist=0.01)
    cuml_model = cuUMAP(n_neighbors=10, min_dist=0.01, verbose=False)
    embedding = model.fit_transform(data)
    cuml_embedding = cuml_model.fit_transform(data, convert_dtype=True)

    trust = trustworthiness(data, embedding, 10)
    cuml_trust = trustworthiness(data, cuml_embedding, 10)

    assert array_equal(trust, cuml_trust, 1e-1, with_sign=True)


@pytest.mark.parametrize('name', [unit_param('digits')])
@pytest.mark.parametrize('nrows', [quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_feats', [quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('should_downcast', [True])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
def test_umap_data_formats(input_type, should_downcast,
                           nrows, n_feats, name):

    dtype = np.float32 if not should_downcast else np.float64
    n_samples = nrows
    n_feats = n_feats

    if name == 'digits':
        # use the digits dataset for unit test
        digits = datasets.load_digits(n_class=9)
        X = digits["data"].astype(dtype)

    else:
        X, y = datasets.make_blobs(n_samples=n_samples,
                                   n_features=n_feats, random_state=0)

    umap = cuUMAP(n_neighbors=3, n_components=2, verbose=False)

    embeds = umap.fit_transform(X)
    assert type(embeds) == np.ndarray


def test_umap_fit_transform_score_default():

    n_samples = 500
    n_features = 20

    data, labels = make_blobs(n_samples=n_samples, n_features=n_features,
                              centers=10, random_state=42)

    model = umap.UMAP()
    cuml_model = cuUMAP(verbose=False)

    embedding = model.fit_transform(data)
    cuml_embedding = cuml_model.fit_transform(data, convert_dtype=True)

    cuml_score = adjusted_rand_score(labels,
                                     KMeans(10).fit_predict(
                                        cuml_embedding))
    score = adjusted_rand_score(labels,
                                KMeans(10).fit_predict(embedding))

    assert array_equal(score, cuml_score, 1e-2, with_sign=True)


def test_umap_fit_transform_against_fit_and_transform():

    n_samples = 500
    n_features = 20

    data, labels = make_blobs(n_samples=n_samples, n_features=n_features,
                              centers=10, random_state=42)

    """
    First test the default option does not hash the input
    """

    cuml_model = cuUMAP(verbose=False)

    ft_embedding = cuml_model.fit_transform(data, convert_dtype=True)
    fit_embedding_same_input = cuml_model.transform(data, convert_dtype=True)

    assert joblib.hash(ft_embedding) != joblib.hash(fit_embedding_same_input)

    """
    Next, test explicitly enabling feature hashes the input
    """

    cuml_model = cuUMAP(hash_input=True, verbose=False)

    ft_embedding = cuml_model.fit_transform(data, convert_dtype=True)
    fit_embedding_same_input = cuml_model.transform(data, convert_dtype=True)

    assert joblib.hash(ft_embedding) == joblib.hash(fit_embedding_same_input)

    fit_embedding_diff_input = cuml_model.transform(data[1:],
                                                    convert_dtype=True)
    assert joblib.hash(ft_embedding) != joblib.hash(fit_embedding_diff_input)
