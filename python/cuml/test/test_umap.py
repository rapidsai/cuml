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

import pytest
from cuml.test.utils import array_equal

from cuml.manifold.umap import UMAP as UMAP_cuml
import umap
import cudf
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.manifold.t_sne import trustworthiness
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets.samples_generator import make_blobs

dataset_names = ['iris', 'digits', 'wine', 'blobs']


def test_umap_fit_transform_score(run_stress, run_quality):

    if run_stress:
        n_samples = 500000
        n_features = 1000

    elif run_quality:
        n_samples = 5000
        n_features = 100

    else:
        n_samples = 500
        n_features = 10

    data, labels = make_blobs(n_samples=n_samples, n_features=n_features,
                              centers=10, random_state=42)

    model = umap.UMAP(n_neighbors=10, min_dist=0.1)
    cuml_model = UMAP_cuml(n_neighbors=10, min_dist=0.01, verbose=True)


    embedding = model.fit_transform(data)
    cuml_embedding = cuml_model.fit_transform(data)

    cuml_score = adjusted_rand_score(labels,
                                     KMeans(10).fit_predict(cuml_embedding))
    score = adjusted_rand_score(labels,
                                KMeans(10).fit_predict(embedding))

    assert array_equal(score, cuml_score, 1e-2, with_sign=True)

def test_supervised_umap_trustworthiness_on_iris():
    iris = datasets.load_iris()
    data = iris.data
    embedding = UMAP(n_neighbors=10, min_dist=0.01).fit_transform(
        data, iris.target
    )
    trust = trustworthiness(iris.data, embedding, 10)
    assert trust >= 0.97


def test_semisupervised_umap_trustworthiness_on_iris():
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target.copy()
    target[25:75] = -1
    embedding = UMAP(n_neighbors=10, min_dist=0.01).fit_transform(
        data, target
    )
    trust = trustworthiness(iris.data, embedding, 10)
    assert trust >= 0.97


def test_umap_trustworthiness_on_iris():
    iris = datasets.load_iris()
    data = iris.data
    embedding = UMAP(n_neighbors=10, min_dist=0.01).fit_transform(data)
    trust = trustworthiness(iris.data, embedding, 10)


@pytest.mark.parametrize('name', dataset_names)
def test_umap_fit_transform_trust(name, run_stress, run_quality):

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
    cuml_model = UMAP_cuml(n_neighbors=10, min_dist=0.01, verbose=True)
    embedding = model.fit_transform(data)
    cuml_embedding = cuml_model.fit_transform(data)

    trust = trustworthiness(data, embedding, 10)
    cuml_trust = trustworthiness(data, cuml_embedding, 10)

    assert array_equal(trust, cuml_trust, 1e-2, with_sign=True)


@pytest.mark.parametrize('should_downcast', [True, False])
@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
def test_umap_data_formats(input_type, should_downcast,
                           run_stress, run_quality):

    dtype = np.float32 if not should_downcast else np.float64
    n_samples = 50000
    n_feats = 50
    if run_stress:
        X, y = datasets.make_blobs(n_samples=n_samples*10,
                                   n_features=n_feats, random_state=0)

    elif run_quality:
        X, y = datasets.make_blobs(n_samples=int(n_samples/10),
                                   n_features=n_feats, random_state=0)

    else:
        # For now, FAISS based nearest_neighbors only supports single precision
        digits = datasets.load_digits(n_class=9)
        X = digits["data"].astype(dtype)

    umap = UMAP_cuml(n_neighbors=3, n_components=2,
                     should_downcast=should_downcast)

    if input_type == 'dataframe':
        X_pd = pd.DataFrame(
               {'fea%d' % i: X[0:, i] for i in range(X.shape[1])})
        X_cudf = cudf.DataFrame.from_pandas(X_pd)
        embeds = umap.fit_transform(X_cudf)
        assert type(embeds) == cudf.DataFrame

    else:
        embeds = umap.fit_transform(X)
        assert type(embeds) == np.ndarray


@pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
def test_umap_downcast_fails(input_type, run_stress, run_quality):
    n_samples = 50000
    n_feats = 50
    if run_stress:
        X, y = datasets.make_blobs(n_samples=n_samples*10,
                                   n_features=n_feats, random_state=0)

    elif run_quality:
        X, y = datasets.make_blobs(n_samples=int(n_samples/10),
                                   n_features=n_feats, random_state=0)

    else:
        X = np.array([[1.0, 1.0], [50.0, 1.0], [51.0, 1.0]],
                     dtype=np.float64)

    # Test fit() fails with double precision when should_downcast set to False
    umap = UMAP_cuml(should_downcast=False)
    if input_type == 'dataframe':
        X = cudf.DataFrame.from_pandas(pd.DataFrame(X))

    with pytest.raises(Exception):
        umap.fit(X, should_downcast=False)

    # Test fit() fails when downcast corrupted data
    X = np.array([[np.finfo(np.float32).max]], dtype=np.float64)

    umap = UMAP_cuml(should_downcast=True)
    if input_type == 'dataframe':
        X = cudf.DataFrame.from_pandas(pd.DataFrame(X))

    with pytest.raises(Exception):
        umap.fit(X, should_downcast=True)
