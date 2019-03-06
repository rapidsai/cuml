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
from cuml.manifold.umap import UMAP
import cudf
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse.csgraph import connected_components
from sklearn.manifold.t_sne import trustworthiness

from nose.tools import assert_greater_equal

from sklearn.cluster import KMeans


from sklearn.utils.testing import (
    assert_equal,
    assert_array_equal,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_raises,
    assert_in,
    assert_not_in,
    assert_no_warnings,
    if_matplotlib,
)

from sklearn.metrics import pairwise_distances, adjusted_rand_score


def test_blobs_cluster():
    data, labels = datasets.make_blobs(n_samples=500, n_features=10, centers=5)
    embedding = UMAP().fit_transform(data)
    assert_equal(adjusted_rand_score(labels, KMeans(5).fit_predict(embedding)), 1.0)


def test_umap_transform_on_iris():
    iris = datasets.load_iris()
    iris_selection = np.random.choice([True, False], 150, replace=True, p=[0.75, 0.25])
    data = iris.data[iris_selection]
    fitter = UMAP(n_neighbors=10, min_dist=0.01)
    fitter.fit(data)

    new_data = iris.data[~iris_selection]
    embedding = fitter.transform(new_data)

    trust = trustworthiness(new_data, embedding, 10)
    assert_greater_equal(
        trust,
        0.89,
        "Insufficiently trustworthy transform for" "iris dataset: {}".format(trust),
    )


def test_umap_trustworthiness_on_iris_random_init():
    iris = datasets.load_iris()
    data = iris.data
    embedding = UMAP(
        n_neighbors=10, min_dist=0.01,  init="random"
    ).fit_transform(data)
    trust = trustworthiness(iris.data, embedding, 10)
    assert_greater_equal(
        trust,
        0.95,
        "Insufficiently trustworthy embedding for" "iris dataset: {}".format(trust),
    )


# @pytest.mark.parametrize('should_downcast', [True, False])
# @pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
# def test_umap_fit(input_type, should_downcast):
#
#     dtype = np.float32 if not should_downcast else np.float64
#
#     # For now, FAISS based nearest_neighbors only supports single precision
#     digits = datasets.load_digits(n_class=9)
#     X = digits["data"].astype(dtype)
#     y = digits["target"]
#
#     print(str(y))
#
#     umap = UMAP(n_neighbors = 3, n_components = 2, should_downcast = should_downcast)
#
#     if input_type == 'dataframe':
#         X = cudf.DataFrame.from_pandas(pd.DataFrame(X))
#         embeds = umap.fit_transform(X)
#
#         assert type(embeds) == cudf.DataFrame
#
#         embeds_arr = np.asarray(embeds.as_gpu_matrix(order="C"))
#     else:
#         embeds = umap.fit_transform(X)
#
#         assert type(embeds) == np.ndarray
#
#         embeds_arr = embeds
#
#     # print(str(embeds_arr))
#
#     # UMAP is stochastic so we cannot compare results directly.
#     # Instead, measure the centroid distance within each label.
#     # Since we are using a low n_neighbors value, we should be
#     # optimizing for local distance, therefore differing labels
#     # *should* be pushed away from each other; hopefully enough
#     # to get a consistent boundary for our tests.
#
#     overall = euclidean_distances(embeds)
#
#     cluster1_dists = euclidean_distances(embeds[y==0])
#     cluster2_dists = euclidean_distances(embeds[y==1])
#
#
#
#     print(str(overall))
#     print(str(connected_components(overall)))
#
#
#     assert 1==0
#
#
# # @pytest.mark.parametrize('input_type', ['dataframe', 'ndarray'])
# # def test_umap_downcast_fails(input_type):
# #
# #     X = np.array([[1.0], [50.0], [51.0]], dtype=np.float64)
# #
# #     # Test fit() fails with double precision when should_downcast set to False
# #     knn_cu = cuKNN()
# #     if input_type == 'dataframe':
# #         X = cudf.DataFrame.from_pandas(pd.DataFrame(X))
# #
# #     with pytest.raises(Exception):
# #         knn_cu.fit(X, should_downcast=False)
# #
# #     # Test fit() fails when downcast corrupted data
# #     X = np.array([[np.finfo(np.float32).max]], dtype=np.float64)
# #
# #     knn_cu = cuKNN()
# #     if input_type == 'dataframe':
# #         X = cudf.DataFrame.from_pandas(pd.DataFrame(X))
# #
# #     with pytest.raises(Exception):
# #         knn_cu.fit(X, should_downcast=True)
