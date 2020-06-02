# Copyright (c) 2020, NVIDIA CORPORATION.
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

import cupy

from cuml.dask.cluster import KMeans
from cuml.dask.naive_bayes.naive_bayes import MultinomialNB
from cuml.test.dask.utils import load_text_corpus

from cuml.dask.datasets import make_blobs


def test_getattr(client):

    # Test getattr on local param
    kmeans_model = KMeans(client=client)

    assert kmeans_model.client is not None

    # Test getattr on local_model param with a non-distributed model

    X, y = make_blobs(n_samples=5,
                      n_features=5,
                      centers=2,
                      n_parts=2,
                      cluster_std=0.01,
                      random_state=10)

    kmeans_model.fit(X)

    assert kmeans_model.cluster_centers_ is not None
    assert isinstance(kmeans_model.cluster_centers_, cupy.core.ndarray)

    # Test getattr on trained distributed model

    X, y = load_text_corpus(client)

    nb_model = MultinomialNB(client=client)
    nb_model.fit(X, y)

    assert nb_model.feature_count_ is not None
    assert isinstance(nb_model.feature_count_, cupy.core.ndarray)
