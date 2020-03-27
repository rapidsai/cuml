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
import cudf

from cuml.dask.cluster import KMeans
from cuml.dask.naive_bayes.naive_bayes import MultinomialNB
from cuml.test.dask.utils import load_text_corpus

from dask.distributed import Client
from dask.distributed import wait

from cuml.dask.datasets import make_blobs


def test_getattr(cluster):

    client = Client(cluster)

    # Test getattr on local param
    kmeans_model = KMeans(client=client)

    assert kmeans_model.client is not None

    # Test getattr on local_model param with a non-distributed model

    X_cudf, y = make_blobs(5, 5, 2, 2, cluster_std=0.01, verbose=False,
                           random_state=10)

    wait(X_cudf)

    kmeans_model.fit(X_cudf)

    assert kmeans_model.cluster_centers_ is not None
    assert isinstance(kmeans_model.cluster_centers_, cudf.DataFrame)

    # Test getattr on trained distributed model

    X, y = load_text_corpus(client)

    print(str(X.compute()))

    nb_model = MultinomialNB(client=client)
    nb_model.fit(X, y)

    assert nb_model.feature_count_ is not None
    assert isinstance(nb_model.feature_count_, cupy.core.ndarray)
