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

import pytest
from dask.distributed import Client

import cupy as cp
import numpy as np
from sklearn.manifold.t_sne import trustworthiness
from cuml.dask.datasets import make_blobs


@pytest.mark.mg
@pytest.mark.parametrize("n_parts", [2, 5])
@pytest.mark.parametrize("sampling_ratio", [0.1, 0.4])
@pytest.mark.parametrize("supervised", [True, False])
def test_umap_mnmg(n_parts, sampling_ratio, supervised, cluster):

    client = Client(cluster)

    try:
        from cuml.manifold import UMAP
        from cuml.dask.manifold import UMAP as MNMG_UMAP

        n_neighbors = 10

        X, y = make_blobs(10000, 10,
                          centers=42,
                          cluster_std=0.1,
                          dtype=cp.float32,
                          n_parts=n_parts,
                          output='array')

        n_samples = X.shape[0]
        n_sampling = int(n_samples * sampling_ratio)

        local_model = UMAP(n_neighbors=n_neighbors)

        selection = np.random.choice(n_samples, n_sampling)
        X_train = X[selection]
        X_train = X_train.compute()

        y_train = None
        if supervised:
            y_train = y[selection]
            y_train = y_train.compute()
        
        local_model.fit(X_train, y=y_train)

        distributed_model = MNMG_UMAP(local_model)
        embedding = distributed_model.transform(X)

        X = cp.asnumpy(X.compute())
        embedding = cp.asnumpy(embedding.compute())

        trust = trustworthiness(X, embedding, n_neighbors)
        assert trust > 0.9

    finally:
        client.close()
