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
from sklearn.datasets import make_blobs


@pytest.mark.mg
@pytest.mark.parametrize("n_parts", [2, 5])
@pytest.mark.parametrize("sampling_ratio", [0.1, 0.2, 0.4])
@pytest.mark.parametrize("supervised", [True, False])
@pytest.mark.parametrize("dataset", ["make_blobs", "digits", "iris"])
def test_umap_mnmg(n_parts, sampling_ratio, supervised, dataset, cluster):

    client = Client(cluster)

    try:
        import dask.array as da
        from cuml.manifold import UMAP
        from cuml.dask.manifold import UMAP as MNMG_UMAP

        n_neighbors = 10

        if dataset == "make_blobs":
            local_X, local_y = make_blobs(n_samples=10000, n_features=10,
                                          centers=200, cluster_std=0.1)
        else:
            if dataset == "digits":
                from sklearn.datasets import load_digits
                local_X, local_y = load_digits(return_X_y=True)
            else:  # dataset == "iris"
                from sklearn.datasets import load_iris
                local_X, local_y = load_iris(return_X_y=True)

        def umap_mnmg_trustworthiness():
            n_samples = local_X.shape[0]
            n_sampling = int(n_samples * sampling_ratio)
            n_samples_per_part = int(n_samples / n_parts)

            local_model = UMAP(n_neighbors=n_neighbors)

            selection = np.random.choice(n_samples, n_sampling)
            X_train = local_X[selection]
            X_transform = local_X[~selection]
            X_transform_d = da.from_array(X_transform,
                                          chunks=(n_samples_per_part, -1))

            y_train = None
            if supervised:
                y_train = local_y[selection]

            local_model.fit(X_train, y=y_train)

            distributed_model = MNMG_UMAP(local_model)
            embedding = distributed_model.transform(X_transform_d)

            embedding = cp.asnumpy(embedding.compute())
            return trustworthiness(X_transform, embedding, n_neighbors)

        def local_umap_trustworthiness():
            local_model = UMAP(n_neighbors=n_neighbors)
            local_model.fit(local_X, local_y)
            embedding = local_model.transform(local_X)
            return trustworthiness(local_X, embedding, n_neighbors)

        loc_umap = local_umap_trustworthiness()
        dist_umap = umap_mnmg_trustworthiness()

        print("\nLocal UMAP trustworthiness score : {:.2f}".format(loc_umap))
        print("UMAP MNMG trustworthiness score : {:.2f}".format(dist_umap))

        if dataset == "make_blobs":
            assert loc_umap > 0.98
            if sampling_ratio <= 0.1:
                assert dist_umap > 0.74
            else:
                assert dist_umap > 0.9
        elif dataset == "digits":
            assert loc_umap > 0.88
            assert dist_umap > 0.8
        else:  # dataset == "iris"
            assert loc_umap > 0.88
            assert dist_umap > 0.78

    finally:
        client.close()
