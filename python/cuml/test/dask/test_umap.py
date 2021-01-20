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

import cupy as cp
import numpy as np
from cuml.common import logger
from cuml.metrics import trustworthiness

import math

from sklearn.datasets import load_digits
from sklearn.datasets import load_iris


def _load_dataset(dataset, n_rows):

    if dataset == "digits":
        local_X, local_y = load_digits(return_X_y=True)

    else:  # dataset == "iris"
        local_X, local_y = load_iris(return_X_y=True)

    local_X = cp.asarray(local_X)
    local_y = cp.asarray(local_y)

    local_X = local_X.repeat(
        math.ceil(n_rows / len(local_X)), axis=0)
    local_y = local_y.repeat(
        math.ceil(n_rows / len(local_y)), axis=0)

    # Add some gaussian noise
    local_X += cp.random.standard_normal(local_X.shape,
                                         dtype=cp.float32)

    return local_X, local_y


def _local_umap_trustworthiness(local_X, local_y,
                                n_neighbors,
                                supervised):
    """
    Train model on all data, report trustworthiness
    """
    from cuml.manifold import UMAP

    local_model = UMAP(n_neighbors=n_neighbors,
                       random_state=42)
    y_train = None
    if supervised:
        y_train = local_y
    local_model.fit(local_X, y=y_train)
    embedding = local_model.transform(local_X)
    return trustworthiness(local_X, embedding,
                           n_neighbors=n_neighbors,
                           batch_size=5000)


def _umap_mnmg_trustworthiness(local_X, local_y,
                               n_neighbors,
                               supervised,
                               n_parts,
                               sampling_ratio):
    """
    Train model on random sample of data, transform in
    parallel, report trustworthiness
    """
    import dask.array as da
    from cuml.dask.manifold import UMAP as MNMG_UMAP

    from cuml.manifold import UMAP

    local_model = UMAP(n_neighbors=n_neighbors,
                       random_state=42)

    n_samples = local_X.shape[0]
    n_samples_per_part = math.ceil(n_samples / n_parts)

    selection = np.random.RandomState(42).choice(
        [True, False], n_samples, replace=True,
        p=[sampling_ratio, 1.0 - sampling_ratio])
    X_train = local_X[selection]
    X_transform = local_X
    X_transform_d = da.from_array(X_transform,
                                  chunks=(n_samples_per_part, -1))

    y_train = None
    if supervised:
        y_train = local_y[selection]

    local_model.fit(X_train, y=y_train)

    distributed_model = MNMG_UMAP(local_model)
    embedding = distributed_model.transform(X_transform_d)

    embedding = embedding.compute()
    return trustworthiness(X_transform, embedding,
                           n_neighbors=n_neighbors,
                           batch_size=5000)


@pytest.mark.mg
@pytest.mark.parametrize("n_parts", [2, 9])
@pytest.mark.parametrize("n_rows", [100, 500])
@pytest.mark.parametrize("sampling_ratio", [0.4, 0.9])
@pytest.mark.parametrize("supervised", [True, False])
@pytest.mark.parametrize("dataset", ["digits", "iris"])
@pytest.mark.parametrize("n_neighbors", [10])
def test_umap_mnmg(n_parts, n_rows, sampling_ratio, supervised,
                   dataset, n_neighbors, client):
    local_X, local_y = _load_dataset(dataset, n_rows)

    dist_umap = _umap_mnmg_trustworthiness(local_X, local_y,
                                           n_neighbors, supervised,
                                           n_parts, sampling_ratio)

    loc_umap = _local_umap_trustworthiness(local_X, local_y,
                                           n_neighbors, supervised)

    logger.debug("\nLocal UMAP trustworthiness score : {:.2f}"
                 .format(loc_umap))
    logger.debug("UMAP MNMG trustworthiness score : {:.2f}"
                 .format(dist_umap))

    trust_diff = loc_umap - dist_umap

    assert trust_diff <= 0.1
