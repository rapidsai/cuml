
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
from cuml.test.utils import unit_param, \
                            quality_param, \
                            stress_param

from cuml.neighbors import KNeighborsClassifier as lKNNClf
from cuml.dask.neighbors import KNeighborsClassifier as dKNNClf

from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

from dask.distributed import Client
import dask.array as da
import numpy as np
from cupy import asnumpy


def generate_dask_array(np_array, n_parts):
    n_samples = np_array.shape[0]
    n_samples_per_part = int(n_samples / n_parts)
    chunks = [n_samples_per_part * n_parts]
    chunks[-1] += n_samples % n_samples_per_part
    chunks = tuple(chunks)
    return da.from_array(np_array, chunks=(chunks, -1))


@pytest.fixture(
    scope="module",
    params=[
        unit_param({'n_samples': 1000, 'n_features': 30,
                    'n_classes': 5, 'n_targets': 2}),
        quality_param({'n_samples': 5000, 'n_features': 100,
                       'n_classes': 12, 'n_targets': 4}),
        stress_param({'n_samples': 50000, 'n_features': 400,
                      'n_classes': 20, 'n_targets': 8})
    ])
def dataset(request):
    X, y = make_multilabel_classification(
        n_samples=int(request.param['n_samples'] * 1.5),
        n_features=request.param['n_features'],
        n_classes=request.param['n_classes'],
        n_labels=request.param['n_classes'],
        length=request.param['n_targets'])
    new_x = []
    new_y = []
    for i in range(y.shape[0]):
        a = np.argwhere(y[i] == 1)[:, 0]
        if len(a) >= request.param['n_targets']:
            new_x.append(i)
            np.random.shuffle(a)
            a = a[:request.param['n_targets']]
            new_y.append(a)
        if len(new_x) >= request.param['n_samples']:
            break
    X = X[new_x]
    y = np.squeeze(np.array(new_y))

    return train_test_split(X, y, test_size=0.33)


def accuracy_score(y_true, y_pred):
    assert y_pred.shape[0] == y_true.shape[0]
    assert y_pred.shape[1] == y_true.shape[1]
    return np.mean(y_pred == y_true)


def match_test(output1, output2):
    l1, i1, d1 = output1
    l2, i2, d2 = output2

    # Check shapes
    assert l1.shape == l2.shape
    assert i1.shape == i2.shape
    assert d1.shape == d2.shape

    # Distances should strictly match
    assert np.array_equal(d1, d2)

    # Indices might differ for equivalent distances
    for i in range(d1.shape[0]):
        idx_set1, idx_set2 = (set(), set())
        dist = 0.
        for j in range(d1.shape[1]):
            if d1[i, j] > dist:
                assert idx_set1 == idx_set2
                idx_set1, idx_set2 = (set(), set())
                dist = d1[i, j]
            idx_set1.add(i1[i, j])
            idx_set2.add(i2[i, j])
        assert idx_set1 == idx_set2

    # As indices might differ, labels can also differ
    assert np.mean(l1 == l2) > 0.9


@pytest.mark.parametrize("n_parts", [None, 1, 3, 16])
@pytest.mark.parametrize("n_neighbors", [1, 3, 8])
def test_knn_classify(dataset, cluster, n_parts, n_neighbors):
    client = Client(cluster)

    try:
        X_train, X_test, y_train, y_test = dataset

        l_model = lKNNClf(n_neighbors=n_neighbors)
        l_model.fit(X_train, y_train)
        l_distances, l_indices = l_model.kneighbors(X_test)
        l_labels = l_model.predict(X_test)

        if not n_parts:
            n_parts = len(client.has_what().keys())

        X_train = generate_dask_array(X_train, n_parts)
        X_test = generate_dask_array(X_test, n_parts)
        y_train = generate_dask_array(y_train, n_parts)

        d_model = dKNNClf(client=client, n_neighbors=n_neighbors)
        d_model.fit(X_train, y_train)
        d_labels, d_indices, d_distances = \
            d_model.predict(X_test, convert_dtype=True)
        d_labels = asnumpy(d_labels.compute())
        d_indices = asnumpy(d_indices.compute())
        d_distances = asnumpy(d_distances.compute())

        local_out = (l_labels, l_indices, l_distances)
        distributed_out = (d_labels, d_indices, d_distances)
        match_test(local_out, distributed_out)
        assert accuracy_score(y_test, d_labels) > 0.15

    finally:
        client.close()
