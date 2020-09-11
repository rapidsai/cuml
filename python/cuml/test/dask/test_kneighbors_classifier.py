
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

import dask.array as da
from cuml.dask.common.dask_arr_utils import to_dask_cudf
from cudf.core.dataframe import DataFrame
import numpy as np


def generate_dask_array(np_array, n_parts):
    n_samples = np_array.shape[0]
    n_samples_per_part = int(n_samples / n_parts)
    chunks = [n_samples_per_part] * n_parts
    chunks[-1] += n_samples % n_samples_per_part
    chunks = tuple(chunks)
    return da.from_array(np_array, chunks=(chunks, -1))


@pytest.fixture(
    scope="module",
    params=[
        unit_param({'n_samples': 1000, 'n_features': 30,
                    'n_classes': 5, 'n_targets': 1}),
        quality_param({'n_samples': 5000, 'n_features': 100,
                       'n_classes': 12, 'n_targets': 4}),
        stress_param({'n_samples': 12000, 'n_features': 40,
                      'n_classes': 5, 'n_targets': 2})
    ])
def dataset(request):
    X, y = make_multilabel_classification(
        n_samples=int(request.param['n_samples'] * 1.2),
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
    y = np.array(new_y)

    return train_test_split(X, y, test_size=0.33)


def accuracy_score(y_true, y_pred):
    assert y_pred.shape[0] == y_true.shape[0]
    assert y_pred.shape[1] == y_true.shape[1]
    return np.mean(y_pred == y_true)


def match_test(output1, output2):
    l1, i1, d1 = output1
    l2, i2, d2 = output2
    l2 = l2.squeeze()

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
        # the last set of indices is not guaranteed

    # As indices might differ, labels can also differ
    # assert np.mean((l1 == l2)) > 0.6


def check_probabilities(l_probas, d_probas):
    assert len(l_probas) == len(d_probas)
    for i in range(len(l_probas)):
        assert l_probas[i].shape == d_probas[i].shape


@pytest.mark.parametrize("datatype", ['dask_array', 'dask_cudf'])
@pytest.mark.parametrize("n_neighbors", [1, 3, 6])
@pytest.mark.parametrize("n_parts", [None, 2, 3, 5])
@pytest.mark.parametrize("batch_size", [256, 512, 1024])
def test_predict(dataset, datatype, n_neighbors, n_parts, batch_size, client):
    X_train, X_test, y_train, y_test = dataset

    l_model = lKNNClf(n_neighbors=n_neighbors)
    l_model.fit(X_train, y_train)
    l_distances, l_indices = l_model.kneighbors(X_test)
    l_labels = l_model.predict(X_test)
    local_out = (l_labels, l_indices, l_distances)

    if not n_parts:
        n_parts = len(client.has_what().keys())

    X_train = generate_dask_array(X_train, n_parts)
    X_test = generate_dask_array(X_test, n_parts)
    y_train = generate_dask_array(y_train, n_parts)

    if datatype == 'dask_cudf':
        X_train = to_dask_cudf(X_train, client)
        X_test = to_dask_cudf(X_test, client)
        y_train = to_dask_cudf(y_train, client)

    d_model = dKNNClf(client=client, n_neighbors=n_neighbors,
                      batch_size=batch_size)
    d_model.fit(X_train, y_train)
    d_labels, d_indices, d_distances = \
        d_model.predict(X_test, convert_dtype=True)
    distributed_out = da.compute(d_labels, d_indices, d_distances)

    if datatype == 'dask_cudf':
        distributed_out = list(map(lambda o: o.as_matrix()
                                   if isinstance(o, DataFrame)
                                   else o.to_array()[..., np.newaxis],
                                   distributed_out))

    match_test(local_out, distributed_out)
    assert accuracy_score(y_test, distributed_out[0]) > 0.12


@pytest.mark.parametrize("datatype", ['dask_array'])
@pytest.mark.parametrize("n_neighbors", [1, 2, 3])
@pytest.mark.parametrize("n_parts", [None, 2, 3, 5])
def test_score(dataset, datatype, n_neighbors, n_parts, client):
    X_train, X_test, y_train, y_test = dataset

    if not n_parts:
        n_parts = len(client.has_what().keys())

    X_train = generate_dask_array(X_train, n_parts)
    X_test = generate_dask_array(X_test, n_parts)
    y_train = generate_dask_array(y_train, n_parts)
    y_test = generate_dask_array(y_test, n_parts)

    if datatype == 'dask_cudf':
        X_train = to_dask_cudf(X_train, client)
        X_test = to_dask_cudf(X_test, client)
        y_train = to_dask_cudf(y_train, client)
        y_test = to_dask_cudf(y_test, client)

    d_model = dKNNClf(client=client, n_neighbors=n_neighbors)
    d_model.fit(X_train, y_train)
    d_labels, d_indices, d_distances = \
        d_model.predict(X_test, convert_dtype=True)
    distributed_out = da.compute(d_labels, d_indices, d_distances)

    if datatype == 'dask_cudf':
        distributed_out = list(map(lambda o: o.as_matrix()
                                   if isinstance(o, DataFrame)
                                   else o.to_array()[..., np.newaxis],
                                   distributed_out))
    cuml_score = d_model.score(X_test, y_test)


    if datatype == 'dask_cudf':
        y_test = y_test.compute().as_matrix()
    else:
        y_test = y_test.compute()
    manual_score = np.mean(y_test == distributed_out[0])

    assert cuml_score == manual_score


@pytest.mark.skip(reason="Need to fix")
@pytest.mark.parametrize("datatype", ['dask_array', 'dask_cudf'])
@pytest.mark.parametrize("n_neighbors", [1, 3, 6])
@pytest.mark.parametrize("n_parts", [None, 2, 3, 5])
def test_predict_proba(dataset, datatype, n_neighbors, n_parts, client):
    X_train, X_test, y_train, y_test = dataset

    l_model = lKNNClf(n_neighbors=n_neighbors)
    l_model.fit(X_train, y_train)
    l_probas = l_model.predict_proba(X_test)

    if not n_parts:
        n_parts = len(client.has_what().keys())

    X_train = generate_dask_array(X_train, n_parts)
    X_test = generate_dask_array(X_test, n_parts)
    y_train = generate_dask_array(y_train, n_parts)

    if datatype == 'dask_cudf':
        X_train = to_dask_cudf(X_train, client)
        X_test = to_dask_cudf(X_test, client)
        y_train = to_dask_cudf(y_train, client)

    d_model = dKNNClf(client=client, n_neighbors=n_neighbors)
    d_model.fit(X_train, y_train)
    d_probas = d_model.predict_proba(X_test, convert_dtype=True)
    d_probas = da.compute(d_probas)[0]

    if datatype == 'dask_cudf':
        d_probas = list(map(lambda o: o.as_matrix()
                            if isinstance(o, DataFrame)
                            else o.to_array()[..., np.newaxis],
                            d_probas))

    check_probabilities(l_probas, d_probas)
