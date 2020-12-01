
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
        unit_param({'n_samples': 3000, 'n_features': 30,
                    'n_classes': 5, 'n_targets': 2}),
        quality_param({'n_samples': 8000, 'n_features': 35,
                       'n_classes': 12, 'n_targets': 3}),
        stress_param({'n_samples': 20000, 'n_features': 40,
                      'n_classes': 12, 'n_targets': 4})
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
    noise = np.random.normal(0, 5., X.shape)
    X += noise
    y = np.array(new_y)

    return train_test_split(X, y, test_size=0.3)


def exact_match(l_outputs, d_outputs):
    # Check shapes
    assert l_outputs.shape == d_outputs.shape

    # Predictions should match
    correct_queries = (l_outputs == d_outputs).all(axis=1)
    assert np.mean(correct_queries) > 0.95


def check_probabilities(l_probas, d_probas):
    assert len(l_probas) == len(d_probas)
    for i in range(len(l_probas)):
        assert l_probas[i].shape == d_probas[i].shape
        assert np.mean(l_probas[i] == d_probas[i]) > 0.95


@pytest.mark.parametrize("datatype", ['dask_array', 'dask_cudf'])
@pytest.mark.parametrize("parameters", [(1, 3, 256),
                                        (8, 8, 256),
                                        (9, 3, 128)])
def test_predict_and_score(dataset, datatype, parameters, client):
    n_neighbors, n_parts, batch_size = parameters
    X_train, X_test, y_train, y_test = dataset

    l_model = lKNNClf(n_neighbors=n_neighbors)
    l_model.fit(X_train, y_train)
    l_outputs = l_model.predict(X_test)
    handmade_local_score = np.mean(y_test == l_outputs)
    handmade_local_score = round(handmade_local_score, 3)

    X_train = generate_dask_array(X_train, n_parts)
    X_test = generate_dask_array(X_test, n_parts)
    y_train = generate_dask_array(y_train, n_parts)
    y_test = generate_dask_array(y_test, n_parts)

    if datatype == 'dask_cudf':
        X_train = to_dask_cudf(X_train, client)
        X_test = to_dask_cudf(X_test, client)
        y_train = to_dask_cudf(y_train, client)
        y_test = to_dask_cudf(y_test, client)

    d_model = dKNNClf(client=client, n_neighbors=n_neighbors,
                      batch_size=batch_size)
    d_model.fit(X_train, y_train)
    d_outputs = d_model.predict(X_test, convert_dtype=True)
    d_outputs = d_outputs.compute()

    d_outputs = d_outputs.as_matrix() \
        if isinstance(d_outputs, DataFrame) \
        else d_outputs

    exact_match(l_outputs, d_outputs)

    distributed_score = d_model.score(X_test, y_test)
    distributed_score = round(distributed_score, 3)
    assert distributed_score == pytest.approx(handmade_local_score, abs=1e-2)


@pytest.mark.parametrize("datatype", ['dask_array', 'dask_cudf'])
@pytest.mark.parametrize("parameters", [(1, 3, 256),
                                        (8, 8, 256),
                                        (9, 3, 128)])
def test_predict_proba(dataset, datatype, parameters, client):
    n_neighbors, n_parts, batch_size = parameters
    X_train, X_test, y_train, y_test = dataset

    l_model = lKNNClf(n_neighbors=n_neighbors)
    l_model.fit(X_train, y_train)
    l_probas = l_model.predict_proba(X_test)

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
