# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

from distributed.protocol.serialize import serialize
from cuml.naive_bayes.naive_bayes import MultinomialNB
from cuml.internals.array_sparse import SparseCumlArray
from cuml.dask.linear_model import LinearRegression
from cuml.internals.safe_imports import gpu_only_import
from dask import array as da
from sklearn.datasets import make_regression
import numpy as np
import pickle

cp = gpu_only_import("cupy")
cupyx = gpu_only_import("cupyx")


def test_register_naive_bayes_serialization():
    """
    Assuming here that the Dask serializers are well-tested.
    This test-case is only validating that register_serialization
    actually provides the expected serializers on the expected
    objects.
    """

    mnb = MultinomialNB()

    X = cupyx.scipy.sparse.random(1, 5)
    y = cp.array([0])

    mnb.fit(X, y)

    # Unfortunately, Dask has no `unregister` function and Pytest
    # shares the same process so cannot test the base-state here.

    stype, sbytes = serialize(mnb, serializers=["cuda"])

    assert stype["serializer"] == "cuda"

    stype, sbytes = serialize(mnb, serializers=["dask"])

    assert stype["serializer"] == "dask"


def test_sparse_cumlarray_serialization():

    X = cupyx.scipy.sparse.random(10, 5, format="csr", density=0.9)

    X_m = SparseCumlArray(X)

    stype, sbytes = serialize(X_m, serializers=["cuda"])

    assert stype["serializer"] == "cuda"

    stype, sbytes = serialize(X_m, serializers=["dask"])

    assert stype["serializer"] == "dask"


def test_serialize_mnmg_model(client):
    X, y = make_regression(n_samples=1000, n_features=20, random_state=0)
    X, y = da.from_array(X), da.from_array(y)

    model = LinearRegression(client=client)
    model.fit(X, y)

    pickled_model = pickle.dumps(model)
    unpickled_model = pickle.loads(pickled_model)

    assert np.allclose(unpickled_model.coef_, model.coef_)
