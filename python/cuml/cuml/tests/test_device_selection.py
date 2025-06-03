# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

import functools

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification, make_regression

import cuml
from cuml.common.device_selection import (
    get_global_device_type,
    set_global_device_type,
    using_device_type,
)
from cuml.internals.device_type import DeviceType
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.mem_type import MemoryType
from cuml.internals.memory_utils import using_memory_type


@pytest.fixture
def reset_fil_device_type():
    """Ensures fil_device_type is reset after a test changing it"""
    orig = GlobalSettings().fil_device_type
    yield
    GlobalSettings().fil_device_type = orig


def test_get_global_device_type_deprecated():
    with pytest.warns(FutureWarning):
        assert get_global_device_type() is DeviceType.device


def test_set_global_device_type_deprecated(reset_fil_device_type):
    with pytest.warns(FutureWarning):
        set_global_device_type("cpu")

    # device_type remains unchanged
    assert GlobalSettings().device_type is DeviceType.device
    # The deprecated method still changes things for FIL
    assert GlobalSettings().fil_device_type is DeviceType.host


def test_using_device_type_deprecated(reset_fil_device_type):
    with pytest.warns(FutureWarning):
        with using_device_type("cpu"):
            # device_type remains unchanged
            assert GlobalSettings().device_type is DeviceType.device
            # The deprecated method still changes things for FIL
            assert GlobalSettings().fil_device_type is DeviceType.host
    # FIL device type is reset
    assert GlobalSettings().fil_device_type is DeviceType.device

    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError):
            with using_device_type("invalid"):
                assert True


@pytest.mark.parametrize(
    "input",
    [
        ("device", MemoryType.device),
        ("host", MemoryType.host),
        ("managed", MemoryType.managed),
        ("mirror", MemoryType.mirror),
    ],
)
def test_memory_type(input):
    initial_memory_type = cuml.global_settings.memory_type
    with using_memory_type(input[0]):
        assert cuml.global_settings.memory_type == input[1]
    assert cuml.global_settings.memory_type == initial_memory_type


def test_memory_type_exception():
    with pytest.raises(ValueError):
        with using_memory_type("wrong_option"):
            assert True


@functools.cache
def get_dataset(estimator_type):
    if estimator_type == "regressor":
        X, y = make_regression(
            n_samples=2000, n_features=20, n_informative=18, random_state=0
        )
        X_train, X_test = X[:1800], X[1800:]
        y_train = y[:1800]
        return (
            X_train.astype(np.float32),
            y_train.astype(np.float32),
            X_test.astype(np.float32),
        )
    elif estimator_type == "classifier":
        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=18,
            n_classes=2,
            random_state=0,
        )
        X_train, X_test = X[:1800], X[1800:]
        y_train = y[:1800]
        return (
            X_train.astype(np.float32),
            y_train.astype(np.float32),
            X_test.astype(np.float32),
        )
    else:
        X, y = make_blobs(
            n_samples=2000,
            n_features=20,
            centers=20,
            random_state=0,
            cluster_std=1.0,
        )
        X_train, X_test = X[:1800], X[1800:]
        y_train = y[:1800]
        return (
            X_train.astype(np.float32),
            y_train.astype(np.float32),
            X_test.astype(np.float32),
        )


@pytest.mark.parametrize(
    "cls",
    [
        "LogisticRegression",
        "LinearRegression",
        "ElasticNet",
        "Lasso",
        "Ridge",
        "PCA",
        "TruncatedSVD",
        "TSNE",
        "KMeans",
        "SVC",
        "SVR",
        "NearestNeighbors",
        "UMAP",
        "KernelRidge",
        "RandomForestClassifier",
        "RandomForestRegressor",
        "HDBSCAN",
        "DBSCAN",
    ],
)
def test_legacy_device_selection_doesnt_error(cls):
    """Check that running in a `using_device_type("cpu")` block warns
    and doesn't fail for classes that used to support CPU execution in
    this manner."""
    model = getattr(cuml, cls)()
    estimator_type = getattr(model, "_estimator_type", None)
    X, y, X_test = get_dataset(estimator_type)

    if hasattr(model, "fit"):
        with pytest.warns(FutureWarning):
            with using_device_type("cpu"):
                model.fit(X, y)

    if hasattr(model, "fit_transform"):
        with pytest.warns(FutureWarning):
            with using_device_type("cpu"):
                res = model.fit_transform(X, y)
        assert type(res) is type(X)

    if hasattr(model, "predict"):
        with pytest.warns(FutureWarning):
            with using_device_type("cpu"):
                res = model.predict(X_test)
        assert type(res) is type(X_test)

    if hasattr(model, "transform"):
        with pytest.warns(FutureWarning):
            with using_device_type("cpu"):
                res = model.transform(X_test)
        assert type(res) is type(X_test)
