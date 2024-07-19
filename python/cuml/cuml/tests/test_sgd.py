# Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from cuml.solvers import SGD as cumlSGD
from cuml.internals.safe_imports import gpu_only_import
import pytest
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")

cudf = gpu_only_import("cudf")


@pytest.mark.parametrize("lrate", ["constant", "invscaling", "adaptive"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("penalty", ["none", "l1", "l2", "elasticnet"])
@pytest.mark.parametrize("loss", ["hinge", "log", "squared_loss"])
@pytest.mark.parametrize("datatype", ["dataframe", "numpy"])
def test_sgd(dtype, lrate, penalty, loss, datatype):

    X, y = make_blobs(n_samples=100, n_features=3, centers=2, random_state=0)
    X = X.astype(dtype)
    y = y.astype(dtype)

    if loss == "hinge" or loss == "squared_loss":
        y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    if datatype == "dataframe":
        X_train = cudf.DataFrame(X_train)
        X_test = cudf.DataFrame(X_test)
        y_train = cudf.Series(y_train)

    cu_sgd = cumlSGD(
        learning_rate=lrate,
        eta0=0.005,
        epochs=2000,
        fit_intercept=True,
        batch_size=4096,
        tol=0.0,
        penalty=penalty,
        loss=loss,
        power_t=0.4,
    )

    cu_sgd.fit(X_train, y_train)
    cu_pred = cu_sgd.predict(X_test)

    if datatype == "dataframe":
        assert isinstance(cu_pred, cudf.Series)
        cu_pred = cu_pred.to_numpy()

    else:
        assert isinstance(cu_pred, np.ndarray)

    if loss == "log":
        cu_pred[cu_pred < 0.5] = 0
        cu_pred[cu_pred >= 0.5] = 1
    elif loss == "squared_loss":
        cu_pred[cu_pred < 0] = -1
        cu_pred[cu_pred >= 0] = 1

    # Adjust for squared loss (we don't need to test for high accuracy,
    # just that the loss function tended towards the expected classes.
    assert np.array_equal(cu_pred, y_test)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("datatype", ["dataframe", "numpy"])
def test_sgd_default(dtype, datatype):

    X, y = make_blobs(n_samples=100, n_features=3, centers=2, random_state=0)
    X = X.astype(dtype)
    y = y.astype(dtype)

    # Default loss is squared_loss
    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    if datatype == "dataframe":
        X_train = cudf.DataFrame(X_train)
        X_test = cudf.DataFrame(X_test)
        y_train = cudf.Series(y_train)

    cu_sgd = cumlSGD()

    cu_sgd.fit(X_train, y_train)
    cu_pred = cu_sgd.predict(X_test)

    if datatype == "dataframe":
        assert isinstance(cu_pred, cudf.Series)
        cu_pred = cu_pred.to_numpy()

    else:
        assert isinstance(cu_pred, np.ndarray)

    # Adjust for squared loss (we don't need to test for high accuracy,
    # just that the loss function tended towards the expected classes.
    cu_pred[cu_pred < 0] = -1
    cu_pred[cu_pred >= 0] = 1

    assert np.array_equal(cu_pred, y_test)
