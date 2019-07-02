# Copyright (c) 2019, NVIDIA CORPORATION.
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
import numpy as np
from cuml.test.utils import get_handle
from sklearn.datasets import make_classification
from cuml.ensemble import RandomForestClassifier as curfc
from sklearn.ensemble import RandomForestClassifier as skrfc
from sklearn.metrics import accuracy_score


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


@pytest.mark.parametrize('nrows', [unit_param(30), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('ncols', [unit_param(10), quality_param(100),
                         stress_param(200)])
@pytest.mark.parametrize('n_info', [unit_param(7), quality_param(50),
                         stress_param(100)])
@pytest.mark.parametrize('datatype', [np.float32, np.float64])
@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('split_algo', [0, 1])
def test_rf_predict_numpy(datatype, use_handle, split_algo,
                          n_info, nrows, ncols):
    train_rows = np.int32(nrows*0.8)
    X, y = make_classification(n_samples=nrows, n_features=ncols,
                               n_clusters_per_class=1, n_informative=n_info,
                               random_state=123, n_classes=5)
    X_test = np.asarray(X[train_rows:, 0:]).astype(datatype)
    y_test = np.asarray(y[train_rows:, ]).astype(np.int32)
    X_train = np.asarray(X[0:train_rows, :]).astype(datatype)
    y_train = np.asarray(y[0:train_rows, ]).astype(np.int32)

    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle)

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc(max_features=1.0,
                       n_bins=8, split_algo=0, split_criterion=0,
                       min_rows_per_node=2,
                       n_estimators=40, handle=handle, max_leaves=-1,
                       max_depth=-1)
    cuml_model.fit(X_train, y_train)
    cu_predict = cuml_model.predict(X_test)
    cu_acc = accuracy_score(y_test, cu_predict)

    if nrows < 500000:
        # sklearn random forest classification model
        # initialization, fit and predict
        sk_model = skrfc(n_estimators=40, max_depth=None,
                         min_samples_split=2, max_features=1.0,
                         random_state=10)
        sk_model.fit(X_train, y_train)
        sk_predict = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_predict)

        # compare the accuracy of the two models
        assert cu_acc >= (sk_acc - 0.07)
