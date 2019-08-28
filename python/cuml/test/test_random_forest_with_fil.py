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
from cuml.fil import ForestInference
from cuml.test.utils import get_handle, array_equal

from cuml.ensemble import RandomForestClassifier as curfc
from cuml.ensemble import RandomForestRegressor as curfr
from sklearn.ensemble import RandomForestClassifier as skrfc
from sklearn.ensemble import RandomForestRegressor as skrfr

from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_california_housing, \
    make_classification, make_regression
from sklearn.metrics import mean_squared_error


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


@pytest.mark.parametrize('nrows', [unit_param(2000), quality_param(5000),
                         stress_param(100000)])
@pytest.mark.parametrize('ncols', [unit_param(100), quality_param(100),
                         stress_param(200)])
@pytest.mark.parametrize('n_info', [unit_param(40), quality_param(60),
                         stress_param(100)])
@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('split_algo', [0, 1])
@pytest.mark.parametrize('max_depth', [16, 1])
def test_rf_classification(datatype, split_algo,
                           n_info, nrows, ncols, max_depth):
    use_handle = True
    if split_algo == 1 and max_depth < 0:
        pytest.xfail("Unlimited depth not supported with quantile")

    train_rows = np.int32(nrows*0.8)
    X, y = make_classification(n_samples=nrows, n_features=ncols,
                               n_clusters_per_class=1, n_informative=n_info,
                               random_state=123, n_classes=2)
    X_test = np.asarray(X[train_rows:, 0:]).astype(datatype)
    y_test = np.asarray(y[train_rows:, ]).astype(np.int32)
    X_train = np.asarray(X[0:train_rows, :]).astype(datatype)
    y_train = np.asarray(y[0:train_rows, ]).astype(np.int32)
    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle)

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc(max_features=1.0,
                       n_bins=8, split_algo=split_algo, split_criterion=0,
                       min_rows_per_node=2,
                       n_estimators=40, handle=handle, max_leaves=-1,
                       max_depth=max_depth)
    cuml_model.fit(X_train, y_train)
    cu_predict = cuml_model.predict(X_test)
    cu_acc = accuracy_score(y_test, cu_predict)
    tl_model = cuml_model.build_treelite_forest(num_features=ncols, task_category=2)
    fm = ForestInference()
    model = fm.load_from_randomforest(tl_model.value, output_class=True,
                                      threshold=0.5, algo='BATCH_TREE_REORG')
    fil_preds = np.around(np.asarray(model.predict(X_test)))
    fil_acc = accuracy_score(y_test, fil_preds)
    assert fil_acc >= cu_acc


@pytest.mark.parametrize('mode', [unit_param('unit'), quality_param('quality'),
                         stress_param('stress')])
@pytest.mark.parametrize('ncols', [unit_param(50), quality_param(100),
                         stress_param(200)])
@pytest.mark.parametrize('n_info', [unit_param(30), quality_param(50),
                         stress_param(100)])
@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('use_handle', [True, False])
@pytest.mark.parametrize('split_algo', [0, 1])
def test_rf_regression(datatype, use_handle, split_algo,
                       n_info, mode, ncols):

    if mode == 'unit':
        X, y = make_regression(n_samples=300, n_features=ncols,
                               n_informative=n_info,
                               random_state=123)

    elif mode == 'quality':
        X, y = fetch_california_housing(return_X_y=True)

    else:
        X, y = make_regression(n_samples=100000, n_features=ncols,
                               n_informative=n_info,
                               random_state=123)

    train_rows = np.int32(X.shape[0]*0.8)
    X_test = np.asarray(X[train_rows:, :]).astype(datatype)
    y_test = np.asarray(y[train_rows:, ]).astype(datatype)
    X_train = np.asarray(X[0:train_rows, :]).astype(datatype)
    y_train = np.asarray(y[0:train_rows, ]).astype(datatype)

    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle)

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfr(max_features=1.0, rows_sample=1.0,
                       n_bins=8, split_algo=split_algo, split_criterion=2,
                       min_rows_per_node=2,
                       n_estimators=50, handle=handle, max_leaves=-1,
                       max_depth=16, accuracy_metric='mse')
    cuml_model.fit(X_train, y_train)
    cu_predict = cuml_model.predict(X_test)
    cu_mse = mean_squared_error(y_test, cu_predict)
    tl_model = cuml_model.build_treelite_forest(num_features=ncols, task_category=2)
    fm = ForestInference()
    model = fm.load_from_randomforest(model_handle=tl_model.value, algo='BATCH_TREE_REORG')
    fil_preds = model.predict(X_test)
    fil_mse = mean_squared_error(y_test, fil_preds)
    assert array_equal(fil_mse, cu_mse, 1e-2)