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

import numpy as np
import pytest
import random

from cuml.ensemble import RandomForestClassifier as curfc
from cuml.ensemble import RandomForestRegressor as curfr
from cuml.metrics import r2_score
from cuml.test.utils import get_handle, unit_param, \
    quality_param, stress_param

from sklearn.ensemble import RandomForestClassifier as skrfc
from sklearn.ensemble import RandomForestRegressor as skrfr
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_california_housing, \
    make_classification, make_regression
from sklearn.model_selection import train_test_split


@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('column_info', [unit_param([20, 10]),
                         quality_param([200, 100]),
                         stress_param([500, 350])])
@pytest.mark.parametrize('rows_sample', [unit_param(1.0), quality_param(0.90),
                         stress_param(0.95)])
@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('split_algo', [0, 1])
@pytest.mark.parametrize('max_features', [1.0, 'auto', 'log2', 'sqrt'])
def test_rf_classification(datatype, split_algo, rows_sample,
                           nrows, column_info, max_features):
    use_handle = True
    ncols, n_info = column_info

    X, y = make_classification(n_samples=nrows, n_features=ncols,
                               n_clusters_per_class=1, n_informative=n_info,
                               random_state=123, n_classes=2)
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)
    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle, n_streams=1)

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc(max_features=max_features, rows_sample=rows_sample,
                       n_bins=16, split_algo=split_algo, split_criterion=0,
                       min_rows_per_node=2, seed=123, n_streams=1,
                       n_estimators=40, handle=handle, max_leaves=-1,
                       max_depth=16)
    cuml_model.fit(X_train, y_train)
    fil_preds = cuml_model.predict(X_test,
                                   predict_model="GPU",
                                   output_class=True,
                                   threshold=0.5,
                                   algo='BATCH_TREE_REORG')
    cu_predict = cuml_model.predict(X_test, predict_model="CPU")
    cuml_acc = accuracy_score(y_test, cu_predict)
    fil_acc = accuracy_score(y_test, fil_preds)

    if nrows < 500000:
        sk_model = skrfc(n_estimators=40,
                         max_depth=16,
                         min_samples_split=2, max_features=max_features,
                         random_state=10)
        sk_model.fit(X_train, y_train)
        sk_predict = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_predict)
        assert fil_acc >= (sk_acc - 0.07)
    assert fil_acc >= (cuml_acc - 0.02)


@pytest.mark.parametrize('mode', [unit_param('unit'), quality_param('quality'),
                         stress_param('stress')])
@pytest.mark.parametrize('column_info', [unit_param([20, 10]),
                         quality_param([200, 50]),
                         stress_param([400, 100])])
@pytest.mark.parametrize('rows_sample', [unit_param(1.0), quality_param(0.90),
                         stress_param(0.95)])
@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('split_algo', [0, 1])
@pytest.mark.parametrize('max_features', [1.0, 'auto', 'log2', 'sqrt'])
def test_rf_regression(datatype, split_algo, mode,
                       column_info, max_features, rows_sample):

    ncols, n_info = column_info
    use_handle = True

    if mode == 'unit':
        X, y = make_regression(n_samples=500, n_features=ncols,
                               n_informative=n_info,
                               random_state=123)

    elif mode == 'quality':
        X, y = fetch_california_housing(return_X_y=True)

    else:
        X, y = make_regression(n_samples=100000, n_features=ncols,
                               n_informative=n_info,
                               random_state=123)
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)

    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle, n_streams=1)

    # Initialize and fit using cuML's random forest regression model
    cuml_model = curfr(max_features=max_features, rows_sample=rows_sample,
                       n_bins=16, split_algo=split_algo, split_criterion=2,
                       min_rows_per_node=2, seed=123, n_streams=1,
                       n_estimators=50, handle=handle, max_leaves=-1,
                       max_depth=16, accuracy_metric='mse')
    cuml_model.fit(X_train, y_train)
    # predict using FIL
    fil_preds = cuml_model.predict(X_test, predict_model="GPU")
    cu_preds = cuml_model.predict(X_test, predict_model="CPU")
    cu_r2 = r2_score(y_test, cu_preds, convert_dtype=datatype)
    fil_r2 = r2_score(y_test, fil_preds, convert_dtype=datatype)
    # Initialize, fit and predict using
    # sklearn's random forest regression model
    if mode != "stress":
        sk_model = skrfr(n_estimators=50, max_depth=16,
                         min_samples_split=2, max_features=max_features,
                         random_state=10)
        sk_model.fit(X_train, y_train)
        sk_predict = sk_model.predict(X_test)
        sk_r2 = r2_score(y_test, sk_predict, convert_dtype=datatype)
        assert fil_r2 >= (sk_r2 - 0.07)
    assert fil_r2 >= (cu_r2 - 0.02)


@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('column_info', [unit_param([20, 10]),
                         quality_param([200, 100]),
                         stress_param([500, 350])])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
def test_rf_classification_default(datatype, column_info, nrows):

    ncols, n_info = column_info
    X, y = make_classification(n_samples=nrows, n_features=ncols,
                               n_clusters_per_class=1, n_informative=n_info,
                               random_state=0, n_classes=2)
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)
    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc()
    cuml_model.fit(X_train, y_train)
    fil_preds = cuml_model.predict(X_test, predict_model="GPU")
    cu_preds = cuml_model.predict(X_test, predict_model="CPU")
    fil_acc = accuracy_score(y_test, fil_preds)
    cu_acc = accuracy_score(y_test, cu_preds)

    score_acc = cuml_model.score(X_test, y_test)
    assert cu_acc == pytest.approx(score_acc)

    # sklearn random forest classification model
    # initialization, fit and predict
    if nrows < 500000:
        sk_model = skrfc(max_depth=16, random_state=10)
        sk_model.fit(X_train, y_train)
        sk_predict = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_predict)
        assert fil_acc >= (sk_acc - 0.07)
    assert fil_acc >= (cu_acc - 0.02)


@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('column_info', [unit_param([20, 10]),
                         quality_param([200, 100]),
                         stress_param([500, 350])])
@pytest.mark.parametrize('nrows', [unit_param(2000), quality_param(25000),
                         stress_param(500000)])
def test_rf_regression_default(datatype, column_info, nrows):

    ncols, n_info = column_info
    X, y = make_regression(n_samples=nrows, n_features=ncols,
                           n_informative=n_info,
                           random_state=123)
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfr()
    cuml_model.fit(X_train, y_train)

    # predict using FIL
    fil_preds = cuml_model.predict(X_test, predict_model="GPU")
    cu_preds = cuml_model.predict(X_test, predict_model="CPU")
    cu_r2 = r2_score(y_test, cu_preds, convert_dtype=datatype)
    fil_r2 = r2_score(y_test, fil_preds, convert_dtype=datatype)

    # score function should be equivalent
    score_mse = cuml_model.score(X_test, y_test)
    manual_mse = ((fil_preds - y_test)**2).mean()
    assert manual_mse == pytest.approx(score_mse)

    # Initialize, fit and predict using
    # sklearn's random forest regression model
    if nrows < 500000:
        sk_model = skrfr(max_depth=16, random_state=10)
        sk_model.fit(X_train, y_train)
        sk_predict = sk_model.predict(X_test)
        sk_r2 = r2_score(y_test, sk_predict, convert_dtype=datatype)
        # XXX Accuracy gap exists with default parameters, requires
        # further investigation for next release
        assert fil_r2 >= (sk_r2 - 0.08)

    assert fil_r2 >= (cu_r2 - 0.02)


@pytest.mark.parametrize('datatype', [np.float32])
@pytest.mark.parametrize('column_info', [unit_param([20, 10]),
                         quality_param([200, 100]),
                         stress_param([500, 350])])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
def test_rf_classification_seed(datatype, column_info, nrows):

    ncols, n_info = column_info
    X, y = make_classification(n_samples=nrows, n_features=ncols,
                               n_clusters_per_class=1, n_informative=n_info,
                               random_state=0, n_classes=2)
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)
    for i in range(20):
        seed = random.randint(100, 1e5)
        # Initialize, fit and predict using cuML's
        # random forest classification model
        cu_class = curfc(seed=seed, n_streams=1)
        cu_class.fit(X_train, y_train)

        # predict using FIL
        fil_preds_orig = cu_class.predict(X_test,
                                          predict_model="GPU").copy_to_host()
        cu_preds_orig = cu_class.predict(X_test,
                                         predict_model="CPU")
        cu_acc_orig = accuracy_score(y_test, cu_preds_orig)
        fil_acc_orig = accuracy_score(y_test, fil_preds_orig)

        # Initialize, fit and predict using cuML's
        # random forest classification model
        cu_class2 = curfc(seed=seed, n_streams=1)
        cu_class2.fit(X_train, y_train)

        # predict using FIL
        fil_preds_rerun = cu_class2.predict(X_test,
                                            predict_model="GPU").copy_to_host()
        cu_preds_rerun = cu_class2.predict(X_test, predict_model="CPU")
        cu_acc_rerun = accuracy_score(y_test, cu_preds_rerun)
        fil_acc_rerun = accuracy_score(y_test, fil_preds_rerun)

        assert fil_acc_orig == fil_acc_rerun
        assert cu_acc_orig == cu_acc_rerun
        assert (fil_preds_orig == fil_preds_rerun).all()
        assert (cu_preds_orig == cu_preds_rerun).all()


@pytest.mark.parametrize('datatype', [(np.float64, np.float32),
                                      (np.float32, np.float64)])
@pytest.mark.parametrize('column_info', [unit_param([20, 10]),
                         quality_param([200, 100]),
                         stress_param([500, 350])])
@pytest.mark.parametrize('nrows', [unit_param(500), quality_param(5000),
                         stress_param(500000)])
def test_rf_classification_float64(datatype, column_info, nrows):

    ncols, n_info = column_info
    X, y = make_classification(n_samples=nrows, n_features=ncols,
                               n_clusters_per_class=1, n_informative=n_info,
                               random_state=0, n_classes=2)
    X = X.astype(datatype[0])
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)
    X_test = X_test.astype(datatype[1])

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc()
    cuml_model.fit(X_train, y_train)
    cu_preds = cuml_model.predict(X_test, predict_model="CPU")
    cu_acc = accuracy_score(y_test, cu_preds)

    # sklearn random forest classification model
    # initialization, fit and predict
    if nrows < 500000:
        sk_model = skrfc(max_depth=16, random_state=10)
        sk_model.fit(X_train, y_train)
        sk_predict = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_predict)
        assert cu_acc >= (sk_acc - 0.07)

    # predict using cuML's GPU based prediction
    if datatype[0] == np.float32:
        fil_preds = cuml_model.predict(X_test, predict_model="GPU")
        fil_acc = accuracy_score(y_test, fil_preds)
        assert fil_acc >= (cu_acc - 0.02)


@pytest.mark.parametrize('datatype', [(np.float64, np.float32),
                                      (np.float32, np.float64)])
@pytest.mark.parametrize('column_info', [unit_param([20, 10]),
                         quality_param([200, 100]),
                         stress_param([500, 350])])
@pytest.mark.parametrize('nrows', [unit_param(5000), quality_param(25000),
                         stress_param(500000)])
def test_rf_regression_float64(datatype, column_info, nrows):

    ncols, n_info = column_info
    X, y = make_regression(n_samples=nrows, n_features=ncols,
                           n_informative=n_info,
                           random_state=123)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        random_state=0)
    X_train = X_train.astype(datatype[0])
    y_train = y_train.astype(datatype[0])
    X_test = X_test.astype(datatype[1])
    y_test = y_test.astype(datatype[1])

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfr()
    cuml_model.fit(X_train, y_train)
    cu_preds = cuml_model.predict(X_test, predict_model="CPU")
    cu_r2 = r2_score(y_test, cu_preds, convert_dtype=datatype[0])

    # sklearn random forest classification model
    # initialization, fit and predict
    if nrows < 500000:
        sk_model = skrfr(max_depth=16, random_state=10)
        sk_model.fit(X_train, y_train)
        sk_predict = sk_model.predict(X_test)
        sk_r2 = r2_score(y_test, sk_predict, convert_dtype=datatype[0])
        assert cu_r2 >= (sk_r2 - 0.09)

    # predict using cuML's GPU based prediction
    if datatype[0] == np.float32:
        fil_preds = cuml_model.predict(X_test, predict_model="GPU")
        fil_r2 = r2_score(y_test, fil_preds, convert_dtype=datatype[0])
        assert fil_r2 >= (cu_r2 - 0.02)
