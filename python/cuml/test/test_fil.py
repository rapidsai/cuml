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
import os

from cuml import FIL
from cuml.utils.import_utils import has_xgboost, has_lightgbm
from numba import cuda
import cudf

from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error

if has_xgboost():
    import xgboost as xgb

    def simulate_data(m, n, k=2, random_state=None, classification=True):
        if classification:
            features, labels = make_classification(n_samples=m,
                                                   n_features=n,
                                                   n_informative=int(n/5),
                                                   n_classes=k,
                                                   random_state=random_state)
        else:
            features, labels = make_regression(n_samples=m,
                                               n_features=n,
                                               n_informative=int(n/5),
                                               n_targets=1,
                                               random_state=random_state)
        return np.c_[labels, features].astype(np.float32)


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


def _build_and_save_xgboost(model_path,
                            X_train,
                            y_train,
                            classification=True,
                            num_rounds=5,
                            xgboost_params={}):
    """Trains a small xgboost classifier and saves it to model_path"""
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # instantiate params
    params = {'silent': 1}

    # learning task params
    if classification:
        params['eval_metric'] = 'error'
        params['objective'] = 'binary:logistic'
    else:
        params['eval_metric'] = 'error'
        params['objective'] = 'reg:squarederror'
        params['base_score'] = 0.0

    params.update(xgboost_params)

    bst = xgb.train(params, dtrain, num_rounds)
    bst.save_model(model_path)
    return bst


@pytest.mark.parametrize('n_rows', [unit_param(100),
                                    quality_param(1000),
                                    stress_param(500000)])
@pytest.mark.parametrize('n_columns', [unit_param(11),
                                       quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('num_rounds', [unit_param(1),
                                        unit_param(5),
                                        quality_param(50),
                                        stress_param(90)])
@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
def test_fil_class(n_rows, n_columns, num_rounds, tmp_path):
    # settings
    classification = True  # change this to false to use regression
    n_rows = n_rows  # we'll use 1 millions rows
    n_columns = n_columns
    n_categories = 2
    random_state = np.random.RandomState(43210)

    dataset = simulate_data(n_rows, n_columns, n_categories,
                            random_state=random_state,
                            classification=classification)
    # identify shape and indices
    n_rows, n_columns = dataset.shape
    train_size = 0.80
    train_index = int(n_rows * train_size)

    # split X, y
    X, y = dataset[:, 1:], dataset[:, 0]
    del dataset

    # split train data
    X_train, y_train = X[:train_index, :], y[:train_index]

    # split validation data
    X_validation, y_validation = X[train_index:, :], y[train_index:]

    model_path = os.path.join(tmp_path, 'xgb_class.model')

    bst = _build_and_save_xgboost(model_path, X_train, y_train,
                                  num_rounds, classification)

    dvalidation = xgb.DMatrix(X_validation, label=y_validation)
    xgb_preds = bst.predict(dvalidation)

    xgb_acc = accuracy_score(y_validation, xgb_preds > 0.5)

    print("Reading the saved xgb model")

    fm = FIL.from_treelite_file(model_path,
                                algo=0,
                                output_class=True,
                                threshold=0.50)
    fil_preds = np.asarray(fm.predict(X_validation))
    fil_acc = accuracy_score(y_validation, fil_preds)

    print("XGB accuracy = ", xgb_acc, " FIL accuracy: ", fil_acc)
    assert fil_acc == pytest.approx(xgb_acc, 0.01)
    assert fil_acc > 0.80


@pytest.mark.parametrize('n_rows', [unit_param(100), quality_param(1000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_columns', [unit_param(11), quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('num_rounds', [unit_param(5), quality_param(10),
                         stress_param(90)])
@pytest.mark.parametrize('max_depth', [unit_param(3),
                                       unit_param(7),
                                       stress_param(11)])
@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
def test_fil_reg(n_rows, n_columns, num_rounds, tmp_path, max_depth):
    # settings
    classification = False  # change this to false to use regression
    n_rows = n_rows  # we'll use 1 millions rows
    n_columns = n_columns
    random_state = np.random.RandomState(43210)

    dataset = simulate_data(n_rows, n_columns,
                            random_state=random_state,
                            classification=classification)
    # identify shape and indices
    n_rows, n_columns = dataset.shape
    train_size = 0.80
    train_index = int(n_rows * train_size)

    # split X, y
    X, y = dataset[:, 1:], dataset[:, 0]
    del dataset

    # split train data
    X_train, y_train = X[:train_index, :], y[:train_index]

    # split validation data
    X_validation, y_validation = X[train_index:, :], y[train_index:]

    model_path = os.path.join(tmp_path, 'xgb_reg.model')
    bst = _build_and_save_xgboost(model_path, X_train,
                                  y_train,
                                  num_rounds,
                                  classification,
                                  xgboost_params={'max_depth': max_depth})

    dvalidation = xgb.DMatrix(X_validation, label=y_validation)
    xgb_preds = bst.predict(dvalidation)

    xgb_mse = mean_squared_error(y_validation, xgb_preds)
    print("Reading the saved xgb model")
    fm = FIL.from_treelite_file(model_path,
                                algo=1,
                                output_class=False,
                                threshold=0.00)
    fil_preds = np.asarray(fm.predict(X_validation))
    fil_mse = mean_squared_error(y_validation, fil_preds)

    print("XGB accuracy = ", xgb_mse, " FIL accuracy: ", fil_mse)
    assert fil_mse == pytest.approx(xgb_mse, 0.01)


@pytest.fixture(scope="session")
def small_classifier_and_preds(tmpdir_factory):
    dataset = simulate_data(100,
                            10,
                            random_state=43210,
                            classification=True)
    X, y = dataset[:, 1:], dataset[:, 0]

    model_path = str(tmpdir_factory.mktemp("models").join("small_class.model"))
    bst = _build_and_save_xgboost(model_path, X, y)
    # just do within-sample since it's not an accuracy test
    dtrain = xgb.DMatrix(X, label=y)
    xgb_preds = bst.predict(dtrain)

    return (model_path, X, xgb_preds)


@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
@pytest.mark.parametrize('algo', [0, 1, 2])
def test_output_algos(algo, small_classifier_and_preds):
    model_path, X, xgb_preds = small_classifier_and_preds
    fm = FIL.from_treelite_file(model_path,
                                algo=algo,
                                output_class=False,
                                threshold=0.50)
    fil_preds = fm.predict(X)
    assert np.allclose(fil_preds, xgb_preds, 1e-3)


@pytest.mark.parametrize('output_class', [True, False])
@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
def test_thresholding(output_class, small_classifier_and_preds):
    model_path, X, xgb_preds = small_classifier_and_preds
    fm = FIL.from_treelite_file(model_path,
                                algo=1,
                                output_class=output_class,
                                threshold=0.50)
    fil_preds = np.asarray(fm.predict(X))
    if output_class:
        assert ((fil_preds != 0.0) & (fil_preds != 1.0)).sum() == 0
    else:
        assert ((fil_preds != 0.0) & (fil_preds != 1.0)).sum() > 0


@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
@pytest.mark.parametrize('format', ['numpy', 'cudf', 'gpuarray'])
def test_output_args(format, small_classifier_and_preds):
    model_path, X, xgb_preds = small_classifier_and_preds
    fm = FIL.from_treelite_file(model_path,
                                algo=1,
                                output_class=False,
                                threshold=0.50)
    if format == 'numpy':
        X = np.asarray(X)
    elif format == 'cudf':
        X = cudf.DataFrame.from_gpu_matrix(
            cuda.to_device(np.ascontiguousarray(X)))
    elif format == 'gpuarray':
        X = cuda.to_device(np.ascontiguousarray(X))
    else:
        assert False

    fil_preds = fm.predict(X)
    assert np.allclose(fil_preds, xgb_preds, 1e-3)


@pytest.mark.skipif(has_lightgbm() is False, reason="need to install lightgbm")
def test_lightgbm(tmp_path):
    import lightgbm as lgb
    dataset = simulate_data(100,
                            10,
                            random_state=43210,
                            classification=True)
    X, y = dataset[:, 1:], dataset[:, 0]

    train_data = lgb.Dataset(X, label=y)
    param = {'objective': 'binary',
             'metric': 'binary_logloss'}
    num_round = 5
    bst = lgb.train(param, train_data, num_round)

    gbm_preds = bst.predict(X)

    model_path = str(os.path.join(tmp_path,
                                  'lgb.model'))
    bst.save_model(model_path)

    fm = FIL.from_treelite_file(model_path,
                                algo=1,
                                output_class=False,
                                threshold=0.00,
                                model_type="lightgbm")
    fil_preds = np.asarray(fm.predict(X))
    assert np.allclose(gbm_preds, fil_preds, 1e-3)
