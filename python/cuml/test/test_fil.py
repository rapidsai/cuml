# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

from cuml import ForestInference
from cuml.test.utils import array_equal, unit_param, \
    quality_param, stress_param
from cuml.common.import_utils import has_xgboost
from cuml.common.import_utils import has_lightgbm

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import GradientBoostingClassifier, \
    GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor, \
    ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split


if has_xgboost():
    import xgboost as xgb


def simulate_data(m, n, k=2, random_state=None, classification=True,
                  bias=0.0):
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
                                           bias=bias,
                                           random_state=random_state)
    return np.c_[features].astype(np.float32), \
        np.c_[labels].astype(np.float32).flatten()


# absolute tolerance for FIL predict_proba
# False is binary classification, True is multiclass
proba_atol = {False: 3e-7, True: 3e-6}


def _build_and_save_xgboost(model_path,
                            X_train,
                            y_train,
                            classification=True,
                            num_rounds=5,
                            n_classes=2,
                            xgboost_params={}):
    """Trains a small xgboost classifier and saves it to model_path"""
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # instantiate params
    params = {'silent': 1}

    # learning task params
    if classification:
        params['eval_metric'] = 'error'
        if n_classes == 2:
            params['objective'] = 'binary:logistic'
        else:
            params['num_class'] = n_classes
            params['objective'] = 'multi:softprob'
    else:
        params['eval_metric'] = 'error'
        params['objective'] = 'reg:squarederror'
        params['base_score'] = 0.0

    params['max_depth'] = 25
    params.update(xgboost_params)
    bst = xgb.train(params, dtrain, num_rounds)
    bst.save_model(model_path)
    return bst


@pytest.mark.parametrize('n_rows', [unit_param(1000),
                                    quality_param(10000),
                                    stress_param(500000)])
@pytest.mark.parametrize('n_columns', [unit_param(30),
                                       quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('num_rounds', [unit_param(1),
                                        unit_param(5),
                                        quality_param(50),
                                        stress_param(90)])
@pytest.mark.parametrize('n_classes', [2, 5, 25])
@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
def test_fil_classification(n_rows, n_columns, num_rounds,
                            n_classes, tmp_path):
    # settings
    classification = True  # change this to false to use regression
    random_state = np.random.RandomState(43210)

    X, y = simulate_data(n_rows, n_columns, n_classes,
                         random_state=random_state,
                         classification=classification)
    # identify shape and indices
    n_rows, n_columns = X.shape
    train_size = 0.80

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, train_size=train_size, random_state=0)

    model_path = os.path.join(tmp_path, 'xgb_class.model')

    bst = _build_and_save_xgboost(model_path, X_train, y_train,
                                  num_rounds=num_rounds,
                                  classification=classification,
                                  n_classes=n_classes)

    dvalidation = xgb.DMatrix(X_validation, label=y_validation)

    if n_classes == 2:
        xgb_preds = bst.predict(dvalidation)
        xgb_preds_int = np.around(xgb_preds)
        xgb_proba = np.stack([1-xgb_preds, xgb_preds], axis=1)
    else:
        xgb_proba = bst.predict(dvalidation)
        xgb_preds_int = xgb_proba.argmax(axis=1)
    xgb_acc = accuracy_score(y_validation, xgb_preds_int)

    fm = ForestInference.load(model_path,
                              algo='auto',
                              output_class=True,
                              threshold=0.50)
    fil_preds = np.asarray(fm.predict(X_validation))
    fil_proba = np.asarray(fm.predict_proba(X_validation))
    fil_acc = accuracy_score(y_validation, fil_preds)

    assert fil_acc == pytest.approx(xgb_acc, abs=0.01)
    assert array_equal(fil_preds, xgb_preds_int)
    np.testing.assert_allclose(fil_proba, xgb_proba,
                               atol=proba_atol[n_classes > 2])


@pytest.mark.parametrize('n_rows', [unit_param(1000), quality_param(10000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_columns', [unit_param(20), quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('num_rounds', [unit_param(5), quality_param(10),
                         stress_param(90)])
@pytest.mark.parametrize('max_depth', [unit_param(3),
                                       unit_param(7),
                                       stress_param(11)])
@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
def test_fil_regression(n_rows, n_columns, num_rounds, tmp_path, max_depth):
    # settings
    classification = False  # change this to false to use regression
    n_rows = n_rows  # we'll use 1 millions rows
    n_columns = n_columns
    random_state = np.random.RandomState(43210)

    X, y = simulate_data(n_rows, n_columns,
                         random_state=random_state,
                         classification=classification, bias=10.0)
    # identify shape and indices
    n_rows, n_columns = X.shape
    train_size = 0.80

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, train_size=train_size, random_state=0)

    model_path = os.path.join(tmp_path, 'xgb_reg.model')
    bst = _build_and_save_xgboost(model_path, X_train,
                                  y_train,
                                  classification=classification,
                                  num_rounds=num_rounds,
                                  xgboost_params={'max_depth': max_depth})

    dvalidation = xgb.DMatrix(X_validation, label=y_validation)
    xgb_preds = bst.predict(dvalidation)

    xgb_mse = mean_squared_error(y_validation, xgb_preds)
    fm = ForestInference.load(model_path,
                              algo='auto',
                              output_class=False)
    fil_preds = np.asarray(fm.predict(X_validation))
    fil_preds = np.reshape(fil_preds, np.shape(xgb_preds))
    fil_mse = mean_squared_error(y_validation, fil_preds)

    assert fil_mse == pytest.approx(xgb_mse, abs=0.01)
    assert np.allclose(fil_preds, xgb_preds, 1e-3)


@pytest.mark.parametrize('n_rows', [1000])
@pytest.mark.parametrize('n_columns', [30])
# Skip depth 20 for dense tests
@pytest.mark.parametrize('max_depth,storage_type',
                         [(2, False), (2, True), (10, False), (10, True),
                          (20, True)])
# FIL not supporting multi-class sklearn RandomForestClassifiers
# When n_classes=25, fit a single estimator only to reduce test time
@pytest.mark.parametrize('n_classes,model_class,n_estimators',
                         [(2, GradientBoostingClassifier, 1),
                          (2, GradientBoostingClassifier, 10),
                          (2, RandomForestClassifier, 1),
                          (2, RandomForestClassifier, 10),
                          (2, ExtraTreesClassifier, 1),
                          (2, ExtraTreesClassifier, 10),
                          (5, GradientBoostingClassifier, 1),
                          (5, GradientBoostingClassifier, 10),
                          (25, GradientBoostingClassifier, 1)])
def test_fil_skl_classification(n_rows, n_columns, n_estimators, max_depth,
                                n_classes, storage_type, model_class):
    # settings
    classification = True  # change this to false to use regression
    random_state = np.random.RandomState(43210)

    X, y = simulate_data(n_rows, n_columns, n_classes,
                         random_state=random_state,
                         classification=classification)
    # identify shape and indices
    train_size = 0.80

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, train_size=train_size, random_state=0)

    init_kwargs = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
    }
    if model_class in [RandomForestClassifier, ExtraTreesClassifier]:
        init_kwargs['max_features'] = 0.3
        init_kwargs['n_jobs'] = -1
    else:
        # model_class == GradientBoostingClassifier
        init_kwargs['init'] = 'zero'

    skl_model = model_class(**init_kwargs, random_state=random_state)
    skl_model.fit(X_train, y_train)

    skl_preds = skl_model.predict(X_validation)
    skl_preds_int = np.around(skl_preds)
    skl_proba = skl_model.predict_proba(X_validation)

    skl_acc = accuracy_score(y_validation, skl_preds_int)

    algo = 'NAIVE' if storage_type else 'BATCH_TREE_REORG'

    fm = ForestInference.load_from_sklearn(skl_model,
                                           algo=algo,
                                           output_class=True,
                                           threshold=0.50,
                                           storage_type=storage_type)
    fil_preds = np.asarray(fm.predict(X_validation))
    fil_preds = np.reshape(fil_preds, np.shape(skl_preds_int))
    fil_acc = accuracy_score(y_validation, fil_preds)
    # fil_acc is within p99 error bars of skl_acc (diff == 0.017 +- 0.012)
    # however, some tests have a delta as big as 0.04.
    # sklearn uses float64 thresholds, while FIL uses float32
    # TODO(levsnv): once FIL supports float64 accuracy, revisit thresholds
    threshold = 1e-5 if n_classes == 2 else 0.1
    assert fil_acc == pytest.approx(skl_acc, abs=threshold)

    if n_classes == 2:
        assert array_equal(fil_preds, skl_preds_int)
    fil_proba = np.asarray(fm.predict_proba(X_validation))
    fil_proba = np.reshape(fil_proba, np.shape(skl_proba))
    np.testing.assert_allclose(fil_proba, skl_proba,
                               atol=proba_atol[n_classes > 2])


@pytest.mark.parametrize('n_rows', [1000])
@pytest.mark.parametrize('n_columns', [20])
@pytest.mark.parametrize('n_classes,model_class,n_estimators',
                         [(1, GradientBoostingRegressor, 1),
                          (1, GradientBoostingRegressor, 10),
                          (1, RandomForestRegressor, 1),
                          (1, RandomForestRegressor, 10),
                          (1, ExtraTreesRegressor, 1),
                          (1, ExtraTreesRegressor, 10),
                          (5, GradientBoostingRegressor, 10)])
@pytest.mark.parametrize('max_depth', [2, 10, 20])
@pytest.mark.parametrize('storage_type', [False, True])
def test_fil_skl_regression(n_rows, n_columns, n_classes, model_class,
                            n_estimators, max_depth, storage_type):

    # skip depth 20 for dense tests
    if max_depth == 20 and not storage_type:
        return

    # settings
    random_state = np.random.RandomState(43210)

    X, y = simulate_data(n_rows, n_columns, n_classes,
                         random_state=random_state,
                         classification=False)
    # identify shape and indices
    train_size = 0.80

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, train_size=train_size, random_state=0)

    init_kwargs = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
    }
    if model_class in [RandomForestRegressor, ExtraTreesRegressor]:
        init_kwargs['max_features'] = 0.3
        init_kwargs['n_jobs'] = -1
    else:
        # model_class == GradientBoostingRegressor
        init_kwargs['init'] = 'zero'

    skl_model = model_class(**init_kwargs)
    skl_model.fit(X_train, y_train)

    skl_preds = skl_model.predict(X_validation)

    skl_mse = mean_squared_error(y_validation, skl_preds)

    algo = 'NAIVE' if storage_type else 'BATCH_TREE_REORG'

    fm = ForestInference.load_from_sklearn(skl_model,
                                           algo=algo,
                                           output_class=False,
                                           storage_type=storage_type)
    fil_preds = np.asarray(fm.predict(X_validation))
    fil_preds = np.reshape(fil_preds, np.shape(skl_preds))

    fil_mse = mean_squared_error(y_validation, fil_preds)

    assert fil_mse <= skl_mse * (1. + 1e-6) + 1e-4
    assert np.allclose(fil_preds, skl_preds, 1.2e-3)


@pytest.fixture(scope="session", params=['binary', 'json'])
def small_classifier_and_preds(tmpdir_factory, request):
    X, y = simulate_data(500, 10,
                         random_state=43210,
                         classification=True)

    ext = 'json' if request.param == 'json' else 'model'
    model_type = 'xgboost_json' if request.param == 'json' else 'xgboost'
    model_path = str(tmpdir_factory.mktemp("models").join(
                 f"small_class.{ext}"))
    bst = _build_and_save_xgboost(model_path, X, y)
    # just do within-sample since it's not an accuracy test
    dtrain = xgb.DMatrix(X, label=y)
    xgb_preds = bst.predict(dtrain)

    return (model_path, model_type, X, xgb_preds)


@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
@pytest.mark.parametrize('algo', ['AUTO', 'NAIVE', 'TREE_REORG',
                                  'BATCH_TREE_REORG',
                                  'auto', 'naive', 'tree_reorg',
                                  'batch_tree_reorg'])
def test_output_algos(algo, small_classifier_and_preds):
    model_path, model_type, X, xgb_preds = small_classifier_and_preds
    fm = ForestInference.load(model_path,
                              model_type=model_type,
                              algo=algo,
                              output_class=True,
                              threshold=0.50)

    xgb_preds_int = np.around(xgb_preds)
    fil_preds = np.asarray(fm.predict(X))
    fil_preds = np.reshape(fil_preds, np.shape(xgb_preds_int))

    assert np.allclose(fil_preds, xgb_preds_int, 1e-3)


@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
@pytest.mark.parametrize('storage_type',
                         [False, True, 'auto', 'dense', 'sparse', 'sparse8'])
def test_output_storage_type(storage_type, small_classifier_and_preds):
    model_path, model_type, X, xgb_preds = small_classifier_and_preds
    fm = ForestInference.load(model_path,
                              model_type=model_type,
                              output_class=True,
                              storage_type=storage_type,
                              threshold=0.50)

    xgb_preds_int = np.around(xgb_preds)
    fil_preds = np.asarray(fm.predict(X))
    fil_preds = np.reshape(fil_preds, np.shape(xgb_preds_int))

    assert np.allclose(fil_preds, xgb_preds_int, 1e-3)


@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
@pytest.mark.parametrize('storage_type', ['dense', 'sparse'])
@pytest.mark.parametrize('blocks_per_sm', [1, 2, 3, 4])
def test_output_blocks_per_sm(storage_type, blocks_per_sm,
                              small_classifier_and_preds):
    model_path, model_type, X, xgb_preds = small_classifier_and_preds
    fm = ForestInference.load(model_path,
                              model_type=model_type,
                              output_class=True,
                              storage_type=storage_type,
                              threshold=0.50,
                              blocks_per_sm=blocks_per_sm)

    xgb_preds_int = np.around(xgb_preds)
    fil_preds = np.asarray(fm.predict(X))
    fil_preds = np.reshape(fil_preds, np.shape(xgb_preds_int))

    assert np.allclose(fil_preds, xgb_preds_int, 1e-3)


@pytest.mark.parametrize('output_class', [True, False])
@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
def test_thresholding(output_class, small_classifier_and_preds):
    model_path, model_type, X, xgb_preds = small_classifier_and_preds
    fm = ForestInference.load(model_path,
                              model_type=model_type,
                              algo='TREE_REORG',
                              output_class=output_class,
                              threshold=0.50)
    fil_preds = np.asarray(fm.predict(X))
    if output_class:
        assert ((fil_preds != 0.0) & (fil_preds != 1.0)).sum() == 0
    else:
        assert ((fil_preds != 0.0) & (fil_preds != 1.0)).sum() > 0


@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
def test_output_args(small_classifier_and_preds):
    model_path, model_type, X, xgb_preds = small_classifier_and_preds
    fm = ForestInference.load(model_path,
                              model_type=model_type,
                              algo='TREE_REORG',
                              output_class=False,
                              threshold=0.50)
    X = np.asarray(X)
    fil_preds = fm.predict(X)
    fil_preds = np.reshape(fil_preds, np.shape(xgb_preds))

    assert array_equal(fil_preds, xgb_preds, 1e-3)


@pytest.mark.parametrize('num_classes', [2, 5])
@pytest.mark.skipif(has_lightgbm() is False, reason="need to install lightgbm")
def test_lightgbm(tmp_path, num_classes):
    import lightgbm as lgb
    X, y = simulate_data(500,
                         10 if num_classes == 2 else 50,
                         num_classes,
                         random_state=43210,
                         classification=True)
    train_data = lgb.Dataset(X, label=y)
    num_round = 5
    model_path = str(os.path.join(tmp_path, 'lgb.model'))

    if num_classes == 2:
        param = {'objective': 'binary',
                 'metric': 'binary_logloss',
                 'num_class': 1}
        bst = lgb.train(param, train_data, num_round)
        bst.save_model(model_path)
        fm = ForestInference.load(model_path,
                                  algo='TREE_REORG',
                                  output_class=True,
                                  model_type="lightgbm")
        # binary classification
        gbm_proba = bst.predict(X)
        fil_proba = fm.predict_proba(X)[:, 1]
        gbm_preds = (gbm_proba > 0.5)
        fil_preds = fm.predict(X)
        assert array_equal(gbm_preds, fil_preds)
        np.testing.assert_allclose(gbm_proba, fil_proba,
                                   atol=proba_atol[num_classes > 2])
    else:
        # multi-class classification
        lgm = lgb.LGBMClassifier(objective='multiclass',
                                 boosting_type='gbdt',
                                 n_estimators=num_round)
        lgm.fit(X, y)
        lgm.booster_.save_model(model_path)
        fm = ForestInference.load(model_path,
                                  algo='TREE_REORG',
                                  output_class=True,
                                  model_type="lightgbm")
        lgm_preds = lgm.predict(X)
        assert array_equal(lgm.booster_.predict(X).argmax(axis=1), lgm_preds)
        assert array_equal(lgm_preds, fm.predict(X))
        # lightgbm uses float64 thresholds, while FIL uses float32
        np.testing.assert_allclose(lgm.predict_proba(X), fm.predict_proba(X),
                                   atol=proba_atol[num_classes > 2])
