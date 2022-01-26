#
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import json
import pytest
import treelite
import numpy as np
import pandas as pd
import cupy as cp
import cudf
from cuml.experimental.explainer.tree_shap import TreeExplainer
from cuml.common.import_utils import has_xgboost, has_lightgbm, has_shap
from cuml.common.import_utils import has_sklearn
from cuml.common.exceptions import NotFittedError
from cuml.ensemble import RandomForestRegressor as curfr
from cuml.ensemble import RandomForestClassifier as curfc

if has_xgboost():
    import xgboost as xgb
if has_lightgbm():
    import lightgbm as lgb
if has_shap():
    import shap
if has_sklearn():
    from sklearn.datasets import make_regression, make_classification
    from sklearn.ensemble import RandomForestRegressor as sklrfr
    from sklearn.ensemble import RandomForestClassifier as sklrfc


def make_classification_with_categorical(
        *, n_samples, n_features, n_categorical, n_informative, n_redundant,
        n_repeated, n_classes, random_state):
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=n_informative,
                               n_redundant=n_redundant, n_repeated=n_repeated,
                               n_classes=n_classes, random_state=random_state)
    X, y = X.astype(np.float32), y.astype(np.float32)

    # Turn some columns into categorical, by taking quartiles
    X = pd.DataFrame({f'f{i}': X[:, i] for i in range(n_features)})
    for i in range(n_categorical):
        column = f'f{i}'
        X[column] = pd.qcut(X[column], 4, labels=range(4))
    return X, y


def make_regression_with_categorical(
        *, n_samples, n_features, n_categorical, n_informative, random_state):
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           n_informative=n_informative, n_targets=1,
                           random_state=random_state)
    X, y = X.astype(np.float32), y.astype(np.float32)

    # Turn some columns into categorical, by taking quartiles
    X = pd.DataFrame({f'f{i}': X[:, i] for i in range(n_features)})
    for i in range(n_categorical):
        column = f'f{i}'
        X[column] = pd.qcut(X[column], 4, labels=range(4))
    return X, y


def count_categorical_split(tl_model):
    model_dump = json.loads(tl_model.dump_as_json(pretty_print=False))
    count = 0
    for tree in model_dump["trees"]:
        for node in tree["nodes"]:
            if "split_type" in node and node["split_type"] == "categorical":
                count += 1
    return count


@pytest.mark.parametrize('objective', ['reg:linear', 'reg:squarederror',
                                       'reg:squaredlogerror',
                                       'reg:pseudohubererror'])
@pytest.mark.skipif(not has_xgboost(), reason="need to install xgboost")
@pytest.mark.skipif(not has_shap(), reason="need to install shap")
@pytest.mark.skipif(not has_sklearn(), reason="need to install scikit-learn")
def test_xgb_regressor(objective):
    n_samples = 100
    X, y = make_regression(n_samples=n_samples, n_features=8, n_informative=8,
                           n_targets=1, random_state=2021)
    # Ensure that the label exceeds -1
    y += (-0.5) - np.min(y)
    assert np.all(y > -1)
    X, y = X.astype(np.float32), y.astype(np.float32)
    dtrain = xgb.DMatrix(X, label=y)
    params = {'objective': objective, 'base_score': 0.5, 'seed': 0,
              'max_depth': 6, 'tree_method': 'gpu_hist',
              'predictor': 'gpu_predictor'}
    num_round = 10
    xgb_model = xgb.train(params, dtrain, num_boost_round=num_round,
                          evals=[(dtrain, 'train')])
    tl_model = treelite.Model.from_xgboost(xgb_model)

    # Insert NaN randomly into X
    X_test = X.copy()
    n_nan = int(np.floor(X.size * 0.1))
    rng = np.random.default_rng(seed=0)
    index_nan = rng.choice(X.size, size=n_nan, replace=False)
    X_test.ravel()[index_nan] = np.nan

    explainer = TreeExplainer(model=tl_model)
    out = explainer.shap_values(X_test)

    ref_explainer = shap.explainers.Tree(model=xgb_model)
    correct_out = ref_explainer.shap_values(X_test)
    np.testing.assert_almost_equal(out, correct_out, decimal=5)
    np.testing.assert_almost_equal(explainer.expected_value,
                                   ref_explainer.expected_value, decimal=5)


@pytest.mark.parametrize('objective,n_classes',
                         [('binary:logistic', 2),
                          ('binary:hinge', 2),
                          ('binary:logitraw', 2),
                          ('count:poisson', 4),
                          ('rank:pairwise', 5),
                          ('rank:ndcg', 5),
                          ('rank:map', 5),
                          ('multi:softmax', 5),
                          ('multi:softprob', 5)],
                         ids=['binary:logistic', 'binary:hinge',
                              'binary:logitraw', 'count:poisson',
                              'rank:pairwise', 'rank:ndcg', 'rank:map',
                              'multi:softmax', 'multi:softprob'])
@pytest.mark.skipif(not has_xgboost(), reason="need to install xgboost")
@pytest.mark.skipif(not has_shap(), reason="need to install shap")
@pytest.mark.skipif(not has_sklearn(), reason="need to install scikit-learn")
def test_xgb_classifier(objective, n_classes):
    n_samples = 100
    X, y = make_classification(n_samples=n_samples, n_features=8,
                               n_informative=8, n_redundant=0, n_repeated=0,
                               n_classes=n_classes, random_state=2021)
    X, y = X.astype(np.float32), y.astype(np.float32)
    num_round = 10
    dtrain = xgb.DMatrix(X, label=y)
    params = {'objective': objective, 'base_score': 0.5, 'seed': 0,
              'max_depth': 6, 'tree_method': 'gpu_hist',
              'predictor': 'gpu_predictor'}
    if objective.startswith('rank:'):
        dtrain.set_group([10] * 10)
    if n_classes > 2 and objective.startswith('multi:'):
        params['num_class'] = n_classes
    xgb_model = xgb.train(params, dtrain=dtrain, num_boost_round=num_round)

    # Insert NaN randomly into X
    X_test = X.copy()
    n_nan = int(np.floor(X.size * 0.1))
    rng = np.random.default_rng(seed=0)
    index_nan = rng.choice(X.size, size=n_nan, replace=False)
    X_test.ravel()[index_nan] = np.nan

    explainer = TreeExplainer(model=xgb_model)
    out = explainer.shap_values(X_test)

    ref_explainer = shap.explainers.Tree(model=xgb_model)
    correct_out = ref_explainer.shap_values(X_test)
    np.testing.assert_almost_equal(out, correct_out, decimal=5)
    np.testing.assert_almost_equal(explainer.expected_value,
                                   ref_explainer.expected_value, decimal=5)


def test_degenerate_cases():
    n_samples = 100
    cuml_model = curfr(max_features=1.0, max_samples=0.1, n_bins=128,
                       min_samples_leaf=2, random_state=123,
                       n_streams=1, n_estimators=10, max_leaves=-1,
                       max_depth=16, accuracy_metric="mse")
    # Attempt to import un-fitted model
    with pytest.raises(NotFittedError):
        TreeExplainer(model=cuml_model)

    # Depth 0 trees
    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal(size=(n_samples, 8), dtype=np.float32)
    y = np.ones(shape=(n_samples,), dtype=np.float32)
    cuml_model.fit(X, y)
    explainer = TreeExplainer(model=cuml_model)
    out = explainer.shap_values(X)
    # Since the output is always 1.0 no matter the input, SHAP values for all
    # features are zero, as feature values don't have any effect on the output.
    # The bias (expected_value) is 1.0.
    assert np.all(out == 0)
    assert explainer.expected_value == 1.0


@pytest.mark.parametrize('input_type', ['numpy', 'cupy', 'cudf'])
@pytest.mark.skipif(not has_sklearn(), reason="need to install scikit-learn")
def test_cuml_rf_regressor(input_type):
    n_samples = 100
    X, y = make_regression(n_samples=n_samples, n_features=8, n_informative=8,
                           n_targets=1, random_state=2021)
    X, y = X.astype(np.float32), y.astype(np.float32)
    if input_type == 'cupy':
        X, y = cp.array(X), cp.array(y)
    elif input_type == 'cudf':
        X, y = cudf.DataFrame(X), cudf.Series(y)
    cuml_model = curfr(max_features=1.0, max_samples=0.1, n_bins=128,
                       min_samples_leaf=2, random_state=123,
                       n_streams=1, n_estimators=10, max_leaves=-1,
                       max_depth=16, accuracy_metric="mse")
    cuml_model.fit(X, y)
    pred = cuml_model.predict(X)

    explainer = TreeExplainer(model=cuml_model)
    out = explainer.shap_values(X)
    if input_type == 'cupy':
        pred = pred.get()
        out = out.get()
        expected_value = explainer.expected_value.get()
    elif input_type == 'cudf':
        pred = pred.to_numpy()
        out = out.get()
        expected_value = explainer.expected_value.get()
    else:
        expected_value = explainer.expected_value
    # SHAP values should add up to predicted score
    shap_sum = np.sum(out, axis=1) + expected_value
    np.testing.assert_almost_equal(shap_sum, pred, decimal=4)


@pytest.mark.parametrize('input_type', ['numpy', 'cupy', 'cudf'])
@pytest.mark.parametrize('n_classes', [2, 5])
@pytest.mark.skipif(not has_sklearn(), reason="need to install scikit-learn")
def test_cuml_rf_classifier(n_classes, input_type):
    n_samples = 100
    X, y = make_classification(n_samples=n_samples, n_features=8,
                               n_informative=8, n_redundant=0, n_repeated=0,
                               n_classes=n_classes, random_state=2021)
    X, y = X.astype(np.float32), y.astype(np.float32)
    if input_type == 'cupy':
        X, y = cp.array(X), cp.array(y)
    elif input_type == 'cudf':
        X, y = cudf.DataFrame(X), cudf.Series(y)
    cuml_model = curfc(max_features=1.0, max_samples=0.1, n_bins=128,
                       min_samples_leaf=2, random_state=123,
                       n_streams=1, n_estimators=10, max_leaves=-1,
                       max_depth=16, accuracy_metric="mse")
    cuml_model.fit(X, y)
    pred = cuml_model.predict_proba(X)

    explainer = TreeExplainer(model=cuml_model)
    out = explainer.shap_values(X)
    if input_type == 'cupy':
        pred = pred.get()
        out = out.get()
        expected_value = explainer.expected_value.get()
    elif input_type == 'cudf':
        pred = pred.to_numpy()
        out = out.get()
        expected_value = explainer.expected_value.get()
    else:
        expected_value = explainer.expected_value
    # SHAP values should add up to predicted score
    expected_value = expected_value.reshape(-1, 1)
    shap_sum = np.sum(out, axis=2) + np.tile(expected_value, (1, n_samples))
    pred = np.transpose(pred, (1, 0))
    np.testing.assert_almost_equal(shap_sum, pred, decimal=4)


@pytest.mark.skipif(not has_shap(), reason="need to install shap")
@pytest.mark.skipif(not has_sklearn(), reason="need to install scikit-learn")
def test_sklearn_rf_regressor():
    n_samples = 100
    X, y = make_regression(n_samples=n_samples, n_features=8, n_informative=8,
                           n_targets=1, random_state=2021)
    X, y = X.astype(np.float32), y.astype(np.float32)
    skl_model = sklrfr(max_features=1.0, max_samples=0.1,
                       min_samples_leaf=2, random_state=123,
                       n_estimators=10, max_depth=16)
    skl_model.fit(X, y)

    explainer = TreeExplainer(model=skl_model)
    out = explainer.shap_values(X)

    ref_explainer = shap.explainers.Tree(model=skl_model)
    correct_out = ref_explainer.shap_values(X)
    np.testing.assert_almost_equal(out, correct_out, decimal=5)
    np.testing.assert_almost_equal(explainer.expected_value,
                                   ref_explainer.expected_value, decimal=5)


@pytest.mark.parametrize('n_classes', [2, 3, 5])
@pytest.mark.skipif(not has_shap(), reason="need to install shap")
@pytest.mark.skipif(not has_sklearn(), reason="need to install scikit-learn")
def test_sklearn_rf_classifier(n_classes):
    n_samples = 100
    X, y = make_classification(n_samples=n_samples, n_features=8,
                               n_informative=8, n_redundant=0, n_repeated=0,
                               n_classes=n_classes, random_state=2021)
    X, y = X.astype(np.float32), y.astype(np.float32)
    skl_model = sklrfc(max_features=1.0, max_samples=0.1,
                       min_samples_leaf=2, random_state=123,
                       n_estimators=10, max_depth=16)
    skl_model.fit(X, y)

    explainer = TreeExplainer(model=skl_model)
    out = explainer.shap_values(X)

    ref_explainer = shap.explainers.Tree(model=skl_model)
    correct_out = np.array(ref_explainer.shap_values(X))
    expected_value = ref_explainer.expected_value
    if n_classes == 2:
        correct_out = correct_out[1, :, :]
        expected_value = expected_value[1:]
    np.testing.assert_almost_equal(out, correct_out, decimal=5)
    np.testing.assert_almost_equal(explainer.expected_value,
                                   expected_value, decimal=5)


@pytest.mark.skipif(not has_xgboost(), reason="need to install xgboost")
def test_xgb_toy_categorical():
    X = pd.DataFrame({'dummy': np.zeros(5, dtype=np.float32),
                      'x': np.array([0, 1, 2, 3, 4], dtype=np.int32)})
    y = np.array([0, 0, 1, 1, 1], dtype=np.float32)
    X['x'] = X['x'].astype("category")
    dtrain = xgb.DMatrix(X, y, enable_categorical=True)
    params = {"tree_method": "gpu_hist", "eval_metric": "error",
              "objective": "binary:logistic", "max_depth": 2,
              "min_child_weight": 0, "lambda": 0}
    xgb_model = xgb.train(params, dtrain, num_boost_round=1,
                          evals=[(dtrain, 'train')])
    explainer = TreeExplainer(model=xgb_model)
    out = explainer.shap_values(X)

    ref_out = xgb_model.predict(dtrain, pred_contribs=True)
    np.testing.assert_almost_equal(out, ref_out[:, :-1], decimal=5)
    np.testing.assert_almost_equal(explainer.expected_value, ref_out[0, -1],
                                   decimal=5)


@pytest.mark.parametrize('n_classes', [2, 3])
@pytest.mark.skipif(not has_xgboost(), reason="need to install xgboost")
@pytest.mark.skipif(not has_sklearn(), reason="need to install scikit-learn")
def test_xgb_classifier_with_categorical(n_classes):
    n_samples = 100
    n_features = 8
    X, y = make_classification_with_categorical(
            n_samples=n_samples, n_features=n_features, n_categorical=4,
            n_informative=n_features, n_redundant=0, n_repeated=0,
            n_classes=n_classes, random_state=2022)

    dtrain = xgb.DMatrix(X, y, enable_categorical=True)
    params = {"tree_method": "gpu_hist", "max_depth": 6,
              "base_score": 0.5, "seed": 0, "predictor": "gpu_predictor"}
    if n_classes == 2:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    else:
        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"
        params["num_class"] = n_classes
    xgb_model = xgb.train(params, dtrain, num_boost_round=10,
                          evals=[(dtrain, 'train')])
    assert count_categorical_split(treelite.Model.from_xgboost(xgb_model)) > 0

    # Insert NaN randomly into X
    X_test = X.values.copy()
    n_nan = int(np.floor(X.size * 0.1))
    rng = np.random.default_rng(seed=0)
    index_nan = rng.choice(X.size, size=n_nan, replace=False)
    X_test.ravel()[index_nan] = np.nan

    explainer = TreeExplainer(model=xgb_model)
    out = explainer.shap_values(X_test)

    dtest = xgb.DMatrix(X_test)
    ref_out = xgb_model.predict(dtest, pred_contribs=True,
                                validate_features=False)
    if n_classes == 2:
        ref_out, ref_expected_value = ref_out[:, :-1], ref_out[0, -1]
    else:
        ref_out = ref_out.transpose((1, 0, 2))
        ref_out, ref_expected_value = ref_out[:, :, :-1], ref_out[:, 0, -1]
    np.testing.assert_almost_equal(out, ref_out, decimal=5)
    np.testing.assert_almost_equal(explainer.expected_value,
                                   ref_expected_value, decimal=5)


@pytest.mark.skipif(not has_xgboost(), reason="need to install xgboost")
@pytest.mark.skipif(not has_sklearn(), reason="need to install scikit-learn")
def test_xgb_regressor_with_categorical():
    n_samples = 100
    n_features = 8
    X, y = make_regression_with_categorical(
            n_samples=n_samples, n_features=n_features, n_categorical=4,
            n_informative=n_features, random_state=2022)

    dtrain = xgb.DMatrix(X, y, enable_categorical=True)
    params = {"tree_method": "gpu_hist", "max_depth": 6,
              "base_score": 0.5, "seed": 0, "predictor": "gpu_predictor",
              "objective": "reg:squarederror", "eval_metric": "rmse"}
    xgb_model = xgb.train(params, dtrain, num_boost_round=10,
                          evals=[(dtrain, 'train')])
    assert count_categorical_split(treelite.Model.from_xgboost(xgb_model)) > 0

    explainer = TreeExplainer(model=xgb_model)
    out = explainer.shap_values(X)

    ref_out = xgb_model.predict(dtrain, pred_contribs=True)
    ref_out, ref_expected_value = ref_out[:, :-1], ref_out[0, -1]
    np.testing.assert_almost_equal(out, ref_out, decimal=5)
    np.testing.assert_almost_equal(explainer.expected_value,
                                   ref_expected_value, decimal=5)


@pytest.mark.skipif(not has_lightgbm(), reason="need to install lightgbm")
@pytest.mark.skipif(not has_sklearn(), reason="need to install scikit-learn")
@pytest.mark.skipif(not has_shap(), reason="need to install shap")
def test_lightgbm_regressor_with_categorical():
    n_samples = 100
    n_features = 8
    n_categorical = 8
    X, y = make_regression_with_categorical(
            n_samples=n_samples, n_features=n_features,
            n_categorical=n_categorical, n_informative=n_features,
            random_state=2022)

    dtrain = lgb.Dataset(X, label=y, categorical_feature=range(n_categorical))
    params = {"num_leaves": 64, "seed": 0, "objective": "regression",
              "metric": "rmse", "min_data_per_group": 1}
    lgb_model = lgb.train(params, dtrain, num_boost_round=10,
                          valid_sets=[dtrain], valid_names=['train'])
    assert count_categorical_split(treelite.Model.from_lightgbm(lgb_model)) > 0

    explainer = TreeExplainer(model=lgb_model)
    out = explainer.shap_values(X)

    ref_explainer = shap.explainers.Tree(model=lgb_model)
    ref_out = ref_explainer.shap_values(X)
    np.testing.assert_almost_equal(out, ref_out, decimal=5)
    np.testing.assert_almost_equal(explainer.expected_value,
                                   ref_explainer.expected_value, decimal=5)


@pytest.mark.parametrize('n_classes', [2, 3])
@pytest.mark.skipif(not has_lightgbm(), reason="need to install lightgbm")
@pytest.mark.skipif(not has_sklearn(), reason="need to install scikit-learn")
@pytest.mark.skipif(not has_shap(), reason="need to install shap")
def test_lightgbm_classifier_with_categorical(n_classes):
    n_samples = 100
    n_features = 8
    n_categorical = 8
    X, y = make_classification_with_categorical(
            n_samples=n_samples, n_features=n_features,
            n_categorical=n_categorical, n_informative=n_features,
            n_redundant=0, n_repeated=0, n_classes=n_classes,
            random_state=2022)

    dtrain = lgb.Dataset(X, label=y, categorical_feature=range(n_categorical))
    params = {"num_leaves": 64, "seed": 0, "min_data_per_group": 1}
    if n_classes == 2:
        params["objective"] = "binary"
        params["metric"] = "binary_logloss"
    else:
        params["objective"] = "multiclass"
        params["metric"] = "multi_logloss"
        params["num_class"] = n_classes
    lgb_model = lgb.train(params, dtrain, num_boost_round=10,
                          valid_sets=[dtrain], valid_names=['train'])
    assert count_categorical_split(treelite.Model.from_lightgbm(lgb_model)) > 0

    # Insert NaN randomly into X
    X_test = X.values.copy()
    n_nan = int(np.floor(X.size * 0.1))
    rng = np.random.default_rng(seed=0)
    index_nan = rng.choice(X.size, size=n_nan, replace=False)
    X_test.ravel()[index_nan] = np.nan

    explainer = TreeExplainer(model=lgb_model)
    out = explainer.shap_values(X_test)

    ref_explainer = shap.explainers.Tree(model=lgb_model)
    ref_out = np.array(ref_explainer.shap_values(X_test))
    if n_classes == 2:
        ref_out = ref_out[1, :, :]
        ref_expected_value = ref_explainer.expected_value[1]
    else:
        ref_expected_value = ref_explainer.expected_value
    np.testing.assert_almost_equal(out, ref_out, decimal=5)
    np.testing.assert_almost_equal(explainer.expected_value,
                                   ref_expected_value, decimal=5)
