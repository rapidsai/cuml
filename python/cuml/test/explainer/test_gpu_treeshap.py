#
# Copyright (c) 2021, NVIDIA CORPORATION.
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
import treelite
import numpy as np
import cupy as cp
import cudf
from cuml.experimental.explainer.tree_shap import TreeExplainer
from cuml.common.import_utils import has_xgboost, has_shap
from cuml.common.exceptions import NotFittedError
from cuml.ensemble import RandomForestRegressor as curfr
from cuml.ensemble import RandomForestClassifier as curfc
from sklearn.datasets import make_regression, make_classification

if has_xgboost():
    import xgboost as xgb
if has_shap():
    import shap


@pytest.mark.parametrize('objective', ['reg:linear', 'reg:squarederror',
                                       'reg:squaredlogerror',
                                       'reg:pseudohubererror'])
@pytest.mark.skipif(not has_xgboost(), reason="need to install xgboost")
@pytest.mark.skipif(not has_shap(), reason="need to install shap")
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

    explainer = TreeExplainer(model=tl_model)
    out = explainer.shap_values(X)

    ref_explainer = shap.TreeExplainer(model=xgb_model)
    correct_out = ref_explainer.shap_values(X)
    np.testing.assert_almost_equal(out, correct_out)
    np.testing.assert_almost_equal(explainer.expected_value,
                                   ref_explainer.expected_value)


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

    explainer = TreeExplainer(model=xgb_model)
    out = explainer.shap_values(X)

    ref_explainer = shap.TreeExplainer(model=xgb_model)
    correct_out = ref_explainer.shap_values(X)
    np.testing.assert_almost_equal(out, correct_out)
    np.testing.assert_almost_equal(explainer.expected_value,
                                   ref_explainer.expected_value)


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
    # SHAP values should add up to predicted score
    shap_sum = np.sum(out, axis=1) + explainer.expected_value
    if input_type == 'cupy':
        pred = pred.get()
        shap_sum = shap_sum.get()
    elif input_type == 'cudf':
        pred = pred.to_numpy()
        shap_sum = shap_sum.get()
    np.testing.assert_almost_equal(shap_sum, pred, decimal=4)


@pytest.mark.parametrize('n_classes', [2, 5])
def test_cuml_rf_classifier(n_classes):
    n_samples = 100
    X, y = make_classification(n_samples=n_samples, n_features=8,
                               n_informative=8, n_redundant=0, n_repeated=0,
                               n_classes=n_classes, random_state=2021)
    X, y = X.astype(np.float32), y.astype(np.float32)
    cuml_model = curfc(max_features=1.0, max_samples=0.1, n_bins=128,
                       min_samples_leaf=2, random_state=123,
                       n_streams=1, n_estimators=10, max_leaves=-1,
                       max_depth=16, accuracy_metric="mse")
    cuml_model.fit(X, y)

    with pytest.raises(RuntimeError):
        # cuML RF classifier is not supported yet
        explainer = TreeExplainer(model=cuml_model)
        explainer.shap_values(X)
