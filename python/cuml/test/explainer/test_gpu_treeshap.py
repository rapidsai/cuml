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
from cuml.explainer.tree_shap import TreeExplainer
from cuml.common.import_utils import has_xgboost
from cuml.common.import_utils import has_lightgbm
from cuml.ensemble import RandomForestClassifier as curfc
from cuml.ensemble import RandomForestRegressor as curfr
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.datasets import make_classification

if has_xgboost():
    import xgboost as xgb

@pytest.mark.parametrize('objective', ['reg:linear', 'reg:squarederror',
                                       'reg:squaredlogerror',
                                       'reg:pseudohubererror'])
@pytest.mark.skipif(not has_xgboost(), reason="need to install xgboost")
def test_xgb_regressor(objective):
    X, y = fetch_california_housing(return_X_y=True)
    dtrain = xgb.DMatrix(X, label=y)
    param = {'max_depth': 8, 'eta': 0.1, 'objective': objective}
    num_round = 10
    xgb_model = xgb.train(param, dtrain, num_boost_round=num_round,
                          evals=[(dtrain, 'train')])
    tl_model = treelite.Model.from_xgboost(xgb_model)

    explainer = TreeExplainer(model=tl_model)
    out = explainer.shap_values(X)
    correct_out = xgb_model.predict(dtrain, pred_contribs=True)
    np.testing.assert_almost_equal(out, correct_out, decimal=5)

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
def test_xgb_classifier(objective, n_classes):
    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, n_features=8,
                               n_informative=8, n_redundant=0, n_repeated=0,
                               n_classes=n_classes, random_state=2021)
    X, y = X.astype(np.float32), y.astype(np.float32)
    num_round = 10
    dtrain = xgb.DMatrix(X, label=y)
    params = {'objective': objective, 'base_score': 0.5, 'seed': 0,
              'max_depth': 6}
    if objective.startswith('rank:'):
        dtrain.set_group([10] * 100)
    if n_classes > 2 and objective.startswith('multi:'):
        params['num_class'] = n_classes
    xgb_model = xgb.train(params, dtrain=dtrain, num_boost_round=num_round)
    tl_model = treelite.Model.from_xgboost(xgb_model)

    explainer = TreeExplainer(model=tl_model)
    out = explainer.shap_values(X)
    correct_out = xgb_model.predict(dtrain, pred_contribs=True)
    np.testing.assert_almost_equal(out, correct_out, decimal=5)

def test_cuml_rf_regressor():
    X, y = fetch_california_housing(return_X_y=True)
    X, y = X.astype(np.float32), y.astype(np.float32)
    cuml_model = curfr(max_features=1.0, max_samples=0.1, n_bins=128,
                       min_samples_leaf=2, random_state=123,
                       n_streams=1, n_estimators=10, max_leaves=-1,
                       max_depth=16, accuracy_metric="mse")
    cuml_model.fit(X, y)
    pred = cuml_model.predict(X)
    tl_model = cuml_model.convert_to_treelite_model()

    explainer = TreeExplainer(model=tl_model)
    out = explainer.shap_values(X)
    # SHAP values should add up to predicted score
    np.testing.assert_almost_equal(np.sum(out, axis=1), pred, decimal=5)
