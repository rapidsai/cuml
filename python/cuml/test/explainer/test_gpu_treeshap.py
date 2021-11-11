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
from sklearn.model_selection import train_test_split

if has_xgboost():
    import xgboost as xgb

@pytest.mark.parametrize('objective', ['reg:linear', 'reg:squarederror',
                                       'reg:squaredlogerror',
                                       'reg:pseudohubererror'])
@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
def test_xgb_regressor(objective):
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        shuffle=True,
                                                        random_state=0)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 8, 'eta': 0.1, 'objective': objective}
    num_round = 10
    xgb_model = xgb.train(param, dtrain, num_boost_round=num_round,
                          evals=[(dtrain, 'train'), (dtest, 'test')])
    tl_model = treelite.Model.from_xgboost(xgb_model)

    explainer = TreeExplainer(model=tl_model)
    out = explainer.shap_values(X_test)
    correct_out = xgb_model.predict(dtest, pred_contribs=True)
    np.testing.assert_almost_equal(out, correct_out, decimal=5)

@pytest.mark.parametrize('objective,max_label',
                         [('binary:logistic', 2),
                          ('binary:hinge', 2),
                          ('binary:logitraw', 2),
                          ('count:poisson', 4),
                          ('rank:pairwise', 5),
                          ('rank:ndcg', 5),
                          ('rank:map', 5)],
                         ids=['binary:logistic', 'binary:hinge',
                              'binary:logitraw', 'count:poisson',
                              'rank:pairwise', 'rank:ndcg', 'rank:map'])
def test_xgb_binary_classifier(objective, max_label):
    nrow = 16
    ncol = 8
    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal(size=(nrow, ncol), dtype=np.float32)
    y = rng.integers(0, max_label, size=nrow)
    assert np.min(y) == 0
    assert np.max(y) == max_label - 1

    num_round = 4
    dtrain = xgb.DMatrix(X, label=y)
    if objective.startswith('rank:'):
        dtrain.set_group([nrow])
    xgb_model = xgb.train({'objective': objective, 'base_score': 0.5,
                           'seed': 0},
                          dtrain=dtrain, num_boost_round=num_round)
    tl_model = treelite.Model.from_xgboost(xgb_model)

    explainer = TreeExplainer(model=tl_model)
    out = explainer.shap_values(X)
    correct_out = xgb_model.predict(dtrain, pred_contribs=True)
    np.testing.assert_almost_equal(out, correct_out)

@pytest.mark.parametrize('objective', ['multi:softmax', 'multi:softprob'])
def test_xgb_multiclass_classifier(objective):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
            shuffle=True, random_state=0)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 6, 'eta': 0.1, 'objective': objective,
             'num_class': 3, 'eval_metric': 'mlogloss',
             'predictor': 'gpu_predictor'}
    num_round = 10

    xgb_model = xgb.train(param, dtrain, num_boost_round=num_round,
                          evals=[(dtrain, 'train'), (dtest, 'test')])
    tl_model = treelite.Model.from_xgboost(xgb_model)

    explainer = TreeExplainer(model=tl_model)
    out = explainer.shap_values(X_test)
    correct_out = xgb_model.predict(dtest, pred_contribs=True)
    np.testing.assert_almost_equal(out, correct_out)

def test_cuml_rf_classifier():
    X, y = load_iris(return_X_y=True)
    X, y = X.astype(np.float32), y.astype(np.int32)
    cuml_model = curfc(max_features=1.0, max_samples=0.1, n_bins=128,
                       min_samples_leaf=2, random_state=123,
                       n_streams=1, n_estimators=10, max_leaves=-1,
                       max_depth=16, accuracy_metric="mse")
    cuml_model.fit(X, y)
    tl_model = cuml_model.convert_to_treelite_model()
    print(dir(tl_model))

    explainer = TreeExplainer(model=tl_model)
    out = explainer.shap_values(X)
    print(out)
    print(out.shape)
