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
import xgboost as xgb

from cuml import FIL as fil
from cuml.utils.import_utils import has_treelite

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

if has_treelite():
    import treelite as tl


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


@pytest.mark.parametrize('n_rows', [unit_param(100), quality_param(1000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_columns', [unit_param(50), quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('n_info', [unit_param(30), quality_param(50),
                         stress_param(500)])
@pytest.mark.parametrize('num_round', [unit_param(20), quality_param(50),
                         stress_param(90)])
@pytest.mark.skipif(has_treelite() is False, reason="need to install treelite")
def test_fil(n_rows, n_columns, n_info, num_round):
    random_state = np.random.RandomState(43210)
    X, y = make_classification(n_samples=n_rows, n_features=n_columns,
                               n_informative=int(n_columns/2),
                               n_classes=2,
                               random_state=random_state)

    # identify shape and indices
    n_rows, n_columns = X.shape
    train_size = 0.80
    train_index = int(n_rows * train_size)

    # split train data
    X_train, y_train = X[:train_index, :], y[:train_index]

    # split validation data
    X_validation, y_validation = X[train_index:, :], y[train_index:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalidation = xgb.DMatrix(X_validation, label=y_validation)

    # instantiate params
    params = {}

    # general params
    general_params = {'silent': 1}
    params.update(general_params)

    # booster params
    booster_params = {}
    params.update(booster_params)

    # learning task params
    learning_task_params = {}
    learning_task_params['eval_metric'] = 'auc'
    learning_task_params['objective'] = 'binary:logistic'

    params.update(learning_task_params)

    # model training settings
    evallist = [(dvalidation, 'validation'), (dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_round, evallist)
    bst.save_model('xgb.model')

    # using treelite for prediction
    tl_model = tl.Model.load('xgb.model', 'xgboost')
    fm = fil()
    fm.from_treelite(tl_model, output_class=True,
                     algo=2, threshold=0.55)
    preds = fm.predict(X_validation).to_array()
    fil_acc = accuracy_score(y_validation, preds)
    assert fil_acc > 0.5