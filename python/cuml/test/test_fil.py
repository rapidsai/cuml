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

from cuml import FIL as fil
from cuml.utils.import_utils import has_treelite, has_xgboost

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

if has_treelite():
    import treelite as tl

if has_xgboost():
    import xgboost as xgb


    def simulate_data(m, n, k=2, random_state=None, classification=True):
        if classification:
            features, labels = make_classification(n_samples=m, n_features=n,
                                                   n_informative=int(n/5), n_classes=k,
                                                  random_state=random_state)
        else:
            features, labels = make_regression(n_samples=m, n_features=n,
                                               n_informative=int(n/5), n_targets=1,
                                               random_state=random_state)
        return np.c_[labels, features].astype(np.float32)


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


@pytest.mark.parametrize('n_rows', [unit_param(100), quality_param(1000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_columns', [unit_param(30), quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('n_info', [unit_param(20), quality_param(50),
                         stress_param(500)])
@pytest.mark.parametrize('num_round', [unit_param(20), quality_param(50),
                         stress_param(90)])
@pytest.mark.skipif(has_treelite() is False, reason="need to install treelite")
@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
def test_fil(n_rows, n_columns, n_info, num_round):


    # settings
    simulate = True
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

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalidation = xgb.DMatrix(X_validation, label=y_validation)

    # instantiate params
    params = {}

    # general params
    general_params = {'silent': 1}
    params.update(general_params)

    # booster params
    n_gpus = 0  # change this to -1 to use all GPUs available or 0 to use the CPU

    #classification=1
    # learning task params
    learning_task_params = {}
    learning_task_params['eval_metric'] = 'error'
    learning_task_params['objective'] = 'binary:logistic'
    params.update(learning_task_params)

    # model training settings
    evallist = [(dvalidation, 'validation'), (dtrain, 'train')]
    num_round = 5

    bst = xgb.train(params, dtrain, num_round, evallist)

    bst.save_model('xgb.model')
    xgb_preds = bst.predict(dvalidation)

    xgb_err = sum(1 for i in range(len(xgb_preds))
                  if xgb_preds[i] != y_validation[i]) / float(len(xgb_preds))
    print(" read the saved xgb modle")
    tl_model = tl.Model.from_xgboost(bst)
    #tl_copy = copy.deepcopy(tl_model)
    fm = fil()
    forest = fm.from_treelite(tl_model, output_class=True, algo=1, threshold=0.5)
    preds_1 = fm.predict(X_validation).to_array()
    fil_acc = accuracy_score(y_validation, preds_1)
    assert fil_acc > 0.8
