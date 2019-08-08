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
from cuml.utils.import_utils import has_xgboost

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

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


def _build_and_save_xgboost(model_path, X_train, y_train):
    """Trains a small xgboost classifier and saves it to model_path"""
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # instantiate params
    params = {'silent': 1}

    # learning task params
    params['eval_metric'] = 'error'
    params['objective'] = 'binary:logistic'

    # model training settings
    num_round = 5

    bst = xgb.train(params, dtrain, num_round)
    bst.save_model(model_path)
    return bst

@pytest.mark.parametrize('n_rows', [unit_param(100), quality_param(1000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_columns', [unit_param(30), quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('n_info', [unit_param(20), quality_param(50),
                         stress_param(500)])
@pytest.mark.parametrize('num_round', [unit_param(20), quality_param(50),
                         stress_param(90)])
@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
def test_fil(n_rows, n_columns, n_info, num_round, tmp_path):
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

    model_path = os.path.join(tmp_path, 'xgb.model')

    bst = _build_and_save_xgboost(model_path, X_train, y_train)

    dvalidation = xgb.DMatrix(X_validation, label=y_validation)
    xgb_preds = bst.predict(dvalidation)

    xgb_acc = accuracy_score(y_validation, xgb_preds > 0.5)

    print("Reading the saved xgb model")
    fm = FIL.from_treelite_file(model_path,
                                algo=0,
                                output_class=True,
                                threshold=0.50)
    preds_1 = np.asarray(fm.predict(X_validation))
    fil_acc = accuracy_score(y_validation, preds_1)

    print("XGB accuracy = ", xgb_acc, " FIL accuracy: ", fil_acc)
    assert fil_acc == pytest.approx(xgb_acc, 0.01)
    assert fil_acc > 0.80
