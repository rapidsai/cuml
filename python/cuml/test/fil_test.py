import numpy as np
import pytest
from cuml.test.utils import get_handle
from cuml.utils.import_utils import has_treelite
import xgboost as xgb
from cuml import FIL as fil
from sklearn.datasets import make_classification, make_regression

if has_treelite():
    import treelite as tl

def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


@pytest.mark.parametrize('n_rows', [unit_param(1000), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_columns', [unit_param(50), quality_param(100),
                         stress_param(1000)])
@pytest.mark.parametrize('n_info', [unit_param(30), unit_param(35), quality_param(50),
                         stress_param(500)])
@pytest.mark.skipif(has_treelite() is False, reason="need to install treelite")
def test_fil(n_rows, n_columns, n_info):
    random_state = np.random.RandomState(43210)
    X, y = make_classification(n_samples=n_rows, n_features=n_columns,
                               n_informative=int(n_columns/5),
                               n_classes=2,
                               random_state=random_state)

    # identify shape and indices
    n_rows, n_columns = X.shape
    train_size = 0.80
    train_index = int(n_rows * train_size)

    # split X, y


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
    # change this to -1 to use all GPUs available or 0 to use the CPU
    n_gpus = 0
    booster_params = {}
    params.update(booster_params)

    # learning task params
    learning_task_params = {}
    learning_task_params['eval_metric'] = 'auc'
    learning_task_params['objective'] = 'binary:logistic'

    params.update(learning_task_params)

    # model training settings
    evallist = [(dvalidation, 'validation'), (dtrain, 'train')]
    num_round = 5

    bst = xgb.train(params, dtrain, num_round, evallist)
    bst.save_model('xgb.model')
    preds_xgboost = bst.predict(dvalidation, output_margin=True)
    # using treelite for prediction
    handle, stream = get_handle(True)
    mod = tl.Model(handle=None)
    tl_model = mod.load('xgb.model', 'xgboost')
    fm = fil(algo=0, threshold=0.55)
    fm.from_treelite(tl_model, output_class=True)
    preds = fm.predict(X_validation)
    
    print(preds)
    print(preds_xgboost)

