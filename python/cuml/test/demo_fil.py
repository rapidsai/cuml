
from cuml import fil
import cuml
import xgboost as xgb
import numpy as np, pandas as pd
from sklearn import metrics
import treelite as tl

# tl = fil.treelite_from_file("xgb.model")
# res = fil.from_treelite_direct(tl)

# model = fil.fil_from_xgboost("xgb.model")
# print("got handle")

def build_model_and_data(n_rows, n_columns, fname, do_fit=True):
    from sklearn import datasets
    random_state = np.random.RandomState(43210)
    X, y = datasets.make_classification(n_samples=n_rows, n_features=n_columns,
                               n_informative=int(n_columns/5),
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

    if do_fit:
        bst = xgb.train(params, dtrain, num_round, evallist)
        bst.save_model(fname)
        preds_xgboost = bst.predict(dvalidation)
        print("xgboost accuracy: ",
              metrics.accuracy_score(y_validation, preds_xgboost > 0.5))
    else:
        bst = None

    return X_validation, y_validation, bst

#
# Using cuml cython wrapper
#
MODEL_NAME = 'demo1.model'
X_test, y_test, bst = build_model_and_data(10000, 11, MODEL_NAME, do_fit=True)
cuml_model = fil.TreeliteModel.from_filename(MODEL_NAME)

print("got model with %d trees, %d features" % (
    cuml_model.num_trees, cuml_model.num_features))
cuml_fm = fil.FIL()
cuml_fm.from_treelite(cuml_model, True, 0, 0.50)

cuml_fm2 = fil.load_fil_from_treelite_file(MODEL_NAME)


print("loaded from cuml model")
cuml_preds = cuml_fm.predict(X_test.astype(np.float32))
cuml_preds2 = cuml_fm2.predict(X_test.astype(np.float32))
print("cuML Accuracy1 = ", metrics.accuracy_score(y_test, np.asarray(cuml_preds)))
print("cuML Accuracy2 = ", metrics.accuracy_score(y_test, np.asarray(cuml_preds2)))

#
# Using treelite python package
#
# mod = tl.Model(handle=None)
# tl_model = mod.load(MODEL_NAME, 'xgboost')
# tl_fm = fil.FIL()
# tl_fm.from_treelite(tl_model, output_class=True,
#                     algo=0, threshold=0.50)
# tl_preds = tl_fm.predict(X_test.astype(np.float32))

# print("Loaded from treelite python and predicted")
# print("TL Accuracy = ", metrics.accuracy_score(y_test, np.asarray(tl_preds)))

