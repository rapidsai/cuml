import numpy as np
import treelite as tl
import xgboost as xgb

from cuml import FIL as fil

from sklearn.datasets import make_classification, make_regression


# helper function for simulating data
def simulate_data(m, n, k=2, random_state=None, classification=True):
    if classification:
        features, labels = make_classification(n_samples=m, n_features=n,
                                               n_informative=int(n/5),
                                               n_classes=k,
                                               random_state=random_state)
    else:
        features, labels = make_regression(n_samples=m, n_features=n,
                                           n_informative=int(n/5),
                                           n_targets=1,
                                           random_state=random_state)
    return np.c_[labels, features].astype(np.float32)


simulate = True
classification = True  # change this to false to use regression
n_rows = int(1e6)  # we'll use 1 millions rows
n_columns = int(50)
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

# check dimensions
print('X_train: ', X_train.shape, X_train.dtype, 'y_train: ',
      y_train.shape, y_train.dtype)
print('X_validation', X_validation.shape, X_validation.dtype,
      'y_validation: ', y_validation.shape, y_validation.dtype)

# check the proportions
total = X_train.shape[0] + X_validation.shape[0]
print('X_train proportion:', X_train.shape[0] / total)
print('X_validation proportion:', X_validation.shape[0] / total)

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

if n_gpus != 0:
    booster_params['tree_method'] = 'gpu_hist'
    booster_params['n_gpus'] = n_gpus
params.update(booster_params)

# learning task params
learning_task_params = {}
if classification:
    learning_task_params['eval_metric'] = 'auc'
    learning_task_params['objective'] = 'binary:logistic'
else:
    learning_task_params['eval_metric'] = 'rmse'
    learning_task_params['objective'] = 'reg:squarederror'
params.update(learning_task_params)
print(params)

# model training settings
evallist = [(dvalidation, 'validation'), (dtrain, 'train')]
num_round = 5

bst = xgb.train(params, dtrain, num_round, evallist)

bst.save_model('xgb.model')

print(" read the saved xgb modle")
tl_model = tl.Model.load('xgb.modle', 'xgboost')
print(" create a fil model")
fm = fil(algo=0, output=0, threshold=0.55)
print(" read data from the model and convert treelite to FIL")
fm.from_treelite(tl_model, output_class=True)
print(" Predict the labels ")
preds = fm.predict(X_validation)
fm.free()
