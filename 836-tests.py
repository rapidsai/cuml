#!/usr/bin/env python
import numpy as np
import os

from cuml.test.utils import array_equal
from cuml.utils.import_utils import has_xgboost

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
    
from cuml import ForestInference
if has_xgboost():
    import xgboost as xgb
else:
    raise ImportError("Please install xgboost using the conda package,"
                      " Use conda install -c conda-forge xgboost "
                      "command to install xgboost")

class XGBoostRFClassifier():
    def train_and_save(self, X, y,
                            num_rounds, model_path):
        # set the xgboost model parameters
        params = {'silent': 1, 'eval_metric':'error',
                  'objective':'binary:logistic',
                  'max_depth': 5}
        dtrain = xgb.DMatrix(X, label=y)
        # train the xgboost model
        bst = xgb.train(params, dtrain, num_rounds)
        self.model = bst

        # save the trained xgboost model
        bst.save_model(model_path)
        self.model_path = model_path

        return bst

    def predict(self, X, y):
        # predict using the xgboost model
        xy = xgb.DMatrix(X, label=y)
        xgb_preds = np.around(self.model.predict(xy))

        # convert the predicted values from xgboost into class labels
        return xgb_preds
    
    def fil(self):
        return ForestInference.load(filename=self.model_path,
                                  algo='BATCH_TREE_REORG',
                                  output_class=True,
                                  model_type='xgboost')

class XGBoostSKLearnAPI():
    def train(self, X, y):
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y)
    
    def predict_(self, X, y, method):
        # predict using the xgboost model
        func = {'predict' : (lambda x: np.around(self.model.predict(x))),
                'predict_proba' : self.model.predict_proba
               }[method]
        xgb_preds = func(x)

        # convert the predicted values from xgboost into class labels
        return xgb_preds
    
    def predict(self, X, y):
        return self.predict_(X, y, 'predict')

    def predict_proba(self, X, y):
        return self.predict_(X, y, 'predict_proba')

class SKLearnRFClassifier():
    def train_and_save(self, X, y,
                            num_rounds, model_path):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(max_depth=5)
        self.model.fit(X, y)

    def predict(self, X):
        return np.around(self.model.predict(X))

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def fil(self):
        return ForestInference.load_from_sklearn(
            self.model, algo='NAIVE', storage_type='sparse', output_class=True, threshold=0.5)

def test_fil(fil_preds, trained_model_preds, name):
    print("The shape of predictions obtained from", name, ": ",(trained_model_preds).shape)
    print("The shape of predictions obtained from FIL : ",(fil_preds).shape)
    print("Are the predictions for", name, "and FIL the same : " ,   array_equal(trained_model_preds, fil_preds))

n_rows = 10000
n_columns = 10
n_categories = 2
random_state = np.random.RandomState(43210)

# enter path to the directory where the trained model will be saved
model_path = 'model'

# num of iterations for which the model is trained
num_rounds = 15

# create the dataset
X, y = make_classification(n_samples=n_rows,
                           n_features=n_columns,
                           n_informative=int(n_columns/5),
                           n_classes=n_categories,
                           random_state=random_state)
train_size = 0.8

# convert the dataset to np.float32
X = X.astype(np.float32)
y = y.astype(np.float32)

# split the dataset into training and validation splits
X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, train_size=train_size)

def try_xgboost():
    xgboost_model = XGBoostRFClassifier()
    xgboost_model.train_and_save(X_train, y_train, num_rounds, model_path)

    # perform prediction on the model loaded from path
    fil = xgboost_model.fil()
    fil_preds = fil.predict(X_validation)

    # test the xgboost model
    trained_model_preds = xgboost_model.predict(X_validation, y_validation)

    test_fil(fil_preds, trained_model_preds, "xgboost")

def try_sk():
    sk = SKLearnRFClassifier()
    sk.train_and_save(X_train, y_train, num_rounds, model_path)
    fil = sk.fil()
    test_fil(fil.predict(X_validation), sk.predict(X_validation), "SK class")
    test_fil(fil.predict_proba(X_validation), sk.predict_proba(X_validation), "SK proba")
    
    def try_with(low, high):
        x = X_validation[low:high,:]
        print(low, high, '\n\n',
              fil.predict(x).      copy_to_host(), '\n\n',
               sk.predict(x), '\n\n',
              fil.predict_proba(x).copy_to_host(), '\n\n',
               sk.predict_proba(x))
    
try_xgboost()
try_sk()
