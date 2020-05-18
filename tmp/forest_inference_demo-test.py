
def my_run_line_magic(*args, **kwargs):
    g=globals()
    l={}
    for a in args:
        try:
            exec(str(a),g,l)
        except Exception as e:
            print('WARNING: %s\n   While executing this magic function code:\n%s\n   continuing...\n' % (e, a))
        else:
            g.update(l)

def my_run_cell_magic(*args, **kwargs):
    my_run_line_magic(*args, **kwargs)

get_ipython().run_line_magic=my_run_line_magic
get_ipython().run_cell_magic=my_run_cell_magic


#!/usr/bin/env python
# coding: utf-8

# # Forest Inference Library (FIL)
# The forest inference library is used to load saved forest models of xgboost, lightgbm or protobuf and perform inference on them. It can be used to perform both classification and regression. In this notebook, we'll begin by fitting a model with XGBoost and saving it. We'll then load the saved model into FIL and use it to infer on new data.
# 
# FIL works in the same way with lightgbm and protobuf model as well.
# 
# The model accepts both numpy arrays and cuDF dataframes. In order to convert your dataset to cudf format please read the cudf documentation on https://docs.rapids.ai/api/cudf/stable. 
# 
# For additional information on the forest inference library please refer to the documentation on https://rapidsai.github.io/projects/cuml/en/stable/api.html#forest-inferencing

# In[ ]:


import numpy as np
import os

from cuml.test.utils import array_equal
from cuml.common.import_utils import has_xgboost

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
    
from cuml import ForestInference


# ### Check for xgboost
# Checks if xgboost is present, if not then it throws an error.

# In[ ]:


if has_xgboost():
    import xgboost as xgb
else:
    raise ImportError("Please install xgboost using the conda package,"
                      " Use conda install -c conda-forge xgboost "
                      "command to install xgboost")


# ## Train helper function
# Defines a simple function that trains the XGBoost model and returns the trained model.
# 
# For additional information on the xgboost library please refer to the documentation on : 
# https://xgboost.readthedocs.io/en/latest/parameter.html

# In[ ]:


def train_xgboost_model(X_train, y_train,
                        num_rounds, model_path):
    # set the xgboost model parameters
    params = {'silent': 1, 'eval_metric':'error',
              'objective':'binary:logistic',
              'max_depth': 25}
    dtrain = xgb.DMatrix(X_train, label=y_train)
    # train the xgboost model
    bst = xgb.train(params, dtrain, num_rounds)

    # save the trained xgboost model
    bst.save_model(model_path)

    return bst


# ## Predict helper function
# Uses the trained xgboost model to perform prediction and return the labels.

# In[ ]:


def predict_xgboost_model(X_validation, y_validation, xgb_model):

    # predict using the xgboost model
    dvalidation = xgb.DMatrix(X_validation, label=y_validation)
    xgb_preds = xgb_model.predict(dvalidation)

    # convert the predicted values from xgboost into class labels
    xgb_preds = np.around(xgb_preds)
    return xgb_preds


# ## Define parameters

# In[ ]:


n_rows = 10000
n_columns = 100
n_categories = 2
random_state = np.random.RandomState(43210)

# enter path to the directory where the trained model will be saved
model_path = 'xgb.model'

# num of iterations for which the model is trained
num_rounds = 15


# ## Generate data

# In[ ]:


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


# ## Train and Predict the model
# Invoke the function to train the model and get predictions so that we can validate them.

# In[ ]:


# train the xgboost model
xgboost_model = train_xgboost_model(X_train, y_train,
                                    num_rounds, model_path)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# test the xgboost model\ntrained_model_preds = predict_xgboost_model(X_validation,\n                                            y_validation,\n                                            xgboost_model)')


# ## Load Forest Inference Library (FIL)
# 
# The load function of the ForestInference class accepts the following parameters:
# 
#        filename : str
#            Path to saved model file in a treelite-compatible format
#            (See https://treelite.readthedocs.io/en/latest/treelite-api.html
#         output_class : bool
#            If true, return a 1 or 0 depending on whether the raw prediction
#            exceeds the threshold. If False, just return the raw prediction.
#         threshold : float
#            Cutoff value above which a prediction is set to 1.0
#            Only used if the model is classification and output_class is True
#         algo : string name of the algo from (from algo_t enum)
#              'NAIVE' - simple inference using shared memory
#              'TREE_REORG' - similar to naive but trees rearranged to be more
#                               coalescing-friendly
#              'BATCH_TREE_REORG' - similar to TREE_REORG but predicting
#                                     multiple rows per thread block
#         model_type : str
#             Format of saved treelite model to load.
#             Can be 'xgboost', 'lightgbm', or 'protobuf'

# ## Loaded the saved model
# Use FIL to load the saved xgboost model

# In[ ]:


fm = ForestInference.load(filename=model_path,
                          algo='BATCH_TREE_REORG',
                          output_class=True,
                          threshold=0.50,
                          model_type='xgboost')


# ## Predict using FIL

# In[ ]:


get_ipython().run_cell_magic('time', '', '# perform prediction on the model loaded from path\nfil_preds = fm.predict(X_validation)')


# ## Evaluate results
# 
# Verify the predictions for the original and FIL model match.

# In[ ]:


print("The shape of predictions obtained from xgboost : ",(trained_model_preds).shape)
print("The shape of predictions obtained from FIL : ",(fil_preds).shape)
print("Are the predictions for xgboost and FIL the same : " ,   array_equal(trained_model_preds, fil_preds))

