
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

# # Random Forest and Pickling
# The Random Forest algorithm is a classification method which builds several decision trees, and aggregates each of their outputs to make a prediction.
# 
# In this notebook we will train a scikit-learn and a cuML Random Forest Classification model. Then we save the cuML model for future use with Python's `pickling` mechanism and demonstrate how to re-load it for prediction. We also compare the results of the scikit-learn, non-pickled and pickled cuML models.
# 
# Note that the underlying algorithm in cuML for tree node splits differs from that used in scikit-learn.
# 
# For information on converting your dataset to cuDF format, refer to the [cuDF documentation](https://docs.rapids.ai/api/cudf/stable)
# 
# For additional information cuML's random forest model: https://rapidsai.github.io/projects/cuml/en/stable/api.html#random-forest

# In[ ]:


import cudf
import numpy as np
import pandas as pd
import pickle

from cuml.ensemble import RandomForestClassifier as curfc
from cuml.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier as skrfc
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# ## Define Parameters

# In[ ]:


n_samples = 2**17
n_features = 399
n_info = 300
data_type = np.float32


# ## Generate Data
# 
# ### Host

# In[ ]:


get_ipython().run_cell_magic('time', '', 'X,y = make_classification(n_samples=n_samples,\n                          n_features=n_features,\n                          n_informative=n_info,\n                          random_state=123, n_classes=2)\n\nX = pd.DataFrame(X.astype(data_type))\n# cuML Random Forest Classifier requires the labels to be integers\ny = pd.Series(y.astype(np.int32))\n\nX_train, X_test, y_train, y_test = train_test_split(X, y,\n                                                    test_size = 0.2,\n                                                    random_state=0)')


# ### GPU

# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_cudf_train = cudf.DataFrame.from_pandas(X_train)\nX_cudf_test = cudf.DataFrame.from_pandas(X_test)\n\ny_cudf_train = cudf.Series(y_train.values)')


# ## Scikit-learn Model
# 
# ### Fit

# In[ ]:


get_ipython().run_cell_magic('time', '', 'sk_model = skrfc(n_estimators=40,\n                 max_depth=16,\n                 max_features=1.0,\n                 random_state=10)\n\nsk_model.fit(X_train, y_train)')


# ### Evaluate

# In[ ]:


get_ipython().run_cell_magic('time', '', 'sk_predict = sk_model.predict(X_test)\nsk_acc = accuracy_score(y_test, sk_predict)')


# ## cuML Model

# ### Fit

# In[ ]:


get_ipython().run_cell_magic('time', '', 'cuml_model = curfc(n_estimators=40,\n                   max_depth=16,\n                   max_features=1.0,\n                   seed=10)\n\ncuml_model.fit(X_cudf_train, y_cudf_train)')


# ### Evaluate

# In[ ]:


get_ipython().run_cell_magic('time', '', 'fil_preds_orig = cuml_model.predict(X_cudf_test)\n\nfil_acc_orig = accuracy_score(y_test.to_numpy(), fil_preds_orig)')


# ## Pickle the cuML random forest classification model

# In[ ]:


filename = 'cuml_random_forest_model.sav'
# save the trained cuml model into a file
pickle.dump(cuml_model, open(filename, 'wb'))
# delete the previous model to ensure that there is no leakage of pointers.
# this is not strictly necessary but just included here for demo purposes.
del cuml_model
# load the previously saved cuml model from a file
pickled_cuml_model = pickle.load(open(filename, 'rb'))


# ### Predict using the pickled model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'pred_after_pickling = pickled_cuml_model.predict(X_cudf_test)\n\nfil_acc_after_pickling = accuracy_score(y_test.to_numpy(), pred_after_pickling)')


# ## Compare Results

# In[ ]:


print("CUML accuracy of the RF model before pickling: %s" % fil_acc_orig)
print("CUML accuracy of the RF model after pickling: %s" % fil_acc_after_pickling)


# In[ ]:


print("SKL accuracy: %s" % sk_acc)
print("CUML accuracy before pickling: %s" % fil_acc_orig)

