
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

# # Linear Regression
# 
# **Linear Regression** is a simple machine learning model where the response y is modelled by a linear combination of the predictors in X.
# 
# The model can take array-like objects, either in host as NumPy arrays or in device (as Numba or cuda_array_interface-compliant), as well as cuDF DataFrames as the input. 
# 
# For information about cuDF, refer to the [cuDF documentation](https://docs.rapids.ai/api/cudf/stable).
# 
# For information about cuML's linear regression API: https://rapidsai.github.io/projects/cuml/en/stable/api.html#cuml.LinearRegression
# 
# **NOTE:** This notebook is not expected to run on a GPU with under 16GB of RAM with its current value for `n_smaples`.  Please change `n_samples` from `2**20` to `2**19`

# ## Imports

# In[ ]:


import cudf
from cuml import make_regression, train_test_split
from cuml.linear_model import LinearRegression as cuLinearRegression
from cuml.metrics.regression import r2_score
from sklearn.linear_model import LinearRegression as skLinearRegression


# ## Define Parameters

# In[ ]:


n_samples = 2**20 #If you are running on a GPU with less than 16GB RAM, please change to 2**19 or you could run out of memory
n_features = 399

random_state = 23


# ## Generate Data

# In[ ]:


get_ipython().run_cell_magic('time', '', 'X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=random_state)\n\nX = cudf.DataFrame.from_gpu_matrix(X)\ny = cudf.DataFrame.from_gpu_matrix(y)[0]\n\nX_cudf, X_cudf_test, y_cudf, y_cudf_test = train_test_split(X, y, test_size = 0.2, random_state=random_state)')


# In[ ]:


# Copy dataset from GPU memory to host memory.
# This is done to later compare CPU and GPU results.
X_train = X_cudf.to_pandas()
X_test = X_cudf_test.to_pandas()
y_train = y_cudf.to_pandas()
y_test = y_cudf_test.to_pandas()


# ## Scikit-learn Model
# 
# ### Fit, predict and evaluate

# In[ ]:


get_ipython().run_cell_magic('time', '', 'ols_sk = skLinearRegression(fit_intercept=True,\n                            normalize=True,\n                            n_jobs=-1)\n\nols_sk.fit(X_train, y_train)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'predict_sk = ols_sk.predict(X_test)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'r2_score_sk = r2_score(y_cudf_test, predict_sk)')


# ## cuML Model

# ### Fit, predict and evaluate

# In[ ]:


get_ipython().run_cell_magic('time', '', "ols_cuml = cuLinearRegression(fit_intercept=True,\n                              normalize=True,\n                              algorithm='eig')\n\nols_cuml.fit(X_cudf, y_cudf)")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'predict_cuml = ols_cuml.predict(X_cudf_test)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'r2_score_cuml = r2_score(y_cudf_test, predict_cuml)')


# ## Compare Results

# In[ ]:


print("R^2 score (SKL):  %s" % r2_score_sk)
print("R^2 score (cuML): %s" % r2_score_cuml)

