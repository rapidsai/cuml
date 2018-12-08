#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import linear_model as sklOLS
from cuml import linear_model as cumlOLS
import cudf
import os


# # Helper Functions

# In[2]:


from timeit import default_timer

class Timer(object):
    def __init__(self):
        self._timer = default_timer
    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        """Start the timer."""
        self.start = self._timer()

    def stop(self):
        """Stop the timer. Calculate the interval in seconds."""
        self.end = self._timer()
        self.interval = self.end - self.start


# In[3]:


import gzip
def load_data(nrows, ncols, cached = 'data/mortgage.npy.gz'):
    if os.path.exists(cached):
        print('use mortgage data')
        with gzip.open(cached) as f:
            X = np.load(f)
        # the 4th column is 'adj_remaining_months_to_maturity'
        # used as the label
        X = X[:,[i for i in range(X.shape[1]) if i!=4]]
        y = X[:,4:5]
        rindices = np.random.randint(0,X.shape[0]-1,nrows)
        X = X[rindices,:ncols]
        y = y[rindices]
    else:
        print('use random data')
        X = np.random.rand(nrows,ncols)
        
    df_X = pd.DataFrame({'fea%d'%i:X[:,i] for i in range(X.shape[1])})
    df_y = pd.DataFrame({'fea%d'%i:y[:,i] for i in range(y.shape[1])})
    
    return df_X, df_y


# In[4]:


from sklearn.metrics import mean_squared_error
def array_equal(a,b,threshold=2e-3,with_sign=True):
    a = to_nparray(a).ravel()
    b = to_nparray(b).ravel()
    if with_sign == False:
        a,b = np.abs(a),np.abs(b)
    error = mean_squared_error(a,b)
    res = error<threshold
    return res

def to_nparray(x):
    if isinstance(x,np.ndarray) or isinstance(x,pd.DataFrame):
        return np.array(x)
    elif isinstance(x,np.float64):
        return np.array([x])
    elif isinstance(x,cudf.DataFrame) or isinstance(x,cudf.Series):
        return x.to_pandas().values
    return x    


# # Run tests

# In[5]:


nrows = 2**10
ncols = 10
X, y = load_data(nrows,ncols)
print('data',X.shape)
print('label',y.shape)


# In[6]:


fit_intercept = True
normalize = False
algorithm = "eig"


# In[7]:


reg_sk = sklOLS.LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
result_sk = reg_sk.fit(X, y)


# In[8]:

X = cudf.DataFrame.from_pandas(X)
y = np.array(y.as_matrix())
y = y[:,0]
y = cudf.Series(y)


# In[9]:

reg_cuml = cumlOLS.LinearRegression(fit_intercept=fit_intercept, normalize=normalize, algorithm=algorithm)
result_cuml = reg_cuml.fit(X, y)


# In[10]:

print(reg_cuml.coef_)
print(reg_sk.coef_)







