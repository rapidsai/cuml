#!/usr/bin/env python
# coding: utf-8

# # Ridge Regression
# 
# This notebook includes code examples of ridge regression using RAPIDS cuDF and cuML. 

# In[ ]:


import numpy as np
import pandas as pd
from scipy import linalg
from sklearn import linear_model as sklGLM
from cuml import LinearRegression as cumlOLS
from cuml import Ridge as cumlRidge
import cudf
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


# ### Helper Functions

# In[ ]:


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


# In[ ]:


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


# In[ ]:


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

def _solve_svd(X, y, alpha):
    U, s, Vt = linalg.svd(X, full_matrices=False)
    idx = s > 1e-15  # same default value as scipy.linalg.pinv
    s_nnz = s[idx][:, np.newaxis]

    #print("s_nnz")
    #print(s_nnz)

    UTy = np.dot(U.T, y)
    
    #print("UTy")
    #print(UTy)

    d = np.zeros((s.size, alpha.size))
    d[idx] = s_nnz / (s_nnz ** 2 + alpha)
    #print("d")
    #print(d)

    d_UT_y = d * UTy
    #print("d_UT_y")
    #print(d_UT_y)
   
    rslt = np.dot(Vt.T, d_UT_y).T

    #print("rslt")
    #print(rslt)

    return rslt

# In[ ]:


nrows = int((2**20) * 1.2)
ncols = 399
X, y = load_data(nrows,ncols)
print('data',X.shape)
print('label',y.shape)

# Even though the ridge regression interface of cuML is very similar to Scikit-Learn's implemetation, cuML doesn't use some of the parameters such as "copy". Also, cuML includes two different implementation of ridge regression using SVD and Eigen decomposition. Eigen decomposition based implementation is very fast but causes very small errors in the coefficients which is negligible for most of the applications. SVD is stable but slower than eigen decomposition based implementation. 

# In[ ]:


fit_intercept = False
normalize = False
alpha = np.array([1.0])
# eig: eigen decomposition based method, 
# svd: singular value decomposition based method,
# cd: coordinate descend.
solver = "eig" 

#coefs = _solve_svd(X, y, alpha)

# In[ ]:

reg_sk = sklGLM.Ridge(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, solver='auto')
result_sk = reg_sk.fit(X, y)

print("coef")
print(reg_sk.coef_[0,0:10])

# In[ ]:

y_sk = reg_sk.predict(X)
error_sk = mean_squared_error(y,y_sk)


# In[ ]:

X_cudf = cudf.DataFrame.from_pandas(X)
y_cudf = np.array(y.as_matrix())
y_cudf = y_cudf[:,0]
y_cudf = cudf.Series(y_cudf)


# In[ ]:

reg_cuml = cumlRidge(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, solver=solver)
result_cuml = reg_cuml.fit(X_cudf, y_cudf)


# In[ ]:

y_cuml = reg_cuml.predict(X_cudf)
y_cuml = to_nparray(y_cuml).ravel()
error_cuml = mean_squared_error(y,y_cuml)


# In[ ]:


#print("SKL MSE(y):")
#print(error_sk)
print("CUML MSE(y):")
print(error_cuml)





