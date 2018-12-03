
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN as skDBSCAN
from cuml import DBSCAN as cumlDBSCAN
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
        X = X[np.random.randint(0,X.shape[0]-1,nrows),:ncols]
    else:
        print('use random data')
        X = np.random.rand(nrows,ncols)
    df = pd.DataFrame({'fea%d'%i:X[:,i] for i in range(X.shape[1])})
    return df


# In[4]:


from sklearn.metrics import mean_squared_error
def array_equal(a,b,threshold=5e-3,with_sign=True):
    a = to_nparray(a)
    b = to_nparray(b)
    if with_sign == False:
        a,b = np.abs(a),np.abs(b)
    res = mean_squared_error(a,b)<threshold
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

# In[21]:


get_ipython().run_cell_magic('time', '', "nrows = 1000\nncols = 128\n\nX = load_data(nrows,ncols)\nprint('data',X.shape)")


# In[22]:


eps = 0.3
min_samples = 2


# In[23]:


get_ipython().run_cell_magic('time', '', 'clustering_sk = skDBSCAN(eps = eps, min_samples = min_samples)\nclustering_sk.fit(X)')


# In[24]:


get_ipython().run_cell_magic('time', '', 'X = cudf.DataFrame.from_pandas(X)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clustering_cuml = cumlDBSCAN(eps = eps, min_samples = min_samples)\nclustering_cuml.fit(X)')


# In[18]:


l = clustering_sk.labels_
print(str(l[l !=0]))


# In[19]:


l2 = clustering_cuml.labels_
print(str(l2[l2 != 0].to_array()))


# In[20]:


passed = array_equal(clustering_sk.labels_,clustering_cuml.labels_)
message = 'compare dbscan: cuml vs sklearn labels_ %s'%('equal'if passed else 'NOT equal')
print(message)

