
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

# # Nearest Neighbors
# 
# Nearest Neighbors enables the query of the k-nearest neighbors from a set of input samples.

# The model can take array-like objects, either in host as NumPy arrays or in device (as Numba or cuda_array_interface-compliant), as well as cuDF DataFrames as the input. 
# 
# For information on converting your dataset to cuDF format, refer to the cuDF documentation: https://docs.rapids.ai/api/cudf/stable
# 
# For additional information on cuML's Nearest Neighbors implementation: https://rapidsai.github.io/projects/cuml/en/stable/api.html#nearest-neighbors

# In[ ]:


import cudf
import numpy as np
from cuml.datasets import make_blobs
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
from sklearn.neighbors import NearestNeighbors as skNearestNeighbors


# ## Define Parameters

# In[ ]:


n_samples = 2**17
n_features = 40

n_query = 2**13
n_neighbors = 4
random_state = 0


# ## Generate Data
# 
# ### GPU

# In[ ]:


get_ipython().run_cell_magic('time', '', 'device_data, _ = make_blobs(n_samples=n_samples,\n                            n_features=n_features,\n                            centers=5,\n                            random_state=random_state)\n\ndevice_data = cudf.DataFrame.from_gpu_matrix(device_data)')


# In[ ]:


# Copy dataset from GPU memory to host memory.
# This is done to later compare CPU and GPU results.
host_data = device_data.to_pandas()


# ## Scikit-learn Model
# 
# ## Fit

# In[ ]:


get_ipython().run_cell_magic('time', '', 'knn_sk = skNearestNeighbors(algorithm="brute",\n                            n_jobs=-1)\nknn_sk.fit(host_data)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'D_sk, I_sk = knn_sk.kneighbors(host_data[:n_query], n_neighbors)')


# ## cuML Model
# 
# ### Fit

# In[ ]:


get_ipython().run_cell_magic('time', '', 'knn_cuml = cuNearestNeighbors()\nknn_cuml.fit(device_data)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'D_cuml, I_cuml = knn_cuml.kneighbors(device_data[:n_query], n_neighbors)')


# ## Compare Results
# 
# cuML currently uses FAISS for exact nearest neighbors search, which limits inputs to single-precision. This results in possible round-off errors when floats of different magnitude are added. As a result, it's very likely that the cuML results will not match Sciklearn's nearest neighbors exactly. You can read more in the [FAISS wiki](https://github.com/facebookresearch/faiss/wiki/FAQ#why-do-i-get-weird-results-with-brute-force-search-on-vectors-with-large-components).
# 
# ### Distances

# In[ ]:


passed = np.allclose(D_sk, D_cuml.as_gpu_matrix(), atol=1e-3)
print('compare knn: cuml vs sklearn distances %s'%('equal'if passed else 'NOT equal'))


# ### Indices

# In[ ]:


sk_sorted = np.sort(I_sk, axis=1)
cuml_sorted = np.sort(I_cuml.as_gpu_matrix(), axis=1)

diff = sk_sorted - cuml_sorted

passed = (len(diff[diff!=0]) / n_samples) < 1e-9
print('compare knn: cuml vs sklearn indexes %s'%('equal'if passed else 'NOT equal'))

