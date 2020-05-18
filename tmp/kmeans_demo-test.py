
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

# # K-Means Demo
# 
# KMeans is a basic but powerful clustering method which is optimized via Expectation Maximization. It randomly selects K data points in X, and computes which samples are close to these points. For every cluster of points, a mean is computed, and this becomes the new centroid.
# 
# cuML’s KMeans supports the scalable KMeans++ intialization method. This method is more stable than randomnly selecting K points.
#     
# The model can take array-like objects, either in host as NumPy arrays or in device (as Numba or cuda_array_interface-compliant), as well as cuDF DataFrames as the input.
# 
# For information about cuDF, refer to the [cuDF documentation](https://docs.rapids.ai/api/cudf/stable).
# 
# For additional information on cuML's k-means implementation: https://rapidsai.github.io/projects/cuml/en/stable/api.html#cuml.KMeans.

# ## Imports

# In[ ]:


import cudf
import cupy
import matplotlib.pyplot as plt
from cuml.cluster import KMeans as cuKMeans
from cuml.datasets import make_blobs
from sklearn.cluster import KMeans as skKMeans
from sklearn.metrics import adjusted_rand_score

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Define Parameters

# In[ ]:


n_samples = 100000
n_features = 2

n_clusters = 5
random_state = 0


# ## Generate Data

# In[ ]:


device_data, device_labels = make_blobs(n_samples=n_samples,
                                        n_features=n_features,
                                        centers=n_clusters,
                                        random_state=random_state,
                                        cluster_std=0.1)

device_data = cudf.DataFrame.from_gpu_matrix(device_data)
device_labels = cudf.Series(device_labels)


# In[ ]:


# Copy dataset from GPU memory to host memory.
# This is done to later compare CPU and GPU results.
host_data = device_data.to_pandas()
host_labels = device_labels.to_pandas()


# ## Scikit-learn model
# 
# ### Fit

# In[ ]:


get_ipython().run_cell_magic('time', '', 'kmeans_sk = skKMeans(init="k-means++",\n                     n_clusters=n_clusters,\n                     n_jobs=-1,\n                    random_state=random_state)\n\nkmeans_sk.fit(host_data)')


# ## cuML Model
# 
# ### Fit

# In[ ]:


get_ipython().run_cell_magic('time', '', 'kmeans_cuml = cuKMeans(init="k-means||",\n                       n_clusters=n_clusters,\n                       oversampling_factor=40,\n                       random_state=random_state)\n\nkmeans_cuml.fit(device_data)')


# ## Visualize Centroids
# 
# Scikit-learn's k-means implementation uses the `k-means++` initialization strategy while cuML's k-means uses `k-means||`. As a result, the exact centroids found may not be exact as the std deviation of the points around the centroids in `make_blobs` is increased.
# 
# *Note*: Visualizing the centroids will only work when `n_features = 2` 

# In[ ]:


fig = plt.figure(figsize=(16, 10))
plt.scatter(host_data.iloc[:, 0], host_data.iloc[:, 1], c=host_labels, s=50, cmap='viridis')

#plot the sklearn kmeans centers with blue filled circles
centers_sk = kmeans_sk.cluster_centers_
plt.scatter(centers_sk[:,0], centers_sk[:,1], c='blue', s=100, alpha=.5)

#plot the cuml kmeans centers with red circle outlines
centers_cuml = kmeans_cuml.cluster_centers_
plt.scatter(cupy.asnumpy(centers_cuml[0].values), 
            cupy.asnumpy(centers_cuml[1].values), 
            facecolors = 'none', edgecolors='red', s=100)

plt.title('cuml and sklearn kmeans clustering')

plt.show()


# ## Compare Results

# In[ ]:


get_ipython().run_cell_magic('time', '', 'cuml_score = adjusted_rand_score(host_labels, kmeans_cuml.labels_.to_array())\nsk_score = adjusted_rand_score(host_labels, kmeans_sk.labels_)')


# In[ ]:


threshold = 1e-4

passed = (cuml_score - sk_score) < threshold
print('compare kmeans: cuml vs sklearn labels_ are ' + ('equal' if passed else 'NOT equal'))

