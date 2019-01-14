# Copyright (c) 2018, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cimport c_knn
import numpy as np
import pandas as pd
import cudf
import ctypes

from librmm_cffi import librmm as rmm
from numba import cuda
from c_knn cimport *

class KNNparams:
    def __init__(self, n_gpus):
        self.n_gpus = n_gpus


cdef class KNN:
    """

    Create a DataFrame, fill it with data, and compute KNN:

    .. code-block:: python

      import cudf
      from cuml import KNN
      import numpy as np

      np_float = np.array([
        [1,2,3], # Point 1
        [1,2,4], # Point 2
        [2,2,4]  # Point 3
      ]).astype('float32')

      gdf_float = cudf.DataFrame()
      gdf_float['dim_0'] = np.ascontiguousarray(np_float[:,0])
      gdf_float['dim_1'] = np.ascontiguousarray(np_float[:,1])
      gdf_float['dim_2'] = np.ascontiguousarray(np_float[:,2])

      print('n_samples = 3, n_dims = 3')
      print(gdf_float)

      knn_float = KNN(n_gpus=1)
      knn_float.fit(gdf_float)
      Distance,Index = knn_float.query(gdf_float,k=3) #get 3 nearest neighbors

      print(Index)
      print(Distance)

    Output:

    .. code-block:: python

      n_samples = 3, n_dims = 3

      dim_0 dim_1 dim_2

      0   1.0   2.0   3.0
      1   1.0   2.0   4.0
      2   2.0   2.0   4.0

      # Index:

               index_neighbor_0 index_neighbor_1 index_neighbor_2
      0                0                1                2
      1                1                0                2
      2                2                1                0
      # Distance:

               distance_neighbor_0 distance_neighbor_1 distance_neighbor_2
      0                 0.0                 1.0                 2.0
      1                 0.0                 1.0                 1.0
      2                 0.0                 1.0                 2.0

    For an additional example see `the KNN notebook <https://github.com/rapidsai/cuml/blob/master/python/notebooks/knn_demo.ipynb>`_. For additional docs, see `scikitlearn's KDtree <http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree>`_.

    """
    cdef kNN *k

    cdef int num_gpus

    cdef uintptr_t X_ctype

    cdef uintptr_t I_ptr
    cdef uintptr_t D_ptr

    def __cinit__(self, num_gpus = 1):
        self.num_gpus = num_gpus

    def _get_ctype_ptr(self, obj):
        # The manner to access the pointers in the gdf's might change, so
        # encapsulating access in the following 3 methods. They might also be
        # part of future gdf versions.
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def _get_gdf_as_matrix_ptr(self, gdf):
        return self._get_ctype_ptr(gdf.as_gpu_matrix())

    def fit(self, X):
        if isinstance(X, cudf.DataFrame):
            X_m = X.as_gpu_matrix(order = "C")
        else:
            X_m = cuda.to_device(X)

        cdef uintptr_t X_ctype = X_m.device_ctypes_pointer.value
        assert len(X.shape) == 2, 'data should be two dimensional'
        n_dims = X.shape[1]

        self.k = new kNN(n_dims)
        self.k.fit(<float*>X_ctype, <int> X.shape[0])

    def query(self, X, k):

        if isinstance(X, cudf.DataFrame):
            X_m = X.as_gpu_matrix(order = "C")
        else:
            X_m = cuda.to_device(X)
        
        cdef uintptr_t X_ctype = self._get_ctype_ptr(X_m)
        N = len(X)

        # Need to establish result matrices for indices (Nxk) and for distances (Nxk)
        I_ndarr = cuda.to_device(np.zeros(N*k, dtype=np.int64))
        D_ndarr = cuda.to_device(np.zeros(N*k, dtype=np.float32))

        cdef uintptr_t I_ptr = self._get_ctype_ptr(I_ndarr)
        cdef uintptr_t D_ptr = self._get_ctype_ptr(D_ndarr)

        self.k.search(<float*>X_ctype, <int> N, <long*>I_ptr, <float*>D_ptr, <int> k)

        I_ndarr = I_ndarr.reshape((N, k)).transpose()
        D_ndarr = D_ndarr.reshape((N, k)).transpose()

        I = cudf.DataFrame()
        for i in range(0, I_ndarr.shape[0]):
            I[str(i)] = I_ndarr[i,:]

        D = cudf.DataFrame()
        for i in range(0, D_ndarr.shape[0]):
            D[str(i)] = D_ndarr[i,:]

        return D, I

    def to_cudf(self, df, col=''):
        # convert pandas dataframe to cudf dataframe
        if isinstance(df,np.ndarray):
            df = pd.DataFrame({'%s_neighbor_%d'%(col, i): df[:, i] for i in range(df.shape[1])})
        pdf = cudf.DataFrame.from_pandas(df)
        return pdf
