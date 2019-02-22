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

cimport knn
import numpy as np
import pandas as pd
import cudf
import ctypes

from librmm_cffi import librmm as rmm
from libc.stdlib cimport malloc, free
from cython.operator cimport dereference as deref
from numba import cuda
from knn cimport *

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
    cpdef kNN *k

    cdef int num_gpus

    cdef uintptr_t X_ctype

    cdef uintptr_t I_ptr
    cdef uintptr_t D_ptr

    cdef bool _should_downcast

    cpdef kNNParams *input



    def __cinit__(self, should_downcast = False):
        """
        Construct the kNN object for training and querying.

        Parameters
        ----------
        should_downcast: Bool
            Currently only single precision is supported in the underlying undex. Setting this to
            true will allow single-precision input arrays to be automatically downcasted to single
            precision. Default = False.
        """
        self._should_downcast = should_downcast
        self.input = <kNNParams*> malloc(sizeof(kNNParams))

    def __dealloc__(self):
        del self.k
        del self.input

    def _get_ctype_ptr(self, obj):
        # The manner to access the pointers in the gdf's might change, so
        # encapsulating access in the following 3 methods. They might also be
        # part of future gdf versions.
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def _get_gdf_as_matrix_ptr(self, gdf):
        return self._get_ctype_ptr(gdf.as_gpu_matrix())


    def _downcast(self, X):

        if isinstance(X, cudf.DataFrame):
            dtype = np.dtype(X[X.columns[0]]._column.dtype)

            if dtype != np.float32:
                if self._should_downcast:

                    new_cols = [(col,X._cols[col].astype(np.float32)) for col in X._cols]
                    overflowed = sum([len(colval[colval >= np.inf])  for colname, colval in new_cols])

                    if overflowed > 0:
                        raise Exception("Downcast to single-precision resulted in data loss.")

                    X = cudf.DataFrame(new_cols)

                else:
                    raise Exception("Input is double precision. Use 'should_downcast=True' "
                                    "if you'd like it to be automatically casted to single precision.")

            X = X.as_gpu_matrix(order="C")
        elif isinstance(X, np.ndarray):
            dtype = X.dtype

            if dtype != np.float32:
                if self._should_downcast:
                    X = X.astype(np.float32)
                    if len(X[X == np.inf]) > 0:
                        raise Exception("Downcast to single-precision resulted in data loss.")
                else:
                    raise Exception("Input is double precision. Use 'should_downcast=True' "
                                    "if you'd like it to be automatically casted to single precision.")

            X = cuda.to_device(X)
        else:
            raise Exception("Received unsupported input type " % type(X))

        return X


    def fit(self, X):
        """
        Fit a KNN index for performing nearest neighbor queries.

        Parameters
        ----------
        X : cuDF DataFrame or numpy ndarray
            Dense matrix (floats or doubles) of shape (n_samples, n_features)
        """

        X_m = self._downcast(X)

        cdef uintptr_t X_ctype = X_m.device_ctypes_pointer.value
        assert len(X.shape) == 2, 'data should be two dimensional'
        n_dims = X.shape[1]

        self.k = new kNN(n_dims)

        params = new kNNParams()
        params.N = <int>len(X)
        params.ptr = <float*>X_ctype

        self.input[0] = deref(params)

        self.k.fit(<kNNParams*> self.input,
                   <int> 1)

    def _fit_mg(self, n_dims, alloc_info):
        """
        Fits a model using multiple GPUs. This method takes in a list of dict objects
        representing the distribution of the underlying device pointers. The device
        information can be extracted from the pointers.

        :param n_dims
            the number of features for each vector
        :param alloc_info
            a list of __cuda_array_interface__ dicts
        :return:
        """
        self.k = new kNN(n_dims)

        del self.input
        self.input = < kNNParams * > malloc(len(alloc_info) * sizeof(kNNParams))

        cdef uintptr_t input_ptr
        for i in range(len(alloc_info)):
            params = new kNNParams()
            params.N = < int > alloc_info[i]["shape"][0]

            input_ptr = alloc_info[i]["data"][0]
            params.ptr = < float * > input_ptr

            self.input[i] = deref(params)

        self.k.fit( < kNNParams * > self.input,
                    < int > len(alloc_info))


    def query(self, X, k):
        """
        Query the KNN index for the k nearest neighbors of row vectors in X.

        Parameters
        ----------
        X : cuDF DataFrame or numpy ndarray
            Dense matrix (floats or doubles) of shape (n_samples, n_features)

        k: Integer
           The number of neighbors

        Returns
        ----------
        distances: cuDF DataFrame or numpy ndarray
            The distances of the k-nearest neighbors for each column vector in X

        indices: cuDF DataFrame of numpy ndarray
            The indices of the k-nearest neighbors for each column vector in X
        """

        X_m = self._downcast(X)

        cdef uintptr_t X_ctype = self._get_ctype_ptr(X_m)
        N = len(X)

        # Need to establish result matrices for indices (Nxk) and for distances (Nxk)
        I_ndarr = cuda.to_device(np.zeros(N*k, dtype=np.int64, order = "F"))
        D_ndarr = cuda.to_device(np.zeros(N*k, dtype=np.float32, order = "F"))

        cdef uintptr_t I_ptr = self._get_ctype_ptr(I_ndarr)
        cdef uintptr_t D_ptr = self._get_ctype_ptr(D_ndarr)

        self._query(X_ctype, N, k, I_ptr, D_ptr)

        I_ndarr = I_ndarr.reshape((N, k))
        D_ndarr = D_ndarr.reshape((N, k))

        I = cudf.DataFrame()
        for i in range(0, I_ndarr.shape[1]):
            I[str(i)] = I_ndarr[:,i]

        D = cudf.DataFrame()
        for i in range(0, D_ndarr.shape[1]):
            D[str(i)] = D_ndarr[:,i]

        if isinstance(X, np.ndarray):
            I = np.asarray(I.as_gpu_matrix())
            D = np.asarray(D.as_gpu_matrix())

        return D, I


    def _query(self, X_ctype, N, k, I_ptr, D_ptr):
        """
        Query the KNN index for the k nearest neighbors of column vectors in X.

        Parameters
        ----------
        X : cuDF DataFrame or numpy ndarray
            Dense matrix (floats or doubles) of shape (n_samples, n_features)

        k: Integer
           The number of neighbors

        Returns
        ----------
        distances: cuDF DataFrame or numpy ndarray
            The distances of the k-nearest neighbors for each column vector in X

        indices: cuDF DataFrame of numpy ndarray
            The indices of the k-nearest neighbors for each column vector in X
        """

        cdef uintptr_t I = I_ptr
        cdef uintptr_t D = D_ptr
        cdef uintptr_t X = X_ctype

        self.k.search(<float*>X,
                      <int> N,
                      <long*>I,
                      <float*>D,
                      <int> k)
