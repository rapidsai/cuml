#
# Copyright (c) 2019, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import numpy as np
import pandas as pd
import cudf
import ctypes
import cuml
import warnings

from cuml.common.base import Base
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros, row_matrix

from cython.operator cimport dereference as deref

from cuml.common.handle cimport cumlHandle


from libcpp cimport bool
from libcpp.memory cimport shared_ptr

import rmm
from libc.stdlib cimport malloc, free

from libc.stdint cimport uintptr_t, int64_t
from libc.stdlib cimport calloc, malloc, free

from numba import cuda
import rmm

cimport cuml.common.handle
cimport cuml.common.cuda

cdef extern from "cuml/cuml.hpp" namespace "ML" nogil:
    cdef cppclass deviceAllocator:
        pass

    cdef cppclass cumlHandle:
        cumlHandle() except +
        void setStream(cuml.common.cuda._Stream s) except +
        void setDeviceAllocator(shared_ptr[deviceAllocator] a) except +
        cuml.common.cuda._Stream getStream() except +

cdef extern from "cuml/neighbors/knn.hpp" namespace "ML":

    void brute_force_knn(
        cumlHandle &handle,
        float **input,
        int *sizes,
        int n_params,
        int D,
        float *search_items,
        int n,
        int64_t *res_I,
        float *res_D,
        int k,
        bool rowMajorIndex,
        bool rowMajorQuery
    ) except +

class NearestNeighbors(Base):
    """
    NearestNeighbors is an queries neighborhoods from a given set of
    datapoints. Currently, cuML supports k-NN queries, which define
    the neighborhood as the closest `k` neighbors to each query point.

    Examples
    ---------
    .. code-block:: python

      import cudf
      from cuml.neighbors import NearestNeighbors
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

      nn_float = NearestNeighbors()
      nn_float.fit(gdf_float)
      # get 3 nearest neighbors
      distances,indices = nn_float.kneighbors(gdf_float,k=3)

      print(indices)
      print(distances)

    Output:

    .. code-block:: python

      import cudf

      # Both import methods supported
      # from cuml.neighbors import NearestNeighbors
      from cuml import NearestNeighbors

      n_samples = 3, n_dims = 3

      dim_0 dim_1 dim_2

      0   1.0   2.0   3.0
      1   1.0   2.0   4.0
      2   2.0   2.0   4.0

      # indices:

               index_neighbor_0 index_neighbor_1 index_neighbor_2
      0                0                1                2
      1                1                0                2
      2                2                1                0
      # distances:

               distance_neighbor_0 distance_neighbor_1 distance_neighbor_2
      0                 0.0                 1.0                 2.0
      1                 0.0                 1.0                 1.0
      2                 0.0                 1.0                 2.0

    Parameters
    ----------
    n_neighbors : int (default = 5)
        The top K closest datapoints you want the algorithm to return.
        Currently, this value must be < 1024.
    verbose : bool print logging
    handle : cuml.Handle cuML handle to use for underlying resource
    metric : string distance metric to use. default = "seuclidean". Currently,
        only "euclidean" and "seuclidean" are supported.

    Notes
    ------

    For an additional example see `the NearestNeighbors notebook
    <https://github.com/rapidsai/notebook/blob/master/python/notebooks/knn_demo.ipynb>`_.

    For additional docs, see `scikitlearn's NearestNeighbors
    <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors>`_.
    """
    def __init__(self,
                 n_neighbors=5,
                 verbose=False,
                 handle=None,
                 algorithm="brute",
                 metric="seuclidean"):
        """
        Construct the NearestNeighbors object for training and querying.

        Parameters
        ----------
        should_downcast: bool (default = None)
            Currently only single precision is supported in the underlying
            index. Setting this to true will allow single-precision input
            arrays to be automatically downcasted to single precision.
        """

        super(NearestNeighbors, self).__init__(handle, verbose)

        if metric != "euclidean" and metric != "seuclidean":
            raise ValueError("Only Euclidean (euclidean) and Squared Euclidean (seuclidean)"
                             "metrics are supported currently")

        self.n_neighbors = n_neighbors
        self.n_indices = 0
        self.metric = metric
        self.algorithm = algorithm

    def __getstate__(self):
        state = self.__dict__.copy()

        del state['handle']

        # Only need to store index if fit() was called
        if self.n_indices == 1:
            state['X_m'] = cudf.DataFrame.from_gpu_matrix(self.X_m)

        return state

    def __setstate__(self, state):
        super(NearestNeighbors, self).__init__(handle=None,
                                               verbose=state['verbose'])

        cdef uintptr_t x_ctype
        # Only need to recover state if model had been previously fit
        if state["n_indices"] == 1:

            state['X_m'] = row_matrix(state['X_m'])

        self.__dict__.update(state)

    def fit(self, X, convert_dtype=True):
        """
        Fit GPU index for performing nearest neighbor queries.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        convert_dtype : bool, optional (default = True)
            When set to True, the fit method will automatically
            convert the inputs to np.float32.
            Note: Convert dtype will be set to False once should_downcast is
                deprecated in 0.10
        """

        if len(X.shape) != 2:
            raise ValueError("data should be two dimensional")

        self.n_dims = X.shape[1]

        self.X_m, X_ctype, n_rows, n_cols, dtype = \
            input_to_dev_array(X, order='F', check_dtype=np.float32,
                               convert_to_dtype=(np.float32
                                                 if convert_dtype
                                                 else None))

        self.n_indices = 1

        return self

    def kneighbors(self, X=None, n_neighbors=None,
                   return_distance=True, convert_dtype=True):
        """
        Query the GPU index for the k nearest neighbors of column vectors in X.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        n_neighbors : Integer
            Number of neighbors to search. If not provided, the n_neighbors
            from the model instance is used (default=10)

        return_distance: Boolean
            If False, distances will not be returned

        convert_dtype : bool, optional (default = True)
            When set to True, the kneighbors method will automatically
            convert the inputs to np.float32.
            Note: Convert dtype will be set to False once should_downcast is
                deprecated in 0.10
        Returns
        ----------
        distances: cuDF DataFrame or numpy ndarray
            The distances of the k-nearest neighbors for each column vector
            in X

        indices: cuDF DataFrame of numpy ndarray
            The indices of the k-nearest neighbors for each column vector in X
        """

        n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors
        X = self.X if X is None else X

        if (n_neighbors is None and self.n_neighbors is None) \
                or n_neighbors <= 0:
            raise ValueError("k or n_neighbors must be a positive integers")

        if X is None:
            raise ValueError("Model needs to be trained "
                             "before calling kneighbors()")

        X_m, X_ctype, N, _, dtype = \
            input_to_dev_array(X, order='F', check_dtype=np.float32,
                               convert_to_dtype=(np.float32 if convert_dtype
                                                 else None))

        # Need to establish result matrices for indices (Nxk)
        # and for distances (Nxk)
        I_ndarr = rmm.to_device(zeros(N*n_neighbors, dtype=np.int64, order="C"))
        D_ndarr = rmm.to_device(zeros(N*n_neighbors, dtype=np.float32, order="C"))

        cdef uintptr_t I_ptr = get_dev_array_ptr(I_ndarr)
        cdef uintptr_t D_ptr = get_dev_array_ptr(D_ndarr)

        cdef float** inputs = <float**> malloc(sizeof(float *))
        cdef int* sizes = <int*> malloc(sizeof(int))

        cdef uintptr_t idx_ptr = get_dev_array_ptr(self.X_m)
        inputs[0] = <float*>idx_ptr
        sizes[0] = <int>self.X_m.shape[0]

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef uintptr_t x_ctype_st = X_ctype

        brute_force_knn(
            handle_[0],
            <float**>inputs,
            <int*>sizes,
            <int>self.n_indices,
            <int>self.n_dims,
            <float*>x_ctype_st,
            <int>N,
            <int64_t*>I_ptr,
            <float*>D_ptr,
            <int>n_neighbors,
            False,
            False
        )

        I_ndarr = I_ndarr.reshape((N, n_neighbors))
        D_ndarr = D_ndarr.reshape((N, n_neighbors))

        if isinstance(X, cudf.DataFrame):
            inds = cudf.DataFrame()
            for i in range(0, I_ndarr.shape[1]):
                inds[str(i)] = I_ndarr[:, i]

            dists = cudf.DataFrame()
            for i in range(0, D_ndarr.shape[1]):
                dists[str(i)] = D_ndarr[:, i]

            return dists, inds

        elif isinstance(X, np.ndarray):
            inds = np.asarray(I_ndarr)
            dists = np.asarray(D_ndarr)

        del I_ndarr
        del D_ndarr
        del X_m

        free(inputs)
        free(sizes)

        return (dists, inds) if return_distance else inds
