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

from cuml.common.base import Base
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros

from cython.operator cimport dereference as deref

from libcpp cimport bool
from libcpp.memory cimport shared_ptr

from librmm_cffi import librmm as rmm
from libc.stdlib cimport malloc, free

from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from numba import cuda

cimport cuml.common.handle
cimport cuml.common.cuda

cdef extern from "cuML.hpp" namespace "ML" nogil:
    cdef cppclass deviceAllocator:
        pass

    cdef cppclass cumlHandle:
        cumlHandle() except +
        void setStream(cuml.common.cuda._Stream s)
        void setDeviceAllocator(shared_ptr[deviceAllocator] a)
        cuml.common.cuda._Stream getStream()

cdef extern from "knn/knn.hpp" namespace "ML":

    void brute_force_knn(
        cumlHandle &handle,
        float **input,
        int *sizes,
        int n_params,
        int D,
        float *search_items,
        int n,
        long *res_I,
        float *res_D,
        int k
    )

    void chunk_host_array(
        cumlHandle &handle,
        const float *ptr,
        int n,
        int D,
        int *devices,
        float **output,
        int *sizes,
        int n_chunks
    )


class NearestNeighbors(Base):
    """
    NearestNeighbors is a unsupervised algorithm where if one wants to find the
    "closest" datapoint(s) to new unseen data, one can calculate a suitable
    "distance" between each and every point, and return the top K datapoints
    which have the smallest distance to it.

    cuML's KNN can take array-like objects, either in host as
    NumPy arrays or in device (as Numba or __cuda_array_interface__ compliant),
    as well as cuDF DataFrames (where automatic chunking will be done in to a
    Numpy Array in a future release). It fits a special data structure first
    to approximate the distance calculations, allowing our querying times to be
    O(plogn) and not the brute force O(np)
    [where p = no(features)]

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
    n_neighbors: int (default = 5)
        The top K closest datapoints you want the algorithm to return.
        If this number is large, then expect the algorithm to run slower.
    should_downcast : bool (default = False)
        Currently only single precision is supported in the underlying undex.
        Setting this to true will allow single-precision input arrays to be
        automatically downcasted to single precision.

    Notes
    ------
    NearestNeighbors is a generative model. This means the data X has to be
    stored in order for inference to occur.

    **Applications of NearestNeighbors**

        Applications of NearestNeighbors include recommendation systems where
        content or colloborative filtering is used. Since NearestNeighbors is a
        relatively simple generative model, it is also used in data
        visualization and regression / classification tasks.

    For an additional example see `the NearestNeighbors notebook
    <https://github.com/rapidsai/notebook/blob/master/python/notebooks/knn_demo.ipynb>`_.

    For additional docs, see `scikitlearn's NearestNeighbors
    <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors>`_.
    """
    def __init__(self, n_neighbors=5, n_gpus=1, devices=None,
                 verbose=False, should_downcast=True, handle=None):
        """
        Construct the NearestNeighbors object for training and querying.

        Parameters
        ----------
        should_downcast: bool (default = False)
            Currently only single precision is supported in the underlying
            index. Setting this to true will allow single-precision input
            arrays to be automatically downcasted to single precision.
        """

        super(NearestNeighbors, self).__init__(handle, verbose)

        self.n_gpus = n_gpus
        self.devices = devices
        self.n_neighbors = n_neighbors
        self._should_downcast = should_downcast

    def fit(self, X):
        """
        Fit GPU index for performing nearest neighbor queries.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        """
        if len(X.shape) != 2:
            raise ValueError("data should be two dimensional")

        self.n_dims = X.shape[1]

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef uintptr_t X_ctype = -1
        cdef uintptr_t dev_ptr = -1

        cdef float** input_arr
        cdef int* sizes_arr

        if isinstance(X, np.ndarray):

            if X.dtype != np.float32:
                if self._should_downcast:
                    X = np.ascontiguousarray(X, np.float32)
                    if len(X[X == np.inf]) > 0:
                        raise ValueError("Downcast to single-precision "
                                         "resulted in data loss.")
                else:
                    raise TypeError("Only single precision floating point is"
                                    " supported for this algorithm. Use "
                                    "'should_downcast=True' if you'd like it "
                                    "to be automatically casted to single "
                                    "precision.")

            sys_devices = set([d.id for d in cuda.gpus])

            if self.devices is not None:
                for d in self.devices:
                    if d not in sys_devices:
                        raise RuntimeError("Device %d is not available" % d)

                final_devices = self.devices

            else:
                n_gpus = min(self.n_gpus, len(sys_devices))
                final_devices = list(sys_devices)[:n_gpus]

            final_devices = np.ascontiguousarray(np.array(final_devices),
                                                 np.int32)

            X_ctype = X.ctypes.data
            dev_ptr = final_devices.ctypes.data

            input_arr = <float**> malloc(len(final_devices) * sizeof(float *))
            sizes_arr = <int*> malloc(len(final_devices) * sizeof(int))

            chunk_host_array(
                handle_[0],
                <float*>X_ctype,
                <int>X.shape[0],
                <int>X.shape[1],
                <int*>dev_ptr,
                <float**>input_arr,
                <int*>sizes_arr,
                <int>len(final_devices)
            )

            self.input = <size_t>input_arr
            self.sizes = <size_t>sizes_arr
            self.n_indices = len(final_devices)

        else:
            if self._should_downcast:
                self.X_m, X_ctype, n_rows, _, dtype = \
                    input_to_dev_array(X, order='C',
                                       convert_to_dtype=np.float32)
            else:
                self.X_m, X_ctype, n_rows, _, dtype = \
                    input_to_dev_array(X, order='C')

            input_arr = <float**> malloc(sizeof(float *))
            sizes_arr = <int*> malloc(sizeof(int))

            sizes_arr[0] = <int>len(X)
            input_arr[0] = <float*>X_ctype

            self.n_indices = 1

            inp = <uintptr_t>deref(input_arr)

            self.sizes = <size_t>sizes_arr
            self.input = <size_t>input_arr

    def _fit_mg(self, n_dims, alloc_info):
        """
        Fits a model using multiple GPUs. This method takes in a list of dict
        objects representing the distribution of the underlying device
        pointers. The device information can be extracted from the pointers.

        :param n_dims
            the number of features for each vector
        :param alloc_info
            a list of __cuda_array_interface__ dicts
        :return:
        """

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef float** input_arr = \
            <float**> malloc(len(alloc_info) * sizeof(float*))
        cdef int* sizes_arr = <int*>malloc(len(alloc_info)*sizeof(int))

        self.n_indices = len(alloc_info)

        cdef uintptr_t input_ptr
        for i in range(len(alloc_info)):
            sizes_arr[i] = < int > alloc_info[i]["shape"][0]

            input_ptr = alloc_info[i]["data"][0]
            input_arr[i] = < float * > input_ptr

        self.sizes = <size_t>sizes_arr
        self.input = <size_t>input_arr

        self.n_dims = n_dims

    def kneighbors(self, X, k=None):
        """
        Query the GPU index for the k nearest neighbors of column vectors in X.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        k: Integer
            Number of neighbors to search

        Returns
        ----------
        distances: cuDF DataFrame or numpy ndarray
            The distances of the k-nearest neighbors for each column vector
            in X

        indices: cuDF DataFrame of numpy ndarray
            The indices of the k-nearest neighbors for each column vector in X
        """

        if k is None:
            k = self.n_neighbors

        if self._should_downcast:
            X_m, X_ctype, N, _, dtype = \
                input_to_dev_array(X, order='C', convert_to_dtype=np.float32)
        else:
            X_m, X_ctype, N, _, dtype = input_to_dev_array(X, order='C')

        # Need to establish result matrices for indices (Nxk)
        # and for distances (Nxk)
        I_ndarr = cuda.to_device(zeros(N*k, dtype=np.int64, order="C"))
        D_ndarr = cuda.to_device(zeros(N*k, dtype=np.float32, order="C"))

        cdef uintptr_t I_ptr = get_dev_array_ptr(I_ndarr)
        cdef uintptr_t D_ptr = get_dev_array_ptr(D_ndarr)

        cdef float** inputs = <float**><size_t>self.input
        cdef int* sizes = <int*><size_t>self.sizes

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
            <long*>I_ptr,
            <float*>D_ptr,
            <int>k
        )

        I_ndarr = I_ndarr.reshape((N, k))
        D_ndarr = D_ndarr.reshape((N, k))

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

        return dists, inds

    def _kneighbors(self, X_ctype, N, I_ptr, D_ptr, k):

        cdef uintptr_t inds = I_ptr
        cdef uintptr_t dists = D_ptr
        cdef uintptr_t x = X_ctype

        cdef uintptr_t input_arr = self.input
        cdef uintptr_t sizes_arr = self.sizes

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        brute_force_knn(
            handle_[0],
            <float**>input_arr,
            <int*>sizes_arr,
            <int>self.n_indices,
            <int>self.n_dims,
            <float*>x,
            <int>N,
            <long*>inds,
            <float*>dists,
            <int>k
        )
