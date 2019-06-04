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

from cuml import numba_utils
from cuml.common.base import Base

from libcpp cimport bool
from libcpp.memory cimport shared_ptr

from librmm_cffi import librmm as rmm
from libc.stdlib cimport malloc, free

from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from numba import cuda

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle

cdef extern from "knn/knn.hpp" namespace "ML" nogil:
    void brute_force_knn(
        cumlHandle &handle,
        float ** input,
        int *sizes,
        int n_params,
        int D,
        float * search_items,
        int n,
        long * res_I,
        float * res_D,
        int k) except +


class NearestNeighbors(Base):
    """
    NearestNeighbors is a unsupervised algorithm where if one wants to find the
    "closest" datapoint(s) to new unseen data, one can calculate a suitable
    "distance" between each and every point, and return the top K datapoints
    which have the smallest distance to it.

    cuML's KNN expects a cuDF DataFrame or a Numpy Array (where automatic
    chunking will be done in to a Numpy Array in a future release), and fits a
    special data structure first to approximate the distance calculations,
    allowing our querying times to be O(plogn) and not the brute force O(np)
    [where p = no(features)]:

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

        super(NearestNeighbors, self).__init__(handle=handle, verbose=False)
        """
        Construct the NearestNeighbors object for training and querying.

        Parameters
        ----------
        should_downcast: bool (default = False)
            Currently only single precision is supported in the underlying
            index. Setting this to true will allow single-precision input
            arrays to be automatically downcasted to single precision.
        """
        self._verbose = verbose
        self.n_gpus = n_gpus
        self.devices = devices
        self.n_neighbors = n_neighbors
        self._should_downcast = should_downcast
        self.handle = handle
        self.D = 0

    def __dealloc__(self):
        del self.input
        del self.sizes

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

                    new_cols = [(col, X._cols[col].astype(np.float32))
                                for col in X._cols]
                    overflowed = sum([len(colval[colval >= np.inf])
                                      for colname, colval in new_cols])

                    if overflowed > 0:
                        raise Exception("Downcast to single-precision resulted"
                                        "in data loss.")

                    X = cudf.DataFrame(new_cols)

                else:
                    raise Exception("Input is double precision. Use"
                                    " 'should_downcast=True' "
                                    "if you'd like it to be automatically"
                                    " casted to single precision.")

            X_m = numba_utils.row_matrix(X)

        elif isinstance(X, np.ndarray):
            dtype = X.dtype

            if dtype != np.float32:
                if self._should_downcast:
                    X = np.ascontiguousarray(X.astype(np.float32))
                    if len(X[X == np.inf]) > 0:
                        raise Exception("Downcast to single-precision resulted"
                                        " in data loss.")
                else:
                    raise Exception("Input is double precision. Use"
                                    " 'should_downcast=True' "
                                    "if you'd like it to be automatically"
                                    " casted to single precision.")

            X_m = cuda.to_device(X)
        else:
            raise Exception("Received unsupported input type " % type(X))

        return X_m

    def fit(self, X):
        """
        Fit GPU index for performing nearest neighbor queries.

        Parameters
        ----------
        X : cuDF DataFrame or numpy ndarray
            Dense matrix (floats or doubles) of shape (n_samples, n_features)
        """
        assert len(X.shape) == 2, 'data should be two dimensional'

        self.D = X.shape[1]

        cdef uintptr_t X_ctype = -1
        cdef uintptr_t dev_ptr = -1
        if isinstance(X, np.ndarray):

            if X.dtype != np.float32:
                if self._should_downcast:
                    X = np.ascontiguousarray(X, np.float32)
                    if len(X[X == np.inf]) > 0:
                        raise Exception("Downcast to single-precision resulted"
                                        " in data loss.")
                else:
                    raise Exception("Only single precision floating point is"
                                    " supported for this algorithm. Use "
                                    "'should_downcast=True' if you'd like it "
                                    "to be automatically casted to single "
                                    "precision.")

            sys_devices = set([d.id for d in cuda.gpus])

            if self.devices is not None:
                for d in self.devices:
                    if d not in sys_devices:
                        raise Exception("Device %d is not available" % d)

                final_devices = self.devices

            else:
                n_gpus = min(self.n_gpus, len(sys_devices))
                final_devices = list(sys_devices)[:n_gpus]

            final_devices = np.ascontiguousarray(np.array(final_devices),
                                                 np.int32)

            # X_ctype = X.ctypes.data
            # dev_ptr = final_devices.ctypes.data
            #
            # # TODO: Call chunk from host here.
            # self.k.fit_from_host(
            #     <float*>X_ctype,
            #     <int>X.shape[0],
            #     <int*>dev_ptr,
            #     <int>len(final_devices)
            # )

        else:
            self.X_m = self._downcast(X)

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

        cdef float** inp = <float**> malloc(len(alloc_info) * sizeof(float*))
        cdef int* sizes = <int*>malloc(len(alloc_info) * sizeof(int))
        n_indices = len(alloc_info)

        cdef uintptr_t input_ptr
        for i in range(len(alloc_info)):

            sizes[i] = < int > alloc_info[i]["shape"][0]

            input_ptr = alloc_info[i]["data"][0]
            inp[i] = < float * > input_ptr


    def kneighbors(self, X, k=None):
        """
        Query the GPU index for the k nearest neighbors of column vectors in X.

        Parameters
        ----------
        X_ctype : Ctypes pointer (row-major)
            Pointer to input data

        N: Intetger
            The number of rows in X

        k: Integer
            Number of neighbors to search

        I_ptr: Ctypes pointer (row-major)
            Pointer to N*k array for output indices

        D_ptr: Ctypes pointer (row-major)
            Pointer to N*k array for output distances

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

        X_m = self._downcast(X)

        cdef uintptr_t input_X_ctype = self.X_m.device_ctypes_pointer.value

        cdef float** input = <float**> malloc(sizeof(float*))
        cdef int* sizes = <int*>malloc(sizeof(int))

        input[0] = < float * > input_X_ctype
        sizes[0] = < int > len(self.X_m)

        cdef uintptr_t X_ctype = self._get_ctype_ptr(X_m)

        N = len(X)

        # Need to establish result matrices for indices (Nxk)
        # and for distances (Nxk)
        I_ndarr = cuda.to_device(np.zeros(N*k, dtype=np.int64, order="C"))
        D_ndarr = cuda.to_device(np.zeros(N*k, dtype=np.float32, order="C"))

        cdef uintptr_t I_ptr = self._get_ctype_ptr(I_ndarr)
        cdef uintptr_t D_ptr = self._get_ctype_ptr(D_ndarr)

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        brute_force_knn(handle_[0],
                        <float**>input,
                        <int*>sizes,
                        <int>1,
                        <int>self.D,
                        <float*>X_ctype,
                        <int> N,
                        <long*>I_ptr,
                        <float*>D_ptr,
                        <int> k)

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

    def _kneighbors(self, inp, sizes, n_indices, X_ctype, N, k, I_ptr, D_ptr):


        cdef uintptr_t inds = I_ptr
        cdef uintptr_t dists = D_ptr
        cdef uintptr_t x = X_ctype

        cdef uintptr_t inp_ptr = inp
        cdef uintptr_t sizes_ptr = sizes

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        brute_force_knn(handle_[0],
                        <float**>inp_ptr,
                        <int*>sizes_ptr,
                        <int>n_indices,
                        <int>self.D,
                        <float*>x,
                        <int> N,
                        <long*>inds,
                        <float*>dists,
                        <int> k)
