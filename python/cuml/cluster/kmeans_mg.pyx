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

import ctypes
import cudf
import numpy as np
import warnings

from numba import cuda

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros, numba_utils

from cuml.cluster import KMeans


cdef extern from "cumlprims/mg/kmeans_mg.hpp" namespace \
        "ML::kmeans::KMeansParams" nogil:
    enum InitMethod:
        KMeansPlusPlus, Random, Array

cdef extern from "cumlprims/mg/kmeans_mg.hpp" namespace "ML::kmeans" nogil:

    cdef struct KMeansParams:
        int n_clusters,
        InitMethod init
        int max_iter,
        double tol,
        int verbose,
        int seed,
        int metric,
        int oversampling_factor,
        int batch_size,
        bool inertia_check

    cdef void fit_mnmg(cumlHandle& handle,
                       KMeansParams& params,
                       const float *X,
                       int n_samples,
                       int n_features,
                       float *centroids,
                       float &inertia,
                       int &n_iter)

    cdef void fit_mnmg(cumlHandle& handle,
                       KMeansParams& params,
                       const double *X,
                       int n_samples,
                       int n_features,
                       double *centroids,
                       double &inertia,
                       int &n_iter)


class KMeansMG(KMeans):

    """
    A Multi-Node Multi-GPU implementation of KMeans

    NOTE: This implementation of KMeans is meant to be used with an
    initialized cumlCommunicator instance inside an existing distributed
    system. Refer to the Dask KMeans implementation in
    `cuml.dask.cluster.kmeans`.
    """

    def __init__(self, handle=None, n_clusters=8, max_iter=300, tol=1e-4,
                 verbose=0, random_state=1, precompute_distances='auto',
                 init='scalable-k-means++', n_init=1, algorithm='auto'):
        super(KMeansMG, self).__init__(handle, n_clusters, max_iter, tol,
                                       verbose, random_state,
                                       precompute_distances,
                                       init, n_init, algorithm)

    def fit(self, X):
        """
        Compute k-means clustering with X in a multi-node multi-GPU setting.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        """

        cdef uintptr_t input_ptr

        X_m, input_ptr, self.n_rows, self.n_cols, self.dtype = \
            input_to_dev_array(X, order='C')

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if (self.init in ['scalable-k-means++', 'k-means||', 'random']):
            clust_cent = zeros(self.n_clusters * self.n_cols,
                               dtype=self.dtype)
            self.cluster_centers_ = cuda.to_device(clust_cent)

        cdef uintptr_t cluster_centers_ptr = \
            get_dev_array_ptr(self.cluster_centers_)

        cdef size_t n_rows = self.n_rows
        cdef size_t n_cols = self.n_cols

        cdef float inertiaf = 0
        cdef double inertiad = 0

        cdef KMeansParams params = self._params
        cdef int n_iter = 0

        if self.dtype == np.float32:
            with nogil:
                fit_mnmg(
                    handle_[0],
                    <KMeansParams> params,
                    <const float*> input_ptr,
                    <size_t> n_rows,
                    <size_t> n_cols,
                    <float*> cluster_centers_ptr,
                    inertiaf,
                    n_iter)
            self.handle.sync()
            self.inertia_ = inertiaf
            self.n_iter_ = n_iter
        elif self.dtype == np.float64:
            with nogil:
                fit_mnmg(
                    handle_[0],
                    <KMeansParams> params,
                    <const double*> input_ptr,
                    <size_t> n_rows,
                    <size_t> n_cols,
                    <double*> cluster_centers_ptr,
                    inertiad,
                    n_iter)
            self.handle.sync()
            self.inertia_ = inertiad
            self.n_iter_ = n_iter
        else:
            raise TypeError('KMeans supports only float32 and float64 input,'
                            'but input type ' + str(self.dtype) +
                            ' passed.')

        self.handle.sync()
        cc_df = cudf.DataFrame()
        for i in range(0, self.n_cols):
            n_c = self.n_clusters
            n_cols = self.n_cols
            cc_df[str(i)] = self.cluster_centers_[i:n_c*n_cols:n_cols]
        self.cluster_centers_ = cc_df

        del(X_m)

        return self
