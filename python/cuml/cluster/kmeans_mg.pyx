#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import rmm

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import input_to_cuml_array

from cuml.cluster import KMeans


cdef extern from "cumlprims/opg/kmeans.hpp" namespace \
        "ML::kmeans::KMeansParams" nogil:
    enum InitMethod:
        KMeansPlusPlus, Random, Array

cdef extern from "cumlprims/opg/kmeans.hpp" namespace \
        "ML::kmeans" nogil:
    cdef struct KMeansParams:
        int n_clusters,
        InitMethod init
        int max_iter,
        double tol,
        int verbose,
        int seed,
        int metric,
        double oversampling_factor,
        int batch_samples,
        int batch_centroids,
        bool inertia_check

cdef extern from "cumlprims/opg/kmeans.hpp" namespace "ML::kmeans::opg" nogil:

    cdef void fit(cumlHandle& handle,
                  KMeansParams& params,
                  const float *X,
                  int n_samples,
                  int n_features,
                  float *centroids,
                  float &inertia,
                  int &n_iter) except +

    cdef void fit(cumlHandle& handle,
                  KMeansParams& params,
                  const double *X,
                  int n_samples,
                  int n_features,
                  double *centroids,
                  double &inertia,
                  int &n_iter) except +


class KMeansMG(KMeans):

    """
    A Multi-Node Multi-GPU implementation of KMeans

    NOTE: This implementation of KMeans is meant to be used with an
    initialized cumlCommunicator instance inside an existing distributed
    system. Refer to the Dask KMeans implementation in
    `cuml.dask.cluster.kmeans`.
    """

    def __init__(self, **kwargs):
        super(KMeansMG, self).__init__(**kwargs)

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

        X_m, self.n_rows, self.n_cols, self.dtype = \
            input_to_cuml_array(X, order='C')

        cdef uintptr_t input_ptr = X_m.ptr

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        if (self.init in ['scalable-k-means++', 'k-means||', 'random']):
            self._cluster_centers_ = CumlArray.zeros(shape=(self.n_clusters,
                                                            self.n_cols),
                                                     dtype=self.dtype,
                                                     order='C')

        cdef uintptr_t cluster_centers_ptr = self._cluster_centers_.ptr

        cdef size_t n_rows = self.n_rows
        cdef size_t n_cols = self.n_cols

        cdef float inertiaf = 0
        cdef double inertiad = 0

        cdef KMeansParams params = self._params
        cdef int n_iter = 0

        if self.dtype == np.float32:
            with nogil:
                fit(
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
                fit(
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

        del(X_m)

        return self
