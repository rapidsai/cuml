#
# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

# distutils: language = c++

from cuml.internals.safe_imports import cpu_only_import
np = cpu_only_import('numpy')

from cuml.internals.safe_imports import gpu_only_import
rmm = gpu_only_import('rmm')

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t, int64_t
from libc.stdlib cimport free

from cuml.internals.array import CumlArray
from pylibraft.common.handle cimport handle_t
from cuml.common import input_to_cuml_array

from cuml.cluster import KMeans
from cuml.cluster.kmeans_utils cimport *


cdef extern from "cuml/cluster/kmeans_mg.hpp" \
        namespace "ML::kmeans::opg" nogil:

    cdef void fit(handle_t& handle,
                  KMeansParams& params,
                  const float *X,
                  int n_samples,
                  int n_features,
                  const float *sample_weight,
                  float *centroids,
                  float &inertia,
                  int &n_iter) except +

    cdef void fit(handle_t& handle,
                  KMeansParams& params,
                  const double *X,
                  int n_samples,
                  int n_features,
                  const double *sample_weight,
                  double *centroids,
                  double &inertia,
                  int &n_iter) except +

    cdef void fit(handle_t& handle,
                  KMeansParams& params,
                  const float *X,
                  int64_t n_samples,
                  int64_t n_features,
                  const float *sample_weight,
                  float *centroids,
                  float &inertia,
                  int64_t &n_iter) except +

    cdef void fit(handle_t& handle,
                  KMeansParams& params,
                  const double *X,
                  int64_t n_samples,
                  int64_t n_features,
                  const double *sample_weight,
                  double *centroids,
                  double &inertia,
                  int64_t &n_iter) except +


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

    def fit(self, X, sample_weight=None) -> "KMeansMG":
        """
        Compute k-means clustering with X in a multi-node multi-GPU setting.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        sample_weight : array-like (device or host) shape = (n_samples,), default=None # noqa
            The weights for each observation in X. If None, all observations
            are assigned equal weight.
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        """

        X_m, self.n_rows, self.n_cols, self.dtype = \
            input_to_cuml_array(X, order='C')

        cdef uintptr_t input_ptr = X_m.ptr
        cdef int n_rows = self.n_rows
        cdef int n_cols = self.n_cols

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        if sample_weight is None:
            sample_weight_m = CumlArray.ones(shape=n_rows, dtype=self.dtype)
        else:
            sample_weight_m, _, _, _ = \
                input_to_cuml_array(sample_weight, order='C',
                                    convert_to_dtype=self.dtype,
                                    check_rows=n_rows)

        cdef uintptr_t sample_weight_ptr = sample_weight_m.ptr

        if (self.init in ['scalable-k-means++', 'k-means||', 'random']):
            self.cluster_centers_ = CumlArray.zeros(shape=(self.n_clusters,
                                                           self.n_cols),
                                                    dtype=self.dtype,
                                                    order='C')

        cdef uintptr_t cluster_centers_ptr = self.cluster_centers_.ptr

        int_dtype = np.int32 if np.int64(n_rows) * np.int64(n_cols) < 2**31-1 else np.int64

        print(str(n_rows * n_cols))

        cdef float inertiaf = 0
        cdef double inertiad = 0

        cdef KMeansParams* params = \
            <KMeansParams*><size_t>self._get_kmeans_params()
        cdef int n_iter = 0
        cdef int64_t n_iter64 = 0

        if self.dtype == np.float32:
            if int_dtype == np.int32:
                with nogil:
                    fit(
                        handle_[0],
                        <KMeansParams> deref(params),
                        <const float*> input_ptr,
                        <int> n_rows,
                        <int> n_cols,
                        <const float *>sample_weight_ptr,
                        <float*> cluster_centers_ptr,
                        inertiaf,
                        n_iter)
                self.n_iter_ = n_iter
            else:
                with nogil:
                    fit(
                        handle_[0],
                        <KMeansParams> deref(params),
                        <const float*> input_ptr,
                        <int64_t> n_rows,
                        <int64_t> n_cols,
                        <const float *>sample_weight_ptr,
                        <float*> cluster_centers_ptr,
                        inertiaf,
                        n_iter64)
                self.n_iter_ = n_iter64
            self.handle.sync()
            self.inertia_ = inertiaf
        elif self.dtype == np.float64:
            if int_dtype == np.int32:
                with nogil:
                    fit(
                        handle_[0],
                        <KMeansParams> deref(params),
                        <const double*> input_ptr,
                        <int> n_rows,
                        <int> n_cols,
                        <const double *>sample_weight_ptr,
                        <double*> cluster_centers_ptr,
                        inertiad,
                        n_iter)
                self.n_iter_ = n_iter
            else:
                with nogil:
                    fit(
                        handle_[0],
                        <KMeansParams> deref(params),
                        <const double*> input_ptr,
                        <int64_t> n_rows,
                        <int64_t> n_cols,
                        <const double *>sample_weight_ptr,
                        <double*> cluster_centers_ptr,
                        inertiad,
                        n_iter64)
                self.n_iter_ = n_iter64
            self.handle.sync()
            self.inertia_ = inertiad
        else:
            raise TypeError('KMeans supports only float32 and float64 input,'
                            'but input type ' + str(self.dtype) +
                            ' passed.')

        self.handle.sync()

        self.labels_, _, _, _ = input_to_cuml_array(self.predict(X_m,
                                                    sample_weight=sample_weight), order='C',
                                                    convert_to_dtype=self.dtype)

        del X_m
        free(params)

        return self
