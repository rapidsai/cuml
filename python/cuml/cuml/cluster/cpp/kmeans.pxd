#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

from libc.stdint cimport int64_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t
from pylibraft.random.cpp.rng_state cimport RngState

from cuml.internals.logger cimport level_enum
from cuml.metrics.distance_type cimport DistanceType


cdef extern from "cuml/cluster/kmeans_params.hpp" namespace "ML::kmeans::KMeansParams" nogil:
    enum class InitMethod:
        KMeansPlusPlus, Random, Array


cdef extern from "cuml/cluster/kmeans_params.hpp" namespace "ML::kmeans" nogil:
    cdef struct KMeansParams:
        DistanceType metric,
        int n_clusters,
        InitMethod init,
        int max_iter,
        double tol,
        level_enum verbosity,
        RngState rng_state,
        int n_init,
        double oversampling_factor,
        int batch_samples,
        int batch_centroids,
        bool inertia_check


cdef extern from "cuml/cluster/kmeans.hpp" namespace "ML::kmeans" nogil:
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

    cdef void predict(handle_t& handle,
                      KMeansParams& params,
                      const float *centroids,
                      const float *X,
                      int n_samples,
                      int n_features,
                      const float *sample_weight,
                      bool normalize_weights,
                      int *labels,
                      float &inertia) except +

    cdef void predict(handle_t& handle,
                      KMeansParams& params,
                      double *centroids,
                      const double *X,
                      int n_samples,
                      int n_features,
                      const double *sample_weight,
                      bool normalize_weights,
                      int *labels,
                      double &inertia) except +

    cdef void transform(handle_t& handle,
                        KMeansParams& params,
                        const float *centroids,
                        const float *X,
                        int n_samples,
                        int n_features,
                        float *X_new) except +

    cdef void transform(handle_t& handle,
                        KMeansParams& params,
                        const double *centroids,
                        const double *X,
                        int n_samples,
                        int n_features,
                        double *X_new) except +

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

    cdef void predict(handle_t& handle,
                      KMeansParams& params,
                      const float *centroids,
                      const float *X,
                      int64_t n_samples,
                      int64_t n_features,
                      const float *sample_weight,
                      bool normalize_weights,
                      int64_t *labels,
                      float &inertia) except +

    cdef void predict(handle_t& handle,
                      KMeansParams& params,
                      double *centroids,
                      const double *X,
                      int64_t n_samples,
                      int64_t n_features,
                      const double *sample_weight,
                      bool normalize_weights,
                      int64_t *labels,
                      double &inertia) except +

    cdef void transform(handle_t& handle,
                        KMeansParams& params,
                        const float *centroids,
                        const float *X,
                        int64_t n_samples,
                        int64_t n_features,
                        float *X_new) except +

    cdef void transform(handle_t& handle,
                        KMeansParams& params,
                        const double *centroids,
                        const double *X,
                        int64_t n_samples,
                        int64_t n_features,
                        double *X_new) except +
