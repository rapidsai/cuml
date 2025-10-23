# Copyright (c) 2025, NVIDIA CORPORATION.
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

from libc.stdint cimport int64_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t
from rmm.librmm.device_uvector cimport device_uvector

from cuml.metrics.distance_type cimport DistanceType


cdef extern from "cuml/cluster/hdbscan.hpp" namespace "ML::HDBSCAN::Common" nogil:

    ctypedef enum CLUSTER_SELECTION_METHOD:
        EOM "ML::HDBSCAN::Common::CLUSTER_SELECTION_METHOD::EOM"
        LEAF "ML::HDBSCAN::Common::CLUSTER_SELECTION_METHOD::LEAF"

    cdef cppclass CondensedHierarchy[value_idx, value_t]:
        CondensedHierarchy(const handle_t &handle, size_t n_leaves) except +

        CondensedHierarchy(const handle_t& handle_,
                           size_t n_leaves_,
                           int _n_edges_,
                           value_idx* parents_,
                           value_idx* children_,
                           value_t* lambdas_,
                           value_idx* sizes_) except +

        value_idx *get_parents() except +
        value_idx *get_children() except +
        value_t *get_lambdas() except +
        value_idx *get_sizes() except +
        value_idx get_n_edges() except +
        value_idx get_n_leaves() except +
        int get_n_clusters() except +

    cdef cppclass hdbscan_output[int64_t, float]:
        hdbscan_output(const handle_t &handle,
                       int n_leaves,
                       int64_t *labels,
                       float *probabilities,
                       int64_t *children,
                       int64_t *sizes,
                       float *deltas,
                       int64_t *mst_src,
                       int64_t *mst_dst,
                       float *mst_weights) except +
        int64_t get_n_leaves() except +
        int get_n_clusters() except +
        float *get_stabilities() except +
        int64_t *get_labels() except +
        int64_t *get_inverse_label_map() except +
        float *get_core_dists() except +
        CondensedHierarchy[int64_t, float] &get_condensed_tree() except +

    cdef cppclass HDBSCANParams:
        int min_samples
        int min_cluster_size
        int max_cluster_size,

        float cluster_selection_epsilon,
        float alpha,

        bool allow_single_cluster,
        CLUSTER_SELECTION_METHOD cluster_selection_method,

    cdef cppclass PredictionData[int64_t, float]:
        PredictionData(const handle_t &handle,
                       int64_t m,
                       int64_t n,
                       float *core_dists) except +

        size_t n_rows
        size_t n_cols

    void generate_prediction_data(const handle_t& handle,
                                  CondensedHierarchy[int64_t, float]& condensed_tree,
                                  int64_t* labels,
                                  int64_t* inverse_label_map,
                                  int n_selected_clusters,
                                  PredictionData[int64_t, float]& prediction_data) except +

cdef extern from "cuml/cluster/hdbscan.hpp" namespace "ML" nogil:

    void hdbscan(const handle_t & handle,
                 const float * X,
                 size_t m, size_t n,
                 DistanceType metric,
                 HDBSCANParams & params,
                 hdbscan_output & output,
                 float * core_dists) except +

    void build_condensed_hierarchy(
      const handle_t &handle,
      const int64_t *children,
      const float *delta,
      const int64_t *sizes,
      int min_cluster_size,
      int n_leaves,
      CondensedHierarchy[int64_t, float] &condensed_tree) except +

    void _extract_clusters(const handle_t &handle, size_t n_leaves,
                           int _n_edges, int64_t *parents, int64_t *children,
                           float *lambdas, int64_t *sizes, int64_t *labels,
                           float *probabilities,
                           CLUSTER_SELECTION_METHOD cluster_selection_method,
                           bool allow_single_cluster, int64_t max_cluster_size,
                           float cluster_selection_epsilon) except +

    void compute_all_points_membership_vectors(
        const handle_t &handle,
        CondensedHierarchy[int64_t, float] &condensed_tree,
        PredictionData[int64_t, float] &prediction_data_,
        float* X,
        DistanceType metric,
        float* membership_vec,
        size_t batch_size) except +

    void compute_membership_vector(
        const handle_t& handle,
        CondensedHierarchy[int64_t, float] &condensed_tree,
        PredictionData[int64_t, float] &prediction_data,
        float* X,
        float* points_to_predict,
        size_t n_prediction_points,
        int min_samples,
        DistanceType metric,
        float* membership_vec,
        size_t batch_size) except +

    void out_of_sample_predict(const handle_t &handle,
                               CondensedHierarchy[int64_t, float] &condensed_tree,
                               PredictionData[int64_t, float] &prediction_data,
                               float* X,
                               int64_t* labels,
                               float* points_to_predict,
                               size_t n_prediction_points,
                               DistanceType metric,
                               int min_samples,
                               int64_t* out_labels,
                               float* out_probabilities) except +

cdef extern from "cuml/cluster/hdbscan.hpp" namespace "ML::HDBSCAN::HELPER" nogil:

    void compute_core_dists(const handle_t& handle,
                            const float* X,
                            float* core_dists,
                            size_t m,
                            size_t n,
                            DistanceType metric,
                            int min_samples) except +

    void compute_inverse_label_map(const handle_t& handle,
                                   CondensedHierarchy[int64_t, float]&
                                   condensed_tree,
                                   size_t n_leaves,
                                   CLUSTER_SELECTION_METHOD
                                   cluster_selection_method,
                                   device_uvector[int64_t]& inverse_label_map,
                                   bool allow_single_cluster,
                                   int64_t max_cluster_size,
                                   float cluster_selection_epsilon) except +
