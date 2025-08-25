/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuml/common/distance_type.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cstddef>

namespace ML {
namespace HDBSCAN {
namespace Common {

/**
 * The Condensed hierarchicy is represented by an edge list with
 * parents as the source vertices, children as the destination,
 * with attributes for the cluster size and lambda value.
 *
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
class CondensedHierarchy {
 public:
  /**
   * Constructs an empty condensed hierarchy object which requires
   * condense() to be called in order to populate the state.
   * @param handle_
   * @param n_leaves_
   */
  CondensedHierarchy(const raft::handle_t& handle_, size_t n_leaves_);

  /**
   * Constructs a condensed hierarchy object with existing arrays
   * which already contain a condensed hierarchy.
   * @param handle_
   * @param n_leaves_
   * @param n_edges_
   * @param parents_
   * @param children_
   * @param lambdas_
   * @param sizes_
   */
  CondensedHierarchy(const raft::handle_t& handle_,
                     size_t n_leaves_,
                     int n_edges_,
                     value_idx* parents_,
                     value_idx* children_,
                     value_t* lambdas_,
                     value_idx* sizes_);

  /**
   * Constructs a condensed hierarchy object by moving
   * rmm::device_uvector. Used to construct cluster trees
   * @param handle_
   * @param n_leaves_
   * @param n_edges_
   * @param n_clusters_
   * @param parents_
   * @param children_
   * @param lambdas_
   * @param sizes_
   */
  CondensedHierarchy(const raft::handle_t& handle_,
                     size_t n_leaves_,
                     int n_edges_,
                     int n_clusters_,
                     rmm::device_uvector<value_idx>&& parents_,
                     rmm::device_uvector<value_idx>&& children_,
                     rmm::device_uvector<value_t>&& lambdas_,
                     rmm::device_uvector<value_idx>&& sizes_);
  /**
   * To maintain a high level of parallelism, the output from
   * Condense::build_condensed_hierarchy() is sparse (the cluster
   * nodes inside any collapsed subtrees will be 0).
   *
   * This function converts the sparse form to a dense form and renumbers
   * the cluster nodes into a topological sort order. The renumbering
   * reverses the values in the parent array since root has the largest value
   * in the single-linkage tree. Then, it makes the combined parent and
   * children arrays monotonic. Finally all of the arrays of the dendrogram
   * are sorted by parent->children->sizes (e.g. topological). The root node
   * will always have an id of 0 and the largest cluster size.
   *
   * Ths single-linkage tree dendrogram is a binary tree and parents/children
   * can be found with simple indexing arithmetic but the condensed tree no
   * longer has this property and so the tree now relies on either
   * special indexing or the topological ordering for efficient traversal.
   */
  void condense(value_idx* full_parents,
                value_idx* full_children,
                value_t* full_lambdas,
                value_idx* full_sizes,
                value_idx size = -1);

  value_idx get_cluster_tree_edges();

  value_idx* get_parents() { return parents.data(); }
  value_idx* get_children() { return children.data(); }
  value_t* get_lambdas() { return lambdas.data(); }
  value_idx* get_sizes() { return sizes.data(); }
  value_idx get_n_edges() { return n_edges; }
  int get_n_clusters() { return n_clusters; }
  value_idx get_n_leaves() const { return n_leaves; }

 private:
  const raft::handle_t& handle;

  rmm::device_uvector<value_idx> parents;
  rmm::device_uvector<value_idx> children;
  rmm::device_uvector<value_t> lambdas;
  rmm::device_uvector<value_idx> sizes;

  size_t n_edges;
  size_t n_leaves;
  int n_clusters;
  value_idx root_cluster;
};

enum CLUSTER_SELECTION_METHOD { EOM = 0, LEAF = 1 };

class RobustSingleLinkageParams {
 public:
  int min_samples      = 5;
  int min_cluster_size = 5;
  int max_cluster_size = 0;

  float cluster_selection_epsilon = 0.0;

  bool allow_single_cluster = false;

  float alpha = 1.0;
};

class HDBSCANParams : public RobustSingleLinkageParams {
 public:
  CLUSTER_SELECTION_METHOD cluster_selection_method = CLUSTER_SELECTION_METHOD::EOM;
};

/**
 * Container object for output information common between
 * robust single linkage variants.
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
class robust_single_linkage_output {
 public:
  /**
   * Construct output object with empty device arrays of
   * known size.
   * @param handle_ raft handle for ordering cuda operations
   * @param n_leaves_  number of data points
   * @param labels_ labels array on device (size n_leaves)
   * @param children_ dendrogram src/dst array (size n_leaves - 1, 2)
   * @param sizes_ dendrogram cluster sizes array (size n_leaves - 1)
   * @param deltas_ dendrogram distances array (size n_leaves - 1)
   * @param mst_src_ min spanning tree source array (size n_leaves - 1)
   * @param mst_dst_ min spanning tree destination array (size n_leaves - 1)
   * @param mst_weights_ min spanninng tree distances array (size n_leaves - 1)
   */
  robust_single_linkage_output(const raft::handle_t& handle_,
                               int n_leaves_,
                               value_idx* labels_,
                               value_idx* children_,
                               value_idx* sizes_,
                               value_t* deltas_,
                               value_idx* mst_src_,
                               value_idx* mst_dst_,
                               value_t* mst_weights_)
    : handle(handle_),
      n_leaves(n_leaves_),
      n_clusters(-1),
      labels(labels_),
      children(children_),
      sizes(sizes_),
      deltas(deltas_),
      mst_src(mst_src_),
      mst_dst(mst_dst_),
      mst_weights(mst_weights_)
  {
  }

  int get_n_leaves() const { return n_leaves; }
  int get_n_clusters() const { return n_clusters; }
  value_idx* get_labels() { return labels; }
  value_idx* get_children() { return children; }
  value_idx* get_sizes() { return sizes; }
  value_t* get_deltas() { return deltas; }
  value_idx* get_mst_src() { return mst_src; }
  value_idx* get_mst_dst() { return mst_dst; }
  value_t* get_mst_weights() { return mst_weights; }

  /**
   * The number of clusters is set by the algorithm once it is known.
   * @param n_clusters_ number of resulting clusters
   */
  void set_n_clusters(int n_clusters_) { n_clusters = n_clusters_; }

 protected:
  const raft::handle_t& get_handle() { return handle; }

  const raft::handle_t& handle;

  int n_leaves;
  int n_clusters;

  value_idx* labels;  // size n_leaves

  // Dendrogram
  value_idx* children;  // size n_leaves * 2
  value_idx* sizes;     // size n_leaves
  value_t* deltas;      // size n_leaves

  // MST (size n_leaves - 1).
  value_idx* mst_src;
  value_idx* mst_dst;
  value_t* mst_weights;
};

/**
 * Plain old container object to consolidate output
 * arrays. This object is intentionally kept simple
 * and straightforward in order to ease its use
 * in the Python layer. For this reason, the MST
 * arrays and renumbered dendrogram array, as well
 * as its aggregated distances/cluster sizes, are
 * kept separate. The condensed hierarchy is computed
 * and populated in a separate object because its size
 * is not known ahead of time. An RMM device vector is
 * held privately and stabilities initialized explicitly
 * since that size is also not known ahead of time.
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
class hdbscan_output : public robust_single_linkage_output<value_idx, value_t> {
 public:
  hdbscan_output(const raft::handle_t& handle_,
                 int n_leaves_,
                 value_idx* labels_,
                 value_t* probabilities_,
                 value_idx* children_,
                 value_idx* sizes_,
                 value_t* deltas_,
                 value_idx* mst_src_,
                 value_idx* mst_dst_,
                 value_t* mst_weights_)
    : robust_single_linkage_output<value_idx, value_t>(
        handle_, n_leaves_, labels_, children_, sizes_, deltas_, mst_src_, mst_dst_, mst_weights_),
      probabilities(probabilities_),
      stabilities(0, handle_.get_stream()),
      condensed_tree(handle_, n_leaves_),
      inverse_label_map(0, handle_.get_stream())
  {
  }

  // Using getters here, making the members private and forcing
  // consistent state with the constructor. This should make
  // it much easier to use / debug.
  value_t* get_probabilities() { return probabilities; }
  value_t* get_stabilities() { return stabilities.data(); }
  value_idx* get_inverse_label_map() { return inverse_label_map.data(); }
  // internal function
  rmm::device_uvector<value_idx>& _get_inverse_label_map() { return inverse_label_map; }

  /**
   * Once n_clusters is known, the stabilities array
   * can be initialized.
   * @param n_clusters_
   */
  void set_n_clusters(int n_clusters_)
  {
    robust_single_linkage_output<value_idx, value_t>::set_n_clusters(n_clusters_);
    stabilities.resize(n_clusters_,
                       robust_single_linkage_output<value_idx, value_t>::get_handle().get_stream());
  }

  CondensedHierarchy<value_idx, value_t>& get_condensed_tree() { return condensed_tree; }

 private:
  value_t* probabilities;  // size n_leaves
  // inversely maps normalized labels to pre-normalized labels
  // used for out-of-sample prediction
  rmm::device_uvector<value_idx> inverse_label_map;  // size n_clusters

  // Size not known ahead of time. Initialize
  // with `initialize_stabilities()` method.
  rmm::device_uvector<value_t> stabilities;

  // Use condensed hierarchy to wrap
  // condensed tree outputs since we do not
  // know the size ahead of time.
  CondensedHierarchy<value_idx, value_t> condensed_tree;
};

template class CondensedHierarchy<int64_t, float>;

/**
 * Container object for computing and storing intermediate information needed later for computing
 * membership vectors and approximate predict. Users are only expected to create an instance of this
 * object, the hdbscan method will do the rest.
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
class PredictionData {
 public:
  PredictionData(const raft::handle_t& handle_, value_idx m, value_idx n, value_t* core_dists_)
    : handle(handle_),
      exemplar_idx(0, handle.get_stream()),
      exemplar_label_offsets(0, handle.get_stream()),
      n_selected_clusters(0),
      selected_clusters(0, handle.get_stream()),
      deaths(0, handle.get_stream()),
      core_dists(core_dists_),
      index_into_children(0, handle.get_stream()),
      n_exemplars(0),
      n_rows(m),
      n_cols(n)
  {
  }
  size_t n_rows;
  size_t n_cols;

  // Using getters here, making the members private and forcing
  // consistent state with the constructor. This should make
  // it much easier to use / debug.
  value_idx get_n_exemplars() { return n_exemplars; }
  value_idx get_n_selected_clusters() { return n_selected_clusters; }
  value_idx* get_exemplar_idx() { return exemplar_idx.data(); }
  value_idx* get_exemplar_label_offsets() { return exemplar_label_offsets.data(); }
  value_idx* get_selected_clusters() { return selected_clusters.data(); }
  value_t* get_deaths() { return deaths.data(); }
  value_t* get_core_dists() { return core_dists; }
  value_idx* get_index_into_children() { return index_into_children.data(); }

  /**
   * Resizes the buffers in the PredictionData object.
   *
   * @param[in] handle raft handle for resource reuse
   * @param[in] n_exemplars_ number of exemplar points
   * @param[in] n_selected_clusters_ number of selected clusters in the final clustering
   * @param[in] n_edges_ number of edges in the condensed hierarchy
   */
  void allocate(const raft::handle_t& handle,
                value_idx n_exemplars_,
                value_idx n_selected_clusters_,
                value_idx n_edges_);

  /**
   * Resize buffers for cluster deaths to n_clusters
   * @param handle raft handle for ordering cuda operations
   * @param n_clusters_ number of clusters
   */
  void set_n_clusters(const raft::handle_t& handle, value_idx n_clusters_)
  {
    deaths.resize(n_clusters_, handle.get_stream());
  }

 private:
  const raft::handle_t& handle;
  rmm::device_uvector<value_idx> exemplar_idx;
  rmm::device_uvector<value_idx> exemplar_label_offsets;
  value_idx n_exemplars;
  value_idx n_selected_clusters;
  rmm::device_uvector<value_idx> selected_clusters;
  rmm::device_uvector<value_t> deaths;
  value_t* core_dists;
  rmm::device_uvector<value_idx> index_into_children;
};

template class PredictionData<int64_t, float>;

void generate_prediction_data(const raft::handle_t& handle,
                              CondensedHierarchy<int64_t, float>& condensed_tree,
                              int64_t* labels,
                              int64_t* inverse_label_map,
                              int n_selected_clusters,
                              PredictionData<int64_t, float>& prediction_data);

};  // namespace Common
};  // namespace HDBSCAN

/**
 * Executes HDBSCAN clustering on an mxn-dimensional input array, X.
 *
 *   Note that while the algorithm is generally deterministic and should
 *   provide matching results between RAPIDS and the Scikit-learn Contrib
 *   versions, the construction of the k-nearest neighbors graph and
 *   minimum spanning tree can introduce differences between the two
 *   algorithms, especially when several nearest neighbors around a
 *   point might have the same distance. While the differences in
 *   the minimum spanning trees alone might be subtle, they can
 *   (and often will) lead to some points being assigned different
 *   cluster labels between the two implementations.
 *
 * @param[in] handle raft handle for resource reuse
 * @param[in] X array (size m, n) on device in row-major format
 * @param m number of rows in X
 * @param n number of columns in X
 * @param metric distance metric to use
 * @param params struct of configuration hyper-parameters
 * @param out struct of output data and arrays on device
 * @param core_dists array (size m, 1) of core distances
 */
void hdbscan(const raft::handle_t& handle,
             const float* X,
             size_t m,
             size_t n,
             ML::distance::DistanceType metric,
             HDBSCAN::Common::HDBSCANParams& params,
             HDBSCAN::Common::hdbscan_output<int64_t, float>& out,
             float* core_dists);

void build_condensed_hierarchy(const raft::handle_t& handle,
                               const int64_t* children,
                               const float* delta,
                               const int64_t* sizes,
                               int min_cluster_size,
                               int n_leaves,
                               HDBSCAN::Common::CondensedHierarchy<int64_t, float>& condensed_tree);

void _extract_clusters(const raft::handle_t& handle,
                       size_t n_leaves,
                       int n_edges,
                       int64_t* parents,
                       int64_t* children,
                       float* lambdas,
                       int64_t* sizes,
                       int64_t* labels,
                       float* probabilities,
                       HDBSCAN::Common::CLUSTER_SELECTION_METHOD cluster_selection_method,
                       bool allow_single_cluster,
                       int64_t max_cluster_size,
                       float cluster_selection_epsilon);

void compute_all_points_membership_vectors(
  const raft::handle_t& handle,
  HDBSCAN::Common::CondensedHierarchy<int64_t, float>& condensed_tree,
  HDBSCAN::Common::PredictionData<int64_t, float>& prediction_data,
  const float* X,
  ML::distance::DistanceType metric,
  float* membership_vec,
  size_t batch_size = 4096);

void compute_membership_vector(const raft::handle_t& handle,
                               HDBSCAN::Common::CondensedHierarchy<int64_t, float>& condensed_tree,
                               HDBSCAN::Common::PredictionData<int64_t, float>& prediction_data,
                               const float* X,
                               const float* points_to_predict,
                               size_t n_prediction_points,
                               int min_samples,
                               ML::distance::DistanceType metric,
                               float* membership_vec,
                               size_t batch_size = 4096);

void out_of_sample_predict(const raft::handle_t& handle,
                           HDBSCAN::Common::CondensedHierarchy<int64_t, float>& condensed_tree,
                           HDBSCAN::Common::PredictionData<int64_t, float>& prediction_data,
                           const float* X,
                           int64_t* labels,
                           const float* points_to_predict,
                           size_t n_prediction_points,
                           ML::distance::DistanceType metric,
                           int min_samples,
                           int64_t* out_labels,
                           float* out_probabilities);

namespace HDBSCAN::HELPER {

/**
 * @brief Compute the core distances for each point in the training matrix
 *
 * @param[in] handle raft handle for resource reuse
 * @param[in] X array (size m, n) on device in row-major format
 * @param[out] core_dists array (size m, 1) of core distances
 * @param m number of rows in X
 * @param n number of columns in X
 * @param metric distance metric to use
 * @param min_samples minimum number of samples to use for computing core distances
 */
void compute_core_dists(const raft::handle_t& handle,
                        const float* X,
                        float* core_dists,
                        size_t m,
                        size_t n,
                        ML::distance::DistanceType metric,
                        int min_samples);

/**
 * @brief Compute the map from final, normalize labels to the labels in the CondensedHierarchy
 *
 * @param[in] handle raft handle for resource reuse
 * @param[in] condensed_tree the Condensed Hierarchy object
 * @param[in] n_leaves number of leaves in the input data
 * @param[in] cluster_selection_method cluster selection method
 * @param[out] inverse_label_map rmm::device_uvector of size 0. It will be resized during the
 * computation
 * @param[in] allow_single_cluster allow single cluster
 * @param[in] max_cluster_size max cluster size
 * @param[in] cluster_selection_epsilon cluster selection epsilon
 */
void compute_inverse_label_map(const raft::handle_t& handle,
                               HDBSCAN::Common::CondensedHierarchy<int64_t, float>& condensed_tree,
                               size_t n_leaves,
                               HDBSCAN::Common::CLUSTER_SELECTION_METHOD cluster_selection_method,
                               rmm::device_uvector<int64_t>& inverse_label_map,
                               bool allow_single_cluster,
                               int64_t max_cluster_size,
                               float cluster_selection_epsilon);

}  // namespace HDBSCAN::HELPER
}  // END namespace ML
