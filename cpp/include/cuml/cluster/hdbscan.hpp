/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <raft/linalg/distance_type.h>
#include <raft/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cstddef>

namespace ML {
namespace HDBSCAN {
namespace Common {

template <typename value_idx, typename value_t>
class CondensedHierarchy {
 public:
  /**
   * Constructs an empty condensed hierarchy object which requires
   * condense() to be called in order to populate the state.
   * @param handle_
   * @param n_leaves_
   */
  CondensedHierarchy(const raft::handle_t &handle_, size_t n_leaves_);

  /**
   * Constructs a condensed hierarchy object with existing arrays
   * which already contain a condensed hierarchy.
   * @param handle_
   * @param n_leaves_
   * @param size_
   * @param n_edges_
   * @param parents_
   * @param children_
   * @param lambdas_
   * @param sizes_
   */
  CondensedHierarchy(const raft::handle_t &handle_, size_t n_leaves_,
                     int n_edges_, value_idx *parents_, value_idx *children_,
                     value_t *lambdas_, value_idx *sizes_);

  /**
   * Constructs a condensed hierarchy object with by moving
   * rmm::device_uvector. Used to construct cluster trees
   * @param handle_
   * @param n_leaves_
   * @param size_
   * @param n_edges_
   * @param parents_
   * @param children_
   * @param lambdas_
   * @param sizes_
   */
  CondensedHierarchy(const raft::handle_t &handle_, size_t n_leaves_,
                     int n_edges_, int n_clusters_,
                     rmm::device_uvector<value_idx> &&parents_,
                     rmm::device_uvector<value_idx> &&children_,
                     rmm::device_uvector<value_t> &&lambdas_,
                     rmm::device_uvector<value_idx> &&sizes_);
  /**
   * Populates the condensed hierarchy object with the output
   * from Condense::condense_hierarchy
   * @param full_parents
   * @param full_children
   * @param full_lambdas
   * @param full_sizes
   */
  void condense(value_idx *full_parents, value_idx *full_children,
                value_t *full_lambdas, value_idx *full_sizes,
                value_idx size = -1);

  value_idx get_cluster_tree_edges();

  value_idx *get_parents() { return parents.data(); }
  value_idx *get_children() { return children.data(); }
  value_t *get_lambdas() { return lambdas.data(); }
  value_idx *get_sizes() { return sizes.data(); }
  value_idx get_n_edges() { return n_edges; }
  int get_n_clusters() { return n_clusters; }
  value_idx get_n_leaves() const { return n_leaves; }

 private:
  const raft::handle_t &handle;

  rmm::device_uvector<value_idx> parents;
  rmm::device_uvector<value_idx> children;
  rmm::device_uvector<value_t> lambdas;
  rmm::device_uvector<value_idx> sizes;

  size_t n_edges;
  size_t n_leaves;
  int n_clusters;
  value_idx root_cluster;
};

enum CLUSTER_SELECTION_METHOD {

  EOM = 0,
  LEAF = 1
};

class HDBSCANParams {
 public:
  int k = 5;
  int min_samples = 5;
  int min_cluster_size = 5;
  int max_cluster_size = 0;

  float cluster_selection_epsilon = 0.0;

  bool allow_single_cluster = false;

  float alpha = 1.0;

  CLUSTER_SELECTION_METHOD cluster_selection_method =
    CLUSTER_SELECTION_METHOD::EOM;
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
class hdbscan_output {
 public:
  hdbscan_output(const raft::handle_t &handle_, int n_leaves_,
                 value_idx *labels_, value_t *probabilities_,
                 value_idx *children_, value_idx *sizes_, value_t *deltas_,
                 value_idx *mst_src_, value_idx *mst_dst_,
                 value_t *mst_weights_)
    : handle(handle_),
      n_leaves(n_leaves_),
      n_clusters(0),
      labels(labels_),
      probabilities(probabilities_),
      children(children_),
      sizes(sizes_),
      deltas(deltas_),
      mst_src(mst_src_),
      mst_dst(mst_dst_),
      mst_weights(mst_weights_),
      stabilities(0, handle_.get_stream()),
      condensed_tree(handle_, n_leaves_) {}

  // Using getters here, making the members private and forcing
  // consistent state with the constructor. This should make
  // it much easier to use / debug.
  int get_n_leaves() const { return n_leaves; }
  int get_n_clusters() const { return n_clusters; }
  value_idx *get_labels() { return labels; }
  value_t *get_probabilities() { return probabilities; }
  value_idx *get_children() { return children; }
  value_idx *get_sizes() { return sizes; }
  value_t *get_deltas() { return deltas; }
  value_idx *get_mst_src() { return mst_src; }
  value_idx *get_mst_dst() { return mst_dst; }
  value_t *get_mst_weights() { return mst_weights; }
  value_t *get_stabilities() {
    ASSERT(stabilities.size() > 0, "stabilities needs to be initialized");
    return stabilities.data();
  }

  /**
   * Once n_clusters is known, the stabilities array
   * can be initialized.
   * @param n_clusters_
   */
  void set_n_clusters(int n_clusters_) {
    n_clusters = n_clusters_;
    stabilities.resize(n_clusters_, handle.get_stream());
  }

  CondensedHierarchy<value_idx, value_t> &get_condensed_tree() {
    return condensed_tree;
  }

 private:
  const raft::handle_t &handle;

  int n_leaves;
  int n_clusters;

  value_idx *labels;       // size n_leaves
  value_t *probabilities;  // size n_leaves

  // Dendrogram
  value_idx *children;  // size n_leaves * 2
  value_idx *sizes;     // size n_leaves
  value_t *deltas;      // size n_leaves

  // MST (size n_leaves - 1).
  value_idx *mst_src;
  value_idx *mst_dst;
  value_t *mst_weights;

  // Size not known ahead of time. Initialize
  // with `initialize_stabilities()` method.
  rmm::device_uvector<value_t> stabilities;

  // Use condensed hierarchy to wrap
  // condensed tree outputs since we do not
  // know the size ahead of time.
  CondensedHierarchy<value_idx, value_t> condensed_tree;
};

template class CondensedHierarchy<int, float>;

};  // namespace Common
};  // namespace HDBSCAN

void hdbscan(const raft::handle_t &handle, const float *X, size_t m, size_t n,
             raft::distance::DistanceType metric,
             HDBSCAN::Common::HDBSCANParams &params,
             HDBSCAN::Common::hdbscan_output<int, float> &out);

void robust_single_linkage(const raft::handle_t &handle, const float *X,
                           size_t m, size_t n,
                           raft::distance::DistanceType metric,
                           HDBSCAN::Common::HDBSCANParams &params,
                           HDBSCAN::Common::hdbscan_output<int, float> &out);
};  // end namespace ML