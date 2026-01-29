/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/ensemble/isolation_forest.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/random/permute.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/transform.h>

#include <decisiontree/batched-levelalgo/quantiles.cuh>
#include <decisiontree/decisiontree.cuh>
#include <decisiontree/treelite_util.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num()  0
#define omp_get_max_threads() 1
#endif

#include <cmath>

namespace ML {

/**
 * @brief Compute the average path length c(n) for normalization.
 *
 * c(n) = 2H(n-1) - 2(n-1)/n
 * where H(i) is the harmonic number ≈ ln(i) + Euler's constant (0.5772156649...)
 *
 * For n <= 1, returns 0.
 * For n == 2, returns 1.
 */
template <typename T>
T compute_c_normalization(int n)
{
  if (n <= 1) return T(0);
  if (n == 2) return T(1);

  constexpr T euler_mascheroni = T(0.5772156649015329);
  T harmonic_n_minus_1         = std::log(T(n - 1)) + euler_mascheroni;
  return T(2) * harmonic_n_minus_1 - T(2) * T(n - 1) / T(n);
}

template <class T>
class IsolationForest {
 private:
  IF_params params;

  /**
   * @brief Get a subsample of row indices for building one tree.
   *
   * For Isolation Forest, we typically subsample without replacement.
   */
  void get_row_sample(int tree_id,
                      int n_rows,
                      int n_sampled_rows,
                      rmm::device_uvector<int>* selected_rows,
                      const cudaStream_t stream)
  {
    raft::common::nvtx::range fun_scope("IF bootstrapping @isolation_forest.cuh");

    // Hash seed with tree_id for uncorrelated samples
    auto rs = DT::fnv1a32_basis;
    rs      = DT::fnv1a32(rs, params.seed);
    rs      = DT::fnv1a32(rs, tree_id);
    raft::random::Rng rng(rs, raft::random::GenPhilox);

    if (params.bootstrap) {
      // Sample with replacement
      rng.uniformInt<int>(selected_rows->data(), n_sampled_rows, 0, n_rows, stream);
    } else {
      // Sample without replacement: generate sequence and shuffle
      thrust::sequence(rmm::exec_policy(stream), selected_rows->begin(), selected_rows->end());

      // If we need all rows, no shuffle needed
      if (n_sampled_rows < n_rows) {
        // Use thrust::shuffle with the RNG
        thrust::default_random_engine g(rs);
        thrust::shuffle(rmm::exec_policy(stream), selected_rows->begin(), selected_rows->end(), g);
      }
    }
  }

  void error_checking(const T* input, int n_rows, int n_cols) const
  {
    ASSERT((n_rows > 0), "Invalid n_rows %d", n_rows);
    ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);
    ASSERT(DT::is_dev_ptr(input), "IF Error: Expected input to be a GPU pointer");
  }

 public:
  IsolationForest(const IF_params& cfg_params) : params(cfg_params) {}

  /**
   * @brief Fit the Isolation Forest model.
   *
   * @param[in] handle raft::handle_t
   * @param[in] input Training data (n_rows x n_cols) in column-major format. Device pointer.
   * @param[in] n_rows Number of training samples
   * @param[in] n_cols Number of features
   * @param[out] model IsolationForestModel to populate
   */
  void fit(const raft::handle_t& handle,
           const T* input,
           int n_rows,
           int n_cols,
           IsolationForestModel<T>* model)
  {
    raft::common::nvtx::range fun_scope("IsolationForest::fit @isolation_forest.cuh");
    this->error_checking(input, n_rows, n_cols);

    // Determine number of samples per tree
    int n_sampled_rows = std::min(params.max_samples, n_rows);

    // Auto-compute max_depth if not specified: ceil(log2(n_sampled_rows))
    int max_depth = params.max_depth;
    if (max_depth <= 0) {
      max_depth = static_cast<int>(std::ceil(std::log2(static_cast<double>(n_sampled_rows))));
      max_depth = std::max(max_depth, 1);  // At least depth 1
    }

    // Compute number of features to use
    int n_features_to_use = n_cols;
    if (params.max_features < 1.0f) {
      n_features_to_use = static_cast<int>(std::round(params.max_features * n_cols));
      n_features_to_use = std::max(n_features_to_use, 1);
    }

    // Set up decision tree parameters for isolation trees
    DT::DecisionTreeParams tree_params;
    tree_params.max_depth            = max_depth;
    tree_params.max_leaves           = -1;  // No leaf limit
    tree_params.max_features         = static_cast<float>(n_features_to_use) / n_cols;
    tree_params.max_n_bins           = 256;  // Standard binning
    tree_params.min_samples_leaf     = 1;    // Allow single-sample leaves
    tree_params.min_samples_split    = 2;    // Standard minimum split
    tree_params.min_impurity_decrease = 0.0f;
    tree_params.max_batch_size       = params.max_batch_size;
    tree_params.split_criterion      = CRITERION::RANDOM;  // Key: use random splits

    int n_streams = params.n_streams;
    ASSERT(static_cast<std::size_t>(n_streams) <= handle.get_stream_pool_size(),
           "IF n_streams (=%d) should be <= handle.n_streams (=%lu)",
           n_streams,
           handle.get_stream_pool_size());

    // Compute quantiles for binning
    auto [quantiles, quantiles_array, n_bins_array] =
      DT::computeQuantiles(handle, input, tree_params.max_n_bins, n_rows, n_cols);

    // Adjust n_streams if fewer trees
    if (params.n_estimators < n_streams) n_streams = params.n_estimators;

    // Allocate row sample buffers for each stream
    std::deque<rmm::device_uvector<int>> selected_rows;
    for (int i = 0; i < n_streams; i++) {
      selected_rows.emplace_back(n_sampled_rows, handle.get_stream_from_stream_pool(i));
    }

    // Dummy labels: Isolation Forest is unsupervised, but the tree builder expects labels.
    // We use dummy labels (all zeros) since RANDOM criterion ignores them.
    rmm::device_uvector<T> dummy_labels(n_rows, handle.get_stream());
    thrust::fill(rmm::exec_policy(handle.get_stream()),
                 dummy_labels.begin(),
                 dummy_labels.end(),
                 T(0));
    handle.sync_stream();

    // Store model metadata
    model->params             = params;
    model->n_features         = n_cols;
    model->n_samples_per_tree = n_sampled_rows;
    model->c_normalization    = compute_c_normalization<T>(n_sampled_rows);
    model->trees.resize(params.n_estimators);

#pragma omp parallel for num_threads(n_streams)
    for (int i = 0; i < params.n_estimators; i++) {
      int stream_id = omp_get_thread_num();
      auto s        = handle.get_stream_from_stream_pool(stream_id);

      // Get subsample for this tree
      this->get_row_sample(i, n_rows, n_sampled_rows, &selected_rows[stream_id], s);

      // Build isolation tree
      // n_unique_labels=1 since we're doing unsupervised isolation
      model->trees[i] = DT::DecisionTree::fit(handle,
                                              s,
                                              input,
                                              n_cols,
                                              n_rows,
                                              dummy_labels.data(),
                                              &selected_rows[stream_id],
                                              1,  // n_unique_labels (dummy)
                                              tree_params,
                                              params.seed,
                                              quantiles,
                                              i);
    }

    // Cleanup
    handle.sync_stream_pool();
    handle.sync_stream();
  }

  /**
   * @brief Compute path lengths for each sample across all trees.
   *
   * The path length is the depth at which a sample reaches a leaf node.
   * For external nodes (leaves) at depth d with n samples, we add c(n) to account
   * for the expected additional depth if the tree were fully grown.
   */
  void compute_path_lengths(const raft::handle_t& handle,
                            const IsolationForestModel<T>* model,
                            const T* input,
                            int n_rows,
                            int n_cols,
                            T* avg_path_lengths) const
  {
    raft::common::nvtx::range fun_scope("IF::compute_path_lengths @isolation_forest.cuh");
    cudaStream_t stream = handle.get_stream();

    // Initialize path lengths to zero
    thrust::fill(rmm::exec_policy(stream), avg_path_lengths, avg_path_lengths + n_rows, T(0));

    // Copy input to host for tree traversal (similar to RF predict)
    std::vector<T> h_input(std::size_t(n_rows) * n_cols);
    raft::update_host(h_input.data(), input, std::size_t(n_rows) * n_cols, stream);
    handle.sync_stream(stream);

    std::vector<T> h_path_lengths(n_rows, T(0));

    // For each tree, compute path length for each sample
    for (int tree_idx = 0; tree_idx < params.n_estimators; tree_idx++) {
      const auto& tree = *model->trees[tree_idx];

      for (int row_id = 0; row_id < n_rows; row_id++) {
        const T* sample = &h_input[row_id * n_cols];

        // Traverse tree to find path length
        T path_length = compute_single_path_length(tree, sample, n_cols, model->n_samples_per_tree);
        h_path_lengths[row_id] += path_length;
      }
    }

    // Average over all trees
    for (int i = 0; i < n_rows; i++) {
      h_path_lengths[i] /= T(params.n_estimators);
    }

    // Copy back to device
    raft::update_device(avg_path_lengths, h_path_lengths.data(), n_rows, stream);
    handle.sync_stream(stream);
  }

  /**
   * @brief Compute anomaly scores from average path lengths.
   *
   * score = 2^(-avg_path_length / c(n))
   */
  void compute_anomaly_scores(const raft::handle_t& handle,
                              const IsolationForestModel<T>* model,
                              const T* avg_path_lengths,
                              int n_rows,
                              T* scores) const
  {
    cudaStream_t stream = handle.get_stream();
    T c_n               = model->c_normalization;

    if (c_n <= T(0)) {
      // Edge case: if c(n) is 0 or negative, all scores are 0.5
      thrust::fill(rmm::exec_policy(stream), scores, scores + n_rows, T(0.5));
    } else {
      thrust::transform(
        rmm::exec_policy(stream),
        avg_path_lengths,
        avg_path_lengths + n_rows,
        scores,
        [c_n] __device__(T path_length) { return std::pow(T(2), -path_length / c_n); });
    }
    handle.sync_stream(stream);
  }

 private:
  /**
   * @brief Compute path length for a single sample in a single tree.
   *
   * Returns the depth plus an adjustment for the expected depth if the
   * external node were expanded further.
   */
  T compute_single_path_length(const DT::TreeMetaDataNode<T, T>& tree,
                               const T* sample,
                               int n_cols,
                               int n_samples_tree) const
  {
    const auto& sparsetree = tree.sparsetree;
    if (sparsetree.empty()) return T(0);

    int node_idx  = 0;  // Start at root
    T path_length = T(0);

    while (true) {
      const auto& node = sparsetree[node_idx];

      if (node.IsLeaf()) {
        // At leaf: add c(n) for the samples in this leaf
        // InstanceCount() returns the number of samples at this node
        int leaf_samples = static_cast<int>(node.InstanceCount());
        leaf_samples     = std::max(leaf_samples, 1);
        path_length += compute_c_normalization<T>(leaf_samples);
        break;
      }

      // Internal node: go left or right based on split
      path_length += T(1);
      int feature_idx = node.ColumnId();
      T threshold     = node.QueryValue();

      if (feature_idx < 0 || feature_idx >= n_cols) {
        // Invalid feature, stop traversal
        break;
      }

      T feature_value = sample[feature_idx];
      if (feature_value <= threshold) {
        node_idx = static_cast<int>(node.LeftChildId());
      } else {
        node_idx = static_cast<int>(node.RightChildId());
      }

      // Safety check
      if (node_idx < 0 || node_idx >= static_cast<int>(sparsetree.size())) {
        break;
      }
    }

    return path_length;
  }
};

// Explicit template instantiations
template class IsolationForest<float>;
template class IsolationForest<double>;

}  // namespace ML
