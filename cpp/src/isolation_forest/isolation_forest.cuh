/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/ensemble/isolation_forest.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/transform.h>

#include "isolation_tree_builder.cuh"

#include <climits>
#include <cmath>

namespace ML {

/** @brief Compute average path length c(n) = 2H(n-1) - 2(n-1)/n for normalization. */
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

  void error_checking(const T* input, size_t n_rows, int n_cols) const
  {
    ASSERT((n_rows > 0), "Invalid n_rows %zu", n_rows);
    ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);
    ASSERT(IsolationTree::is_dev_ptr(input), "IF Error: Expected input to be a GPU pointer");
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
           size_t n_rows,
           int n_cols,
           IsolationForestModel<T>* model)
  {
    raft::common::nvtx::range fun_scope("IsolationForest::fit @isolation_forest.cuh");
    this->error_checking(input, n_rows, n_cols);
    
    int n_sampled_rows = std::min(params.max_samples, static_cast<int>(std::min(n_rows, static_cast<size_t>(INT_MAX))));
    int max_depth = params.max_depth;
    if (max_depth <= 0) {
      max_depth = static_cast<int>(std::ceil(std::log2(static_cast<double>(n_sampled_rows))));
      max_depth = std::max(max_depth, 1);
    }
    
    model->params             = params;
    model->n_features         = n_cols;
    model->n_samples_per_tree = n_sampled_rows;
    model->c_normalization    = compute_c_normalization<T>(n_sampled_rows);
    
    auto stream = handle.get_stream();
    size_t trees_size = params.n_estimators * sizeof(IsolationTree::IFTree<T>);
    model->fast_trees = rmm::device_buffer(trees_size, stream);
    
    IsolationTree::build_isolation_forest(
        handle, input, n_rows, n_cols,
        params.n_estimators, n_sampled_rows, max_depth,
        params.seed,
        static_cast<IsolationTree::IFTree<T>*>(model->fast_trees.data()));
    
    handle.sync_stream();
  }

  /** @brief Compute average path lengths for each sample. Input must be row-major. */
  void compute_path_lengths(const raft::handle_t& handle,
                            const IsolationForestModel<T>* model,
                            const T* input,
                            size_t n_rows,
                            int n_cols,
                            T* avg_path_lengths) const
  {
    raft::common::nvtx::range fun_scope("IF::compute_path_lengths @isolation_forest.cuh");
    cudaStream_t stream = handle.get_stream();

    auto* fast_trees = static_cast<const IsolationTree::IFTree<T>*>(model->fast_trees.data());
    int threads = 256;
    size_t blocks = (n_rows + threads - 1) / threads;
    
    IsolationTree::compute_path_lengths_kernel<T><<<blocks, threads, 0, stream>>>(
        input, n_rows, n_cols, fast_trees, params.n_estimators, avg_path_lengths);
    
    RAFT_CUDA_TRY(cudaGetLastError());
    handle.sync_stream(stream);
  }

  /** @brief Compute anomaly scores: score = 2^(-avg_path_length / c(n)). */
  void compute_anomaly_scores(const raft::handle_t& handle,
                              const IsolationForestModel<T>* model,
                              const T* avg_path_lengths,
                              size_t n_rows,
                              T* scores) const
  {
    cudaStream_t stream = handle.get_stream();
    T c_n               = model->c_normalization;

    if (c_n <= T(0)) {
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
};

template class IsolationForest<float>;
template class IsolationForest<double>;

}  // namespace ML
