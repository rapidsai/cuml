/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "isolation_forest.cuh"

#include <cuml/ensemble/isolation_forest.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

namespace ML {

// Explicit instantiation of compute_c_normalization
template float compute_c_normalization<float>(int n);
template double compute_c_normalization<double>(int n);

void fit(const raft::handle_t& handle,
         IsolationForestF* forest,
         const float* input,
         int n_rows,
         int n_cols,
         const IF_params& params,
         rapids_logger::level_enum verbosity)
{
  ML::default_logger().set_level(verbosity);
  IsolationForest<float> if_model(params);
  if_model.fit(handle, input, n_rows, n_cols, forest);
}

void fit(const raft::handle_t& handle,
         IsolationForestD* forest,
         const double* input,
         int n_rows,
         int n_cols,
         const IF_params& params,
         rapids_logger::level_enum verbosity)
{
  ML::default_logger().set_level(verbosity);
  IsolationForest<double> if_model(params);
  if_model.fit(handle, input, n_rows, n_cols, forest);
}

void score_samples(const raft::handle_t& handle,
                   const IsolationForestF* forest,
                   const float* input,
                   int n_rows,
                   int n_cols,
                   float* scores,
                   rapids_logger::level_enum verbosity)
{
  ML::default_logger().set_level(verbosity);
  IsolationForest<float> if_model(forest->params);

  // Compute average path lengths
  rmm::device_uvector<float> avg_path_lengths(n_rows, handle.get_stream());
  if_model.compute_path_lengths(handle, forest, input, n_rows, n_cols, avg_path_lengths.data());

  // Convert to anomaly scores
  if_model.compute_anomaly_scores(handle, forest, avg_path_lengths.data(), n_rows, scores);
}

void score_samples(const raft::handle_t& handle,
                   const IsolationForestD* forest,
                   const double* input,
                   int n_rows,
                   int n_cols,
                   double* scores,
                   rapids_logger::level_enum verbosity)
{
  ML::default_logger().set_level(verbosity);
  IsolationForest<double> if_model(forest->params);

  // Compute average path lengths
  rmm::device_uvector<double> avg_path_lengths(n_rows, handle.get_stream());
  if_model.compute_path_lengths(handle, forest, input, n_rows, n_cols, avg_path_lengths.data());

  // Convert to anomaly scores
  if_model.compute_anomaly_scores(handle, forest, avg_path_lengths.data(), n_rows, scores);
}

void predict(const raft::handle_t& handle,
             const IsolationForestF* forest,
             const float* input,
             int n_rows,
             int n_cols,
             int* predictions,
             float threshold,
             rapids_logger::level_enum verbosity)
{
  ML::default_logger().set_level(verbosity);
  cudaStream_t stream = handle.get_stream();

  // First compute anomaly scores
  rmm::device_uvector<float> scores(n_rows, stream);
  score_samples(handle, forest, input, n_rows, n_cols, scores.data(), verbosity);

  // Convert scores to predictions: 1 for anomaly (score >= threshold), -1 for normal
  thrust::transform(rmm::exec_policy(stream),
                    scores.data(),
                    scores.data() + n_rows,
                    predictions,
                    [threshold] __device__(float score) { return score >= threshold ? 1 : -1; });

  handle.sync_stream(stream);
}

void predict(const raft::handle_t& handle,
             const IsolationForestD* forest,
             const double* input,
             int n_rows,
             int n_cols,
             int* predictions,
             double threshold,
             rapids_logger::level_enum verbosity)
{
  ML::default_logger().set_level(verbosity);
  cudaStream_t stream = handle.get_stream();

  // First compute anomaly scores
  rmm::device_uvector<double> scores(n_rows, stream);
  score_samples(handle, forest, input, n_rows, n_cols, scores.data(), verbosity);

  // Convert scores to predictions: 1 for anomaly (score >= threshold), -1 for normal
  thrust::transform(rmm::exec_policy(stream),
                    scores.data(),
                    scores.data() + n_rows,
                    predictions,
                    [threshold] __device__(double score) { return score >= threshold ? 1 : -1; });

  handle.sync_stream(stream);
}

template <class T>
void build_treelite_forest(TreeliteModelHandle* model,
                           const IsolationForestModel<T>* forest,
                           int n_features)
{
  // TODO: Implement treelite export for FIL inference
  // This will require setting up the proper leaf output format
  // for anomaly scoring
  ASSERT(false, "Treelite export not yet implemented for IsolationForest");
}

// Explicit template instantiations
template void build_treelite_forest<float>(TreeliteModelHandle* model,
                                           const IsolationForestModel<float>* forest,
                                           int n_features);
template void build_treelite_forest<double>(TreeliteModelHandle* model,
                                            const IsolationForestModel<double>* forest,
                                            int n_features);

}  // namespace ML
