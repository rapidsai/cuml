/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/logger.hpp>
#include <rmm/device_buffer.hpp>

namespace raft {
class handle_t;
}

namespace ML {

/**
 * @brief Isolation Forest hyperparameters
 */
struct IF_params {
  int n_estimators = 100;   ///< Number of isolation trees
  int max_samples = 256;    ///< Samples per tree (default 256)
  int max_depth = -1;       ///< Max depth (-1 = auto: ceil(log2(max_samples)))
  uint64_t seed = 0;        ///< Random seed
};

/** @brief Trained Isolation Forest model */
template <class T>
struct IsolationForestModel {
  IF_params params;            ///< Hyperparameters used for training
  int n_features = 0;          ///< Number of features in training data
  int n_samples_per_tree = 0;  ///< Samples used per tree (for c(n) calculation)
  T c_normalization = 0;       ///< Precomputed c(n) normalization constant
  rmm::device_buffer fast_trees;  ///< Device memory for trees (IFTree<T>*)
};

typedef IsolationForestModel<float> IsolationForestF;
typedef IsolationForestModel<double> IsolationForestD;

/** @brief Compute c(n) = 2H(n-1) - 2(n-1)/n normalization constant. */
template <typename T>
T compute_c_normalization(int n);

/**
 * @brief Fit an Isolation Forest model.
 *
 * @param[in]  handle    RAFT handle for GPU resources
 * @param[out] forest    Model to populate with trained trees
 * @param[in]  input     Training data, column-major [n_rows × n_cols], device pointer
 * @param[in]  n_rows    Number of training samples
 * @param[in]  n_cols    Number of features
 * @param[in]  params    Hyperparameters (n_estimators, max_samples, max_depth, seed)
 * @param[in]  verbosity Logging level
 */
void fit(const raft::handle_t& handle,
         IsolationForestF* forest,
         const float* input,
         int n_rows,
         int n_cols,
         const IF_params& params,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

void fit(const raft::handle_t& handle,
         IsolationForestD* forest,
         const double* input,
         int n_rows,
         int n_cols,
         const IF_params& params,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

/**
 * @brief Compute anomaly scores.
 *
 * Returns scores following the original paper convention (Liu et al. 2008):
 * - Score ≈ 1.0: anomaly (isolated quickly)
 * - Score ≈ 0.5: normal (average isolation depth)
 * - Score ≈ 0.0: very normal (hard to isolate)
 *
 * @param[in]  handle    RAFT handle for GPU resources
 * @param[in]  forest    Trained Isolation Forest model
 * @param[in]  input     Test data, row-major [n_rows × n_cols], device pointer
 * @param[in]  n_rows    Number of test samples
 * @param[in]  n_cols    Number of features (must match training)
 * @param[out] scores    Anomaly scores [n_rows], device pointer
 * @param[in]  verbosity Logging level
 */
void score_samples(const raft::handle_t& handle,
                   const IsolationForestF* forest,
                   const float* input,
                   int n_rows,
                   int n_cols,
                   float* scores,
                   rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

void score_samples(const raft::handle_t& handle,
                   const IsolationForestD* forest,
                   const double* input,
                   int n_rows,
                   int n_cols,
                   double* scores,
                   rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

/**
 * @brief Predict anomaly labels.
 *
 * @param[in]  handle      RAFT handle for GPU resources
 * @param[in]  forest      Trained Isolation Forest model
 * @param[in]  input       Test data, row-major [n_rows × n_cols], device pointer
 * @param[in]  n_rows      Number of test samples
 * @param[in]  n_cols      Number of features (must match training)
 * @param[out] predictions Labels [n_rows]: 1 = anomaly, -1 = normal, device pointer
 * @param[in]  threshold   Score threshold (default 0.5, higher = more anomalies)
 * @param[in]  verbosity   Logging level
 */
void predict(const raft::handle_t& handle,
             const IsolationForestF* forest,
             const float* input,
             int n_rows,
             int n_cols,
             int* predictions,
             float threshold = 0.5f,
             rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

void predict(const raft::handle_t& handle,
             const IsolationForestD* forest,
             const double* input,
             int n_rows,
             int n_cols,
             int* predictions,
             double threshold = 0.5,
             rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

}  // namespace ML
