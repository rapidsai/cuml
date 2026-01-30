/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/logger.hpp>

namespace raft {
class handle_t;
}

namespace ML {

/**
 * @brief Isolation Forest hyperparameters
 */
struct IF_params {
  int n_estimators = 100;      ///< Number of isolation trees
  int max_samples = 256;       ///< Samples per tree (default 256)
  int max_depth = -1;          ///< Max depth (-1 = auto: ceil(log2(max_samples)))
  float max_features = 1.0f;   ///< Fraction of features per tree
  bool bootstrap = false;      ///< Sample with replacement
  uint64_t seed = 0;           ///< Random seed
  int n_streams = 4;           ///< Parallel GPU streams (unused in fast builder)
  int max_batch_size = 4096;   ///< Batch size (unused in fast builder)
};

/**
 * @brief Trained Isolation Forest model
 */
template <class T>
struct IsolationForestModel {
  IF_params params;
  int n_features = 0;
  int n_samples_per_tree = 0;
  T c_normalization = 0;
  void* fast_trees = nullptr;  ///< Device memory for trees (IFTree<T>*)
};

typedef IsolationForestModel<float> IsolationForestF;
typedef IsolationForestModel<double> IsolationForestD;

/** @brief Compute c(n) = 2H(n-1) - 2(n-1)/n normalization constant. */
template <typename T>
T compute_c_normalization(int n);

/**
 * @brief Fit an Isolation Forest model.
 * @param input Column-major training data [n_rows x n_cols]
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
 * @brief Compute anomaly scores (original paper convention: higher = more anomalous).
 * @param input Row-major test data [n_rows x n_cols]
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
 * @brief Predict anomaly labels (1=anomaly, -1=normal).
 * @param input Row-major test data [n_rows x n_cols]
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
