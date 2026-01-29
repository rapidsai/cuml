/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/logger.hpp>
#include <cuml/ensemble/treelite_defs.hpp>
#include <cuml/tree/decisiontree.hpp>

#include <memory>
#include <vector>

namespace raft {
class handle_t;  // forward decl
}

namespace ML {

/**
 * @brief Isolation Forest hyperparameters
 */
struct IF_params {
  /**
   * Number of isolation trees in the forest.
   * Default: 100
   */
  int n_estimators = 100;

  /**
   * Number of samples to draw from X to train each isolation tree.
   * - If int, draw max_samples samples.
   * - If float, draw max_samples * n_samples samples.
   * Default: 256 (or n_samples if smaller)
   */
  int max_samples = 256;

  /**
   * Maximum depth of each isolation tree.
   * If -1, max_depth is set to ceil(log2(max_samples)).
   * Default: -1 (auto)
   */
  int max_depth = -1;

  /**
   * Number of features to draw from X to train each isolation tree.
   * - If 1.0, use all features.
   * - If < 1.0, use that fraction of features.
   * Default: 1.0 (all features)
   */
  float max_features = 1.0f;

  /**
   * If True, individual trees are fit on random subsets of the training data
   * sampled with replacement. Otherwise, sampling without replacement.
   * Default: false (sampling without replacement for IF)
   */
  bool bootstrap = false;

  /**
   * Random seed for reproducibility.
   * Default: 0 (use system random)
   */
  uint64_t seed = 0;

  /**
   * Number of concurrent GPU streams for parallel tree building.
   * Default: 4
   */
  int n_streams = 4;

  /**
   * Maximum batch size for tree building.
   * Default: 4096
   */
  int max_batch_size = 4096;
};

/**
 * @brief Metadata for a trained Isolation Forest model
 */
template <class T>
struct IsolationForestModel {
  /** Trained decision trees */
  std::vector<std::shared_ptr<DT::TreeMetaDataNode<T, T>>> trees;
  /** Hyperparameters used for training */
  IF_params params;
  /** Number of features in training data */
  int n_features = 0;
  /** Number of samples used per tree */
  int n_samples_per_tree = 0;
  /** c(n) normalization constant for anomaly scoring */
  T c_normalization = 0;
};

// Type aliases for float and double precision
typedef IsolationForestModel<float> IsolationForestF;
typedef IsolationForestModel<double> IsolationForestD;

/**
 * @brief Compute the average path length c(n) for normalization.
 *
 * c(n) = 2H(n-1) - 2(n-1)/n
 * where H(i) is the harmonic number = ln(i) + Euler's constant (0.5772...)
 *
 * @param n Number of samples
 * @return c(n) normalization constant
 */
template <typename T>
T compute_c_normalization(int n);

/**
 * @brief Fit an Isolation Forest model.
 *
 * @param[in] handle raft::handle_t
 * @param[out] forest Pointer to IsolationForestModel to populate
 * @param[in] input Training data (n_rows x n_cols) in column-major format. Device pointer.
 * @param[in] n_rows Number of training samples
 * @param[in] n_cols Number of features
 * @param[in] params Isolation Forest hyperparameters
 * @param[in] verbosity Logging verbosity level
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
 * @brief Build a Treelite model from a trained Isolation Forest.
 *
 * @param[out] model Treelite model handle
 * @param[in] forest Trained Isolation Forest model
 * @param[in] n_features Number of features
 */
template <class T>
void build_treelite_forest(TreeliteModelHandle* model,
                           const IsolationForestModel<T>* forest,
                           int n_features);

/**
 * @brief Compute anomaly scores for input samples.
 *
 * Score formula: s(x, n) = 2^(-E[h(x)] / c(n))
 * where:
 *   - h(x) is the path length to isolate sample x
 *   - E[h(x)] is the average path length across all trees
 *   - c(n) is the normalization factor
 *
 * Scores close to 1 indicate anomalies, scores close to 0.5 indicate normal points.
 *
 * @param[in] handle raft::handle_t
 * @param[in] forest Trained Isolation Forest model
 * @param[in] input Test data (n_rows x n_cols) in row-major format. Device pointer.
 * @param[in] n_rows Number of test samples
 * @param[in] n_cols Number of features
 * @param[out] scores Anomaly scores for each sample (n_rows). Device pointer.
 * @param[in] verbosity Logging verbosity level
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
 * @brief Predict anomaly labels (1 for anomaly, -1 for normal).
 *
 * Uses a threshold of 0.5 on the anomaly score by default.
 *
 * @param[in] handle raft::handle_t
 * @param[in] forest Trained Isolation Forest model
 * @param[in] input Test data (n_rows x n_cols) in row-major format. Device pointer.
 * @param[in] n_rows Number of test samples
 * @param[in] n_cols Number of features
 * @param[out] predictions Predictions for each sample: 1=anomaly, -1=normal. Device pointer.
 * @param[in] threshold Score threshold for anomaly classification (default 0.5)
 * @param[in] verbosity Logging verbosity level
 */
void predict(const raft::handle_t& handle,
             const IsolationForestF* forest,
             const float* input,
             int n_rows,
             int n_cols,
             int* predictions,
             float threshold                     = 0.5f,
             rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

void predict(const raft::handle_t& handle,
             const IsolationForestD* forest,
             const double* input,
             int n_rows,
             int n_cols,
             int* predictions,
             double threshold                    = 0.5,
             rapids_logger::level_enum verbosity = rapids_logger::level_enum::info);

}  // namespace ML
