/*
 * Copyright (c) 2018-2025, NVIDIA CORPORATION.
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
#include <cuml/common/logger.hpp>

#include <cstddef>
#include <cstdint>

namespace raft {
class handle_t;
}

namespace ML {
namespace Dbscan {

enum EpsNnMethod { BRUTE_FORCE, RBC };

/**
 * @defgroup DbscanCpp C++ implementation of Dbscan algo
 * @brief Fits a DBSCAN model on an input feature matrix and outputs the labels
 *        and core_sample_indices.
 * @param[in] handle cuml handle to use across the algorithm
 * @param[in] input row-major input feature matrix or distance matrix
 * @param[in] n_rows number of samples in the input feature matrix
 * @param[in] n_cols number of features in the input feature matrix
 * @param[in] eps epsilon value to use for epsilon-neighborhood determination
 * @param[in] min_pts minimum number of points to determine a cluster
 * @param[in] metric metric type (or precomputed)
 * @param[out] labels (size n_rows) output labels array
 * @param[out] core_sample_indices (size n_rows) output array containing the
 *             indices of each core point. If the number of core points is less
 *             than n_rows, the right will be padded with -1. Setting this to
 *             NULL will prevent calculating the core sample indices
 * @param[in] sample_weight (size n_rows) input array containing the
 *            weight of each sample to be taken instead of a plain sum to
 *            fulfill the min_pts criteria for core points.
 *            NULL will default to weights of 1 for all samples
 * @param[in] max_bytes_per_batch the maximum number of megabytes to be used for
 *            each batch of the pairwise distance calculation. This enables the
 *            trade off between memory usage and algorithm execution time.
 * @param[in] eps_nn_method method for computing epsilon neighborhood
 * @param[in] verbosity verbosity level for logging messages during execution
 * @param[in] opg whether we are running in a multi-node multi-GPU context
 * @{
 */

void fit(const raft::handle_t& handle,
         float* input,
         int n_rows,
         int n_cols,
         float eps,
         int min_pts,
         ML::distance::DistanceType metric,
         int* labels,
         int* core_sample_indices            = nullptr,
         float* sample_weight                = nullptr,
         size_t max_bytes_per_batch          = 0,
         EpsNnMethod eps_nn_method           = BRUTE_FORCE,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info,
         bool opg                            = false);
void fit(const raft::handle_t& handle,
         double* input,
         int n_rows,
         int n_cols,
         double eps,
         int min_pts,
         ML::distance::DistanceType metric,
         int* labels,
         int* core_sample_indices            = nullptr,
         double* sample_weight               = nullptr,
         size_t max_bytes_per_batch          = 0,
         EpsNnMethod eps_nn_method           = BRUTE_FORCE,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info,
         bool opg                            = false);

void fit(const raft::handle_t& handle,
         float* input,
         int64_t n_rows,
         int64_t n_cols,
         float eps,
         int min_pts,
         ML::distance::DistanceType metric,
         int64_t* labels,
         int64_t* core_sample_indices        = nullptr,
         float* sample_weight                = nullptr,
         size_t max_bytes_per_batch          = 0,
         EpsNnMethod eps_nn_method           = BRUTE_FORCE,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info,
         bool opg                            = false);
void fit(const raft::handle_t& handle,
         double* input,
         int64_t n_rows,
         int64_t n_cols,
         double eps,
         int min_pts,
         ML::distance::DistanceType metric,
         int64_t* labels,
         int64_t* core_sample_indices        = nullptr,
         double* sample_weight               = nullptr,
         size_t max_bytes_per_batch          = 0,
         EpsNnMethod eps_nn_method           = BRUTE_FORCE,
         rapids_logger::level_enum verbosity = rapids_logger::level_enum::info,
         bool opg                            = false);

/** @} */

}  // namespace Dbscan
}  // namespace ML
