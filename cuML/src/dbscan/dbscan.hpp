/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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
#include <cuML.hpp>

namespace ML{

/**
 * @brief Fits a DBSCAN model on an input feature matrix and outputs the labels.
 * @param handle: cuml handle to use across the algorithm
 * @param input: row-major input feature matrix
 * @param n_rows: number of samples in the input feature matrix
 * @param n_cols: number of features in the input feature matrix
 * @param eps: the epsilon value to use for epsilon-neighborhood determination
 * @param min_pts: minimum number of points to determine a cluster
 * @param labels: (size n_rows) output labels array
 * @param max_mem_bytes: the maximum number of bytes to be used for each batch of
 *          the pairwise distance calculation. This enables the trade off between
 *          memory usage and algorithm execution time.
 * @param verbose: print useful information as algorithm executes
 */
void dbscanFit(const cumlHandle& handle, float *input, int n_rows, int n_cols, float eps, int min_pts,
		       int *labels, size_t max_bytes_per_batch, bool verbose = false);

/**
 * @brief Fits a DBSCAN model on an input feature matrix and outputs the labels.
 * @param handle: cuml handle to use across the algorithm
 * @param input: row-major input feature matrix
 * @param n_rows: number of samples in the input feature matrix
 * @param n_cols: number of features in the input feature matrix
 * @param eps: the epsilon value to use for epsilon-neighborhood determination
 * @param min_pts: minimum number of points to determine a cluster
 * @param labels: (size n_rows) output labels array
 * @param max_mem_bytes: the maximum number of bytes to be used for each batch of
 *          the pairwise distance calculation. This enables the trade off between
 *          memory usage and algorithm execution time.
 * @param verbose: print useful information as algorithm executes
 */
void dbscanFit(const cumlHandle& handle, double *input, int n_rows, int n_cols, double eps, int min_pts,
		       int *labels, size_t max_bytes_per_batch, bool verbose = false);

}

