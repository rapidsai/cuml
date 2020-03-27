/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuml/cuml.hpp>

namespace ML {

/**
 * Batch division step 1: build an index of the position of each series
 * in its new batch and measure the size of each sub-batch
 *
 * @param[in]  handle     cuML handle
 * @param[in]  d_mask     Boolean mask
 * @param[out] d_index    Index of each series in its new batch
 * @param[in]  batch_size Batch size
 * @param[in]  n_obs      Number of data points per series
 * @return The number of 'true' series in the mask
 */
int divide_batch_build_index(const cumlHandle& handle, const bool* d_mask,
                             int* d_index, int batch_size, int n_obs);

/**
 * Batch division step 2: create both sub-batches from the mask and index
 *
 * @param[in]  handle     cuML handle
 * @param[in]  d_in       Input batch. Each series is a contiguous chunk
 * @param[in]  d_mask     Boolean mask
 * @param[in]  d_index    Index of each series in its new batch
 * @param[out] d_out0     The sub-batch for the 'false' members
 * @param[out] d_out1     The sub-batch for the 'true' members
 * @param[in]  batch_size Batch size
 * @param[in]  n_obs      Number of data points per series
 */
void divide_batch_execute(const cumlHandle& handle, const float* d_in,
                          const bool* d_mask, const int* d_index, float* d_out0,
                          float* d_out1, int batch_size, int n_obs);
void divide_batch_execute(const cumlHandle& handle, const double* d_in,
                          const bool* d_mask, const int* d_index,
                          double* d_out0, double* d_out1, int batch_size,
                          int n_obs);
void divide_batch_execute(const cumlHandle& handle, const int* d_in,
                          const bool* d_mask, const int* d_index, int* d_out0,
                          int* d_out1, int batch_size, int n_obs);

}  // namespace ML
