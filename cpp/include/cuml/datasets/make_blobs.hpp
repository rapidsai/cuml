/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cstdint>

namespace raft {
class handle_t;
}

namespace ML {
namespace Datasets {

/**
 * @defgroup MakeBlobs scikit-learn-esq make_blobs
 *
 * @brief GPU-equivalent of sklearn.datasets.make_blobs
 *
 * @param[out] out                generated data [on device]
 *                                [dim = n_rows x n_cols]
 * @param[out] labels             labels for the generated data [on device]
 *                                [len = n_rows]
 * @param[in]  n_rows             number of rows in the generated data
 * @param[in]  n_cols             number of columns in the generated data
 * @param[in]  n_clusters         number of clusters (or classes) to generate
 * @param[in]  row_major          whether input `centers` and output `out`
 *                                buffers are to be stored in row or column
 *                                major layout
 * @param[in]  centers            centers of each of the cluster, pass a nullptr
 *                                if you need this also to be generated randomly
 *                                [on device] [dim = n_clusters x n_cols]
 * @param[in]  cluster_std        standard deviation of each cluster center,
 *                                pass a nullptr if this is to be read from the
 *                                `cluster_std_scalar`. [on device]
 *                                [len = n_clusters]
 * @param[in]  cluster_std_scalar if 'cluster_std' is nullptr, then use this as
 *                                the std-dev across all dimensions.
 * @param[in]  shuffle            shuffle the generated dataset and labels
 * @param[in]  center_box_min     min value of box from which to pick cluster
 *                                centers. Useful only if 'centers' is nullptr
 * @param[in]  center_box_max     max value of box from which to pick cluster
 *                                centers. Useful only if 'centers' is nullptr
 * @param[in]  seed               seed for the RNG
 * @{
 */
void make_blobs(const raft::handle_t& handle,
                float* out,
                int64_t* labels,
                int64_t n_rows,
                int64_t n_cols,
                int64_t n_clusters,
                bool row_major                 = true,
                const float* centers           = nullptr,
                const float* cluster_std       = nullptr,
                const float cluster_std_scalar = 1.f,
                bool shuffle                   = true,
                float center_box_min           = -10.f,
                float center_box_max           = 10.f,
                uint64_t seed                  = 0ULL);
void make_blobs(const raft::handle_t& handle,
                double* out,
                int64_t* labels,
                int64_t n_rows,
                int64_t n_cols,
                int64_t n_clusters,
                bool row_major                  = true,
                const double* centers           = nullptr,
                const double* cluster_std       = nullptr,
                const double cluster_std_scalar = 1.0,
                bool shuffle                    = true,
                double center_box_min           = -10.0,
                double center_box_max           = 10.0,
                uint64_t seed                   = 0ULL);
void make_blobs(const raft::handle_t& handle,
                float* out,
                int* labels,
                int n_rows,
                int n_cols,
                int n_clusters,
                bool row_major                 = true,
                const float* centers           = nullptr,
                const float* cluster_std       = nullptr,
                const float cluster_std_scalar = 1.f,
                bool shuffle                   = true,
                float center_box_min           = -10.f,
                float center_box_max           = 10.0,
                uint64_t seed                  = 0ULL);
void make_blobs(const raft::handle_t& handle,
                double* out,
                int* labels,
                int n_rows,
                int n_cols,
                int n_clusters,
                bool row_major                  = true,
                const double* centers           = nullptr,
                const double* cluster_std       = nullptr,
                const double cluster_std_scalar = 1.0,
                bool shuffle                    = true,
                double center_box_min           = -10.0,
                double center_box_max           = 10.0,
                uint64_t seed                   = 0ULL);
/** @} */

}  // namespace Datasets
}  // namespace ML
