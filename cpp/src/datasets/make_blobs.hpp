/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

namespace ML {
namespace Datasets {

/**
 * @defgroup MakeBlobs
 * @{
 * @brief GPU-equivalent of sklearn.datasets.make_blobs as documented here:
 * https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
 * @tparam DataT output data type
 * @tparam IdxT indexing arithmetic type
 * @param out the generated data on device (dim = n_rows x n_cols) in row-major
 * layout
 * @param labels labels for the generated data on device (dim = n_rows x 1)
 * @param n_rows number of rows in the generated data
 * @param n_cols number of columns in the generated data
 * @param n_cluster number of clusters (or classes) to generate
 * @param allocator device allocator to help allocate temporary buffers
 * @param stream cuda stream to schedule the work on
 * @param centers centers of each of the cluster, pass a nullptr if you need
 * this also to be generated randomly (dim = n_clusters x n_cols). This is
 * expected to be on device
 * @param cluster_std standard deviation of each of the cluster center, pass a
 * nullptr if you need this to be read from 'cluster_std_scalar'.
 * (dim = n_clusters x 1) This is expected to be on device
 * @param cluster_std_scalar if 'cluster_std' is nullptr, then use this as the
 * standard deviation across all dimensions.
 * @param shuffle shuffle the generated dataset and labels
 * @param center_box_min min value of the box from which to pick the cluster
 * centers. Useful only if 'centers' is nullptr
 * @param center_box_max max value of the box from which to pick the cluster
 * centers. Useful only if 'centers' is nullptr
 * @param seed seed for the RNG
 */
void make_blobs(const cumlHandle& handle, float* out, int* labels, int n_rows,
                int n_cols, int n_clusters, const float* centers = nullptr,
                const float* cluster_std = nullptr,
                const float cluster_std_scalar = 1.f, bool shuffle = true,
                float center_box_min = 10.f, float center_box_max = 10.f,
                uint64_t seed = 0ULL);

void make_blobs(const cumlHandle& handle, double* out, int* labels, int n_rows,
                int n_cols, int n_clusters, const double* centers = nullptr,
                const double* cluster_std = nullptr,
                const double cluster_std_scalar = 1.f, bool shuffle = true,
                double center_box_min = 10.f, double center_box_max = 10.f,
                uint64_t seed = 0ULL);
/** @} */

}  // namespace Datasets
}  // namespace ML
