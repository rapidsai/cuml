/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cuml/datasets/make_blobs.hpp>
#include <random/make_blobs.cuh>

namespace ML {
namespace Datasets {

void make_blobs(const raft::handle_t& handle, float* out, int64_t* labels,
                int64_t n_rows, int64_t n_cols, int64_t n_clusters,
                bool row_major, const float* centers, const float* cluster_std,
                const float cluster_std_scalar, bool shuffle,
                float center_box_min, float center_box_max, uint64_t seed) {
  MLCommon::Random::make_blobs(
    out, labels, n_rows, n_cols, n_clusters, handle.get_device_allocator(),
    handle.get_stream(), row_major, centers, cluster_std, cluster_std_scalar,
    shuffle, center_box_min, center_box_max, seed);
}

void make_blobs(const raft::handle_t& handle, double* out, int64_t* labels,
                int64_t n_rows, int64_t n_cols, int64_t n_clusters,
                bool row_major, const double* centers,
                const double* cluster_std, const double cluster_std_scalar,
                bool shuffle, double center_box_min, double center_box_max,
                uint64_t seed) {
  MLCommon::Random::make_blobs(
    out, labels, n_rows, n_cols, n_clusters, handle.get_device_allocator(),
    handle.get_stream(), row_major, centers, cluster_std, cluster_std_scalar,
    shuffle, center_box_min, center_box_max, seed);
}

void make_blobs(const raft::handle_t& handle, float* out, int* labels,
                int n_rows, int n_cols, int n_clusters, bool row_major,
                const float* centers, const float* cluster_std,
                const float cluster_std_scalar, bool shuffle,
                float center_box_min, float center_box_max, uint64_t seed) {
  MLCommon::Random::make_blobs(
    out, labels, n_rows, n_cols, n_clusters, handle.get_device_allocator(),
    handle.get_stream(), row_major, centers, cluster_std, cluster_std_scalar,
    shuffle, center_box_min, center_box_max, seed);
}

void make_blobs(const raft::handle_t& handle, double* out, int* labels,
                int n_rows, int n_cols, int n_clusters, bool row_major,
                const double* centers, const double* cluster_std,
                const double cluster_std_scalar, bool shuffle,
                double center_box_min, double center_box_max, uint64_t seed) {
  MLCommon::Random::make_blobs(
    out, labels, n_rows, n_cols, n_clusters, handle.get_device_allocator(),
    handle.get_stream(), row_major, centers, cluster_std, cluster_std_scalar,
    shuffle, center_box_min, center_box_max, seed);
}
}  // namespace Datasets
}  // namespace ML
