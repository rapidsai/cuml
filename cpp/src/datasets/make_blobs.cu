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

#include "make_blobs.hpp"
#include "random/make_blobs.h"

namespace ML {
namespace Metrics {

void make_blobs(const cumlHandle& handle, float* out, int* labels, int n_rows,
                int n_cols, int n_clusters, const float* centers = nullptr,
                const float* cluster_std = nullptr,
                const float cluster_std_scalar = 1.f, bool shuffle = true,
                float center_box_min = 10.f, float center_box_max = 10.f,
                uint64_t seed = 0ULL) {
  MLCommon::Random::make_blobs(out, labels, n_rows, n_cols, n_clusters,
                               handle.getDeviceAllocator(), handle.getStream(),
                               centers, cluster_std, cluster_std_scalar,
                               shuffle, center_box_min, center_box_max, seed);
}

void make_blobs(const cumlHandle& handle, double* out, int* labels, int n_rows,
                int n_cols, int n_clusters, const double* centers = nullptr,
                const double* cluster_std = nullptr,
                const double cluster_std_scalar = 1.f, bool shuffle = true,
                double center_box_min = 10.f, double center_box_max = 10.f,
                uint64_t seed = 0ULL) {
  MLCommon::Random::make_blobs(out, labels, n_rows, n_cols, n_clusters,
                               handle.getDeviceAllocator(), handle.getStream(),
                               centers, cluster_std, cluster_std_scalar,
                               shuffle, center_box_min, center_box_max, seed);
}

}  // end namespace Metrics
}  // end namespace ML
