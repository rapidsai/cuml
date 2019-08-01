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
namespace Datasets {

void make_blobs(const cumlHandle& handle, float* out, int* labels, int n_rows,
                int n_cols, int n_clusters, const float* centers,
                const float* cluster_std,
                const float cluster_std_scalar, bool shuffle,
                float center_box_min, float center_box_max,
                uint64_t seed) {
  MLCommon::Random::make_blobs(out, labels, n_rows, n_cols, n_clusters,
                               handle.getDeviceAllocator(), handle.getStream(),
                               centers, cluster_std, cluster_std_scalar,
                               shuffle, center_box_min, center_box_max, seed);
}

void make_blobs(const cumlHandle& handle, double* out, int* labels, int n_rows,
                int n_cols, int n_clusters, const double* centers,
                const double* cluster_std,
                const double cluster_std_scalar, bool shuffle,
                double center_box_min, double center_box_max,
                uint64_t seed) {
  MLCommon::Random::make_blobs(out, labels, n_rows, n_cols, n_clusters,
                               handle.getDeviceAllocator(), handle.getStream(),
                               centers, cluster_std, cluster_std_scalar,
                               shuffle, center_box_min, center_box_max, seed);
}

}  // end namespace Metrics
}  // end namespace ML
