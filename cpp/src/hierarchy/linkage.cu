/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <cuml/cuml_api.h>
#include <common/cumlHandle.hpp>

#include <cuml/cluster/linkage.hpp>
#include <hierarchy/runner.cuh>

namespace ML {

void single_linkage(const raft::handle_t &handle, const float *X, size_t m,
                    size_t n, raft::distance::DistanceType metric,
                    LinkageDistance dist_type, linkage_output<int, float> *out,
                    int c, int n_clusters) {
  Linkage::_single_linkage<int, float>(handle, X, m, n, metric, dist_type, out,
                                       c, n_clusters);
}

};  // end namespace ML