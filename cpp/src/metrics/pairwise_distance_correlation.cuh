
/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <raft/distance/distance.hpp>
#include <raft/distance/specializations.hpp>
#include <raft/handle.hpp>

namespace ML {

namespace Metrics {
void pairwise_distance_correlation(const raft::handle_t& handle,
                                   const double* x,
                                   const double* y,
                                   double* dist,
                                   int m,
                                   int n,
                                   int k,
                                   bool isRowMajor,
                                   double metric_arg);

void pairwise_distance_correlation(const raft::handle_t& handle,
                                   const float* x,
                                   const float* y,
                                   float* dist,
                                   int m,
                                   int n,
                                   int k,
                                   bool isRowMajor,
                                   float metric_arg);

}  // namespace Metrics
}  // namespace ML
