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

#include "distance_common.cuh"

namespace MLCommon {
namespace Bench {
namespace Distance {

DIST_BENCH_REGISTER(DistanceL2Sq, raft::distance::DistanceType::L2Expanded);
DIST_BENCH_REGISTER(DistanceL2Sqrt, raft::distance::DistanceType::L2SqrtExpanded);

}  // namespace Distance
}  // namespace Bench
}  // namespace MLCommon
