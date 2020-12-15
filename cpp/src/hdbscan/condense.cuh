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

#pragma once

#include <cuml/cuml_api.h>
#include <raft/cudart_utils.h>
#include <common/cumlHandle.hpp>

#include <raft/mr/device/buffer.hpp>
#include <raft/sparse/mst/mst.cuh>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <distance/distance.cuh>

#include <cuml/neighbors/knn.hpp>

namespace ML {
namespace HDBSCAN {
namespace Condense {



};  // end namespace Condense
};  // end namespace HDBSCAN
};  // end namespace ML