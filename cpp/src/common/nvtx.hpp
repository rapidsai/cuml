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

#include <raft/core/nvtx.hpp>

namespace ML {

/**
 * @brief Synchronize CUDA stream and push a named nvtx range
 * @param name range name
 * @param stream stream to synchronize
 */
[[deprecated("Use new raft::common::nvtx::push_range from <raft/core/nvtx.hpp>")]] inline void
PUSH_RANGE(const char* name, cudaStream_t stream)
{
  raft::common::nvtx::push_range(name);
}

/**
 * @brief Synchronize CUDA stream and pop the latest nvtx range
 * @param stream stream to synchronize
 */
[[deprecated("Use new raft::common::nvtx::pop_range from <raft/core/nvtx.hpp>")]] inline void
POP_RANGE(cudaStream_t stream)
{
  raft::common::nvtx::pop_range();
}

/**
 * @brief Push a named nvtx range
 * @param name range name
 */
[[deprecated("Use new raft::common::nvtx::push_range from <raft/core/nvtx.hpp>")]] inline void
PUSH_RANGE(const char* name)
{
  raft::common::nvtx::push_range(name);
}

/** Pop the latest range */
[[deprecated("Use new raft::common::nvtx::pop_range from <raft/core/nvtx.hpp>")]] inline void
POP_RANGE()
{
  raft::common::nvtx::pop_range();
}

}  // end namespace ML
