/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
