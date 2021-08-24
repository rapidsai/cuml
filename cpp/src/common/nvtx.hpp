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

#pragma once

#include <raft/cudart_utils.h>

namespace ML {

/**
 * @brief Synchronize CUDA stream and push a named nvtx range
 * @param name range name
 * @param stream stream to synchronize
 */
void PUSH_RANGE(const char* name, cudaStream_t stream);

/**
 * @brief Synchronize CUDA stream and pop the latest nvtx range
 * @param stream stream to synchronize
 */
void POP_RANGE(cudaStream_t stream);

/**
 * @brief Push a named nvtx range
 * @param name range name
 */
void PUSH_RANGE(const char* name);

/** Pop the latest range */
void POP_RANGE();

}  // end namespace ML
