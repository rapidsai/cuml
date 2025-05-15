/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
#include <stdint.h>

#include <cstddef>
#include <exception>

namespace raft_proto {
#ifdef CUML_ENABLE_GPU
auto constexpr static const GPU_ENABLED = true;
#else
auto constexpr static const GPU_ENABLED = false;
#endif

#ifdef __CUDACC__
#define HOST   __host__
#define DEVICE __device__
auto constexpr static const GPU_COMPILATION = true;
#else
#define HOST
#define DEVICE
auto constexpr static const GPU_COMPILATION = false;
#endif

#ifndef DEBUG
auto constexpr static const DEBUG_ENABLED = false;
#elif DEBUG == 0
auto constexpr static const DEBUG_ENABLED = false;
#else
auto constexpr static const DEBUG_ENABLED = true;
#endif

struct gpu_unsupported : std::exception {
  gpu_unsupported() : gpu_unsupported("GPU functionality invoked in non-GPU build") {}
  gpu_unsupported(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

}  // namespace raft_proto
