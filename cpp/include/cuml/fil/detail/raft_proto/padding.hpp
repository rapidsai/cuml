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
#include <cuml/fil/detail/raft_proto/gpu_support.hpp>

namespace raft_proto {

/* Return the value that must be added to val to equal the next multiple of
 * alignment greater than or equal to val */
template <typename T, typename U>
HOST DEVICE auto padding_size(T val, U alignment)
{
  auto result = val;
  if (alignment != 0) {
    auto remainder = val % alignment;
    result         = alignment - remainder;
    result *= (remainder != 0);
  }
  return result;
}

/* Return the next multiple of alignment >= val */
template <typename T, typename U>
HOST DEVICE auto padded_size(T val, U alignment)
{
  return val + padding_size(val, alignment);
}

/* Return the value that must be added to val to equal the next multiple of
 * alignment less than or equal to val */
template <typename T, typename U>
HOST DEVICE auto downpadding_size(T val, U alignment)
{
  auto result = val;
  if (alignment != 0) { result = val % alignment; }
  return result;
}

/* Return the next multiple of alignment <= val */
template <typename T, typename U>
HOST DEVICE auto downpadded_size(T val, U alignment)
{
  return val - downpadding_size(val, alignment);
}

}  // namespace raft_proto
