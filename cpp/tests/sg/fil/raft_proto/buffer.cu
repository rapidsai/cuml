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

#include <cuml/common/utils.hpp>
#include <cuml/fil/detail/raft_proto/buffer.hpp>
#include <cuml/fil/detail/raft_proto/cuda_check.hpp>
#include <cuml/fil/detail/raft_proto/cuda_stream.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>

#include <cuda_runtime_api.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>

namespace raft_proto {

CUML_KERNEL void check_buffer_access(int* buf)
{
  if (buf[0] == 1) { buf[0] = 4; }
  if (buf[1] == 2) { buf[1] = 5; }
  if (buf[2] == 3) { buf[2] = 6; }
}

TEST(Buffer, device_buffer_access)
{
  auto data     = std::vector<int>{1, 2, 3};
  auto expected = std::vector<int>{4, 5, 6};
  auto buf      = buffer<int>(
    buffer<int>(data.data(), data.size(), device_type::cpu), device_type::gpu, 0, cuda_stream{});
  check_buffer_access<<<1, 1>>>(buf.data());
  auto data_out = std::vector<int>(expected.size());
  auto host_buf = buffer<int>(data_out.data(), data_out.size(), device_type::cpu);
  copy<true>(host_buf, buf);
  ASSERT_EQ(cudaStreamSynchronize(cuda_stream{}), cudaSuccess);
  EXPECT_THAT(data_out, testing::ElementsAreArray(expected));
}

}  // namespace raft_proto
