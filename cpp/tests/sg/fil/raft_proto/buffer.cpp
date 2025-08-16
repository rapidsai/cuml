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

#include <cuml/fil/detail/raft_proto/buffer.hpp>
#include <cuml/fil/detail/raft_proto/cuda_check.hpp>
#include <cuml/fil/detail/raft_proto/cuda_stream.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>
#include <cuml/fil/detail/raft_proto/exceptions.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace raft_proto {

TEST(Buffer, default_buffer)
{
  auto buf = buffer<int>();
  EXPECT_EQ(buf.memory_type(), device_type::cpu);
  EXPECT_EQ(buf.size(), 0);
  EXPECT_EQ(buf.device_index(), 0);
}

TEST(Buffer, device_buffer)
{
  auto data         = std::vector<int>{1, 2, 3};
  auto test_buffers = std::vector<buffer<int>>{};
  test_buffers.emplace_back(data.size(), device_type::gpu, 0, cuda_stream{});
  test_buffers.emplace_back(data.size(), device_type::gpu, 0);
  test_buffers.emplace_back(data.size(), device_type::gpu);

  for (auto& buf : test_buffers) {
    ASSERT_EQ(buf.memory_type(), device_type::gpu);
    ASSERT_EQ(buf.size(), data.size());
#ifdef CUML_ENABLE_GPU
    ASSERT_NE(buf.data(), nullptr);

    auto data_out = std::vector<int>(data.size());
    cudaMemcpy(static_cast<void*>(buf.data()),
               static_cast<void*>(data.data()),
               sizeof(int) * data.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(static_cast<void*>(data_out.data()),
               static_cast<void*>(buf.data()),
               sizeof(int) * data.size(),
               cudaMemcpyDeviceToHost);
    EXPECT_THAT(data_out, testing::ElementsAreArray(data));
#endif
  }
}

TEST(Buffer, non_owning_device_buffer)
{
  auto data   = std::vector<int>{1, 2, 3};
  auto* ptr_d = static_cast<int*>(nullptr);
#ifdef CUML_ENABLE_GPU
  cudaMalloc(reinterpret_cast<void**>(&ptr_d), sizeof(int) * data.size());
  cudaMemcpy(static_cast<void*>(ptr_d),
             static_cast<void*>(data.data()),
             sizeof(int) * data.size(),
             cudaMemcpyHostToDevice);
#endif
  auto test_buffers = std::vector<buffer<int>>{};
  test_buffers.emplace_back(ptr_d, data.size(), device_type::gpu, 0);
  test_buffers.emplace_back(ptr_d, data.size(), device_type::gpu);
#ifdef CUML_ENABLE_GPU

  for (auto& buf : test_buffers) {
    ASSERT_EQ(buf.memory_type(), device_type::gpu);
    ASSERT_EQ(buf.size(), data.size());
    ASSERT_EQ(buf.data(), ptr_d);

    auto data_out = std::vector<int>(data.size());
    cudaMemcpy(static_cast<void*>(data_out.data()),
               static_cast<void*>(buf.data()),
               sizeof(int) * data.size(),
               cudaMemcpyDeviceToHost);
    EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
  }
  cudaFree(reinterpret_cast<void*>(ptr_d));
#endif
}

TEST(Buffer, host_buffer)
{
  auto data         = std::vector<int>{1, 2, 3};
  auto test_buffers = std::vector<buffer<int>>{};
  test_buffers.emplace_back(data.size(), device_type::cpu, 0, cuda_stream{});
  test_buffers.emplace_back(data.size(), device_type::cpu, 0);
  test_buffers.emplace_back(data.size(), device_type::cpu);
  test_buffers.emplace_back(data.size());

  for (auto& buf : test_buffers) {
    ASSERT_EQ(buf.memory_type(), device_type::cpu);
    ASSERT_EQ(buf.size(), data.size());
    ASSERT_NE(buf.data(), nullptr);

    std::copy(data.begin(), data.end(), buf.data());

    auto data_out = std::vector<int>(buf.data(), buf.data() + buf.size());
    EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
  }
}

TEST(Buffer, host_buffer_from_iters)
{
  auto data         = std::vector<int>{1, 2, 3};
  auto test_buffers = std::vector<buffer<int>>{};
  test_buffers.emplace_back(std::begin(data), std::end(data));

  for (auto& buf : test_buffers) {
    ASSERT_EQ(buf.memory_type(), device_type::cpu);
    ASSERT_EQ(buf.size(), data.size());
    ASSERT_NE(buf.data(), nullptr);

    std::copy(data.begin(), data.end(), buf.data());

    auto data_out = std::vector<int>(buf.data(), buf.data() + buf.size());
    EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
  }
}

TEST(Buffer, device_buffer_from_iters)
{
  auto data         = std::vector<int>{1, 2, 3};
  auto test_buffers = std::vector<buffer<int>>{};
  test_buffers.emplace_back(std::begin(data), std::end(data), device_type::gpu);
  test_buffers.emplace_back(std::begin(data), std::end(data), device_type::gpu, 0);
  test_buffers.emplace_back(std::begin(data), std::end(data), device_type::gpu, 0, cuda_stream{});

  for (auto& buf : test_buffers) {
    ASSERT_EQ(buf.memory_type(), device_type::gpu);
    ASSERT_EQ(buf.size(), data.size());
#ifdef CUML_ENABLE_GPU
    ASSERT_NE(buf.data(), nullptr);

    auto data_out = std::vector<int>(data.size());
    cudaMemcpy(static_cast<void*>(buf.data()),
               static_cast<void*>(data.data()),
               sizeof(int) * data.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(static_cast<void*>(data_out.data()),
               static_cast<void*>(buf.data()),
               sizeof(int) * data.size(),
               cudaMemcpyDeviceToHost);
    EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
#endif
  }
}

TEST(Buffer, non_owning_host_buffer)
{
  auto data = std::vector<int>{1, 2, 3};
  std::vector<buffer<int>> test_buffers;
  test_buffers.emplace_back(data.data(), data.size(), device_type::cpu, 0);
  ASSERT_EQ(test_buffers.back().memory_type(), device_type::cpu);
  ASSERT_EQ(test_buffers.back().size(), data.size());
  ASSERT_EQ(test_buffers.back().data(), data.data());
  test_buffers.emplace_back(data.data(), data.size(), device_type::cpu);
  ASSERT_EQ(test_buffers.back().memory_type(), device_type::cpu);
  ASSERT_EQ(test_buffers.back().size(), data.size());
  ASSERT_EQ(test_buffers.back().data(), data.data());
  test_buffers.emplace_back(data.data(), data.size());
  ASSERT_EQ(test_buffers.back().memory_type(), device_type::cpu);
  ASSERT_EQ(test_buffers.back().size(), data.size());
  ASSERT_EQ(test_buffers.back().data(), data.data());

  for (auto& buf : test_buffers) {
    ASSERT_EQ(buf.memory_type(), device_type::cpu);
    ASSERT_EQ(buf.size(), data.size());
    ASSERT_EQ(buf.data(), data.data());

    auto data_out = std::vector<int>(buf.data(), buf.data() + buf.size());
    EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
  }
}

TEST(Buffer, copy_buffer)
{
  auto data        = std::vector<int>{1, 2, 3};
  auto orig_buffer = buffer<int>(data.data(), data.size(), device_type::cpu);

  auto test_buffers = std::vector<buffer<int>>{};
  test_buffers.emplace_back(orig_buffer);
  test_buffers.emplace_back(orig_buffer, device_type::cpu);
  test_buffers.emplace_back(orig_buffer, device_type::cpu, 0);
  test_buffers.emplace_back(orig_buffer, device_type::cpu, 0, cuda_stream{});

  for (auto& buf : test_buffers) {
    ASSERT_EQ(buf.memory_type(), device_type::cpu);
    ASSERT_EQ(buf.size(), data.size());
    ASSERT_NE(buf.data(), orig_buffer.data());

    auto data_out = std::vector<int>(buf.data(), buf.data() + buf.size());
    EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));

#ifdef CUML_ENABLE_GPU
    auto test_dev_buffers = std::vector<buffer<int>>{};
    test_dev_buffers.emplace_back(orig_buffer, device_type::gpu);
    test_dev_buffers.emplace_back(orig_buffer, device_type::gpu, 0);
    test_dev_buffers.emplace_back(orig_buffer, device_type::gpu, 0, cuda_stream{});
    for (auto& dev_buf : test_dev_buffers) {
      data_out = std::vector<int>(data.size());
      cuda_check(cudaMemcpy(static_cast<void*>(data_out.data()),
                            static_cast<void*>(dev_buf.data()),
                            dev_buf.size() * sizeof(int),
                            cudaMemcpyDefault));
      EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));

      auto test_dev_copies = std::vector<buffer<int>>{};
      test_dev_copies.emplace_back(dev_buf, device_type::gpu);
      test_dev_copies.emplace_back(dev_buf, device_type::gpu, 0);
      test_dev_copies.emplace_back(dev_buf, device_type::gpu, 0, cuda_stream{});
      for (auto& copy_buf : test_dev_copies) {
        data_out = std::vector<int>(data.size());
        cuda_check(cudaMemcpy(static_cast<void*>(data_out.data()),
                              static_cast<void*>(copy_buf.data()),
                              copy_buf.size() * sizeof(int),
                              cudaMemcpyDefault));
        EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
      }

      auto test_host_buffers = std::vector<buffer<int>>{};
      test_host_buffers.emplace_back(dev_buf, device_type::cpu);
      test_host_buffers.emplace_back(dev_buf, device_type::cpu, 0);
      test_host_buffers.emplace_back(dev_buf, device_type::cpu, 0, cuda_stream{});
      for (auto& host_buf : test_host_buffers) {
        data_out = std::vector<int>(host_buf.data(), host_buf.data() + host_buf.size());
        EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
      }
    }
#endif
  }
}

TEST(Buffer, move_buffer)
{
  auto data         = std::vector<int>{1, 2, 3};
  auto test_buffers = std::vector<buffer<int>>{};
  test_buffers.emplace_back(buffer<int>(data.data(), data.size(), device_type::cpu));
  test_buffers.emplace_back(buffer<int>(data.data(), data.size(), device_type::cpu),
                            device_type::cpu);
  test_buffers.emplace_back(
    buffer<int>(data.data(), data.size(), device_type::cpu), device_type::cpu, 0);
  test_buffers.emplace_back(
    buffer<int>(data.data(), data.size(), device_type::cpu), device_type::cpu, 0, cuda_stream{});

  for (auto& buf : test_buffers) {
    ASSERT_EQ(buf.memory_type(), device_type::cpu);
    ASSERT_EQ(buf.size(), data.size());
    ASSERT_EQ(buf.data(), data.data());

    auto data_out = std::vector<int>(buf.data(), buf.data() + buf.size());
    EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
  }
#ifdef CUML_ENABLE_GPU
  test_buffers = std::vector<buffer<int>>{};
  test_buffers.emplace_back(buffer<int>(data.data(), data.size(), device_type::cpu),
                            device_type::gpu);
  test_buffers.emplace_back(
    buffer<int>(data.data(), data.size(), device_type::cpu), device_type::gpu, 0);
  test_buffers.emplace_back(
    buffer<int>(data.data(), data.size(), device_type::cpu), device_type::gpu, 0, cuda_stream{});
  for (auto& buf : test_buffers) {
    ASSERT_EQ(buf.memory_type(), device_type::gpu);
    ASSERT_EQ(buf.size(), data.size());
    ASSERT_NE(buf.data(), data.data());

    auto data_out = std::vector<int>(buf.size());
    cuda_check(cudaMemcpy(static_cast<void*>(data_out.data()),
                          static_cast<void*>(buf.data()),
                          buf.size() * sizeof(int),
                          cudaMemcpyDefault));
    EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
  }
#endif
}

TEST(Buffer, move_assignment_buffer)
{
  auto data = std::vector<int>{1, 2, 3};

#ifdef CUML_ENABLE_GPU
  auto buf = buffer<int>{data.data(), data.size() - 1, device_type::gpu};
#else
  auto buf = buffer<int>{data.data(), data.size() - 1, device_type::cpu};
#endif
  buf = buffer<int>{data.size(), device_type::cpu};

  ASSERT_EQ(buf.memory_type(), device_type::cpu);
  ASSERT_EQ(buf.size(), data.size());
}

TEST(Buffer, partial_buffer_copy)
{
  auto data1    = std::vector<int>{1, 2, 3, 4, 5};
  auto data2    = std::vector<int>{0, 0, 0, 0, 0};
  auto expected = std::vector<int>{0, 3, 4, 5, 0};
#ifdef CUML_ENABLE_GPU
  auto buf1 =
    buffer<int>{buffer<int>{data1.data(), data1.size(), device_type::cpu}, device_type::gpu};
#else
  auto buf1 = buffer<int>{data1.data(), data1.size(), device_type::cpu};
#endif
  auto buf2 = buffer<int>{data2.data(), data2.size(), device_type::cpu};
  copy<true>(buf2, buf1, 1, 2, 3, cuda_stream{});
  copy<false>(buf2, buf1, 1, 2, 3, cuda_stream{});
  EXPECT_THROW(copy<true>(buf2, buf1, 1, 2, 4, cuda_stream{}), out_of_bounds);
}

TEST(Buffer, buffer_copy_overloads)
{
  auto data             = std::vector<int>{1, 2, 3};
  auto expected         = data;
  auto orig_host_buffer = buffer<int>(data.data(), data.size(), device_type::cpu);
  auto orig_dev_buffer  = buffer<int>(orig_host_buffer, device_type::gpu);
  auto copy_dev_buffer  = buffer<int>(data.size(), device_type::gpu);

  // copying host to host
  auto data_out         = std::vector<int>(data.size());
  auto copy_host_buffer = buffer<int>(data_out.data(), data.size(), device_type::cpu);
  copy<true>(copy_host_buffer, orig_host_buffer);
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));

  // copying host to host with stream
  data_out         = std::vector<int>(data.size());
  copy_host_buffer = buffer<int>(data_out.data(), data.size(), device_type::cpu);
  copy<true>(copy_host_buffer, orig_host_buffer, cuda_stream{});
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));

  // copying host to host with offset
  data_out         = std::vector<int>(data.size() + 1);
  copy_host_buffer = buffer<int>(data_out.data(), data.size(), device_type::cpu);
  copy<true>(copy_host_buffer, orig_host_buffer, 2, 1, 1, cuda_stream{});
  expected = std::vector<int>{0, 0, 2, 0};
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));

#ifdef CUML_ENABLE_GPU
  // copy device to host
  data_out         = std::vector<int>(data.size());
  copy_host_buffer = buffer<int>(data_out.data(), data.size(), device_type::cpu);
  copy<true>(copy_host_buffer, orig_dev_buffer);
  expected = data;
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));

  // copy device to host with stream
  data_out         = std::vector<int>(data.size());
  copy_host_buffer = buffer<int>(data_out.data(), data.size(), device_type::cpu);
  copy<true>(copy_host_buffer, orig_dev_buffer, cuda_stream{});
  expected = data;
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));

  // copy device to host with offset
  data_out         = std::vector<int>(data.size() + 1);
  copy_host_buffer = buffer<int>(data_out.data(), data.size(), device_type::cpu);
  copy<true>(copy_host_buffer, orig_dev_buffer, 2, 1, 1, cuda_stream{});
  expected = std::vector<int>{0, 0, 2, 0};
  EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));
#endif
}

}  // namespace raft_proto
