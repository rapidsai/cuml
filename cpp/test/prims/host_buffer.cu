/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iterator>

#include <gtest/gtest.h>
#include "common/host_buffer.hpp"

namespace MLCommon {

TEST(HostBufferTest, ctor) {
  std::shared_ptr<hostAllocator> allocator(new defaultHostAllocator);
  cudaStream_t stream = 0;

  const int size = 4;
  host_buffer<int> buffer(allocator, stream, size);
  ASSERT_EQ(size, buffer.size());
}

TEST(HostBufferTest, clear) {
  std::shared_ptr<hostAllocator> allocator(new defaultHostAllocator);
  cudaStream_t stream = 0;

  const int size = 8;
  host_buffer<int> buffer(allocator, stream, size);
  ASSERT_EQ(size, buffer.size());
  buffer.clear();
  ASSERT_EQ(0, buffer.size());
}

TEST(HostBufferTest, itiface) {
  std::shared_ptr<hostAllocator> allocator(new defaultHostAllocator);
  cudaStream_t stream = 0;

  const int size = 8;
  host_buffer<int> buffer(allocator, stream, size);
  ASSERT_EQ(std::distance(buffer.begin(), buffer.end()), buffer.size());
}

TEST(HostBufferTest, reserve) {
  std::shared_ptr<hostAllocator> allocator(new defaultHostAllocator);
  cudaStream_t stream = 0;

  constexpr int size = 8;
  constexpr int capacity = 16;
  static_assert(capacity > size,
                "capacity must be larger than size for test to work");

  host_buffer<int> buffer(allocator, stream, 0);
  buffer.reserve(capacity, stream);
  ASSERT_NE(nullptr, buffer.data());

  const int* const data_ptr = buffer.data();
  buffer.resize(size, stream);

  ASSERT_EQ(data_ptr, buffer.data());
}

TEST(HostBufferTest, resize) {
  std::shared_ptr<hostAllocator> allocator(new defaultHostAllocator);
  cudaStream_t stream = 0;

  std::srand(std::time(nullptr));
  const int random_variable = std::rand();

  const int size = 1;
  host_buffer<int> buffer(allocator, stream, size);
  buffer[0] = random_variable;

  const int* const data_ptr = buffer.data();
  buffer.resize(4, stream);

  ASSERT_EQ(random_variable, buffer[0]);
  ASSERT_NE(data_ptr, buffer.data());
}

TEST(HostBufferTest, release) {
  std::shared_ptr<hostAllocator> allocator(new defaultHostAllocator);
  cudaStream_t stream = 0;

  const int size = 8;
  host_buffer<int> buffer(allocator, stream, size);
  ASSERT_EQ(size, buffer.size());
  ASSERT_NE(nullptr, buffer.data());

  buffer.release(stream);
  ASSERT_EQ(0, buffer.size());
  ASSERT_EQ(nullptr, buffer.data());
}

}  // end namespace MLCommon
