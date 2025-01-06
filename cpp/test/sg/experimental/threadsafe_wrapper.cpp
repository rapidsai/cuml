/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cuml/experimental/threadsafe_wrapper.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

namespace ML {
namespace experimental {

struct threadsafe_test_struct {
  auto access() const
  {
    access_counter_.fetch_add(1);
    return access_counter_.fetch_sub(1) > 0 && !modification_in_progress_.load();
  }

  auto modify()
  {
    auto being_modified = modification_in_progress_.exchange(true);
    return !being_modified && access_counter_.load() == 0;
  }

 private:
  mutable std::atomic<int> access_counter_    = int{};
  std::atomic<bool> modification_in_progress_ = false;
};

TEST(ThreadsafeWrapper, threadsafe_wrapper)
{
  auto test_obj = threadsafe_wrapper<threadsafe_test_struct>{};
  // Choose a prime number large enough to cause contention. We use a prime
  // number to allow us to easily produce different patterns of access in
  // each thread.
  auto const num_threads = 61;
  auto threads           = std::vector<std::thread>{};
  for (auto thread_id = 0; thread_id < num_threads; ++thread_id) {
    threads.emplace_back(
      [thread_id](auto& obj) {
        for (auto i = 0; i < num_threads; ++i) {
          if (i % (thread_id + 1) == 0) {
            EXPECT(obj->modify());
          } else {
            EXPECT(std::as_const(obj)->access());
          }
        }
      },
      test_obj);
  }
  for (auto thread_id = 0; thread_id < num_threads; ++thread_id) {
    threads[thread_id].join();
  }
}

}  // namespace experimental
}  // namespace ML
