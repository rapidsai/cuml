/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include "cuda_utils.h"

namespace MLCommon {



TEST(Utils, Assert) {
  ASSERT_NO_THROW(ASSERT(1 == 1, "Should not assert!"));
  ASSERT_THROW(ASSERT(1 != 1, "Should assert!"), Exception);
}

TEST(Utils, CudaCheck) { ASSERT_NO_THROW(CUDA_CHECK(cudaFree(nullptr))); }

// we want the functions like 'log2' to work both at compile and runtimes!
static const int log2Of1024 = log2(1024);
static const int log2Of1023 = log2(1023);
TEST(Utils, log2) {
  ASSERT_EQ(10, log2(1024));
  ASSERT_EQ(9, log2(1023));
  ASSERT_EQ(10, log2Of1024);
  ASSERT_EQ(9, log2Of1023);
}

}  // end namespace MLCommon
