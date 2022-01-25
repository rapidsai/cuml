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

#include <common/seive.cuh>
#include <gtest/gtest.h>

namespace MLCommon {

TEST(Seive, Test)
{
  Seive s1(32);
  ASSERT_TRUE(s1.isPrime(17));
  ASSERT_FALSE(s1.isPrime(28));

  Seive s2(1024 * 1024);
  ASSERT_TRUE(s2.isPrime(107));
  ASSERT_FALSE(s2.isPrime(111));
  ASSERT_TRUE(s2.isPrime(6047));
}

}  // end namespace MLCommon
