/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cuml/fil/fnv_hash.h>
#include <gtest/gtest.h>
#include <raft/error.hpp>

struct fnv_vec_t {
  std::vector<char> input;
  unsigned long long correct_64bit;
  uint32_t correct_32bit;
};

class FNVHashTest : public testing::TestWithParam<fnv_vec_t> {
 protected:
  void SetUp() override { param = GetParam(); }

  void check()
  {
    unsigned long long hash_64bit =
      fowler_noll_vo_fingerprint64(param.input.begin(), param.input.end());
    ASSERT(hash_64bit == param.correct_64bit, "Wrong hash computed");
    unsigned long hash_32bit =
      fowler_noll_vo_fingerprint64_32(param.input.begin(), param.input.end());
    ASSERT(hash_32bit == param.correct_32bit, "Wrong xor-folded hash computed");
  }

  fnv_vec_t param;
};

std::vector<fnv_vec_t> fnv_vecs = {
  {{}, 14695981039346656037ull, 0xcbf29ce4 ^ 0x84222325},  // test #0
  // 32-bit output is xor-folded 64-bit output. The format below makes this obvious.
  {{0}, 0xaf63bd4c8601b7df, 0xaf63bd4c ^ 0x8601b7df},
  {{1}, 0xaf63bd4c8601b7de, 0xaf63bd4c ^ 0x8601b7de},
  {{2}, 0xaf63bd4c8601b7dd, 0xaf63bd4c ^ 0x8601b7dd},
  {{3}, 0xaf63bd4c8601b7dc, 0xaf63bd4c ^ 0x8601b7dc},
  {{1, 2}, 0x08328707b4eb6e38, 0x08328707 ^ 0xb4eb6e38},  // test #5
  {{2, 1}, 0x08328607b4eb6c86, 0x08328607 ^ 0xb4eb6c86},
  {{1, 2, 3}, 0xd949aa186c0c492b, 0xd949aa18 ^ 0x6c0c492b},
  {{1, 3, 2}, 0xd949ab186c0c4ad9, 0xd949ab18 ^ 0x6c0c4ad9},
  {{2, 1, 3}, 0xd94645186c0967b1, 0xd9464518 ^ 0x6c0967b1},
  {{2, 3, 1}, 0xd94643186c09644d, 0xd9464318 ^ 0x6c09644d},  // test #10
  {{3, 1, 2}, 0xd942e1186c0687ed, 0xd942e118 ^ 0x6c0687ed},
  {{3, 2, 1}, 0xd942e2186c0689a3, 0xd942e218 ^ 0x6c0689a3},
};

TEST_P(FNVHashTest, Import) { check(); }
INSTANTIATE_TEST_CASE_P(FilTests, FNVHashTest, testing::ValuesIn(fnv_vecs));
