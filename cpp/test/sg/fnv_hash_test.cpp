#include <cuml/fil/fnv_hash.h>
#include <gtest/gtest.h>
#include <raft/error.hpp>

typedef std::tuple<std::vector<char>, unsigned long long, unsigned long>
  fnv_vec_t;

class FNVHashTest : public testing::TestWithParam<fnv_vec_t> {
 protected:
  void SetUp() override { std::tie(input, correct64, correct32) = GetParam(); }

  void check() {
    unsigned long long real64 =
      fowler_noll_vo_fingerprint64(input.begin(), input.end());
    ASSERT(real64 == correct64, "Wrong hash computed");
    unsigned long real32 =
      fowler_noll_vo_fingerprint64_32(input.begin(), input.end());
    ASSERT(real32 == correct32, "Wrong xor-folded hash computed");
  }

  // parameters
  std::vector<char> input;
  unsigned long long correct64;
  unsigned long correct32;
};

std::vector<fnv_vec_t> fnv_vecs = {
  {{}, 14695981039346656037ull, 0xcbf29ce4 ^ 0x84222325},  // test #0
  {{0},
   0xaf63bd4c8601b7df,
   0xaf63bd4c ^ 0x8601b7df},  // __style__don't__wrap__me__please__
  {{1},
   0xaf63bd4c8601b7de,
   0xaf63bd4c ^ 0x8601b7de},  // __style__don't__wrap__me__please__
  {{2},
   0xaf63bd4c8601b7dd,
   0xaf63bd4c ^ 0x8601b7dd},  // __style__don't__wrap__me__please__
  {{3},
   0xaf63bd4c8601b7dc,
   0xaf63bd4c ^ 0x8601b7dc},  // __style__don't__wrap__me__please__
  {{1, 2}, 0x08328707b4eb6e38, 0x08328707 ^ 0xb4eb6e38},  // test #5
  {{2, 1},
   0x08328607b4eb6c86,
   0x08328607 ^ 0xb4eb6c86},  // __style__don't__wrap__me__please__
  {{1, 2, 3},
   0xd949aa186c0c492b,
   0xd949aa18 ^ 0x6c0c492b},  // __style__don't__wrap__me__please__
  {{1, 3, 2},
   0xd949ab186c0c4ad9,
   0xd949ab18 ^ 0x6c0c4ad9},  // __style__don't__wrap__me__please__
  {{2, 1, 3},
   0xd94645186c0967b1,
   0xd9464518 ^ 0x6c0967b1},  // __style__don't__wrap__me__please__
  {{2, 3, 1}, 0xd94643186c09644d, 0xd9464318 ^ 0x6c09644d},  // test #10
  {{3, 1, 2},
   0xd942e1186c0687ed,
   0xd942e118 ^ 0x6c0687ed},  // __style__don't__wrap__me__please__
  {{3, 2, 1},
   0xd942e2186c0689a3,
   0xd942e218 ^ 0x6c0689a3},  // __style__don't__wrap__me__please__
};

TEST_P(FNVHashTest, Import) { check(); }
INSTANTIATE_TEST_CASE_P(FilTests, FNVHashTest, testing::ValuesIn(fnv_vecs));
