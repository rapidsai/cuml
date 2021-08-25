/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include "../../src/fil/internal.cuh"
#define __host__
#define __device__

#include <test_utils.h>

#include <cuml/fil/fil.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <utility>

namespace ML {

using namespace fil;

template <typename fil_node_t>
struct ChildIdxTestParams {
  fil_node_t node     = {};
  int parent_node_idx = 0;
  cat_sets_owner cso  = {};
  val_t input         = {.idx = 0};  // == { . f = 0.0f }
  int correct         = INT_MAX;
};

/** mechanism to use named aggregate initialization before C++20, and also use
    the struct defaults. Using it directly only works if all defaulted
    members come after ones explicitly mentioned.
**/
#define CHILD_IDX_TEST_PARAMS(...)                                    \
  []() {                                                              \
    struct NonDefaultChildIdxTestParams : public ChildIdxTestParams { \
      NonDefaultChildIdxTestParams() { __VA_ARGS__; }                 \
    };                                                                \
    return ChildIdxTestParams(NonDefaultChildIdxTestParams());        \
  }()

// proto inner node
struct pin {
  bool def_left = false, is_categorical = false;
  int fid      = 0;  // feature id
  int set      = 0;
  float thresh = 0.0f;
  int left     = 1;  // left child idx
  operator sparse_node16()
  {
    val_t split = is_categorical ? {.idx = set} : {.f = thresh};
    return {{.idx = left}, split, fid, def_left, false, is_categorical};
  }
  operator sparse_node8()
  {
    val_t split = is_categorical ? {.idx = set} : {.f = thresh};
    return {{.idx = left}, split, fid, def_left, false, is_categorical};
  }
  operator dense_node()
  {
    val_t split = is_categorical ? {.idx = set} : {.f = thresh};
    return {{}, split, fid, def_left, false, is_categorical};
  }
};

#define PIN(...)                        \
  []() {                                \
    struct NonDefaultPin : public pin { \
      NonDefaultPin() { __VA_ARGS__; }  \
    };                                  \
    return pin(NonDefaultPin());        \
  }()

// proto category sets for one node
struct PCS {
  // each bit set for each feature id is in a separate vector
  // read each uint8_t from right to left, and the vector(s) - from left to right
  std::vector<std::vector<uint8_t>> bits;
  std::vector<int> max_matching;
  operator cat_sets_owner()
  {
    ASSERT(bits.size() == max_matching.size(),
           "internal error: PCS::bits.size() != PCS::max_matching.size()");
    std::vector<uint8_t> flat;
    for (std::vector<uint8_t> v : bits) {
      for (uint8_t b : v)
        flat.push_back(b);
    }
    return {.bits = flat, .max_matching = max_matching};
  }
};

template <typename fil_node_t>
class BaseFilTest : public testing::TestWithParam<FilTestParams> {
 protected:
  void SetUp() override { param = GetParam(); }
  void check() override
  {
    tree_base tree = param.cso.accessor();
    // nan -> !def_left, categorical -> if matches, numerical -> input >= threshold
    int test_idx = tree.child_index<true>(param.node, parent_node_idx, param.input);
    ASSERT(test_idx == correct, "child index test: actual %d != correct %d", test_idx, correct);
  }

  TestParams param;
};

/* for dense nodes, left (false) == parent * 2 + 1, right (true) == parent * 2 + 2
   E.g. see tree below:
 0 -> 1, 2
 1 -> 3, 4
 2 -> 5, 6
 3 -> 7, 8
 4 -> 9, 10
 */
std::vector<ChildIdxTestParams<dense_node>> params = {
  CHILD_IDX_TEST_PARAMS(PIN(thresh = 0.0f), input.f = -INF, correct = 1),
  CHILD_IDX_TEST_PARAMS(PIN(thresh = 0.0f), input.f = 0.0f, correct = 2),
  CHILD_IDX_TEST_PARAMS(PIN(thresh = 0.0f), input.f = +INF, correct = 2),
  CHILD_IDX_TEST_PARAMS(PIN(thresh = 0.0f), input.f = nan, correct = 1),
  CHILD_IDX_TEST_PARAMS(PIN(thresh = nan), input.f = nan, correct = 1),
  CHILD_IDX_TEST_PARAMS(PIN(thresh = nan), input.f = 0.0f, correct = 1),
  CHILD_IDX_TEST_PARAMS(PIN(thresh = 0.0f), input.f = -INF, parent_node_idx = 1, correct = 3),
  CHILD_IDX_TEST_PARAMS(PIN(thresh = 0.0f), input.f = 0.0f, parent_node_idx = 1, correct = 4),
  CHILD_IDX_TEST_PARAMS(PIN(thresh = 0.0f), input.f = -INF, parent_node_idx = 2, correct = 5),
  CHILD_IDX_TEST_PARAMS(PIN(thresh = 0.0f), input.f = 0.0f, parent_node_idx = 2, correct = 6),
  CHILD_IDX_TEST_PARAMS(PIN(thresh = 0.0f), input.f = -INF, parent_node_idx = 3, correct = 7),
  CHILD_IDX_TEST_PARAMS(PIN(thresh = 0.0f), input.f = 0.0f, parent_node_idx = 3, correct = 8),
  CHILD_IDX_TEST_PARAMS(PIN(thresh = 0.0f), input.f = -INF, parent_node_idx = 4, correct = 9),
  CHILD_IDX_TEST_PARAMS(PIN(thresh = 0.0f), input.f = 0.0f, parent_node_idx = 4, correct = 10),
  CHILD_IDX_TEST_PARAMS(
    PIN(is_categorical = true), input.idx = 0, cso.bits = {}, cso.max_matching = {-1}, correct = 1),
  CHILD_IDX_TEST_PARAMS(PIN(is_categorical = true),
                        input.idx        = 0,
                        cso.bits         = {0b0000'0000},
                        cso.max_matching = {0},
                        correct          = 2)};

TEST_P(TreeliteThrowSparse8FilTest, Import) { check(); }

INSTANTIATE_TEST_CASE_P(FilTests,
                        TreeliteThrowSparse8FilTest,
                        testing::ValuesIn(import_throw_sparse8_inputs));
}  // namespace ML
