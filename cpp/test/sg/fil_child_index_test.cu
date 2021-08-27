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

struct proto_inner_node {
  bool def_left = false, is_categorical = false;
  int fid      = 0;     // feature id
  int set      = 0;     // which bit set represents the matching category list
  float thresh = 0.0f;  // threshold
  int left     = 1;     // left child idx
  val_t split()
  {
    val_t split;
    if (is_categorical)
      split.idx = set;
    else
      split.f = thresh;
    return split;
  }
  operator sparse_node16()
  {
    return sparse_node16({}, split(), fid, def_left, false, is_categorical, left);
  }
  operator sparse_node8()
  {
    return sparse_node8({}, split(), fid, def_left, false, is_categorical, left);
  }
  operator dense_node() { return dense_node({}, split(), fid, def_left, false, is_categorical); }
};

std::ostream& operator<<(std::ostream& os, const proto_inner_node& node)
{
  os << "def_left " << node.def_left << " is_categorical " << node.is_categorical << " fid "
     << node.fid << " set " << node.set << " thresh " << node.thresh << " left " << node.left;
  return os;
}

// proto inner node
#define NODE(...)                                               \
  []() {                                                        \
    struct NonDefaultProtoInnerNode : public proto_inner_node { \
      NonDefaultProtoInnerNode() { __VA_ARGS__; }               \
    };                                                          \
    return proto_inner_node(NonDefaultProtoInnerNode());        \
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
    return {flat, max_matching};
  }
};

struct ChildIdxTestParams {
  proto_inner_node node;
  int parent_node_idx = 0;
  cat_sets_owner cso;
  float input = 0.0f;
  int correct = INT_MAX;
};

std::ostream& operator<<(std::ostream& os, const ChildIdxTestParams& ps)
{
  os << "node = {\n"
     << ps.node << "\n} "
     << "parent_node_idx = " << ps.parent_node_idx << " cat_sets_owner = {\n"
     << ps.cso << "\n} input = " << ps.input << " correct = " << ps.correct;
  return os;
}

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

template <typename fil_node_t>
class ChildIdxTest : public testing::TestWithParam<ChildIdxTestParams> {
 protected:
  void check()
  {
    ChildIdxTestParams param = GetParam();
    tree_base tree{param.cso.accessor()};
    // nan -> !def_left, categorical -> if matches, numerical -> input >= threshold
    int test_idx =
      tree.child_index<true>((fil_node_t)param.node, param.parent_node_idx, param.input);
    ASSERT(test_idx == param.correct,
           "child index test: actual %d != correct %d",
           test_idx,
           param.correct);
  }
};

typedef ChildIdxTest<fil::dense_node> ChildIdxTestDense;

/* for dense nodes, left (false) == parent * 2 + 1, right (true) == parent * 2 + 2
   E.g. see tree below:
 0 -> 1, 2
 1 -> 3, 4
 2 -> 5, 6
 3 -> 7, 8
 4 -> 9, 10
 */
const float INF = std::numeric_limits<float>::infinity();

std::vector<ChildIdxTestParams> dense_params = {
  CHILD_IDX_TEST_PARAMS(node.thresh = 0.0f, input = -INF, correct = 1),   // val !>= thresh
  CHILD_IDX_TEST_PARAMS(node.thresh = 0.0f, input = 0.0f, correct = 2),   // val >= thresh
  CHILD_IDX_TEST_PARAMS(node.thresh = 0.0f, input = +INF, correct = 2),   // val >= thresh
  CHILD_IDX_TEST_PARAMS(node.thresh = 0.0f, input = NAN, correct = 2),    // !def_left
  CHILD_IDX_TEST_PARAMS(node.def_left = true, input = NAN, correct = 1),  // !def_left
  CHILD_IDX_TEST_PARAMS(node.thresh = NAN, input = NAN, correct = 2),     // !def_left
  CHILD_IDX_TEST_PARAMS(
    node.def_left = true, node.thresh = NAN, input = NAN, correct = 1),  // !def_left
  CHILD_IDX_TEST_PARAMS(node.thresh = NAN, input = 0.0f, correct = 1),   // val !>= thresh
  CHILD_IDX_TEST_PARAMS(node.thresh = 0.0f, parent_node_idx = 1, input = -INF, correct = 3),
  CHILD_IDX_TEST_PARAMS(node.thresh = 0.0f, parent_node_idx = 1, input = 0.0f, correct = 4),
  CHILD_IDX_TEST_PARAMS(node.thresh = 0.0f, parent_node_idx = 2, input = -INF, correct = 5),
  CHILD_IDX_TEST_PARAMS(node.thresh = 0.0f, parent_node_idx = 2, input = 0.0f, correct = 6),
  CHILD_IDX_TEST_PARAMS(node.thresh = 0.0f, parent_node_idx = 3, input = -INF, correct = 7),
  CHILD_IDX_TEST_PARAMS(node.thresh = 0.0f, parent_node_idx = 3, input = 0.0f, correct = 8),
  CHILD_IDX_TEST_PARAMS(node.thresh = 0.0f, parent_node_idx = 4, input = -INF, correct = 9),
  CHILD_IDX_TEST_PARAMS(node.thresh = 0.0f, parent_node_idx = 4, input = 0.0f, correct = 10),
  CHILD_IDX_TEST_PARAMS(parent_node_idx = 4, input = NAN, correct = 10),  // !def_left
  CHILD_IDX_TEST_PARAMS(
    node.def_left = true, input = NAN, parent_node_idx = 4, correct = 9),  // !def_left
  // cannot match ( > max_matching)
  CHILD_IDX_TEST_PARAMS(
    node.is_categorical = true, cso.bits = {}, cso.max_matching = {-1}, input = 0, correct = 1),
  // doesn't match (bits[category] == 0, category == 0)
  CHILD_IDX_TEST_PARAMS(node.is_categorical = true,
                        cso.bits            = {0b0000'0000},
                        cso.max_matching    = {0},
                        input               = 0,
                        correct             = 1),
  // matches
  CHILD_IDX_TEST_PARAMS(node.is_categorical = true,
                        cso.bits            = {0b0000'0001},
                        cso.max_matching    = {0},
                        input               = 0,
                        correct             = 2),
  // matches
  CHILD_IDX_TEST_PARAMS(node.is_categorical = true,
                        cso.bits            = {0b0000'0101},
                        cso.max_matching    = {2, -1},
                        input               = 2,
                        correct             = 2),
  // doesn't match (bits[category] == 0, category > 0)
  CHILD_IDX_TEST_PARAMS(node.is_categorical = true,
                        cso.bits            = {0b0000'0101},
                        cso.max_matching    = {2},
                        input               = 1,
                        correct             = 1),
  // canot match (max_matching[fid=1] == -1)
  CHILD_IDX_TEST_PARAMS(node.is_categorical = true,
                        node.fid            = 1,
                        cso.bits            = {0b0000'0101},
                        cso.max_matching    = {2, -1},
                        input               = 2,
                        correct             = 1),
};

TEST_P(ChildIdxTestDense, Predict) { check(); }

INSTANTIATE_TEST_CASE_P(FilTests, ChildIdxTestDense, testing::ValuesIn(dense_params));
}  // namespace ML
