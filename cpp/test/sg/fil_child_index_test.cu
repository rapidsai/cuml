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
  bool def_left       = false;  // default left, see base_node::def_left
  bool is_categorical = false;  // see base_node::is_categorical
  int fid             = 0;      // feature id, see base_node::fid
  int set             = 0;      // which bit set represents the matching category list
  float thresh        = 0.0f;   // threshold, see base_node::thresh
  int left            = 1;      // left child idx, see sparse_node*::left_index()
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

/** mechanism to use named aggregate initialization before C++20, and also use
    the struct defaults. Using it directly only works if all defaulted
    members come after ones explicitly mentioned. C++ doesn't have reflection,
    so any non-macro alternative would need a separate list of member accessors.
**/
// proto inner node
#define NODE(...)                                               \
  []() {                                                        \
    struct NonDefaultProtoInnerNode : public proto_inner_node { \
      NonDefaultProtoInnerNode() { __VA_ARGS__; }               \
    };                                                          \
    return proto_inner_node(NonDefaultProtoInnerNode());        \
  }()

// proto category sets for one node
struct ProtoCategorySets {
  // each bit set for each feature id is in a separate vector
  // read each uint8_t from right to left, and the vector(s) - from left to right
  std::vector<std::vector<uint8_t>> bits;
  std::vector<float> fid_num_cats;
  operator cat_sets_owner()
  {
    ASSERT(bits.size() == fid_num_cats.size(),
           "internal error: ProtoCategorySets::bits.size() != "
           "ProtoCategorySets::fid_num_cats.size()");
    std::vector<uint8_t> flat;
    for (std::vector<uint8_t> v : bits) {
      for (uint8_t b : v)
        flat.push_back(b);
    }
    return {flat, fid_num_cats};
  }
};

struct ChildIndexTestParams {
  proto_inner_node node;
  int parent_node_idx = 0;
  cat_sets_owner cso;
  float input = 0.0f;
  int correct = INT_MAX;
};

std::ostream& operator<<(std::ostream& os, const ChildIndexTestParams& ps)
{
  os << "node = {\n"
     << ps.node << "\n} "
     << "parent_node_idx = " << ps.parent_node_idx << " cat_sets_owner = {\n"
     << ps.cso << "\n} input = " << ps.input << " correct = " << ps.correct;
  return os;
}

/** mechanism to use named aggregate initialization before C++20, and also use
    the struct defaults. Using it directly only works if all defaulted
    members come after ones explicitly mentioned. C++ doesn't have reflection,
    so any non-macro alternative would need a separate list of member accessors.
**/
#define CHILD_INDEX_TEST_PARAMS(...)                                      \
  []() {                                                                  \
    struct NonDefaultChildIndexTestParams : public ChildIndexTestParams { \
      NonDefaultChildIndexTestParams() { __VA_ARGS__; }                   \
    };                                                                    \
    return ChildIndexTestParams(NonDefaultChildIndexTestParams());        \
  }()

template <typename fil_node_t>
class ChildIndexTest : public testing::TestWithParam<ChildIndexTestParams> {
 protected:
  void check()
  {
    ChildIndexTestParams param = GetParam();
    tree_base tree{param.cso.accessor()};
    if (!std::is_same<fil_node_t, fil::dense_node>::value) {
      // test that the logic uses node.left instead of parent_node_idx
      param.node.left       = param.parent_node_idx * 2 + 1;
      param.parent_node_idx = INT_MIN;
    }
    // nan -> !def_left, categorical -> if matches, numerical -> input >= threshold
    int test_idx =
      tree.child_index<true>((fil_node_t)param.node, param.parent_node_idx, param.input);
    ASSERT(test_idx == param.correct,
           "child index test: actual %d != correct %d",
           test_idx,
           param.correct);
  }
};

typedef ChildIndexTest<fil::dense_node> ChildIndexTestDense;
typedef ChildIndexTest<fil::sparse_node16> ChildIndexTestSparse16;
typedef ChildIndexTest<fil::sparse_node8> ChildIndexTestSparse8;

/* for dense nodes, left (false) == parent * 2 + 1, right (true) == parent * 2 + 2
   E.g. see tree below:
 0 -> 1, 2
 1 -> 3, 4
 2 -> 5, 6
 3 -> 7, 8
 4 -> 9, 10
 */
const float INF = std::numeric_limits<float>::infinity();

std::vector<ChildIndexTestParams> params = {
  CHILD_INDEX_TEST_PARAMS(node = NODE(thresh = 0.0f), input = -INF, correct = 1),  // val !>= thresh
  CHILD_INDEX_TEST_PARAMS(node = NODE(thresh = 0.0f), input = 0.0f, correct = 2),  // val >= thresh
  CHILD_INDEX_TEST_PARAMS(node = NODE(thresh = 0.0f), input = +INF, correct = 2),  // val >= thresh
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 1.0f), input = -3.141592f, correct = 1),  // val !>= thresh
  CHILD_INDEX_TEST_PARAMS(                                         // val >= thresh (e**pi > pi**e)
    node    = NODE(thresh = 22.459158f),
    input   = 23.140693f,
    correct = 2),
  CHILD_INDEX_TEST_PARAMS(  // val >= thresh for both negative
    node    = NODE(thresh = -0.37f),
    input   = -0.36f,
    correct = 2),                                                                   // val >= thresh
  CHILD_INDEX_TEST_PARAMS(node = NODE(thresh = -INF), input = 0.36f, correct = 2),  // val >= thresh
  CHILD_INDEX_TEST_PARAMS(node = NODE(thresh = 0.0f), input = NAN, correct = 2),    // !def_left
  CHILD_INDEX_TEST_PARAMS(node = NODE(def_left = true), input = NAN, correct = 1),  // !def_left
  CHILD_INDEX_TEST_PARAMS(node = NODE(thresh = NAN), input = NAN, correct = 2),     // !def_left
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(def_left = true, thresh = NAN), input = NAN, correct = 1),        // !def_left
  CHILD_INDEX_TEST_PARAMS(node = NODE(thresh = NAN), input = 0.0f, correct = 1),  // val !>= thresh
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 0.0f), parent_node_idx = 1, input = -INF, correct = 3),
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 0.0f), parent_node_idx = 1, input = 0.0f, correct = 4),
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 0.0f), parent_node_idx = 2, input = -INF, correct = 5),
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 0.0f), parent_node_idx = 2, input = 0.0f, correct = 6),
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 0.0f), parent_node_idx = 3, input = -INF, correct = 7),
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 0.0f), parent_node_idx = 3, input = 0.0f, correct = 8),
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 0.0f), parent_node_idx = 4, input = -INF, correct = 9),
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 0.0f), parent_node_idx = 4, input = 0.0f, correct = 10),
  CHILD_INDEX_TEST_PARAMS(parent_node_idx = 4, input = NAN, correct = 10),  // !def_left
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(def_left = true), input = NAN, parent_node_idx = 4, correct = 9),  // !def_left
  // cannot match ( < 0 and realistic fid_num_cats)
  CHILD_INDEX_TEST_PARAMS(node             = NODE(is_categorical = true),
                          cso.bits         = {},
                          cso.fid_num_cats = {11.0f},
                          input            = -5,
                          correct          = 1),
  // Skipping category < 0 and dummy categorical node: fid_num_cats == 0. Prevented by FIL
  // import. cannot match ( > INT_MAX)
  CHILD_INDEX_TEST_PARAMS(node             = NODE(is_categorical = true),
                          cso.bits         = {0b1111'1111},
                          cso.fid_num_cats = {8.0f},
                          input            = (float)(1ll << 33ll),
                          correct          = 1),
  // cannot match ( >= fid_num_cats and integer)
  CHILD_INDEX_TEST_PARAMS(node             = NODE(is_categorical = true),
                          cso.bits         = {0b1111'1111},
                          cso.fid_num_cats = {2.0f},
                          input            = 2,
                          correct          = 1),
  // matches ( < fid_num_cats because comparison is floating-point and there's no rounding)
  CHILD_INDEX_TEST_PARAMS(node             = NODE(is_categorical = true),
                          cso.bits         = {0b1111'1111},
                          cso.fid_num_cats = {2.0f},
                          input            = 1.8f,
                          correct          = 2),
  // cannot match ( >= fid_num_cats)
  CHILD_INDEX_TEST_PARAMS(node             = NODE(is_categorical = true),
                          cso.bits         = {0b1111'1111},
                          cso.fid_num_cats = {2.0f},
                          input            = 2.1f,
                          correct          = 1),
  // does not match (bits[category] == 0, category == 0)
  CHILD_INDEX_TEST_PARAMS(node             = NODE(is_categorical = true),
                          cso.bits         = {0b0000'0000},
                          cso.fid_num_cats = {1.0f},
                          input            = 0,
                          correct          = 1),
  // matches (negative zero)
  CHILD_INDEX_TEST_PARAMS(node             = NODE(is_categorical = true),
                          cso.bits         = {0b0000'0001},
                          cso.fid_num_cats = {1.0f},
                          input            = -0.0f,
                          correct          = 2),
  // matches (positive zero)
  CHILD_INDEX_TEST_PARAMS(node             = NODE(is_categorical = true),
                          cso.bits         = {0b0000'0001},
                          cso.fid_num_cats = {1.0f},
                          input            = 0,
                          correct          = 2),
  // matches
  CHILD_INDEX_TEST_PARAMS(node             = NODE(is_categorical = true),
                          cso.bits         = {0b0000'0101},
                          cso.fid_num_cats = {3.0f, 1.0f},
                          input            = 2,
                          correct          = 2),
  // does not match (bits[category] == 0, category > 0)
  CHILD_INDEX_TEST_PARAMS(node             = NODE(is_categorical = true),
                          cso.bits         = {0b0000'0101},
                          cso.fid_num_cats = {3.0f},
                          input            = 1,
                          correct          = 1),
  // cannot match (fid_num_cats[fid=1] <= input)
  CHILD_INDEX_TEST_PARAMS(node             = NODE(is_categorical = true),
                          node.fid         = 1,
                          cso.bits         = {0b0000'0101},
                          cso.fid_num_cats = {3.0f, 1.0f},
                          input            = 2,
                          correct          = 1),
  // default left
  CHILD_INDEX_TEST_PARAMS(node             = NODE(is_categorical = true, def_left = true),
                          cso.bits         = {0b0000'0101},
                          cso.fid_num_cats = {3.0f},
                          input            = NAN,
                          correct          = 1),
  // default right
  CHILD_INDEX_TEST_PARAMS(node             = NODE(is_categorical = true, def_left = false),
                          cso.bits         = {0b0000'0101},
                          cso.fid_num_cats = {3.0f},
                          input            = NAN,
                          correct          = 2),
};

TEST_P(ChildIndexTestDense, Predict) { check(); }
TEST_P(ChildIndexTestSparse16, Predict) { check(); }
TEST_P(ChildIndexTestSparse8, Predict) { check(); }

INSTANTIATE_TEST_CASE_P(FilTests, ChildIndexTestDense, testing::ValuesIn(params));
INSTANTIATE_TEST_CASE_P(FilTests, ChildIndexTestSparse16, testing::ValuesIn(params));
INSTANTIATE_TEST_CASE_P(FilTests, ChildIndexTestSparse8, testing::ValuesIn(params));
}  // namespace ML
