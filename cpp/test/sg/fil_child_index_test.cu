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
  double thresh       = 0.0;    // threshold, see base_node::thresh
  int left            = 1;      // left child idx, see sparse_node*::left_index()
  template <typename real_t>
  val_t<real_t> split()
  {
    val_t<real_t> split;
    if (is_categorical)
      split.idx = set;
    else if (std::isnan(thresh))
      split.f = std::numeric_limits<real_t>::quiet_NaN();
    else
      split.f = static_cast<real_t>(thresh);
    return split;
  }
  template <typename real_t>
  operator dense_node<real_t>()
  {
    return dense_node<real_t>({}, split<real_t>(), fid, def_left, false, is_categorical);
  }
  template <typename real_t>
  operator sparse_node16<real_t>()
  {
    return sparse_node16<real_t>({}, split<real_t>(), fid, def_left, false, is_categorical, left);
  }
  operator sparse_node8()
  {
    return sparse_node8({}, split<float>(), fid, def_left, false, is_categorical, left);
  }
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
  double input  = 0.0;
  int correct   = INT_MAX;
  bool skip_f32 = false;  // if true, the test only runs for float64
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
  using real_t = typename fil_node_t::real_type;

 protected:
  void check()
  {
    ChildIndexTestParams param = GetParam();

    // skip tests that require float64 to work correctly
    if (std::is_same_v<real_t, float> && param.skip_f32) return;

    tree_base tree{param.cso.accessor()};
    if constexpr (!std::is_same_v<fil_node_t, fil::dense_node<real_t>>) {
      // test that the logic uses node.left instead of parent_node_idx
      param.node.left       = param.parent_node_idx * 2 + 1;
      param.parent_node_idx = INT_MIN;
    }
    real_t input = isnan(param.input) ? std::numeric_limits<real_t>::quiet_NaN()
                                      : static_cast<real_t>(param.input);
    // nan -> !def_left, categorical -> if matches, numerical -> input >= threshold
    int test_idx = tree.child_index<true>((fil_node_t)param.node, param.parent_node_idx, input);
    ASSERT_EQ(test_idx, param.correct)
      << "child index test: actual " << test_idx << "  != correct %d" << param.correct;
  }
};

using ChildIndexTestDenseFloat32    = ChildIndexTest<fil::dense_node<float>>;
using ChildIndexTestDenseFloat64    = ChildIndexTest<fil::dense_node<double>>;
using ChildIndexTestSparse16Float32 = ChildIndexTest<fil::sparse_node16<float>>;
using ChildIndexTestSparse16Float64 = ChildIndexTest<fil::sparse_node16<double>>;
using ChildIndexTestSparse8         = ChildIndexTest<fil::sparse_node8>;

/* for dense nodes, left (false) == parent * 2 + 1, right (true) == parent * 2 + 2
   E.g. see tree below:
 0 -> 1, 2
 1 -> 3, 4
 2 -> 5, 6
 3 -> 7, 8
 4 -> 9, 10
 */
const double INF  = std::numeric_limits<double>::infinity();
const double QNAN = std::numeric_limits<double>::quiet_NaN();

std::vector<ChildIndexTestParams> params = {
  CHILD_INDEX_TEST_PARAMS(node = NODE(thresh = 0.0), input = -INF, correct = 1),  // val !>= thresh
  CHILD_INDEX_TEST_PARAMS(node = NODE(thresh = 0.0), input = 0.0, correct = 2),   // val >= thresh
  CHILD_INDEX_TEST_PARAMS(node = NODE(thresh = 0.0), input = +INF, correct = 2),  // val >= thresh
  // the following two tests only work for float64
  CHILD_INDEX_TEST_PARAMS(node = NODE(thresh = 0.0), input = -1e-50, correct = 1, skip_f32 = true),
  CHILD_INDEX_TEST_PARAMS(node = NODE(thresh = 1e-50), input = 0.0, correct = 1, skip_f32 = true),
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 1.0), input = -3.141592, correct = 1),  // val !>= thresh
  CHILD_INDEX_TEST_PARAMS(                                       // val >= thresh (e**pi > pi**e)
    node    = NODE(thresh = 22.459158),
    input   = 23.140693,
    correct = 2),
  CHILD_INDEX_TEST_PARAMS(  // val >= thresh for both negative
    node    = NODE(thresh = -0.37),
    input   = -0.36,
    correct = 2),                                                                  // val >= thresh
  CHILD_INDEX_TEST_PARAMS(node = NODE(thresh = -INF), input = 0.36, correct = 2),  // val >= thresh
  CHILD_INDEX_TEST_PARAMS(node = NODE(thresh = 0.0f), input = QNAN, correct = 2),  // !def_left
  CHILD_INDEX_TEST_PARAMS(node = NODE(def_left = true), input = QNAN, correct = 1),  // !def_left
  CHILD_INDEX_TEST_PARAMS(node = NODE(thresh = QNAN), input = QNAN, correct = 2),    // !def_left
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(def_left = true, thresh = QNAN), input = QNAN, correct = 1),      // !def_left
  CHILD_INDEX_TEST_PARAMS(node = NODE(thresh = QNAN), input = 0.0, correct = 1),  // val !>= thresh
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 0.0), parent_node_idx = 1, input = -INF, correct = 3),
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 0.0), parent_node_idx = 1, input = 0.0f, correct = 4),
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 0.0), parent_node_idx = 2, input = -INF, correct = 5),
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 0.0), parent_node_idx = 2, input = 0.0f, correct = 6),
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 0.0), parent_node_idx = 3, input = -INF, correct = 7),
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 0.0), parent_node_idx = 3, input = 0.0f, correct = 8),
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 0.0), parent_node_idx = 4, input = -INF, correct = 9),
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(thresh = 0.0), parent_node_idx = 4, input = 0.0, correct = 10),
  CHILD_INDEX_TEST_PARAMS(parent_node_idx = 4, input = QNAN, correct = 10),  // !def_left
  CHILD_INDEX_TEST_PARAMS(
    node = NODE(def_left = true), input = QNAN, parent_node_idx = 4, correct = 9),  // !def_left
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
                          input            = QNAN,
                          correct          = 1),
  // default right
  CHILD_INDEX_TEST_PARAMS(node             = NODE(is_categorical = true, def_left = false),
                          cso.bits         = {0b0000'0101},
                          cso.fid_num_cats = {3.0f},
                          input            = QNAN,
                          correct          = 2),
};

TEST_P(ChildIndexTestDenseFloat32, Predict) { check(); }
TEST_P(ChildIndexTestDenseFloat64, Predict) { check(); }
TEST_P(ChildIndexTestSparse16Float32, Predict) { check(); }
TEST_P(ChildIndexTestSparse16Float64, Predict) { check(); }
TEST_P(ChildIndexTestSparse8, Predict) { check(); }

INSTANTIATE_TEST_CASE_P(FilTests, ChildIndexTestDenseFloat32, testing::ValuesIn(params));
INSTANTIATE_TEST_CASE_P(FilTests, ChildIndexTestDenseFloat64, testing::ValuesIn(params));
INSTANTIATE_TEST_CASE_P(FilTests, ChildIndexTestSparse16Float32, testing::ValuesIn(params));
INSTANTIATE_TEST_CASE_P(FilTests, ChildIndexTestSparse16Float64, testing::ValuesIn(params));
INSTANTIATE_TEST_CASE_P(FilTests, ChildIndexTestSparse8, testing::ValuesIn(params));
}  // namespace ML
