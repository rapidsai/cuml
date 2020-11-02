/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cuml/gp/node.h>
#include <raft/cudart_utils.h>

namespace cuml {
namespace gp {

TEST(GP, node_test) {
  node feature(1);
  ASSERT_EQ(feature.t,  node::type::variable);
  ASSERT_TRUE(feature.is_terminal());
  ASSERT_FALSE(feature.is_nonterminal());
  ASSERT_EQ(feature.arity(), 0);
  ASSERT_EQ(feature.u.fid, 1);

  node constval(0.1f);
  ASSERT_EQ(constval.t,  node::type::constant);
  ASSERT_TRUE(constval.is_terminal());
  ASSERT_FALSE(constval.is_nonterminal());
  ASSERT_EQ(constval.arity(), 0);
  ASSERT_EQ(constval.u.val, 0.1f);

  node func1(node::type::add);
  ASSERT_EQ(func1.t,  node::type::add);
  ASSERT_FALSE(func1.is_terminal());
  ASSERT_TRUE(func1.is_nonterminal());
  ASSERT_EQ(func1.arity(), 2);
  ASSERT_EQ(func1.u.fid, node::kInvalidFeatureId);

  node func2(node::type::cosh);
  ASSERT_EQ(func2.t,  node::type::cosh);
  ASSERT_FALSE(func2.is_terminal());
  ASSERT_TRUE(func2.is_nonterminal());
  ASSERT_EQ(func2.arity(), 1);
  ASSERT_EQ(func2.u.fid, node::kInvalidFeatureId);
}

TEST(GP, node_from_str) {
  ASSERT_EQ(node::from_str("add"), node::type::add);
  ASSERT_EQ(node::from_str("tanh"), node::type::tanh);
  ASSERT_THROW(node::from_str("bad_type"), raft::exception);
}

}  // namespace gp
}  // namespace cuml
