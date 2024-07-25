/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "node.cuh"

#include <cuml/common/utils.hpp>

namespace cuml {
namespace genetic {

const int node::kInvalidFeatureId = -1;

node::node() {}

node::node(node::type ft) : t(ft)
{
  ASSERT(is_nonterminal(), "node: ctor with `type` argument expects functions type only!");
  u.fid = kInvalidFeatureId;
}

node::node(int fid) : t(node::type::variable) { u.fid = fid; }

node::node(float val) : t(node::type::constant) { u.val = val; }

node::node(const node& src) : t(src.t), u(src.u) {}

node& node::operator=(const node& src)
{
  t = src.t;
  u = src.u;
  return *this;
}

bool node::is_terminal() const { return detail::is_terminal(t); }

bool node::is_nonterminal() const { return detail::is_nonterminal(t); }

int node::arity() const { return detail::arity(t); }

#define CASE(str, val) \
  if (#val == str) return node::type::val
node::type node::from_str(const std::string& ntype)
{
  CASE(ntype, variable);
  CASE(ntype, constant);
  // note: keep the case statements in alphabetical order under each category of
  // operators.
  // binary operators
  CASE(ntype, add);
  CASE(ntype, atan2);
  CASE(ntype, div);
  CASE(ntype, fdim);
  CASE(ntype, max);
  CASE(ntype, min);
  CASE(ntype, mul);
  CASE(ntype, pow);
  CASE(ntype, sub);
  // unary operators
  CASE(ntype, abs);
  CASE(ntype, acos);
  CASE(ntype, asin);
  CASE(ntype, atan);
  CASE(ntype, acosh);
  CASE(ntype, asinh);
  CASE(ntype, atanh);
  CASE(ntype, cbrt);
  CASE(ntype, cos);
  CASE(ntype, cosh);
  CASE(ntype, cube);
  CASE(ntype, exp);
  CASE(ntype, inv);
  CASE(ntype, log);
  CASE(ntype, neg);
  CASE(ntype, rcbrt);
  CASE(ntype, rsqrt);
  CASE(ntype, sq);
  CASE(ntype, sqrt);
  CASE(ntype, sin);
  CASE(ntype, sinh);
  CASE(ntype, tan);
  CASE(ntype, tanh);
  ASSERT(false, "node::from_str: Bad type passed '%s'!", ntype.c_str());
}
#undef CASE

}  // namespace genetic
}  // namespace cuml
