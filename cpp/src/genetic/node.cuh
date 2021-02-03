/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#pragma once

#include <cuml/genetic/node.h>
#include <raft/cuda_utils.cuh>

namespace cuml {
namespace genetic {
namespace detail {

static constexpr float MIN_VAL = 0.001f;

HDI bool is_terminal(node::type t) {
  return t == node::type::variable || t == node::type::constant;
}

HDI bool is_nonterminal(node::type t) { return !is_terminal(t); }

HDI int arity(node::type t) {
  if (node::type::unary_begin <= t && t <= node::type::unary_end) {
    return 1;
  }
  if (node::type::binary_begin <= t && t <= node::type::binary_end) {
    return 2;
  }
  return 0;
}

// `data` assumed to be stored in col-major format
DI float evaluate_node(const node& n, const float* data, size_t stride,
                       float inval, float inval1) {
  if (n.t == node::type::constant) {
    return n.u.val;
  } else if (n.t == node::type::variable) {
    return n.u.fid != node::kInvalidFeatureId ? data[n.u.fid * stride] : 0.f;
  } else {
    auto abs_inval = fabsf(inval), abs_inval1 = fabsf(inval1);
    auto small = abs_inval < MIN_VAL;
    // note: keep the case statements in alphabetical order under each category
    // of operators.
    switch (n.t) {
      // binary operators
      case node::type::add:
        return inval + inval1;
      case node::type::atan2:
        return atan2f(inval, inval1);
      case node::type::div:
        return abs_inval1 < MIN_VAL ? 1.f : fdividef(inval, inval1);
      case node::type::fdim:
        return fdimf(inval, inval1);
      case node::type::max:
        return fmaxf(inval, inval1);
      case node::type::min:
        return fminf(inval, inval1);
      case node::type::mul:
        return inval * inval1;
      case node::type::pow:
        return powf(inval, inval1);
      case node::type::sub:
        return inval - inval1;
      // unary operators
      case node::type::abs:
        return abs_inval;
      case node::type::acos:
        return acosf(inval);
      case node::type::acosh:
        return acoshf(inval);
      case node::type::asin:
        return asinf(inval);
      case node::type::asinh:
        return asinhf(inval);
      case node::type::atan:
        return atanf(inval);
      case node::type::atanh:
        return atanhf(inval);
      case node::type::cbrt:
        return cbrtf(inval);
      case node::type::cos:
        return cosf(inval);
      case node::type::cosh:
        return coshf(inval);
      case node::type::cube:
        return inval * inval * inval;
      case node::type::exp:
        return expf(inval);
      case node::type::inv:
        return abs_inval < MIN_VAL ? 0.f : 1.f / inval;
      case node::type::log:
        return abs_inval < MIN_VAL ? 0.f : logf(abs_inval);
      case node::type::neg:
        return -inval;
      case node::type::rcbrt:
        return rcbrtf(inval);
      case node::type::rsqrt:
        return rsqrtf(abs_inval);
      case node::type::sin:
        return sinf(inval);
      case node::type::sinh:
        return sinhf(inval);
      case node::type::sq:
        return inval * inval;
      case node::type::sqrt:
        return sqrtf(abs_inval);
      case node::type::tan:
        return tanf(inval);
      case node::type::tanh:
        return tanhf(inval);
      // shouldn't reach here!
      default:
        return 0.f;
    };
  }
}

}  // namespace detail
}  // namespace genetic
}  // namespace cuml
