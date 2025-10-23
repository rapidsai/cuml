/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/linalg/unary_op.cuh>

namespace MLCommon {
namespace Functions {

template <typename math_t>
void softThres(
  math_t* out, const math_t* in, const math_t thres, const int len, cudaStream_t stream)
{
  raft::linalg::unaryOp(
    out,
    in,
    len,
    [thres] __device__(math_t in) {
      if (in > math_t(0) && thres < raft::abs(in))
        return in - thres;
      else if (in < math_t(0) && thres < raft::abs(in))
        return in + thres;
      else
        return math_t(0);
    },
    stream);
}

};  // namespace Functions
};  // namespace MLCommon
// end namespace ML
