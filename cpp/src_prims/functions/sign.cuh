/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/linalg/unary_op.cuh>

namespace MLCommon {
namespace Functions {

template <typename math_t, typename idx_type = int>
void sign(
  math_t* out, const math_t* in, const math_t scalar, const idx_type len, cudaStream_t stream)
{
  raft::linalg::unaryOp(
    out,
    in,
    len,
    [scalar] __device__(math_t in) {
      if (in < math_t(0))
        return (math_t(-1) * scalar);
      else if (in > math_t(0))
        return (math_t(1) * scalar);
      else
        return math_t(0);
    },
    stream);
}

template <typename math_t, typename idx_type = int>
void sign(math_t* out, const math_t* in, const idx_type n_len, cudaStream_t stream)
{
  math_t scalar = math_t(1);
  sign(out, in, scalar, n_len, stream);
}

};  // namespace Functions
};  // namespace MLCommon
// end namespace ML
