/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/linalg/unary_op.cuh>

namespace MLCommon {
namespace Functions {

template <typename T, typename IdxType = int>
void f_log(T* out, T* in, T scalar, IdxType len, cudaStream_t stream)
{
  raft::linalg::unaryOp(
    out, in, len, [scalar] __device__(T in) { return raft::log(in) * scalar; }, stream);
}

};  // end namespace Functions
};  // end namespace MLCommon
