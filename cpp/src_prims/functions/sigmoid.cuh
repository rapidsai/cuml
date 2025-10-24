/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>

namespace MLCommon {
namespace Functions {

template <typename T, typename IdxType = int>
void sigmoid(T* out, T* in, IdxType len, cudaStream_t stream)
{
  T one = T(1);
  raft::linalg::unaryOp(
    out, in, len, [one] __device__(T in) { return one / (one + raft::exp(-in)); }, stream);
}

};  // end namespace Functions
};  // end namespace MLCommon
