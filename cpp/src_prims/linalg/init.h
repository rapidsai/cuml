/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cudart_utils.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

namespace MLCommon {
namespace LinAlg {

namespace {

/**
 * @brief Like Python range.
 *
 * Fills the output as out[i] = i.
 *
 * \param [out] out device array, size [end-start]
 * \param [in] start of the range
 * \param [in] end of range (exclusive)
 * \param [in] stream cuda stream
 */
template <typename T>
void range(T* out, int start, int end, cudaStream_t stream)
{
  thrust::counting_iterator<int> first(start);
  thrust::counting_iterator<int> last = first + (end - start);
  thrust::device_ptr<T> ptr(out);
  thrust::copy(thrust::cuda::par.on(stream), first, last, ptr);
}

/**
 * @brief Like Python range.
 *
 * Fills the output as out[i] = i.
 *
 * \param [out] out device array, size [n]
 * \param [in] n length of the array
 * \param [in] stream cuda stream
 */
template <typename T, int TPB = 256>
void range(T* out, int n, cudaStream_t stream)
{
  range(out, 0, n, stream);
}

/**
 * @brief Zeros the output.
 *
 * \param [out] out device array, size [n]
 * \param [in] n length of the array
 * \param [in] stream cuda stream
 */
template <typename T>
void zero(T* out, int n, cudaStream_t stream)
{
  RAFT_CUDA_TRY(cudaMemsetAsync(static_cast<void*>(out), 0, n * sizeof(T), stream));
}

}  // unnamed namespace
}  // namespace LinAlg
}  // namespace MLCommon
