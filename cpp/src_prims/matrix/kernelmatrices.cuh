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

#pragma once

#include "grammatrix.cuh"
#include <raft/cuda_utils.cuh>
#include <raft/distance/distance.hpp>
#include <raft/linalg/gemm.hpp>

namespace MLCommon {
namespace Matrix {

using namespace MLCommon;

/** Epiloge function for polynomial kernel without padding.
 * Calculates output = (gain*in + offset)^exponent
 * @param inout device vector in column major format, size [len]
 * @param len array length
 * @param exponent
 * @param gain
 * @param offset
 */
template <typename math_t, typename exp_t>
__global__ void polynomial_kernel_nopad(
  math_t* inout, size_t len, exp_t exponent, math_t gain, math_t offset)
{
  for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < len;
       tid += blockDim.x * gridDim.x) {
    inout[tid] = pow(gain * inout[tid] + offset, exponent);
  }
}

/** Epiloge function for polynomial kernel with padding.
 * Calculates output = (gain*input + offset)^exponent
 * @param inout device vector in column major format, size [ld * cols]
 * @param ld leading dimension of the inout buffer
 * @param rows number of rows (rows <= ld)
 * @param cols number of colums
 * @param exponent
 * @param gain
 * @param offset
 */
template <typename math_t, typename exp_t>
__global__ void polynomial_kernel(
  math_t* inout, int ld, int rows, int cols, exp_t exponent, math_t gain, math_t offset)
{
  for (size_t tidy = threadIdx.y + blockIdx.y * blockDim.y; tidy < cols;
       tidy += blockDim.y * gridDim.y)
    for (size_t tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < rows;
         tidx += blockDim.x * gridDim.x) {
      inout[tidx + tidy * ld] = pow(gain * inout[tidx + tidy * ld] + offset, exponent);
    }
}

/** Epiloge function for tanh kernel without padding.
 * Calculates output = tanh(gain*input + offset)
 * @param inout device vector, size [len]
 * @param len length of the input vector
 * @param gain
 * @param offset
 */
template <typename math_t>
__global__ void tanh_kernel_nopad(math_t* inout, size_t len, math_t gain, math_t offset)
{
  for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < len;
       tid += blockDim.x * gridDim.x) {
    inout[tid] = tanh(gain * inout[tid] + offset);
  }
}

/** Epiloge function for tanh kernel without padding.
 * Calculates output = tanh(gain*input + offset)
 * @param inout device vector in column major format, size [ld * cols]
 * @param ld leading dimension of the inout buffer
 * @param rows number of rows (rows <= ld)
 * @param cols number of colums
 * @param gain
 * @param offset
 */
template <typename math_t>
__global__ void tanh_kernel(math_t* inout, int ld, int rows, int cols, math_t gain, math_t offset)
{
  for (size_t tidy = threadIdx.y + blockIdx.y * blockDim.y; tidy < cols;
       tidy += blockDim.y * gridDim.y)
    for (size_t tidx = threadIdx.x + blockIdx.x * blockDim.x; tidx < rows;
         tidx += blockDim.x * gridDim.x) {
      inout[tidx + tidy * ld] = tanh(gain * inout[tidx + tidy * ld] + offset);
    }
}

/**
 * Create a kernel matrix using polynomial kernel function.
 */
template <typename math_t, typename exp_t>
class PolynomialKernel : public GramMatrixBase<math_t> {
  exp_t exponent;
  math_t gain;
  math_t offset;

  void applyKernel(
    math_t* inout, int ld, int rows, int cols, bool is_row_major, cudaStream_t stream)
  {
    const int n_minor = is_row_major ? cols : rows;
    if (ld == n_minor) {
      polynomial_kernel_nopad<<<raft::ceildiv<size_t>((size_t)rows * cols, 128), 128, 0, stream>>>(
        inout, rows * cols, exponent, gain, offset);
    } else {
      int n1 = is_row_major ? cols : rows;
      int n2 = is_row_major ? rows : cols;
      polynomial_kernel<<<dim3(raft::ceildiv(n1, 32), raft::ceildiv(n2, 4), 1),
                          dim3(32, 4, 1),
                          0,
                          stream>>>(inout, ld, n1, n2, exponent, gain, offset);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

 public:
  /**
   * Constructs a polynomial kernel object.
   * It evaluates the kernel matrix using the following formula:
   * K_ij = (gain*<x1_i, x2_k> + offset)^exponent
   *
   * @tparam math_t floating point type
   * @tparam exp_t type of exponent
   * @param exponent
   * @param gain
   * @param offset
   * @param cublas_handle
   */
  PolynomialKernel(exp_t exponent, math_t gain, math_t offset, cublasHandle_t cublas_handle)
    : GramMatrixBase<math_t>(cublas_handle), exponent(exponent), gain(gain), offset(offset)
  {
  }

  /** Evaluate kernel matrix using polynomial kernel.
   *
   * output[i,k] = (gain*<x1_i, x2_k> + offset)^exponent,
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and < , > denotes dot product.
   *
   * @param [in] x1 device array of vectors, size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of features in x1 and x2
   * @param [in] x2 device array of vectors, size [n2*cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] is_row_major whether the input and output matrices are in row
   *        major format
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1
   * @param ld2 leading dimension of x2
   * @param ld_out leading dimension of out
   */
  void evaluate(const math_t* x1,
                int n1,
                int n_cols,
                const math_t* x2,
                int n2,
                math_t* out,
                bool is_row_major,
                cudaStream_t stream,
                int ld1,
                int ld2,
                int ld_out)
  {
    GramMatrixBase<math_t>::linear(
      x1, n1, n_cols, x2, n2, out, is_row_major, stream, ld1, ld2, ld_out);
    applyKernel(out, ld_out, n1, n2, is_row_major, stream);
  }
};

/**
 * Create a kernel matrix using tanh kernel function.
 */
template <typename math_t>
class TanhKernel : public GramMatrixBase<math_t> {
  math_t gain, offset;

  void applyKernel(
    math_t* inout, int ld, int rows, int cols, bool is_row_major, cudaStream_t stream)
  {
    const int n_minor = is_row_major ? cols : rows;
    if (ld == n_minor) {
      tanh_kernel_nopad<<<raft::ceildiv<size_t>((size_t)rows * cols, 128), 128, 0, stream>>>(
        inout, rows * cols, gain, offset);
    } else {
      int n1 = is_row_major ? cols : rows;
      int n2 = is_row_major ? rows : cols;
      tanh_kernel<<<dim3(raft::ceildiv(n1, 32), raft::ceildiv(n2, 4), 1),
                    dim3(32, 4, 1),
                    0,
                    stream>>>(inout, ld, n1, n2, gain, offset);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

 public:
  /**
   * Constructs a tanh kernel object.
   * It evaluates the kernel matrix using the following formula:
   * K_ij = tanh(gain*<x1_i, x2_k> + offset)
   *
   * @tparam math_t floating point type
   * @param gain
   * @param offset
   * @param cublas_handle
   */
  TanhKernel(math_t gain, math_t offset, cublasHandle_t cublas_handle)
    : GramMatrixBase<math_t>(cublas_handle), gain(gain), offset(offset)
  {
  }

  /** Evaluate kernel matrix using tanh kernel.
   *
   * output_[i + k*n1] = (gain*<x1_i, x2_k> + offset)^exponent,
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and < , > denotes dot product.
   *
   * @param [in] x1 device array of vectors,
   *  size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of features in x1 and x2
   * @param [in] x2 device array of vectors,
   *   size [n2*n_cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] is_row_major whether the input and output matrices are in row
   *        major format
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1 (usually it is n1)
   * @param ld2 leading dimension of x2 (usually it is n2)
   * @param ld_out leading dimension of out (usually it is n1)
   */
  void evaluate(const math_t* x1,
                int n1,
                int n_cols,
                const math_t* x2,
                int n2,
                math_t* out,
                bool is_row_major,
                cudaStream_t stream,
                int ld1,
                int ld2,
                int ld_out)
  {
    GramMatrixBase<math_t>::linear(
      x1, n1, n_cols, x2, n2, out, is_row_major, stream, ld1, ld2, ld_out);
    applyKernel(out, ld_out, n1, n2, is_row_major, stream);
  }
};

/**
 * Create a kernel matrix using RBF kernel function.
 */
template <typename math_t>
class RBFKernel : public GramMatrixBase<math_t> {
  math_t gain;

  void applyKernel(
    math_t* inout, int ld, int rows, int cols, bool is_row_major, cudaStream_t stream)
  {
    const int n_minor = is_row_major ? cols : rows;
    if (ld == n_minor) {
      rbf_kernel_nopad<<<raft::ceildiv<size_t>((size_t)rows * cols, 128), 128, 0, stream>>>(
        inout, rows * cols, gain);
    } else {
      int n1 = is_row_major ? cols : rows;
      int n2 = is_row_major ? rows : cols;
      rbf_kernel<<<dim3(raft::ceildiv(n1, 32), raft::ceildiv(n2, 4), 1),
                   dim3(32, 4, 1),
                   0,
                   stream>>>(inout, ld, n1, n2, gain);
    }
  }

 public:
  /**
   * Constructs a RBF kernel object.
   * It evaluates the kernel matrix using the following formula:
   * K_ij = exp(-gain*|x1_i- x2_k|^2)
   *
   * @tparam math_t floating point type
   * @param gain
   */
  RBFKernel(math_t gain) : GramMatrixBase<math_t>(NULL), gain(gain) {}

  /** Evaluate kernel matrix using RBF kernel.
   *
   * output_[i + k*n1] = exp(-gain*|x1_i - x2_k|^2),
   * where x1_i is the i-th vector from the x1 set, and x2_k is k-th vector
   * in the x2 set, and | | euclidean distance.
   *
   * @param [in] x1 device array of vectors, size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of features in x1 and x2
   * @param [in] x2 device array of vectors, size [n2*n_cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix, size [n1*n2]
   * @param [in] is_row_major whether the input and output matrices are in row
   *        major format
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1, currently only ld1 == n1 is supported
   * @param ld2 leading dimension of x2, currently only ld2 == n2 is supported
   * @param ld_out leading dimension of out, only ld_out == n1 is supported
   */
  void evaluate(const math_t* x1,
                int n1,
                int n_cols,
                const math_t* x2,
                int n2,
                math_t* out,
                bool is_row_major,
                cudaStream_t stream,
                int ld1,
                int ld2,
                int ld_out)
  {
    int minor1    = is_row_major ? n_cols : n1;
    int minor2    = is_row_major ? n_cols : n2;
    int minor_out = is_row_major ? n2 : n1;
    ASSERT(ld1 == minor1, "RBF Kernel distance does not support ld1 parameter");
    ASSERT(ld2 == minor2, "RBF Kernel distance does not support ld2 parameter");
    ASSERT(ld_out == minor_out, "RBF Kernel distance does not support ld_out parameter");
    distance(x1, n1, n_cols, x2, n2, out, is_row_major, stream, ld1, ld2, ld_out);
  }

  /** Customize distance function withe RBF epilogue */
  void distance(const math_t* x1,
                int n1,
                int n_cols,
                const math_t* x2,
                int n2,
                math_t* out,
                bool is_row_major,
                cudaStream_t stream,
                int ld1,
                int ld2,
                int ld_out)
  {
    math_t gain   = this->gain;
    using index_t = int64_t;

    auto fin_op = [gain] __device__(math_t d_val, index_t idx) { return exp(-gain * d_val); };
    raft::distance::distance<raft::distance::DistanceType::L2Unexpanded,
                             math_t,
                             math_t,
                             math_t,
                             decltype(fin_op),
                             index_t>(const_cast<math_t*>(x1),
                                      const_cast<math_t*>(x2),
                                      out,
                                      n1,
                                      n2,
                                      n_cols,
                                      NULL,
                                      0,
                                      fin_op,
                                      stream,
                                      is_row_major);
  }
};

};  // end namespace Matrix
};  // namespace MLCommon
