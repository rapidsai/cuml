/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cuda_utils.h>
#include <distance/distance.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/gemm.h>

namespace MLCommon {
namespace GramMatrix {

/**
 * Base class for general Gram matrices
 * A Gram matrix is the Hermitian matrix of inner probucts G_ik = <x_i, x_k>
 * Here, the  inner product is evaluated for all elements from vectors sets X1,
 * and X2.
 *
 * To be more precise, on exit the output buffer will store:
 * out[j+k*n1] = <x1_j, x2_k> where x1_j is the j-th vector from the x1 set
 * and x2_k is the k-th vector from the x2 set.
 */
template <typename math_t>
class GramMatrixBase {
  cublasHandle_t cublas_handle;

 public:
  GramMatrixBase(cublasHandle_t cublas_handle) : cublas_handle(cublas_handle){};

  virtual ~GramMatrixBase(){};

  /** Convenience function to evaluate the Gram matrix for two vector sets.
  *
  * @param [in] x1 device array of vectors in column major format,
  *  size [n1*n_cols]
  * @param [in] n1 number vectors in x1
  * @param [in] n_cols number of columns (features) in x1 and x2
  * @param [in] x2 device array of vectors in column major format,
  *   size [n2*n_cols]
  * @param [in] n2 number vectors in x2
  * @param [out] out device buffer to store the Gram matrix in column major
  *   format, size [n1*n2]
  * @param [in] stream cuda stream
  * @param ld1 leading dimension of x1 (usually it is n1)
  * @param ld2 leading dimension of x2 (usually it is n2)
  * @param ld_out leading dimension of out (usually it is n1)
  */
  virtual void operator()(const math_t *x1, int n1, int n_cols,
                          const math_t *x2, int n2, math_t *out,
                          cudaStream_t stream, int ld1 = 0, int ld2 = 0,
                          int ld_out = 0) {
    if (ld1 <= 0) {
      ld1 = n1;
    }
    if (ld2 <= 0) {
      ld2 = n2;
    }
    if (ld_out <= 0) {
      ld_out = n1;
    }
    evaluate(x1, n1, n_cols, x2, n2, out, stream, ld1, ld2, ld_out);
  }

  /** Evaluate the Gram matrix for two vector sets using simple dot product.
  *
  * @param [in] x1 device array of vectors in column major format,
  *  size [n1*n_cols]
  * @param [in] n1 number vectors in x1
  * @param [in] n_cols number of columns (features) in x1 and x2
  * @param [in] x2 device array of vectors in column major format,
  *   size [n2*n_cols]
  * @param [in] n2 number vectors in x2
  * @param [out] out device buffer to store the Gram matrix in column major
  *   format, size [n1*n2]
  * @param [in] stream cuda stream
  * @param ld1 leading dimension of x1 (usually it is n1)
  * @param ld2 leading dimension of x2 (usually it is n2)
  * @param ld_out leading dimension of out (usually it is n1)
  */
  virtual void evaluate(const math_t *x1, int n1, int n_cols, const math_t *x2,
                        int n2, math_t *out, cudaStream_t stream, int ld1,
                        int ld2, int ld_out) {
    linear(x1, n1, n_cols, x2, n2, out, stream, ld1, ld2, ld_out);
  }

  //private:
  // The following methods should be private, they are kept public to avoid:
  // "error: The enclosing parent function ("distance") for an extended
  // __device__ lambda cannot have private or protected access within its class"

  /** Calculates the Gram matrix using simple dot product between vector sets.
   *
   * Can be used as a building block for more complex kernel functions.
   *
   * @param [in] x1 device array of vectors in column major format,
   *  size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of colums (features) in x1 and x2
   * @param [in] x2 device array of vectors in column major format,
   *   size [n2*n_cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix in column major
   *   format, size [n1*n2]
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1 (usually it is n1)
   * @param ld2 leading dimension of x2 (usually it is n2)
   * @param ld_out leading dimension of out (usually it is n1)
   */
  void linear(const math_t *x1, int n1, int n_cols, const math_t *x2, int n2,
              math_t *out, cudaStream_t stream, int ld1, int ld2, int ld_out) {
    math_t alpha = 1.0;
    math_t beta = 0.0;
    CUBLAS_CHECK(LinAlg::cublasgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, n1,
                                    n2, n_cols, &alpha, x1, ld1, x2, ld2, &beta,
                                    out, ld_out, stream));
  }

  /** Calculates the Gram matrix using Euclidean distance.
   *
   * Can be used as a building block for more complex kernel functions.
   *
   * @param [in] x1 device array of vectors in column major format,
   *  size [n1*n_cols]
   * @param [in] n1 number vectors in x1
   * @param [in] n_cols number of columns (features) in x1 and x2
   * @param [in] x2 device array of vectors in column major format,
   *   size [n2*n_cols]
   * @param [in] n2 number vectors in x2
   * @param [out] out device buffer to store the Gram matrix in column major
   *   format, size [n1*n2]
   * @param [in] stream cuda stream
   * @param ld1 leading dimension of x1 (usually it is n1)
   * @param ld2 leading dimension of x2 (usually it is n2)
   * @param ld_out leading dimension of out (usually it is n1)
   */
  virtual void distance(const math_t *x1, int n1, int n_cols, const math_t *x2,
                        int n2, math_t *out, cudaStream_t stream, int ld1,
                        int ld2, int ld_out) {
    typedef cutlass::Shape<8, 128, 128> OutputTile_t;
    auto fin_op = [] __device__(math_t d_val, int idx) { return d_val; };
    Distance::distance<Distance::EucUnexpandedL2, math_t, math_t, math_t,
                       OutputTile_t>(x1, x2, out, n1, n2, n_cols, NULL, 0,
                                     fin_op, stream, false);
  }
};
};  // end namespace GramMatrix
};  // end namespace MLCommon
