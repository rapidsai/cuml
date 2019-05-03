/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include "linalg/matrix_vector_op.h"
#include "linalg/binary_op.h"
#include "linalg/unary_op.h"
#include "linalg/map_then_reduce.h"
#include "device_allocator.h"

namespace MLCommon {
namespace Matrix {

/**
 * @defgroup power math operation on the input matrix. Power of every element in
 * the input matrix
 * @param in: input matrix
 * @param out: output matrix. The result is stored in the out matrix
 * @param scalar: every element is multiplied with scalar.
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t>
void power(math_t *in, math_t *out, math_t scalar, int len, cudaStream_t stream) {
  auto d_src = in;
  auto d_dest = out;

  MLCommon::LinAlg::binaryOp(d_dest, d_src, d_src, len,
                             [=] __device__(math_t a, math_t b)
                             {return scalar * a * b;}, stream);

}

/**
 * @defgroup power math operation on the input matrix. Power of every element in
 * the input matrix
 * @param inout: input matrix and also the result is stored
 * @param scalar: every element is multiplied with scalar.
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t>
void power(math_t *inout, math_t scalar, int len, cudaStream_t stream) {
  power(inout, inout, scalar, len, stream);
}

/**
 * @defgroup overloaded power math operation on the input matrix. Power of every
 * element in the input matrix
 * @param inout: input matrix and also the result is stored
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t>
void power(math_t *inout, int len, cudaStream_t stream) {
  math_t scalar = 1.0;
  power(inout, scalar, len, stream);
}

/**
 * @defgroup power math operation on the input matrix. Power of every element in
 * the input matrix
 * @param in: input matrix
 * @param out: output matrix. The result is stored in the out matrix
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t>
void power(math_t *in, math_t *out, int len, cudaStream_t stream) {
  math_t scalar = 1.0;
  power(in, out, scalar, len, stream);
}


/**
 * @defgroup square root math operation on the input matrix. Square root of every element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param in: input matrix and also the result is stored
 * @param out: output matrix. The result is stored in the out matrix
 * @param scalar: every element is multiplied with scalar
 * @param len: number elements of input matrix
 * @{
 */
template<typename math_t, typename IdxType = int>
void seqRoot(math_t* in, math_t* out, math_t scalar, IdxType len, cudaStream_t stream,
              bool set_neg_zero = false) {

	auto d_src = in;
	auto d_dest = out;

  MLCommon::LinAlg::unaryOp(d_dest, d_src, len,
                            [=] __device__(math_t a)
                            {
                              if (set_neg_zero) {
                                if (a < math_t(0)) {
                                  return math_t(0);
                                } else {
                                  return sqrt(a * scalar);
                                }
                              } else {
                                return sqrt(a * scalar);
                              }
                            },
                            stream);
}

/**
 * @defgroup square root math operation on the input matrix. Square root of every element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param inout: input matrix and also the result is stored
 * @param scalar: every element is multiplied with scalar
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t, typename IdxType = int>
void seqRoot(math_t* inout, math_t scalar, IdxType len, cudaStream_t stream,
              bool set_neg_zero = false) {
  seqRoot(inout, inout, scalar, len, stream, set_neg_zero);
}


/**
 * @defgroup square root math operation on the input matrix. Square root of every element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param in: input matrix and also the result is stored
 * @param out: output matrix. The result is stored in the out matrix
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t, typename IdxType = int>
void seqRoot(math_t* in, math_t* out, IdxType len, cudaStream_t stream) {
	math_t scalar = 1.0;
	seqRoot(in, out, scalar, len, stream);
}

template <typename math_t, typename IdxType = int>
void seqRoot(math_t* inout, IdxType len, cudaStream_t stream) {
	math_t scalar = 1.0;
	seqRoot(inout, inout, scalar, len, stream);
}


template <typename math_t, typename IdxType = int>
void setSmallValuesZero(math_t* out, const math_t* in, IdxType len,
                         cudaStream_t stream, math_t thres = 1e-15) {
  MLCommon::LinAlg::unaryOp(out, in, len, [=] __device__(math_t a)
                                             {
                                               if(a <= thres && -a <= thres) {
                                                 return math_t(0);
                                               }
                                               else {
                                                 return a;
                                               }
                                             },
                                             stream);
}

/**
 * @defgroup sets the small values to zero based on a defined threshold
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param inout: input matrix and also the result is stored
 * @param len: number elements of input matrix
 * @param thres: threshold
 * @{
 */
template <typename math_t, typename IdxType = int>
void setSmallValuesZero(math_t* inout, IdxType len, cudaStream_t stream,
                         math_t thres = 1e-15) {
  setSmallValuesZero(inout, inout, len, stream, thres);
}



/**
 * @defgroup inverse math operation on the input matrix. Reciprocal of every
 * element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param in: input matrix and also the result is stored
 * @param out: output matrix. The result is stored in the out matrix
 * @param scalar: every element is multiplied with scalar
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t, typename IdxType = int>
void reciprocal(math_t *in, math_t *out, math_t scalar, int len,
                cudaStream_t stream, bool setzero = false, math_t thres = 1e-15) {
  auto d_src = in;
  auto d_dest = out;

  MLCommon::LinAlg::unaryOp(d_dest, d_src, len, [=]__device__(math_t a){
                                                  if (setzero) {
                                                    if (abs(a) <= thres) {
                                                      return math_t(0);
                                                    } else {
                                                      return scalar / a;
                                                    }
                                                  }
                                                  else {
                                                    return scalar / a;
                                                  }
                                                },
                                                stream);
}

/**
 * @defgroup inverse math operation on the input matrix. Reciprocal of every
 * element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param inout: input matrix and also the result is stored
 * @param scalar: every element is multiplied with scalar
 * @param len: number elements of input matrix
 * @param setzero: (default false) when true and |value|<thres, avoid dividing by (almost) zero
 * @param thres: Threshold to avoid dividing by zero (|value| < thres -> result = 0)
 * @{
 */
template <typename math_t, typename IdxType = int>
void reciprocal(math_t* inout, math_t scalar, IdxType len, cudaStream_t stream,
                 bool setzero = false, math_t thres = 1e-15) {
  reciprocal(inout, inout, scalar, len, stream, setzero, thres);
}


/**
 * @defgroup overloaded reciprocal math operation on the input matrix.
 * Reciprocal of every element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param inout: input matrix and also the result is stored
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t, typename IdxType = int>
void reciprocal(math_t *inout, IdxType len, cudaStream_t stream) {
  math_t scalar = 1.0;
  reciprocal(inout, scalar, len, stream);
}

/**
 * @defgroup inverse math operation on the input matrix. Reciprocal of every
 * element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param in: input matrix and also the result is stored
 * @param out: output matrix. The result is stored in the out matrix
 * @param len: number elements of input matrix
 * @{
 */
template <typename math_t, typename IdxType = int>
void reciprocal(math_t *in, math_t *out, IdxType len, cudaStream_t stream) {
  math_t scalar = 1.0;
  reciprocal(in, out, scalar, len, stream);
}

template <typename math_t>
void setValue(math_t* out, const math_t* in, math_t scalar, int len, cudaStream_t stream = 0) {
	MLCommon::LinAlg::unaryOp(out, in, len,
			          [scalar] __device__(math_t in) { return scalar; },
			          stream);
}

/**
 * @defgroup ratio math operation on the input matrix. ratio of every element
 * over sum of input vector is calculated
 *           Used in PCA.
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param src: input matrix
 * @param dest: output matrix. The result is stored in the dest matrix
 * @param len: number elements of input matrix
 * @{
 */

template <typename math_t, typename IdxType = int>
void ratio(math_t *src, math_t *dest, IdxType len,
           DeviceAllocator &mgr, cudaStream_t stream) {
  auto d_src = src;
  auto d_dest = dest;

  math_t* d_sum = (math_t*)mgr.alloc(sizeof(math_t));
  
  auto no_op = [] __device__(math_t in) { return in; };
  MLCommon::LinAlg::mapThenSumReduce(d_sum, len, no_op, stream, src);

  MLCommon::LinAlg::unaryOp(d_dest, d_src, len, [=] __device__(math_t a)
                                                { return a / (*d_sum); },
                                                stream);

  mgr.free(d_sum);
}

// Computes the argmax(d_in) column-wise in a DxN matrix
template <typename T, int TPB>
__global__ void argmaxKernel(const T *d_in, int D, int N, T *argmax) {
  typedef cub::BlockReduce<cub::KeyValuePair<int, T>, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // compute maxIndex=argMax (with abs()) index for column
  using KVP = cub::KeyValuePair<int, T>;
  int rowStart = blockIdx.x * D;
  KVP thread_data(0, 0);
  for (int i = threadIdx.x; i < D; i += TPB) {
    int idx = rowStart + i;
    thread_data = cub::ArgMax()(thread_data, KVP(idx, d_in[idx]));
  }
  auto maxKV = BlockReduce(temp_storage).Reduce(thread_data, cub::ArgMax());

  if (threadIdx.x == 0) {
    argmax[blockIdx.x] = d_in[maxKV.key];
  }
}
/**
 * @brief Argmax: find the row idx with maximum value for each column
 * @param in: input matrix
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param out: output vector of size n_cols
 */
template <typename math_t>
void argmax(const math_t *in, int n_rows, int n_cols, math_t *out,
              cudaStream_t stream) {
  int D = n_rows;
  int N = n_cols;
  auto data = in;
  if (D <= 32) {
    argmaxKernel<math_t, 32><<<N, 32, 0, stream>>>(data, D, N, out);
  } else if (D <= 64) {
    argmaxKernel<math_t, 64><<<N, 64, 0, stream>>>(data, D, N, out);
  } else if (D <= 128) {
    argmaxKernel<math_t, 128><<<N, 128, 0, stream>>>(data, D, N, out);
  } else {
    argmaxKernel<math_t, 256><<<N, 256, 0, stream>>>(data, D, N, out);
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

// Utility kernel needed for signFlip.
// Computes the argmax(abs(d_in)) column-wise in a DxN matrix followed by
// flipping the sign if the |max| value for each column is negative.
template <typename T, int TPB>
__global__ void signFlipKernel(T* d_in, int D, int N) {
  typedef cub::BlockReduce<cub::KeyValuePair<int, T>, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // compute maxIndex=argMax (with abs()) index for column
  using KVP = cub::KeyValuePair<int, T>;
  int rowStart = blockIdx.x * D;
  KVP thread_data(0,0);
  for(int i = threadIdx.x; i < D; i += TPB) {
    int idx = rowStart + i;
    thread_data = cub::ArgMax()(thread_data, KVP(idx, abs(d_in[idx])));
  }
  auto maxKV = BlockReduce(temp_storage).Reduce(thread_data, cub::ArgMax());

  // flip column sign if d_in[maxIndex] < 0
  __shared__ bool need_sign_flip;
  if(threadIdx.x == 0) {
    need_sign_flip = d_in[maxKV.key] < T(0);
  }
  __syncthreads();

  if(need_sign_flip) {
    for(int i = threadIdx.x; i < D; i += TPB) {
      int idx = rowStart + i;
      d_in[idx] = -d_in[idx];
    }    
  }
}

/**
 * @defgroup sign flip for PCA. This is used to stabilize the sign of column
 * major eigen vectors. Flips the sign if the column has negative |max|.
 * @param inout: input matrix. Result also stored in this parameter
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @{
 */
template <typename math_t>
void signFlip(math_t *inout, int n_rows, int n_cols, cudaStream_t stream) {
  int D = n_rows;
  int N = n_cols;
  auto data = inout;
  if (D <= 32) {
    signFlipKernel<math_t, 32><<<N, 32, 0, stream>>>(data, D, N);
  } else if(D <= 64) {
    signFlipKernel<math_t, 64><<<N, 64, 0, stream>>>(data, D, N);
  } else if(D <= 128) {
    signFlipKernel<math_t, 128><<<N, 128, 0, stream>>>(data, D, N);
  } else {
    signFlipKernel<math_t, 256><<<N, 256, 0, stream>>>(data, D, N);
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename Type, typename IdxType = int, int TPB = 256>
void matrixVectorBinaryMult(Type *data, const Type *vec, IdxType n_row,
                            IdxType n_col, bool rowMajor, bool bcastAlongRows,
                            cudaStream_t stream) {
    LinAlg::matrixVectorOp(data, data, vec, n_col, n_row, rowMajor, bcastAlongRows,
                 [] __device__(Type a, Type b) { return a * b; }, stream);
}

template <typename Type, typename IdxType = int, int TPB = 256>
void matrixVectorBinaryMultSkipZero(Type *data, const Type *vec, IdxType n_row,
                                    IdxType n_col, bool rowMajor,
                                    bool bcastAlongRows, cudaStream_t stream) {
    LinAlg::matrixVectorOp(data, data, vec, n_col, n_row, rowMajor, bcastAlongRows,
                 [] __device__(Type a, Type b) {
                   if (b == Type(0))
                     return a;
                   else
                     return a * b;
                 },
                 stream);
}

template <typename Type, typename IdxType = int, int TPB = 256>
void matrixVectorBinaryDiv(Type *data, const Type *vec, IdxType n_row,
                           IdxType n_col, bool rowMajor, bool bcastAlongRows,
                           cudaStream_t stream) {
    LinAlg::matrixVectorOp(data, data, vec, n_col, n_row, rowMajor, bcastAlongRows,
                 [] __device__(Type a, Type b) { return a / b; }, stream);
}

template <typename Type, typename IdxType = int, int TPB=256>
void matrixVectorBinaryDivSkipZero(Type* data, const Type* vec, IdxType n_row,
                                   IdxType n_col, bool rowMajor, bool bcastAlongRows,
                                   cudaStream_t stream, bool return_zero = false) {

	if (return_zero) {
            LinAlg::matrixVectorOp(data, data, vec, n_col, n_row, rowMajor, bcastAlongRows,
				        		[] __device__ (Type a, Type b) {
				                       if (myAbs(b) < Type(1e-10))
				                      	   return Type(0);
				                       else
                                                           return a / b;
                		    },
                        stream);
	} else {
	    LinAlg::matrixVectorOp(data, data, vec, n_col, n_row, rowMajor, bcastAlongRows,
		        		       [] __device__ (Type a, Type b) {
				                       if (myAbs(b) < Type(1e-10))
				                      	   return a;
				                       else
                                                           return a / b;
                        },
                        stream);
	}
}

template <typename Type, typename IdxType = int, int TPB = 256>
void matrixVectorBinaryAdd(Type *data, const Type *vec, IdxType n_row,
                           IdxType n_col, bool rowMajor, bool bcastAlongRows,
                           cudaStream_t stream) {
    LinAlg::matrixVectorOp(data, data, vec, n_col, n_row, rowMajor, bcastAlongRows,
                 [] __device__(Type a, Type b) { return a + b; }, stream);
}

template <typename Type, typename IdxType = int, int TPB = 256>
void matrixVectorBinarySub(Type *data, const Type *vec, IdxType n_row,
                           IdxType n_col, bool rowMajor, bool bcastAlongRows,
                           cudaStream_t stream) {
    LinAlg::matrixVectorOp(data, data, vec, n_col, n_row, rowMajor, bcastAlongRows,
                 [] __device__(Type a, Type b) { return a - b; }, stream);
}

}; // end namespace Matrix
}; // end namespace MLCommon
