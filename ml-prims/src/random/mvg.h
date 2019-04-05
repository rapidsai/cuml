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
#include <stdio.h>
#include <cmath>
#include "cuda_utils.h"
#include "curand_wrappers.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/cusolver_wrappers.h"
#include "linalg/matrix_vector_op.h"
#include "linalg/unary_op.h"

// mvg.h takes in matrices that are colomn major (as in fortan)
#define IDX2C(i, j, ld) (j * ld + i)

namespace MLCommon {
namespace Random {

enum Filler : unsigned char {
  LOWER, // = 0
  UPPER  // = 1
};       // used in memseting upper/lower matrix

/**
 * @brief Reset values within the epsilon absolute range to zero
 * @tparam T the data type
 * @param eig the array
 * @param epsilon the range
 * @param size length of the array
 */
template <typename T>
void epsilonToZero(T *eig, T epsilon, int size, cudaStream_t stream) {
  LinAlg::unaryOp(eig, eig, size, [epsilon] __device__(T in) {
    return (in < epsilon && in > -epsilon) ? T(0.0) : in;
  },
  stream);
}

/**
 * @brief Broadcast addition of vector onto a matrix
 * @tparam the data type
 * @param out the output matrix
 * @param in_m the input matrix
 * @param in_v the input vector
 * @param rows number of rows in the input matrix
 * @param cols number of cols in the input matrix
 */
template <typename T>
void matVecAdd(T *out, const T *in_m, const T *in_v, T scalar, int rows,
               int cols, cudaStream_t stream) {
  LinAlg::matrixVectorOp(
    out, in_m, in_v, cols, rows, true, true,
    [=] __device__(T mat, T vec) { return mat + scalar * vec; }, stream);
}

// helper kernels
template <typename T>
__global__ void combined_dot_product(int rows, int cols, const T *W, T *matrix,
                                     int *check) {
  int m_i = threadIdx.x + blockDim.x * blockIdx.x;
  int Wi = m_i / cols;
  if (m_i < cols * rows) {
    if (W[Wi] >= 0.0)
      matrix[m_i] = pow(W[Wi], 0.5) * (matrix[m_i]);
    else
      check[0] = Wi; // reports Wi'th eigen values is negative.
  }
}

template <typename T> // if uplo = 0, lower part of dim x dim matrix set to
// value
__global__ void fill_uplo(int dim, Filler uplo, T value, T *A) {
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < dim && j < dim) {
    // making off-diagonals == value
    if (i < j) {
      if (uplo == 1)
        A[IDX2C(i, j, dim)] = value;
    } else if (i > j) {
      if (uplo == 0)
        A[IDX2C(i, j, dim)] = value;
    }
  }
}


template <typename T>
class MultiVarGaussian {
public:
  enum Decomposer : unsigned char { chol_decomp, jacobi, qr };

private:
  // adjustable stuff
  const int dim;
  const int nPoints = 1;
  const double tol = 1.e-7;
  const T epsilon = 1.e-12;
  const int max_sweeps = 100;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  const Decomposer method;

  // not so much
  T *P = 0, *X = 0, *x = 0, *workspace_decomp = 0, *eig = 0;
  int *info, Lwork, info_h;
  syevjInfo_t syevj_params = NULL;
  curandGenerator_t gen;
  cublasHandle_t cublasHandle;
  cusolverDnHandle_t cusolverHandle;
  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  cudaStream_t cudaStream;
  bool deinitilized = false;

  size_t give_buffer_size() {
    // malloc workspace_decomp
    size_t granuality = 256, offset = 0;
    workspace_decomp = (T *)offset;
    offset += alignTo(sizeof(T) * Lwork, granuality);
    eig = (T *)offset;
    offset += alignTo(sizeof(T) * dim, granuality);
    info = (int *)offset;
    offset += alignTo(sizeof(int), granuality);
    return offset;
  }

public: // functions
  MultiVarGaussian() = delete;
  MultiVarGaussian(const int dim, Decomposer method)
    : dim(dim), method(method) {}

  size_t init(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH,
                cudaStream_t stream) {
    cublasHandle = cublasH;
    cusolverHandle = cusolverH;
    cudaStream = stream;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 28)); // SEED
    if (method == chol_decomp) {
      CUSOLVER_CHECK(LinAlg::cusolverDnpotrf_bufferSize(cusolverHandle, uplo,
                                                        dim, P, dim, &Lwork));
    } else if (method == jacobi) { // jacobi init
      CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));
      CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params, tol));
      CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps));
      CUSOLVER_CHECK(LinAlg::cusolverDnsyevj_bufferSize(
        cusolverHandle, jobz, uplo, dim, P, dim, eig, &Lwork, syevj_params));
    } else { // method == qr
      CUSOLVER_CHECK(LinAlg::cusolverDnsyevd_bufferSize(
        cusolverHandle, jobz, uplo, dim, P, dim, eig, &Lwork));
    }
    return give_buffer_size();
  }

  void set_workspace(T *workarea) {
    workspace_decomp = (T *)((size_t)workspace_decomp + (size_t)workarea);
    eig = (T *)((size_t)eig + (size_t)workarea);
    info = (int *)((size_t)info + (size_t)workarea);
  }

  void give_gaussian(const int nPoints, T *P, T *X, const T *x = 0) {
    if (method == chol_decomp) {
      // lower part will contains chol_decomp
      CUSOLVER_CHECK(LinAlg::cusolverDnpotrf(cusolverHandle, uplo, dim, P, dim,
                                             workspace_decomp, Lwork, info, cudaStream));
    } else if (method == jacobi) {
      CUSOLVER_CHECK(LinAlg::cusolverDnsyevj(
        cusolverHandle, jobz, uplo, dim, P, dim, eig, workspace_decomp, Lwork,
        info, syevj_params, cudaStream)); // vectors stored as cols. & col major
    } else {                  // qr
      CUSOLVER_CHECK(LinAlg::cusolverDnsyevd(cusolverHandle, jobz, uplo, dim, P,
                                             dim, eig, workspace_decomp, Lwork,
                                             info, cudaStream));
    }
    updateHost(&info_h, info, 1);
    ASSERT(info_h == 0, "mvg: error in syevj/syevd/potrf, info=%d | expected=0",
           info_h);
    T mean = 0.0, stddv = 1.0;
    // generate nxN gaussian nums in X
    CURAND_CHECK(curandGenerateNormal(
      gen, X, (nPoints * dim) + (nPoints * dim) % 2, mean, stddv));
    T alfa = 1.0, beta = 0.0;
    if (method == chol_decomp) {
      // upper part (0) being filled with 0.0
      dim3 block(32, 32);
      dim3 grid(ceildiv(dim, (int)block.x), ceildiv(dim, (int)block.y));
      fill_uplo<T><<<grid, block>>>(dim, UPPER, (T)0.0, P);
      CUDA_CHECK(cudaPeekAtLastError());

      // P is lower triangular chol decomp mtrx
      CUBLAS_CHECK(LinAlg::cublasgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                      dim, nPoints, dim, &alfa, P, dim, X, dim,
                                      &beta, X, dim, cudaStream));
    } else {
      epsilonToZero(eig, epsilon, dim, cudaStream);
      dim3 block(64);
      dim3 grid(ceildiv(dim, (int)block.x));
      CUDA_CHECK(cudaMemset(info, 0, sizeof(int)));
      grid.x = ceildiv(dim * dim, (int)block.x);
      combined_dot_product<T><<<grid, block>>>(dim, dim, eig, P, info);
      CUDA_CHECK(cudaPeekAtLastError());

      // checking if any eigen vals were negative
      updateHost(&info_h, info, 1);
      ASSERT(info_h == 0, "mvg: Cov matrix has %dth Eigenval negative", info_h);

      // Got Q = eigvect*eigvals.sqrt in P, Q*X in X below
      CUBLAS_CHECK(LinAlg::cublasgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                      dim, nPoints, dim, &alfa, P, dim, X, dim,
                                      &beta, X, dim, cudaStream));
    }
    // working to make mean not 0
    // since we are working with column-major, nPoints and dim are swapped
    if (x != NULL)
      matVecAdd(X, X, x, T(1.0), nPoints, dim, cudaStream);
  }

  void deinit() {
    if (deinitilized)
      return;
    CURAND_CHECK(curandDestroyGenerator(gen));
    CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params));
    deinitilized = true;
  }

  ~MultiVarGaussian() { deinit(); }
}; // end of MultiVarGaussian

}; // end of namespace Random
}; // end of namespace MLCommon
