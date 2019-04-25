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
#include <iostream>
#include <vector>

#include <common/cumlHandle.hpp>
#include <common/device_buffer.hpp>
#include <cuda_utils.h>
#include <glm/qn/csr_mat.h>
#include <linalg/binary_op.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/map_then_reduce.h>
#include <linalg/norm.h>
#include <linalg/ternary_op.h>
#include <linalg/unary_op.h>
#include <sparse/cusparse_wrappers.h>

namespace ML {
using MLCommon::Sparse::cusparseCsrmm;
using MLCommon::Sparse::cusparseGemmi;

template <typename T> struct SimpleMat;
template <typename T> struct SimpleVec;
template <typename T> struct CsrMat;

enum STORAGE_ORDER { COL_MAJOR = 0, ROW_MAJOR = 1 };

// dispatch C = alpha * op(A) * op(B) + beta * C
template <typename T, class MatA, class MatB, class MatC> struct Gemm {
  static inline void gemm_() {
    ASSERT(false, "simple_mat.h: Combination of matrix types not implemented.");
  }
};
template <typename T> struct Gemm<T, SimpleMat<T>, SimpleMat<T>, SimpleMat<T>> {
  static inline void gemm_(const cumlHandle_impl &handle, SimpleMat<T> &C,
                           const T alpha, const SimpleMat<T> &A,
                           const bool transA, const SimpleMat<T> &B,
                           const bool transB, const T beta,
                           cudaStream_t stream) {

    int kA = A.n;
    int kB = B.m;

    if (transA) {
      ASSERT(A.n == C.m, "GEMM invalid dims: m");
      kA = A.m;
    } else {
      ASSERT(A.m == C.m, "GEMM invalid dims: m");
    }

    if (transB) {
      ASSERT(B.m == C.n, "GEMM invalid dims: n");
      kB = B.n;
    } else {
      ASSERT(B.n == C.n, "GEMM invalid dims: n");
    }
    ASSERT(kA == kB, "GEMM invalid dims: k");

    if (C.ord == COL_MAJOR && A.ord == COL_MAJOR &&
        B.ord == COL_MAJOR) {                                // base case
      MLCommon::LinAlg::cublasgemm(handle.getCublasHandle(), // handle
                                   transA ? CUBLAS_OP_T : CUBLAS_OP_N, // transA
                                   transB ? CUBLAS_OP_T : CUBLAS_OP_N, // transB
                                   C.m, C.n, kA, // dimensions m,n,k
                                   &alpha, A.data,
                                   A.m,         // lda
                                   B.data, B.m, // ldb
                                   &beta, C.data,
                                   C.m, // ldc,
                                   stream);
      return;
    }
    if (A.ord == ROW_MAJOR) {
      SimpleMat<T> Acm(A.data, A.n, A.m, COL_MAJOR);
      gemm_(handle, C, alpha, Acm, !transA, B, transB, beta, stream);
      return;
    }
    if (B.ord == ROW_MAJOR) {
      SimpleMat<T> Bcm(B.data, B.n, B.m, COL_MAJOR);
      gemm_(handle, C, alpha, A, transA, Bcm, !transB, beta, stream);
      return;
    }
    if (C.ord == ROW_MAJOR) {
      SimpleMat<T> Ccm(C.data, C.n, C.m, COL_MAJOR);
      gemm_(handle, Ccm, alpha, B, !transB, A, !transA, beta, stream);
      return;
    }
  }
};

template <typename T> struct Gemm<T, SimpleMat<T>, SimpleVec<T>, SimpleMat<T>> {
  typedef SimpleMat<T> Mat;
  static inline void gemm_(const cumlHandle_impl &handle, Mat &C, const T alpha,
                           const Mat &A, const bool transA,
                           const SimpleVec<T> &B, const bool transB,
                           const T beta, cudaStream_t stream) {
    Gemm<T, Mat, Mat, Mat>::gemm_(handle, C, alpha, A, transA, B, transB, beta,
                                  stream);
  }
};

template <typename T> struct Gemm<T, SimpleMat<T>, CsrMat<T>, SimpleMat<T>> {
  // we implement only two cases, essential to running QN GLM:
  // Case 1: C_cm = alpha * A_cm * B_csr' + beta * C_cm
  // - we implement it by reinterpreting B_csr' as B_csc and using
  // cusparseGemmi
  //
  // Case 2: C_cm = alpha * A_cm * B_csr
  // - we implement it as C_cm = ( alpha * B_csr' * A_cm' )' using
  // cusparseCsrmm
  //
  // TODO constraints: beta = 0, number of outputs C_cm.m == 1
  // Therefore, we dont need to transpose C before and after the multiplication
  // If we wanted to support these cases, we would need dynamic allocs here.
  // Once we have cuml poolig allocators, this might be viable.
  // Passing in workspace instead for this case would make the API semantics
  // awkward
  static inline void gemm_(const cumlHandle_impl &handle, SimpleMat<T> &C,
                           const T alpha, const SimpleMat<T> &A,
                           const bool transA, const CsrMat<T> &B,
                           const bool transB, const T beta,
                           cudaStream_t stream) {
    int kA = A.n;
    int kB = B.m;

    if (transA) {
      ASSERT(A.n == C.m, "GEMM invalid dims: m");
      kA = A.m;
    } else {
      ASSERT(A.m == C.m, "GEMM invalid dims: m");
    }

    if (transB) {
      ASSERT(B.m == C.n, "GEMM invalid dims: n");
      kB = B.n;
    } else {
      ASSERT(B.n == C.n, "GEMM invalid dims: n");
    }
    ASSERT(kA == kB, "GEMM invalid dims: k");

    ASSERT(C.ord == COL_MAJOR && A.ord == COL_MAJOR,
           "simple_mat.h: Storage orders of dense matrices.");

    ASSERT(C.m == 1, "simple_mat.h: multiple outputs not yet supported.");
    // Check that we are either in case 1 or 2
    // if (!transA && transB) { // case 1
    //    printf("m=%d,n=%d,k=%d, nnz=%d, lda=%d, ldc=%d\n",
    //            C.m,C.n,A.n, B.nnz, A.m, C.m
    //            );
    //  CUSPARSE_CHECK(cusparseGemmi(cuml.getcusparseHandle(),
    //                               C.m, // m = C = W.m
    //                               C.n, // n = N = X.m
    //                               A.n, // k = X.n = D
    //                               B.nnz, &alpha, A.data,
    //                               A.m, // lda = C
    //                               B.csrVal.data, B.csrRowPtr.data,
    //                               B.csrColInd.data, &beta, C.data,
    //                               C.m // ldc = C
    //                               ));

    //} else
    { // case 2

      // if beta != 0, we would also have to transpose C first
      // ASSERT(beta == 0, "simple_mat.h: requested configuration not
      // implemented.");

      // TODO If C > 1, we would need to transpose the output of the cusparse
      // call  However, here we do not have the necessary scratch space  We
      // could allocate it using the cuml handle and rely on RMM's mempool  or
      // pass in the scratch space explicitely (which is bad for the API)
      // inverting transposes!
      cusparseOperation_t opB = transB ? CUSPARSE_OPERATION_NON_TRANSPOSE
                                       : CUSPARSE_OPERATION_TRANSPOSE;

      // if B is transposed, it will not be transposed in this formulation
      int ldc = transB ? B.m : B.n;
      int ldb = A.n;

      // printf("m=%d,n=%d,k=%d, nnz=%d, lda=%d, ldc=%d\n", B.m,1,B.n, B.nnz,
      // ldb, ldc);

      CUSPARSE_CHECK(
          cusparseCsrmm(handle.getcusparseHandle(),
                        opB,           // B, the sparse matrix is A
                        B.m,           // flip for computing the transpose
                        1,             // A.m
                        B.n,           // number of columns of the sparse matrix
                        B.nnz,         // nnz
                        &alpha,        // factor
                        B.descr,       // no structure
                        B.csrVal.data, // csr stuff
                        B.csrRowPtr.data, // csr stuff
                        B.csrColInd.data, // csr stuff
                        A.data,           // dense
                        ldb,              // ldb
                        &beta,            // beta=0
                        C.data,           // out data
                        ldc               // ldc flipped
                        ));
    }
  }
}; // namespace ML

template <typename T> struct SimpleMat {
  int m, n;
  T *data;
  int len;

  STORAGE_ORDER ord; // storage order: runtime param for compile time sake

  SimpleMat(STORAGE_ORDER order = COL_MAJOR)
      : data(nullptr), len(0), ord(order) {}

  SimpleMat(T *data, int m, int n, STORAGE_ORDER order = COL_MAJOR)
      : data(data), len(m * n), m(m), n(n), ord(order) {}

  SimpleMat(const SimpleMat<T> &other)
      : data(other.data), m(other.m), n(other.n), len(other.len),
        ord(other.ord) {}

  void reset(T *data_, int m_, int n_) {
    m = m_;
    n = n_;
    data = data_;
    len = m * n;
  }

  void print() const { std::cout << (*this) << std::endl; }

  template <typename MatB>
  inline void assign_gemm(const cumlHandle_impl &handle, const T alpha,
                          const SimpleMat<T> &A, const bool transA,
                          const MatB &B, const bool transB, const T beta,
                          cudaStream_t stream) {

    Gemm<T, SimpleMat<T>, MatB, SimpleMat<T>>::gemm_(
        handle, *this, alpha, A, transA, B, transB, beta, stream);
  }

  // this = a*x
  inline void ax(const T a, const SimpleMat<T> &x, cudaStream_t stream) {
    ASSERT(ord == x.ord, "SimpleMat::ax: Storage orders must match");

    auto scale = [a] __device__(const T x) { return a * x; };
    MLCommon::LinAlg::unaryOp(data, x.data, len, scale, stream);
  }

  // this = a*x + y
  inline void axpy(const T a, const SimpleMat<T> &x, const SimpleMat<T> &y,
                   cudaStream_t stream) {
    ASSERT(ord == x.ord, "SimpleMat::axpy: Storage orders must match");
    ASSERT(ord == y.ord, "SimpleMat::axpy: Storage orders must match");

    auto axpy = [a] __device__(const T x, const T y) { return a * x + y; };
    MLCommon::LinAlg::binaryOp(data, x.data, y.data, len, axpy, stream);
  }

  template <typename Lambda>
  inline void assign_unary(const SimpleMat<T> &other, Lambda &f,
                           cudaStream_t stream) {
    ASSERT(ord == other.ord,
           "SimpleMat::assign_unary: Storage orders must match");

    MLCommon::LinAlg::unaryOp(data, other.data, len, f, stream);
  }

  template <typename Lambda>
  inline void assign_binary(const SimpleMat<T> &other1,
                            const SimpleMat<T> &other2, Lambda &f,
                            cudaStream_t stream) {

    ASSERT(ord == other1.ord,
           "SimpleMat::assign_binary: Storage orders must match");
    ASSERT(ord == other2.ord,
           "SimpleMat::assign_binary: Storage orders must match");

    MLCommon::LinAlg::binaryOp(data, other1.data, other2.data, len, f, stream);
  }

  template <typename Lambda>
  inline void
  assign_ternary(const SimpleMat<T> &other1, const SimpleMat<T> &other2,
                 const SimpleMat<T> &other3, Lambda &f, cudaStream_t stream) {
    ASSERT(ord == other1.ord,
           "SimpleMat::assign_ternary: Storage orders must match");
    ASSERT(ord == other2.ord,
           "SimpleMat::assign_ternary: Storage orders must match");
    ASSERT(ord == other3.ord,
           "SimpleMat::assign_ternary: Storage orders must match");

    MLCommon::LinAlg::ternaryOp(data, other1.data, other2.data, other3.data,
                                len, f, stream);
  }

  inline void fill(const T val, cudaStream_t stream) {
    // TODO this reads data unnecessary, though it's mostly used for testing
    auto f = [val] __device__(const T x) { return val; };
    MLCommon::LinAlg::unaryOp(data, data, len, f, stream);
  }

  inline void copy(const SimpleMat<T> &other, cudaStream_t stream) {
    ASSERT((ord == other.ord) && (m == other.m) && (n == other.n),
           "SimpleMat::copy: matrices not compatible");

    CUDA_CHECK(cudaMemcpyAsync(data, other.data, len * sizeof(T),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void operator=(const SimpleMat<T> &other) = delete;
};

template <typename T> struct SimpleVec : SimpleMat<T> {
  typedef SimpleMat<T> Super;

  SimpleVec(T *data, const int n) : Super(data, n, 1, COL_MAJOR) {}
  // this = alpha * A * x + beta * this
  void assign_gemv(const cumlHandle_impl &handle, const T alpha,
                   const SimpleMat<T> &A, bool transA, const SimpleVec<T> &x,
                   const T beta, cudaStream_t stream) {
    Super::assign_gemm(handle, alpha, A, transA, x, false, beta, stream);
  }

  SimpleVec() : Super(COL_MAJOR) {}

  inline void reset(T *new_data, int n) { Super::reset(new_data, n, 1); }
};

template <typename T>
inline void col_ref(const SimpleMat<T> &mat, SimpleVec<T> &mask_vec, int c) {
  ASSERT(mat.ord == COL_MAJOR, "col_ref only available for column major mats");
  T *tmp = &mat.data[mat.m * c];
  mask_vec.reset(tmp, mat.m);
}

template <typename T>
inline void col_slice(const SimpleMat<T> &mat, SimpleMat<T> &mask_mat,
                      int c_from, int c_to) {
  ASSERT(c_from >= 0 && c_from < mat.n, "col_slice: invalid from");
  ASSERT(c_to >= 0 && c_to <= mat.n, "col_slice: invalid to");

  ASSERT(mat.ord == COL_MAJOR, "col_ref only available for column major mats");
  ASSERT(mask_mat.ord == COL_MAJOR,
         "col_ref only available for column major mask");
  T *tmp = &mat.data[mat.m * c_from];
  mask_mat.reset(tmp, mat.m, c_to - c_from);
}

// Reductions such as dot or norm require an additional location in dev mem
// to hold the result. We don't want to deal with this in the SimpleVec class
// as it  impedes thread safety and constness

template <typename T>
inline T dot(const SimpleVec<T> &u, const SimpleVec<T> &v, T *tmp_dev,
             cudaStream_t stream) {
  auto f = [] __device__(const T x, const T y) { return x * y; };
  MLCommon::LinAlg::mapThenSumReduce(tmp_dev, u.len, f, stream, u.data, v.data);
  T tmp_host;
  MLCommon::updateHostAsync(&tmp_host, tmp_dev, 1, stream);
  cudaStreamSynchronize(stream);
  return tmp_host;
}

template <typename T>
inline T squaredNorm(const SimpleVec<T> &u, T *tmp_dev, cudaStream_t stream) {
  return dot(u, u, tmp_dev, stream);
}

template <typename T>
inline T nrm2(const SimpleVec<T> &u, T *tmp_dev, cudaStream_t stream) {
  return MLCommon::mySqrt<T>(squaredNorm(u, tmp_dev, stream));
}

template <typename T>
inline T nrm1(const SimpleVec<T> &u, T *tmp_dev, cudaStream_t stream) {
  MLCommon::LinAlg::rowNorm(tmp_dev, u.data, u.len, 1, MLCommon::LinAlg::L1Norm,
                            true, stream, MLCommon::Nop<T>());
  T tmp_host;
  MLCommon::updateHostAsync(&tmp_host, tmp_dev, 1, stream);
  cudaStreamSynchronize(stream);
  return tmp_host;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const SimpleVec<T> &v) {
  std::vector<T> out(v.len);
  MLCommon::updateHost(&out[0], v.data, v.len);
  int it = 0;
  for (; it < v.len - 1;) {
    os << out[it] << " ";
    it++;
  }
  os << out[it];
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const SimpleMat<T> &mat) {
  os << "ord=" << (mat.ord == COL_MAJOR ? "CM" : "RM") << "\n";
  std::vector<T> out(mat.len);
  MLCommon::updateHost(&out[0], mat.data, mat.len);
  if (mat.ord == COL_MAJOR) {
    for (int r = 0; r < mat.m; r++) {
      int idx = r;
      for (int c = 0; c < mat.n - 1; c++) {
        os << out[idx] << ",";
        idx += mat.m;
      }
      os << out[idx] << std::endl;
    }
  } else {
    for (int c = 0; c < mat.m; c++) {
      int idx = c * mat.n;
      for (int r = 0; r < mat.n - 1; r++) {
        os << out[idx] << ",";
        idx += 1;
      }
      os << out[idx] << std::endl;
    }
  }

  return os;
}

template <typename T> struct SimpleVecOwning : SimpleVec<T> {
  typedef SimpleVec<T> Super;
  typedef MLCommon::device_buffer<T> Buffer;
  std::unique_ptr<Buffer> buf;

  SimpleVecOwning() : Super() {}

  SimpleVecOwning(const cumlHandle_impl &handle, int n, cudaStream_t stream)
      : Super() {
    reset(handle, n, stream);
  }

  void reset(const cumlHandle_impl &handle, int n, cudaStream_t stream) {
    buf.reset(new Buffer(handle.getDeviceAllocator(), stream, n));
    Super::reset(buf->data(), n);
  }

  void operator=(const SimpleVec<T> &other) = delete;

  SimpleVecOwning(const SimpleMat<T> &other) = delete;
};

template <typename T> struct SimpleMatOwning : SimpleMat<T> {
  typedef SimpleMat<T> Super;
  typedef MLCommon::device_buffer<T> Buffer;
  std::unique_ptr<Buffer> buf;
  using Super::m;
  using Super::n;
  using Super::ord;

  SimpleMatOwning(STORAGE_ORDER order = COL_MAJOR) : Super(order) {}

  SimpleMatOwning(const cumlHandle_impl &handle, int m, int n,
                  cudaStream_t stream, STORAGE_ORDER order = COL_MAJOR)
      : Super(order) {
    reset(handle, m, n, stream);
  }

  void reset(const cumlHandle_impl &handle, int m, int n, cudaStream_t stream) {
    buf.reset(new Buffer(handle.getDeviceAllocator(), stream, m * n));
    Super::reset(buf->data(), m, n);
  }

  void operator=(const SimpleVec<T> &other) = delete;
  SimpleMatOwning(const SimpleMat<T> &other) = delete;
};

}; // namespace ML
