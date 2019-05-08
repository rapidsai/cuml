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

#include <glm/qn/simple_mat.h>

namespace ML {
using MLCommon::Sparse::cusparseCsrmm2;

enum COMPRESSED_STORAGE_FORMAT { CSR = 0, CSC = 1 };

// non-allocating light-weight wrapper for compressed storage format sparse
// matrices used to dispatch gemm calls

template <typename T> struct CSMat {

  COMPRESSED_STORAGE_FORMAT format;

  SimpleVec<T> data;
  SimpleVec<int> idxPtr;
  SimpleVec<int> indices;

  cusparseMatDescr_t descr;

  int m, n, nnz;

  CSMat(const SimpleVec<T> &data, const SimpleVec<int> &idxPtr,
        const SimpleVec<int> &indices, const int m, const int n,
        COMPRESSED_STORAGE_FORMAT format = CSR)
      : m(m), n(n), nnz(data.len), data(data), idxPtr(idxPtr), indices(indices),
        format(format) {
    check();
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
  }

  CSMat(const CSMat &other)
      : m(other.m), n(other.n), nnz(other.nnz), format(other.format),
        descr(other.descr), data(other.data), idxPtr(other.idxPtr),
        indices(other.indices) {
    check();
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
  }

  ~CSMat() { CUSPARSE_CHECK(cusparseDestroyMatDescr(descr)); }

  void check() {

    ASSERT(nnz >= 0 && indices.len == nnz, "CSMat::check(): inconsistent nnz");
    ASSERT(indices.len == data.len,
           "CSMat::check(): data and indices must be both of lenght nnz");
    int dim = format == CSR ? m : n;
    ASSERT(idxPtr.len == dim + 1, "CSMat::check(): idxPtr invalid length");
  }
};

// Providing GEMM handle for B of type CSMat
template <typename T> struct Gemm<T, SimpleMat<T>, CSMat<T>, SimpleMat<T>> {
  static inline void gemm_(const cumlHandle_impl &handle, SimpleMat<T> &C,
                           const T alpha, const SimpleMat<T> &A,
                           const bool transA, const CSMat<T> &B,
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
           "cs_mat.h: Storage orders of dense matrices.");

    ASSERT(C.m == 1, "cs_mat.h: multiple outputs not yet supported.");
    if (B.format == CSR) {
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
                        B.data.data,   // csr stuff
                        B.idxPtr.data, // csr stuff
                        B.indices.data, // csr stuff
                        A.data,         // dense
                        ldb,            // ldb
                        &beta,          // beta=0
                        C.data,         // out data
                        ldc             // ldc flipped
                        ));
    } else {
      CSMat<T> Btmp(B);
      std::swap(Btmp.m, Btmp.n);
      Btmp.format = CSR;
      gemm_(handle, C, alpha, A, transA, Btmp, !transB, beta, stream);
    }
  }
};

// Providing GEMM handle for B of type CSMat
template <typename T> struct Gemm<T, CSMat<T>, SimpleMat<T>, SimpleMat<T>> {
  static inline void gemm_(const cumlHandle_impl &handle, SimpleMat<T> &C,
                           const T alpha, const CSMat<T> &A, const bool transA,
                           const SimpleMat<T> &B, const bool transB,
                           const T beta, cudaStream_t stream) {
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

    ASSERT(C.ord == COL_MAJOR, "cs_mat.h: Output storage order");

    if (A.format == CSR && B.ord == COL_MAJOR) {
      cusparseOperation_t opA = transA ? CUSPARSE_OPERATION_TRANSPOSE
                                       : CUSPARSE_OPERATION_NON_TRANSPOSE;

      cusparseOperation_t opB = transB ? CUSPARSE_OPERATION_TRANSPOSE
                                       : CUSPARSE_OPERATION_NON_TRANSPOSE;

      //printf("m=%d,n=%d,k=%d, nnz=%d, ldb=%d, ldc=%d transA=%d transB=%d "
      //       "invalid: %d\n",
      //       A.m, C.n, A.n, A.nnz, B.m, C.m, transA, transB,
      //       CUSPARSE_STATUS_INVALID_VALUE);

      CUSPARSE_CHECK(cusparseCsrmm2(handle.getcusparseHandle(),
                                    opA,            // transA
                                    opB,            // transB
                                    A.m,            // m: rows A
                                    C.n,            // n: cols op(B)/C
                                    A.n,            // k: cols A
                                    A.nnz,          //
                                    &alpha,         //
                                    A.descr,        // sparse mat descr
                                    A.data.data,    // csrValA
                                    A.idxPtr.data,  // csrRowPtrA
                                    A.indices.data, // csrColIndA
                                    B.data,         //
                                    B.m,            // ldb
                                    &beta,          //
                                    C.data,         //
                                    C.m             //
                                    ));
    } else if (A.format == CSC) {
      CSMat<T> Atmp(A);
      std::swap(Atmp.m, Atmp.n);
      Atmp.format = CSR;
      gemm_(handle, C, alpha, Atmp, !transA, B, transB, beta, stream);
    } else if (B.ord == ROW_MAJOR) {
      SimpleMat<T> Btmp(B.data, B.n, B.m, COL_MAJOR);
      gemm_(handle, C, alpha, A, transA, Btmp, !transB, beta, stream);
    } else {
      ASSERT(false, "cs_mat.h: Invalid config");
    }
  }
};

} // namespace ML
