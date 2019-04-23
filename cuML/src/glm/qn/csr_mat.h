#pragma once

#include <glm/qn/simple_mat.h>

namespace ML {

// non-allocating light-weight wrapper for Csr format used to dispatch
// gemm calls
template <typename T> struct CsrMat {
  SimpleVec<T> csrVal;
  SimpleVec<int> csrRowPtr;
  SimpleVec<int> csrColInd;

  cusparseMatDescr_t descr;

  int m, n, nnz;

  CsrMat(T *vals, int *rowPtr, int *colInd, int m, int n, int nnz)
      : m(m), n(n), csrVal(vals, nnz), csrRowPtr(rowPtr, m + 1),
        csrColInd(colInd, nnz), nnz(nnz) {
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr));
  }

  CsrMat(const CsrMat &other)
      : m(other.m), n(other.n), csrVal(other.csrVal.data, other.csrVal.len),
        csrRowPtr(other.csrRowPtr.data, other.csrRowPtr.len),
        csrColInd(other.csrColInd.data, other.csrColInd.len),
        descr(other.descr), nnz(other.nnz) {}
};

} // namespace ML
