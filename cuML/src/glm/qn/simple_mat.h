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
#include <linalg/binary_op.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/map_then_reduce.h>
#include <linalg/norm.h>
#include <linalg/ternary_op.h>
#include <linalg/unary_op.h>

namespace ML {

enum STORAGE_ORDER { COL_MAJOR = 0, ROW_MAJOR = 1 };

template <typename T> struct SimpleMat {
  int m, n;
  T *data;
  int len;

  STORAGE_ORDER ord; // storage order: runtime param for compile time sake

  SimpleMat(STORAGE_ORDER order = COL_MAJOR)
      : data(nullptr), len(0), ord(order) {}

  SimpleMat(T *data, int m, int n, STORAGE_ORDER order = COL_MAJOR)
      : data(data), len(m * n), m(m), n(n), ord(order) {}

  void reset(T *data_, int m_, int n_) {
    m = m_;
    n = n_;
    data = data_;
    len = m * n;
  }

  void print() const { std::cout << (*this) << std::endl; }

  inline void assign_gemm(const cumlHandle_impl &handle, const T alpha,
                          const SimpleMat<T> &A, const bool transA,
                          const SimpleMat<T> &B, const bool transB,
                          const T beta, cudaStream_t stream) {

    int kA = A.n;
    int kB = B.m;

    if (transA) {
      ASSERT(A.n == this->m, "GEMM invalid dims: m");
      kA = A.m;
    } else {
      ASSERT(A.m == this->m, "GEMM invalid dims: m");
    }

    if (transB) {
      ASSERT(B.m == this->n, "GEMM invalid dims: n");
      kB = B.n;
    } else {
      ASSERT(B.n == this->n, "GEMM invalid dims: n");
    }
    ASSERT(kA == kB, "GEMM invalid dims: k");

    if (ord == COL_MAJOR && A.ord == COL_MAJOR &&
        B.ord == COL_MAJOR) {                                // base case
      MLCommon::LinAlg::cublasgemm(handle.getCublasHandle(), // handle
                                   transA ? CUBLAS_OP_T : CUBLAS_OP_N, // transA
                                   transB ? CUBLAS_OP_T : CUBLAS_OP_N, // transB
                                   this->m, this->n, kA, // dimensions m,n,k
                                   &alpha, A.data,
                                   A.m,         // lda
                                   B.data, B.m, // ldb
                                   &beta, this->data,
                                   this->m, // ldc,
                                   stream);
      return;
    }
    if (A.ord == ROW_MAJOR) {
      SimpleMat<T> Acm(A.data, A.n, A.m, COL_MAJOR);
      assign_gemm(handle, alpha, Acm, !transA, B, transB, beta, stream);
      return;
    }
    if (B.ord == ROW_MAJOR) {
      SimpleMat<T> Bcm(B.data, B.n, B.m, COL_MAJOR);
      assign_gemm(handle, alpha, A, transA, Bcm, !transB, beta, stream);
      return;
    }
    if (ord == ROW_MAJOR) {
      SimpleMat<T> Ccm(this->data, n, m, COL_MAJOR);
      Ccm.assign_gemm(handle, alpha, B, !transB, A, !transA, beta, stream);
      return;
    }
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

  inline void copy_async(const SimpleMat<T> &other, cudaStream_t stream) {
    ASSERT((ord == other.ord) && (m == other.m) && (n == other.n),
           "SimpleMat::copy: matrices not compatible");

    CUDA_CHECK(cudaMemcpyAsync(data, other.data, len * sizeof(T),
                               cudaMemcpyDeviceToDevice, stream));
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

  SimpleVecOwning(std::shared_ptr<deviceAllocator> allocator, int n, cudaStream_t stream)
      : Super() {
    reset(allocator, n, stream);
  }

  void reset(std::shared_ptr<deviceAllocator> allocator, int n, cudaStream_t stream) {
    buf.reset(new Buffer(allocator, stream, n));
    Super::reset(buf->data(), n);
  }

  void operator=(const SimpleVec<T> &other) = delete;
};

template <typename T> struct SimpleMatOwning : SimpleMat<T> {
  typedef SimpleMat<T> Super;
  typedef MLCommon::device_buffer<T> Buffer;
  std::unique_ptr<Buffer> buf;
  using Super::m;
  using Super::n;
  using Super::ord;

  SimpleMatOwning(STORAGE_ORDER order = COL_MAJOR) : Super(order) {}

  SimpleMatOwning(std::shared_ptr<deviceAllocator> allocator, int m, int n,
                  cudaStream_t stream, STORAGE_ORDER order = COL_MAJOR)
      : Super(order) {
    reset(allocator, m, n, stream);
  }

  void reset(std::shared_ptr<deviceAllocator> allocator, int m, int n, cudaStream_t stream) {
    buf.reset(new Buffer(allocator, stream, m * n));
    Super::reset(buf->data(), m, n);
  }

  void operator=(const SimpleVec<T> &other) = delete;
};

}; // namespace ML
