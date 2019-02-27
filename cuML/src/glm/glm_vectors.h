#pragma once
#include <iostream>
#include <vector>

#include <cuda_utils.h>
#include <linalg/binary_op.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/map_then_reduce.h>
#include <linalg/ternary_op.h>
#include <linalg/unary_op.h>

#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace ML {

enum STORAGE_ORDER { COL_MAJOR = 0, ROW_MAJOR = 1 };

template <typename T> struct SimpleMat;

template <typename T> struct SimpleVec {
  typedef thrust::device_ptr<T> Ptr;

  T *data;
  int len;
  bool free_data; // I am the owner of the data
  Ptr p;

  SimpleVec() : data(nullptr), len(0), free_data(false) {}

  SimpleVec(T *data, int len) : SimpleVec() { reset(data, len); }

  SimpleVec(int len, const T val = 0) : SimpleVec() {
    reset(len);
    fill(val);
  }

  virtual ~SimpleVec() {
    if (free_data) {
      CUDA_CHECK(cudaFree(data));
    }
  }

  inline void reset(int n) {
    if (free_data)
      CUDA_CHECK(cudaFree(data));

    len = n;
    MLCommon::allocate(data, len);
    free_data = true;
    p = thrust::device_pointer_cast(data);
  }

  inline void reset(T *new_data, int n) {
    if (free_data)
      CUDA_CHECK(cudaFree(data));

    free_data = false;
    len = n;
    data = new_data;
    p = thrust::device_pointer_cast(data);
  }

  inline const T *ptr() const { return data; }

  inline T *ptr() { return data; }

  inline int size() const { return len; }

  inline void fill(const T val) {
    auto f = [val] __device__(const T x) { return val; };
    MLCommon::LinAlg::unaryOp(data, data, len, f);
  }

  inline void operator=(const T val) { fill(val); }

  inline void operator=(const SimpleVec<T> &other) {
    CUDA_CHECK(cudaMemcpy(data, other.data, len * sizeof(T),
                          cudaMemcpyDeviceToDevice));
  }

  inline T operator[](int pos) const {
    T tmp;
    MLCommon::updateHost(&tmp, &data[pos], 1);
    return tmp;
  }

  inline void assign_pos(int pos, T val) { updateDevice(&data[pos], &val, 1); }

  // this = a*x
  inline void ax(const T a, const SimpleVec<T> &x) {
    auto scale = [a] __device__(const T x) { return a * x; };
    MLCommon::LinAlg::unaryOp(data, x.ptr(), len, scale);
  }

  // this = a*x + y
  inline void axpy(const T a, const SimpleVec<T> &x, const SimpleVec<T> &y) {
    auto axpy = [a] __device__(const T x, const T y) { return a * x + y; };
    MLCommon::LinAlg::binaryOp(data, x.data, y.data, len, axpy);
  }

  // this = a*x + b*y
  inline void axpby(const T a, const SimpleVec<T> &x, const T b,
                    const SimpleVec<T> &y) {
    auto axpby = [a, b] __device__(const T x, const T y) {
      return a * x + b * y;
    };
    MLCommon::LinAlg::binaryOp(data, x.data, y.data, len, axpby);
  }

  template <typename Lambda>
  inline void assign_unary(const SimpleVec<T> &other, Lambda &f) {
    MLCommon::LinAlg::unaryOp(data, other.data, len, f);
  }

  template <typename Lambda>
  inline void assign_binary(const SimpleVec<T> &other1,
                            const SimpleVec<T> &other2, Lambda &f) {
    MLCommon::LinAlg::binaryOp(data, other1.data, other2.data, len, f);
  }

  template <typename Lambda>
  inline void assign_ternary(const SimpleVec<T> &other1,
                             const SimpleVec<T> &other2,
                             const SimpleVec<T> &other3, Lambda &f) {
    MLCommon::LinAlg::ternaryOp(data, other1.data, other2.data, other3.data,
                                len, f);
  }

  template <typename Op, typename... Vectors>
  inline void assign_k_ary(Op &op, const Vectors &... args) {
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(args.p...));

    auto end =
        thrust::make_zip_iterator(thrust::make_tuple((args.p + args.len)...));
    thrust::transform(begin, end, p, op);
  }

  inline void print() const { std::cout << (*this) << std::endl; }

  inline void assign_gemv(const T alpha, const SimpleMat<T> &A,
                          const SimpleVec<T> &x, const T beta,
                          cublasHandle_t &cublas) {
    // this <- alpha * A * x + beta * this
    if (A.ord == COL_MAJOR) {

      MLCommon::LinAlg::cublasgemv(cublas, CUBLAS_OP_N, A.m, A.n, &alpha,
                                   A.data, A.m, x.data, 1, &beta, this->data,
                                   1);
    } else {

      MLCommon::LinAlg::cublasgemv(cublas, CUBLAS_OP_T, A.n, A.m, &alpha,
                                   A.data, A.n, x.data, 1, &beta, this->data,
                                   1);
    }
  }

  inline void assign_gemvT(const T alpha, const SimpleMat<T> &A,
                           const SimpleVec<T> &x, const T beta,
                           cublasHandle_t &cublas) {
    // this <- alpha * A * x + beta * this
    if (A.ord == COL_MAJOR) {
      MLCommon::LinAlg::cublasgemv(cublas, CUBLAS_OP_T, A.m, A.n, &alpha,
                                   A.data, A.m, x.data, 1, &beta, this->data,
                                   1);

    } else {
      MLCommon::LinAlg::cublasgemv(cublas, CUBLAS_OP_N, A.n, A.m, &alpha,
                                   A.data, A.n, x.data, 1, &beta, this->data,
                                   1);
    }
  }
};

template <typename T> struct SimpleMat : SimpleVec<T> {
  typedef SimpleVec<T> Super;
  int m, n;

  STORAGE_ORDER ord; // storage order: runtime param for compile time sake

  SimpleMat(STORAGE_ORDER order = COL_MAJOR) : Super(), ord(order) {}

  SimpleMat(T *data, int m, int n, STORAGE_ORDER order = COL_MAJOR)
      : Super(data, m * n), m(m), n(n), ord(order) {}

  SimpleMat(int m, int n, STORAGE_ORDER order = COL_MAJOR, const T val = 0)
      : Super(m * n, val), m(m), n(n), ord(order) {}

  void reset(int m_, int n_) {
    m = m_;
    n = n_;
    Super::reset(m * n);
  }
  void reset(T *data_, int m_, int n_) {
    m = m_;
    n = n_;
    Super::reset(data_, m * n);
  }

  void print() const { std::cout << (*this) << std::endl; }

  void assign_gemm(const T alpha, const SimpleMat<T> &A, const SimpleMat<T> &B,
                   const T beta, cublasHandle_t &cublas) {

    ASSERT(A.n == B.m, "GEMM invalid dims");
    ASSERT(A.m == this->m, "GEMM invalid dims");
    ASSERT(B.n == this->n, "GEMM invalid dims");

    ASSERT(ord == COL_MAJOR, "GEMM for row-major C not implemented");
    ASSERT(A.ord == COL_MAJOR, "GEMM for row-major A not implemented");

    if (B.ord == COL_MAJOR) {
      MLCommon::LinAlg::cublasgemm(cublas, CUBLAS_OP_N,
                                   CUBLAS_OP_N,           // transA, transB
                                   this->m, this->n, A.n, // dimensions m,n,k
                                   &alpha, A.data,
                                   A.m,         // lda
                                   B.data, B.m, // ldb
                                   &beta, this->data,
                                   this->m // ldc
      );

    } else {
      MLCommon::LinAlg::cublasgemm(cublas,
                                   CUBLAS_OP_N,           // tranA
                                   CUBLAS_OP_T,           // transB
                                   this->m, this->n, A.n, // dimensions m,n,k
                                   &alpha, A.data,
                                   A.m,         // lda
                                   B.data, B.n, // ldb
                                   &beta, this->data, this->m);
    }
  }
  void assign_gemmBT(const T alpha, const SimpleMat<T> &A,
                     const SimpleMat<T> &B, const T beta,
                     cublasHandle_t &cublas) {

    ASSERT(A.n == B.n, "GEMM BT invalid dims");
    ASSERT(A.m == this->m, "GEMM BT invalid dims");
    ASSERT(B.m == this->n, "GEMM BT invalid dims");

    ASSERT(ord == COL_MAJOR, "GEMM BT for row-major C not implemented");
    ASSERT(A.ord == COL_MAJOR, "GEMM BT for row-major A not implemented");
    if (B.ord == COL_MAJOR) {
      MLCommon::LinAlg::cublasgemm(cublas, CUBLAS_OP_N,   // transA
                                   CUBLAS_OP_T,           // transB
                                   this->m, this->n, A.n, // dimensions m,n,k
                                   &alpha, A.data,
                                   A.m,         // lda
                                   B.data, B.m, // ldb
                                   &beta, this->data, this->m);

    } else {
      MLCommon::LinAlg::cublasgemm(cublas,
                                   CUBLAS_OP_N,           // tranA
                                   CUBLAS_OP_N,           // transB
                                   this->m, this->n, A.n, // dimensions m,n,k
                                   &alpha, A.data,
                                   A.m,         // lda
                                   B.data, B.n, // ldb
                                   &beta, this->data, this->m);
    }
  }
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
             cudaStream_t stream = 0) {
  auto f = [] __device__(const T x, const T y) { return x * y; };
  MLCommon::LinAlg::mapThenSumReduce(tmp_dev, u.len, f, stream, u.data, v.data);
  T tmp_host;
  MLCommon::updateHost(&tmp_host, tmp_dev, 1);
  return tmp_host;
}

template <typename T>
inline T squaredNorm(const SimpleVec<T> &u, T *tmp_dev,
             cudaStream_t stream = 0) {
    return dot(u, u, tmp_dev, stream);
}

template <typename T>
inline T nrm2(const SimpleVec<T> &u, T *tmp_dev,
             cudaStream_t stream = 0) {
    return MLCommon::mySqrt<T>(squaredNorm(u, tmp_dev, stream));
}

template <typename T>
inline T nrm1(const SimpleVec<T> &u, T *tmp_dev,
             cudaStream_t stream = 0) {
  auto f = [] __device__(const T x) { return MLCommon::myAbs<T>(x); };
  MLCommon::LinAlg::mapThenSumReduce(tmp_dev, u.len, f, stream, u.data);
  T tmp_host;
  MLCommon::updateHost(&tmp_host, tmp_dev, 1);
  return tmp_host;
}



/*
template <typename T>
inline void dot(T *out_dev, const SimpleVec<T> &u, const SimpleVec<T> &v,
                cudaStream_t stream = 0) {
  auto f = [] __device__(const T x, const T y) { return x * y; };
  MLCommon::LinAlg::mapThenSumReduce(out_dev, u.len, f, stream, u.data, v.data);
}

template <typename T>
inline void squaredNorm(T *out_dev, const SimpleVec<T> &v,
                        cudaStream_t stream = 0) {
  auto f = [] __device__(const T x) { return x * x; };
  MLCommon::LinAlg::mapThenSumReduce(out_dev, v.len, f, stream, v.data);
}
template <typename T> struct inner_product {
  SimpleVec<T> out;

  inner_product() : out(1) {}
  inner_product(T *data) : out(data, 1) {}

  inline T operator()(const SimpleVec<T> &u, const SimpleVec<T> &v,
                      cudaStream_t stream = 0) {
    dot(out.data, u, v, stream);
    return out[0];
  }
};

template <typename T, class I> struct norm {
  SimpleVec<T> out;

  norm() : out(1) {}
  norm(T *data) : out(data, 1) {}

  inline T operator()(const SimpleVec<T> &u, cudaStream_t stream = 0) {
    static_cast<I *>(this)->get(out.data, u, stream);
    T ret = 0;
    MLCommon::updateHost(&ret, out.data, 1);
    return ret;
  }
};

template <typename T> struct norm2 : norm<T, norm2<T>> {
  inline void get(T *out_dev, const SimpleVec<T> &v, cudaStream_t stream = 0) {
    squaredNorm(out_dev, v, stream);
    auto f = [] __device__(const T x) { return MLCommon::mySqrt<T>(x); };
    MLCommon::LinAlg::unaryOp(out_dev, out_dev, 1, f, stream);
  }
};

template <typename T> struct norm1 : norm<T, norm1<T>> {
  inline void get(T *out_dev, const SimpleVec<T> &v, cudaStream_t stream = 0) {
    auto f = [] __device__(const T x) { return abs(x); };
    MLCommon::LinAlg::mapThenSumReduce(out_dev, v.len, f, stream, v.data);
  }
};

template <typename T> struct norm0 : norm<T, norm0<T>> {
  inline void get(T *out_dev, const SimpleVec<T> &v, cudaStream_t stream = 0) {
    auto f = [] __device__(const T x) { return x == T(0); };
    MLCommon::LinAlg::mapThenSumReduce(out_dev, v.len, f, stream, v.data);
  }
};
*/

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

}; // namespace ML
