#pragma once
#include <cuda_utils.h>
#include <linalg/binary_op.h>
#include <linalg/map_then_reduce.h>
#include <linalg/ternary_op.h>
#include <linalg/unary_op.h>
#include <iostream>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include "linalg/cublas_wrappers.h"


namespace ML {

using namespace MLCommon;

enum STORAGE_ORDER { COL_MAJOR = 0, ROW_MAJOR = 1 };

template <typename T>
struct SimpleVec;

template <typename T, STORAGE_ORDER>
struct SimpleMat;

template <typename T>
HDI int sgn(const T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename T>
struct op_ax {
  T a;
  op_ax(T a) : a(a) {}

  HDI T operator()(T x) const { return a * x; }
};

template <typename T>
struct op_axpy {
  T a;
  op_axpy(T a) : a(a) {}
  HDI T operator()(T x, T y) const { return a * x + y; }
};

template <typename T>
struct op_axpby {
  T a;
  T b;
  op_axpby(T a, T b) : a(a), b(b) {}
  HDI T operator()(T x, T y) const { return a * x + b * y; }
};

template <typename T, STORAGE_ORDER Storage>
struct gemv_helper {
  static void gemv(SimpleVec<T> &v, const T alpha,
                   const SimpleMat<T, COL_MAJOR> &A, const SimpleVec<T> &x,
                   const T beta, cublasHandle_t &cublas) {}
  static void gemvT(SimpleVec<T> &v, const T alpha,
                    const SimpleMat<T, COL_MAJOR> &A, const SimpleVec<T> &x,
                    const T beta, cublasHandle_t &cublas) {}
};

template <typename T>
struct SimpleVec {
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
    allocate(data, len);
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
    LinAlg::unaryOp(data, data, len, f);
  }

  inline void operator=(const T val) { fill(val); }

  inline void operator=(const SimpleVec<T> &other) {
    CUDA_CHECK(
      cudaMemcpy(data, other.data, len * sizeof(T), cudaMemcpyDeviceToDevice));
  }

  inline T operator[](int pos) const {
    T tmp;
    updateHost(&tmp, &data[pos], 1);
    return tmp;
  }

  inline void assign_pos(int pos, T val) { updateDevice(&data[pos], &val, 1); }

  // this = a*x
  inline void ax(const T a, const SimpleVec<T> &x) {
    auto scale = [a] __device__(T x) { return a * x; };
    LinAlg::unaryOp(data, x.ptr(), len, scale);
  }

  // this = a*x + y
  inline void axpy(const T a, const SimpleVec<T> &x, const SimpleVec<T> &y) {
    auto axpy = [a] __device__(T x, T y) { return a * x + y; };
    LinAlg::binaryOp(data, x.data, y.data, len, axpy);
  }

  // this = a*x + b*y
  inline void axpby(const T a, const SimpleVec<T> &x, const T b,
                    const SimpleVec<T> &y) {
    auto axpby = [a, b] __device__(T x, T y) { return a * x + b * y; };
    LinAlg::binaryOp(data, x.data, y.data, len, axpby);
  }

  template <typename Lambda>
  inline void assign_unary(const SimpleVec<T> &other, Lambda &f) {
    LinAlg::unaryOp(data, other.data, len, f);
  }

  template <typename Lambda>
  inline void assign_binary(const SimpleVec<T> &other1,
                            const SimpleVec<T> &other2, Lambda &f) {
    LinAlg::binaryOp(data, other1.data, other2.data, len, f);
  }

  template <typename Lambda>
  inline void assign_ternary(const SimpleVec<T> &other1,
                             const SimpleVec<T> &other2,
                             const SimpleVec<T> &other3, Lambda &f) {
    LinAlg::ternaryOp(data, other1.data, other2.data, other3.data, len, f);
  }

  template <typename Op, typename... Vectors>
  inline void assign_k_ary(Op &op, const Vectors &... args) {
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(args.p...));

    auto end =
      thrust::make_zip_iterator(thrust::make_tuple((args.p + args.len)...));
    thrust::transform(begin, end, p, op);
  }

  inline void print() const { std::cout << (*this) << std::endl; }

  template <STORAGE_ORDER Storage>
  inline void assign_gemv(const T alpha, const SimpleMat<T, Storage> &A,
                          const SimpleVec<T> &x, const T beta,
                          cublasHandle_t &cublas) {
    // this <- alpha * A * x + beta * this
    gemv_helper<T, Storage>::gemv(*this, alpha, A, x, beta, cublas);
  }

  template <STORAGE_ORDER Storage>
  inline void assign_gemvT(const T alpha, const SimpleMat<T, Storage> &A,
                           const SimpleVec<T> &x, const T beta,
                           cublasHandle_t &cublas) {
    // this <- alpha * A * x + beta * this
    gemv_helper<T, Storage>::gemvT(*this, alpha, A, x, beta, cublas);
  }
};

template <typename T>
struct gemv_helper<T, COL_MAJOR> {
  static void gemv(SimpleVec<T> &v, const T alpha,
                   const SimpleMat<T, COL_MAJOR> &A, const SimpleVec<T> &x,
                   const T beta, cublasHandle_t &cublas) {
    MLCommon::LinAlg::cublasgemv(cublas, CUBLAS_OP_N, A.m, A.n, &alpha, A.data,
                                 A.m, x.data, 1, &beta, v.data, 1);
  }
  static void gemvT(SimpleVec<T> &v, const T alpha,
                    const SimpleMat<T, COL_MAJOR> &A, const SimpleVec<T> &x,
                    const T beta, cublasHandle_t &cublas) {
    MLCommon::LinAlg::cublasgemv(cublas, CUBLAS_OP_T, A.m, A.n, &alpha, A.data,
                                 A.m, x.data, 1, &beta, v.data, 1);
  }
};

template <typename T>
struct gemv_helper<T, ROW_MAJOR> {
  static void gemv(SimpleVec<T> &v, const T alpha,
                   const SimpleMat<T, ROW_MAJOR> &A, const SimpleVec<T> &x,
                   const T beta, cublasHandle_t &cublas) {
    MLCommon::LinAlg::cublasgemv(cublas, CUBLAS_OP_T, A.n, A.m, &alpha, A.data,
                                 A.n, x.data, 1, &beta, v.data, 1);
  }
  static void gemvT(SimpleVec<T> &v, const T alpha,
                    const SimpleMat<T, ROW_MAJOR> &A, const SimpleVec<T> &x,
                    const T beta, cublasHandle_t &cublas) {
    MLCommon::LinAlg::cublasgemv(cublas, CUBLAS_OP_N, A.n, A.m, &alpha, A.data,
                                 A.n, x.data, 1, &beta, v.data, 1);
  }
};

template <typename T, STORAGE_ORDER Storage = COL_MAJOR>
struct SimpleMat : SimpleVec<T> {
  typedef SimpleVec<T> Super;
  int m, n;

  SimpleMat(T *data, int m, int n) : Super(data, m * n), m(m), n(n) {}
  SimpleMat(T *data, int m, int n, const T val)
    : Super(data, m * n, val), m(m), n(n) {}

  SimpleMat(int m, int n, const T val = 0) : Super(m * n, val), m(m), n(n) {}

  void reset(T *data_, int m_, int n_) {
    m = m_;
    n = n_;
    Super::reset(data_, m * n);
  }

  void print() const { std::cout << (*this) << std::endl; }
};

template <typename T>
struct ColMajorMat : SimpleMat<T, COL_MAJOR> {
  typedef SimpleMat<T, COL_MAJOR> Super;
  using Super::m;
  using Super::n;
  ColMajorMat(T *data, int m, int n) : Super(data, m , n) {}

  ColMajorMat(int m, int n, const T val = 0) : Super(m, n, val) {}

  void col_ref(SimpleVec<T> &mask_vec, int c) {
    T *tmp = &Super::data[m * c];
    mask_vec.reset(tmp, m);
  }
};

// Reductions such as dot or norm require an additional location in dev mem
// to hold the result. We don't want to deal with this in the SimpleVec class as
// it  impedes thread safety and constness

template <typename T>
inline void dot(T *out_dev, const SimpleVec<T> &u, const SimpleVec<T> &v,
                cudaStream_t stream = 0) {
  auto f = [] __device__(const T x, const T y) { return x * y; };
  LinAlg::mapThenSumReduce(out_dev, u.len, f, stream, u.data, v.data);
}

template <typename T>
inline void squaredNorm(T *out_dev, const SimpleVec<T> &v,
                        cudaStream_t stream = 0) {
  auto f = [] __device__(const T x) { return x * x; };
  LinAlg::mapThenSumReduce(out_dev, v.len, f, stream, v.data);
}
template <typename T>
struct inner_product {
  SimpleVec<T> out;

  inner_product() : out(1) {}
  inner_product(T *data) : out(data, 1) {}

  inline T operator()(const SimpleVec<T> &u, const SimpleVec<T> &v,
                      cudaStream_t stream = 0) {
    dot(out.data, u, v, stream);
    return out[0];
  }
};

template <typename T, class I>
struct norm {
  SimpleVec<T> out;

  norm() : out(1) {}
  norm(T *data) : out(data, 1) {}

  inline T operator()(const SimpleVec<T> &u, cudaStream_t stream = 0) {
    static_cast<I *>(this)->get(out.data, u, stream);
    T ret = 0;
    updateHost(&ret, out.data, 1);
    return ret;
  }
};

template <typename T>
struct norm2 : norm<T, norm2<T>> {
  inline void get(T *out_dev, const SimpleVec<T> &v, cudaStream_t stream = 0) {
    squaredNorm(out_dev, v, stream);
    auto f = [] __device__(const T x) { return mySqrt<T>(x); };
    LinAlg::unaryOp(out_dev, out_dev, 1, f, stream);
  }
};

template <typename T>
struct norm1 : norm<T, norm1<T>> {
  inline void get(T *out_dev, const SimpleVec<T> &v, cudaStream_t stream = 0) {
    auto f = [] __device__(const T x) { return abs(x); };
    LinAlg::mapThenSumReduce(out_dev, v.len, f, stream, v.data);
  }
};

template <typename T>
struct norm0 : norm<T, norm0<T>> {
  inline void get(T *out_dev, const SimpleVec<T> &v, cudaStream_t stream = 0) {
    auto f = [] __device__(const T x) { return x == T(0); };
    LinAlg::mapThenSumReduce(out_dev, v.len, f, stream, v.data);
  }
};


template <typename T>
std::ostream &operator<<(std::ostream &os, const SimpleVec<T> &v) {
  std::vector<T> out(v.len);
  updateHost(&out[0], v.data, v.len);
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
  updateHost(&out[0], mat.data, mat.len);
  for (int r = 0; r < mat.m; r++) {
    int idx = r;
    for (int c = 0; c < mat.n - 1; c++) {
      os << out[idx] << " ";
      idx += mat.m;
    }
    os << out[idx] << std::endl;
  }

  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const SimpleMat<T, ROW_MAJOR> &mat) {
  std::vector<T> out(mat.len);
  updateHost(&out[0], mat.data, mat.len);
  for (int c = 0; c < mat.m; c++) {
    int idx = c * mat.n;
    for (int r = 0; r < mat.n - 1; r++) {
      os << out[idx] << " ";
      idx += 1;
    }
    os << out[idx] << std::endl;
  }

  return os;
}


}; // namespace ML
