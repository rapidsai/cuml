#pragma once

#include <tuple>
#include <vector>
#include <stdexcept>
#include <functional>

#include <linalg/cublas_wrappers.h>
#include <linalg/binary_op.h>
#include <memory>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

namespace MLCommon {
namespace Matrix {

std::shared_ptr<double*>
init(double* A, std::pair<int, int> shape, int num_batches, bool gpu=true);

template<typename T>
void cudaFreeT(T *ptr) { CUDA_CHECK(cudaFree(ptr)); }

class BatchedMatrix {
public:
  // Create a BatchedMatrix.
  BatchedMatrix(double* A, std::pair<int, int> shape, int num_batches, bool gpu=true)
    : m_gpu(gpu), m_shape(shape), m_num_batches(num_batches) {
    if(!gpu) {
      throw std::runtime_error("CPU-only not supported");
    }
    m_A_batches = init(A, shape, num_batches, gpu);
    m_A_dense = std::shared_ptr<double>(A, cudaFreeT<double>);
  }

  BatchedMatrix(int m, int n, int num_batches, bool initZero=false, bool gpu=true) : m_gpu(gpu), m_num_batches(num_batches) {
    if(!gpu) {
      throw std::runtime_error("CPU-only not supported");
    }
    m_shape = std::make_pair(m, n);

    double* A_all;

    allocate(A_all, m*n*num_batches);
    m_A_dense = std::shared_ptr<double>(A_all, cudaFreeT<double>);

    m_A_batches = init(A_all, std::make_pair(m, n), num_batches, gpu);
  }

  bool onGPU() const { return m_gpu; }
  size_t batches() const { return m_num_batches; }
  const std::pair<int, int>& shape() const { return m_shape; }

  // TODO: probably should add a const on returned type
  double** data() const {return m_A_batches.get();}

  double* operator[](int id) const {
    return &(m_A_dense.get()[id*m_shape.first*m_shape.second]);
  }

private:

  // decides where data is stored and where operations are computed.
  bool m_gpu;

  // Shape (rows, cols) of matrices. We assume all matrices in batch have same shape.
  std::pair<int, int> m_shape;

  // Array(pointer) to each matrix.
  std::shared_ptr<double*> m_A_batches;

  // Data pointer to first element of consecutive matrix data
  std::shared_ptr<double> m_A_dense;

  // batch information
  size_t m_num_batches;


};

// shared initialization function
std::shared_ptr<double*>
init(double* A, std::pair<int, int> shape, int num_batches, bool gpu) {
  double** raw_m_A_data;
  allocate(raw_m_A_data, num_batches);

  int m = shape.first;
  int n = shape.second;

  // fill array of pointers to each batch matrix.
  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(counting, counting + num_batches,
                   [=]__device__(int bid){
                     raw_m_A_data[bid] = &(A[bid*m*n]);
                   });

  // when the reference count goes to zero it will free the underlying arrays.
  return std::shared_ptr<double*>(raw_m_A_data, cudaFreeT<double*>);
}


// Multiplies each matrix in a batch-A with it's batch-B counterpart.
// A = [A1,A2,A3], B=[B1,B2,B3]
// return [A1*B1, A2*B2, A3*B3]
BatchedMatrix b_gemm(const BatchedMatrix& A,
                     const BatchedMatrix& B, bool aT=false, bool bT=false) {
  if(!(A.onGPU() && B.onGPU())) {
    throw std::runtime_error("CPU not currently supported");
  }
  if(A.batches() != B.batches()) {
    throw std::runtime_error("A & B must have same number of batches");
  }

  // logic for matrix dimensions with optional transpose
  // m = number of rows of matrix op(A) and C.
  int m = !aT ? A.shape().first : A.shape().second;

  // n = number of columns of matrix op(B) and C.
  int n = !bT ? B.shape().second : B.shape().first;

  // k = number of columns of op(A) and rows of op(B).
  int k = !aT ? A.shape().second : A.shape().first;
  int kB = !bT ? B.shape().first : B.shape().second;

  if(k != kB) {
    throw std::runtime_error("Matrix-Multiplication dimensions don't match!");
  }

  auto num_batches = A.batches();
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  
  // set transpose
  cublasOperation_t opA = aT ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = bT ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Create C(m,n)
  BatchedMatrix C(m, n, num_batches);

  double alpha = 1.0;
  double beta = 0.0;

  // [C1,C2,C3] = [A1*B1, A2*B2, A3*B3]
  CUBLAS_CHECK(cublasDgemmBatched(handle,
                                  opA, // A.T?
                                  opB, // B.T?
                                  m, // rows op(A), C
                                  n, // cols of op(B), C
                                  k, // cols of op(A), rows of op(B)
                                  &alpha, // alpha * A * B
                                  A.data(),
                                  A.shape().first, // rows of A
                                  B.data(),
                                  B.shape().first, // rows of B
                                  &beta, // + beta * C
                                  C.data(),
                                  C.shape().first, // rows of C
                                  num_batches));
  CUBLAS_CHECK(cublasDestroy(handle));

  return C;
}

template <typename F>
BatchedMatrix b_aA_op_B(const BatchedMatrix &A, const BatchedMatrix &B,
                        F binary_op) {
  if(A.shape().first != B.shape().first && A.shape().second != B.shape().second) {
    throw std::runtime_error("Batched Matrix Addition ERROR: Matrices must be same size");
  }
  if(A.batches() != B.batches()) {
    throw std::runtime_error("A & B must have same number of batches");
  }

  auto num_batches = A.batches();
  int m = A.shape().first;
  int n = A.shape().second;

  BatchedMatrix C(m, n, num_batches);

  for(int i=0; i<num_batches; i++) {
    // LinAlg::binaryOp(C.A()[i].get(), A.A()[i].get(), B.A()[i].get(), m*n, binary_op);
    LinAlg::binaryOp(C[i], A[i], B[i], m*n, binary_op);
  }
  return C;
}

// Multiplies each matrix in a batch-A with it's batch-B counterpart.
// A = [A1,A2,A3], B=[B1,B2,B3]
// return [A1*B1, A2*B2, A3*B3]
BatchedMatrix
operator*(const BatchedMatrix &A, const BatchedMatrix &B) {
  return b_gemm(A,B);
}

BatchedMatrix operator+(const BatchedMatrix& A, const BatchedMatrix& B) {
  return b_aA_op_B(A, B, [] __device__ (double a, double b) {return a + b;});
}

BatchedMatrix operator-(const BatchedMatrix& A, const BatchedMatrix& B) {
  return b_aA_op_B(A, B,  [] __device__ (double a, double b) {return a - b;});
}

}
}
