#pragma once

#include <tuple>
#include <vector>
#include <stdexcept>

#include <linalg/cublas_wrappers.h>

namespace MLCommon {
namespace Matrix {
class BatchedMatrix {
public:
  // Create a BatchedMatrix.
  BatchedMatrix(const std::vector<double*>& A, std::pair<int, int> shape, bool gpu=true)
    : m_gpu(gpu), m_shape(shape) {
    init(A, shape, gpu);
  }

  BatchedMatrix(int m, int n, int num_batches, bool gpu=true) : m_gpu(gpu) {
    if(!gpu) {
      throw std::runtime_error("CPU-only not supported");
    }
    m_shape = std::make_pair(m, n);
    std::vector<double*> C_data;
    for(int i=0;i<num_batches;i++) {
      double* d_ptr;
      cudaMalloc(&d_ptr, sizeof(double)*m*n);
      C_data.push_back(d_ptr);
    }
    init(C_data, std::make_pair(m, n), gpu);
  }

  bool onGPU() const { return m_gpu; }
  size_t batches() const { return m_num_batches; }
  const std::pair<int, int>& shape() const { return m_shape; }
  double** data() const {return m_A_data;}

  const std::vector<double*>& A() {
    if (m_A.size() == 0) {
      double** h_ptr = new double*[m_num_batches];
      updateHost(h_ptr, m_A_data, m_num_batches);
      for(int i=0;i<m_num_batches;i++) {
        m_A.push_back(h_ptr[i]);
      }
    }
    return m_A;
  }

private:

  // shared initialization function
  void init(const std::vector<double*>& A, std::pair<int, int> shape, bool gpu=true) {
    m_A = A;
    m_num_batches = A.size();
    if(!gpu) {
      throw std::runtime_error("CPU-only not supported");
    }
    CUDA_CHECK(cudaMalloc(&m_A_data, sizeof(double*) * m_num_batches));
    CUDA_CHECK(cudaMemcpy(m_A_data, A.data(), sizeof(double*) * m_num_batches, cudaMemcpyHostToDevice));
  }

  // host-stored pointers to (device) matrices
  std::vector<double*> m_A;

  // decides where data is stored and where operations are computed.
  bool m_gpu;

  // Shape (rows, cols) of matrices. We assume all matrices in batch have same shape.
  std::pair<int, int> m_shape;



  // Raw Data pointer to batched matrices. Is stored in CPU or GPU depending on `m_gpu`
  double** m_A_data;

  // batch information
  size_t m_num_batches;


};

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

  // Create C(m,k)
  BatchedMatrix C(m, k, num_batches);

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
  return C;
}

// Multiplies each matrix in a batch-A with it's batch-B counterpart.
// A = [A1,A2,A3], B=[B1,B2,B3]
// return [A1*B1, A2*B2, A3*B3]
BatchedMatrix operator*(const BatchedMatrix& A,
                        const BatchedMatrix& B) {

  return b_gemm(A,B);

}
}
}
