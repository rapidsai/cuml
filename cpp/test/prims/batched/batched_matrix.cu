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

#include <gtest/gtest.h>
#include <algorithm>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/cuml.hpp>
#include <random>
#include <vector>

#include "add.h"
#include "batched_matrix.h"
#include "linalg/batched/batched_matrix.hpp"
#include "test_utils.h"

namespace MLCommon {
namespace LinAlg {
namespace Batched {

enum BatchedMatrixOperation {
  AB_op,
  AZT_op,
  ZA_op,
  ApB_op,
  AmB_op,
  AkB_op,
  AsolveZ_op,
  LaggedZ_op,
  CopyA_op,
  CopyA2D_op,
  DiffA_op
};

template <typename T>
struct BatchedMatrixInputs {
  BatchedMatrixOperation operation;
  int n_batches;
  int m;  // Usually the dimensions of A and/or Z
  int n;
  int p;  // Usually the dimensions of B or other parameters
  int q;
  int s;  // Additional parameters for operations that need more than 4
  int t;
  T tolerance;
};

template <typename T>
class BatchedMatrixTest
  : public ::testing::TestWithParam<BatchedMatrixInputs<T>> {
 protected:
  void SetUp() override {
    using std::vector;
    params = ::testing::TestWithParam<BatchedMatrixInputs<T>>::GetParam();

    // Find out whether A, B and Z will be used (depending on the operation)
    bool use_A = (params.operation != LaggedZ_op);
    bool use_B = (params.operation == AB_op) || (params.operation == ApB_op) ||
                 (params.operation == AmB_op) || (params.operation == AkB_op);
    bool use_Z = (params.operation == AZT_op) || (params.operation == ZA_op) ||
                 (params.operation == AsolveZ_op) ||
                 (params.operation == LaggedZ_op);
    bool Z_col = (params.operation == AsolveZ_op);
    int r = params.operation == AZT_op ? params.n : params.m;

    // Check if the dimensions are valid and compute the output dimensions
    int m_r, n_r;
    switch (params.operation) {
      case AB_op:
        ASSERT_TRUE(params.n == params.p);
        m_r = params.m;
        n_r = params.q;
        break;
      case ApB_op:
      case AmB_op:
        ASSERT_TRUE(params.m == params.p && params.n == params.q);
        m_r = params.m;
        n_r = params.n;
        break;
      case AkB_op:
        m_r = params.m * params.p;
        n_r = params.n * params.q;
        break;
      case AZT_op:
        m_r = params.m;
        n_r = 1;
        break;
      case ZA_op:
        m_r = 1;
        n_r = params.n;
        break;
      case AsolveZ_op:
        ASSERT_TRUE(params.n == params.m);
        // For this test we multiply A by the solution and check against Z
        m_r = params.m;
        n_r = 1;
        break;
      case LaggedZ_op:
        // For this operation params.n holds the number of lags
        m_r = params.m - params.n;
        n_r = params.n;
        break;
      case CopyA_op:
        m_r = params.m;
        n_r = params.n;
        break;
      case CopyA2D_op:
        // For this operation p and q are the dimensions of the copy window
        m_r = params.p;
        n_r = params.q;
        break;
      case DiffA_op:
        // Note: A can represent either a row or column vector
        ASSERT_TRUE(params.m == 1 || params.n == 1);
        m_r = std::max(1, params.m - 1);
        n_r = std::max(1, params.n - 1);
    }

    // Create test matrices and vector
    std::vector<T> A;
    std::vector<T> B;
    std::vector<T> Z;
    if (use_A) A.resize(params.n_batches * params.m * params.n);
    if (use_B) B.resize(params.n_batches * params.p * params.q);
    if (use_Z) Z.resize(params.n_batches * r);

    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> udis(-1.0, 3.0);
    for (int i = 0; i < A.size(); i++) A[i] = udis(gen);
    for (int i = 0; i < B.size(); i++) B[i] = udis(gen);
    for (int i = 0; i < Z.size(); i++) Z[i] = udis(gen);

    // Create handles, stream, allocator
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto allocator = std::make_shared<MLCommon::defaultDeviceAllocator>();

    // Created batched matrices
    BatchedMatrix<T> AbM(params.m, params.n, params.n_batches, handle,
                         allocator, stream);
    BatchedMatrix<T> BbM(params.p, params.q, params.n_batches, handle,
                         allocator, stream);
    BatchedMatrix<T> ZbM(Z_col ? r : 1, Z_col ? 1 : r, params.n_batches, handle,
                         allocator, stream);

    // Copy the data to the device
    if (use_A) updateDevice(AbM.raw_data(), A.data(), A.size(), stream);
    if (use_B) updateDevice(BbM.raw_data(), B.data(), B.size(), stream);
    if (use_Z) updateDevice(ZbM.raw_data(), Z.data(), Z.size(), stream);

    // Create fake batched matrices to be overwritten by results
    res_bM = new BatchedMatrix<T>(1, 1, 1, handle, allocator, stream);

    // Compute the tested results
    switch (params.operation) {
      case AB_op:
        *res_bM = AbM * BbM;
        break;
      case ApB_op:
        *res_bM = AbM + BbM;
        break;
      case AmB_op:
        *res_bM = AbM - BbM;
        break;
      case AkB_op:
        *res_bM = b_kron(AbM, BbM);
        break;
      case AZT_op:
        *res_bM = b_gemm(AbM, ZbM, false, true);
        break;
      case ZA_op:
        *res_bM = ZbM * AbM;
        break;
      case AsolveZ_op:
        // A * A\Z -> should be Z
        *res_bM = AbM * b_solve(AbM, ZbM);
        break;
      case LaggedZ_op:
        *res_bM = b_lagged_mat(ZbM, params.n);
        break;
      case CopyA_op:
        *res_bM = AbM.deepcopy();
        break;
      case CopyA2D_op:
        *res_bM = b_2dcopy(AbM, params.s, params.t, params.p, params.q);
        break;
      case DiffA_op:
        *res_bM = AbM.difference();
        break;
    }

    // Compute the expected results
    res_h.resize(params.n_batches * m_r * n_r);
    switch (params.operation) {
      case AB_op:
        for (int bid = 0; bid < params.n_batches; bid++) {
          naiveMatMul(res_h.data() + bid * m_r * n_r,
                      A.data() + bid * params.m * params.n,
                      B.data() + bid * params.p * params.q, params.m, params.n,
                      params.q);
        }
        break;
      case ApB_op:
        naiveAdd(res_h.data(), A.data(), B.data(), A.size());
        break;
      case AmB_op:
        naiveAdd(res_h.data(), A.data(), B.data(), A.size(), T(-1.0));
        break;
      case AkB_op:
        for (int bid = 0; bid < params.n_batches; bid++) {
          naiveKronecker(res_h.data() + bid * m_r * n_r,
                         A.data() + bid * params.m * params.n,
                         B.data() + bid * params.p * params.q, params.m,
                         params.n, params.p, params.q);
        }
        break;
      case AZT_op:
        for (int bid = 0; bid < params.n_batches; bid++) {
          naiveMatMul(res_h.data() + bid * m_r * n_r,
                      A.data() + bid * params.m * params.n, Z.data() + bid * r,
                      params.m, params.n, 1);
        }
        break;
      case ZA_op:
        for (int bid = 0; bid < params.n_batches; bid++) {
          naiveMatMul(res_h.data() + bid * m_r * n_r, Z.data() + bid * r,
                      A.data() + bid * params.m * params.n, 1, params.m,
                      params.n);
        }
        break;
      case AsolveZ_op:
        // Simply copy Z in the result
        memcpy(res_h.data(), Z.data(), r * params.n_batches * sizeof(T));
        break;
      case LaggedZ_op:
        for (int bid = 0; bid < params.n_batches; bid++) {
          naiveLaggedMat(res_h.data() + bid * m_r * n_r,
                         Z.data() + bid * params.m, params.m, params.n);
        }
        break;
      case CopyA_op:
        memcpy(res_h.data(), A.data(),
               params.m * params.n * params.n_batches * sizeof(T));
        break;
      case CopyA2D_op:
        for (int bid = 0; bid < params.n_batches; bid++) {
          naive2DCopy(res_h.data() + bid * m_r * n_r,
                      A.data() + bid * params.m * params.n, params.s, params.t,
                      params.m, m_r, n_r);
        }
        break;
      case DiffA_op:
        int len = params.m * params.n;
        for (int bid = 0; bid < params.n_batches; bid++) {
          naiveDiff(res_h.data() + bid * (len - 1), A.data() + bid * len, len);
        }
        break;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override {
    delete res_bM;
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  BatchedMatrixInputs<T> params;
  BatchedMatrix<T> *res_bM;
  std::vector<T> res_h;
  cublasHandle_t handle;
  cudaStream_t stream;
};

// Test parameters (op, n_batches, m, n, p, q, tolerance)
const std::vector<BatchedMatrixInputs<double>> inputsd = {
  {AB_op, 7, 15, 37, 37, 11, 0, 0, 1e-6},
  {AZT_op, 5, 33, 65, 1, 1, 0, 0, 1e-6},
  {ZA_op, 8, 12, 41, 1, 1, 0, 0, 1e-6},
  {ApB_op, 4, 16, 48, 16, 48, 0, 0, 1e-6},
  {AmB_op, 17, 9, 3, 9, 3, 0, 0, 1e-6},
  {AkB_op, 5, 3, 13, 31, 8, 0, 0, 1e-6},
  {AkB_op, 3, 7, 12, 31, 15, 0, 0, 1e-6},
  {AkB_op, 2, 11, 2, 8, 46, 0, 0, 1e-6},
  {AsolveZ_op, 6, 17, 17, 1, 1, 0, 0, 1e-6},
  {LaggedZ_op, 5, 31, 9, 1, 1, 0, 0, 1e-6},
  {LaggedZ_op, 7, 129, 3, 1, 1, 0, 0, 1e-6},
  {CopyA_op, 7, 35, 43, 1, 1, 0, 0, 1e-6},
  {CopyA2D_op, 11, 31, 63, 17, 14, 5, 9, 1e-6},
  {CopyA2D_op, 4, 33, 7, 30, 4, 3, 0, 1e-6},
  {DiffA_op, 5, 11, 1, 1, 1, 0, 0, 1e-6},
  {DiffA_op, 15, 1, 37, 1, 1, 0, 0, 1e-6}};

// Test parameters (op, n_batches, m, n, p, q, tolerance)
const std::vector<BatchedMatrixInputs<float>> inputsf = {
  {AB_op, 7, 15, 37, 37, 11, 0, 0, 1e-2},
  {AZT_op, 5, 33, 65, 1, 1, 0, 0, 1e-2},
  {ZA_op, 8, 12, 41, 1, 1, 0, 0, 1e-2},
  {ApB_op, 4, 16, 48, 16, 48, 0, 0, 1e-2},
  {AmB_op, 17, 9, 3, 9, 3, 0, 0, 1e-2},
  {AkB_op, 5, 3, 13, 31, 8, 0, 0, 1e-2},
  {AkB_op, 3, 7, 12, 31, 15, 0, 0, 1e-2},
  {AkB_op, 2, 11, 2, 8, 46, 0, 0, 1e-2},
  {AsolveZ_op, 6, 17, 17, 1, 1, 0, 0, 1e-2},
  {LaggedZ_op, 5, 31, 9, 1, 1, 0, 0, 1e-5},
  {LaggedZ_op, 7, 129, 3, 1, 1, 0, 0, 1e-5},
  {CopyA_op, 7, 35, 43, 1, 1, 0, 0, 1e-5},
  {CopyA2D_op, 11, 31, 63, 17, 14, 5, 9, 1e-5},
  {CopyA2D_op, 4, 33, 7, 30, 4, 3, 0, 1e-5},
  {DiffA_op, 5, 11, 1, 1, 1, 0, 0, 1e-2},
  {DiffA_op, 15, 1, 37, 1, 1, 0, 0, 1e-2}};

using BatchedMatrixTestD = BatchedMatrixTest<double>;
using BatchedMatrixTestF = BatchedMatrixTest<float>;
TEST_P(BatchedMatrixTestD, Result) {
  ASSERT_TRUE(devArrMatchHost(res_h.data(), res_bM->raw_data(), res_h.size(),
                              CompareApprox<double>(params.tolerance), stream));
}
TEST_P(BatchedMatrixTestF, Result) {
  ASSERT_TRUE(devArrMatchHost(res_h.data(), res_bM->raw_data(), res_h.size(),
                              CompareApprox<float>(params.tolerance), stream));
}

INSTANTIATE_TEST_CASE_P(BatchedMatrixTests, BatchedMatrixTestD,
                        ::testing::ValuesIn(inputsd));
INSTANTIATE_TEST_CASE_P(BatchedMatrixTests, BatchedMatrixTestF,
                        ::testing::ValuesIn(inputsf));

}  // namespace Batched
}  // namespace LinAlg
}  // namespace MLCommon
