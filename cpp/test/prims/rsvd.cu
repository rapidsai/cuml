/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <common/cudart_utils.h>
#include <gtest/gtest.h>
#include <cuda_utils.cuh>
#include <linalg/rsvd.cuh>
#include <raft/handle.hpp>
#include <random/rng.cuh>
#include "test_utils.h"

namespace MLCommon {
namespace LinAlg {

template <typename T>
struct RsvdInputs {
  T tolerance;
  int n_row;
  int n_col;
  T PC_perc;
  T UpS_perc;
  int k;
  int p;
  bool use_bbt;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const RsvdInputs<T> &dims) {
  return os;
}

template <typename T>
class RsvdTest : public ::testing::TestWithParam<RsvdInputs<T>> {
 protected:
  void SetUp() override {
    raft::handle_t handle;
    stream = handle.get_stream();

    params = ::testing::TestWithParam<RsvdInputs<T>>::GetParam();
    // rSVD seems to be very sensitive to the random number sequence as well!
    raft::random::Rng r(params.seed, raft::random::GenTaps);
    int m = params.n_row, n = params.n_col;
    T eig_svd_tol = 1.e-7;
    int max_sweeps = 100;

    T mu = 0.0, sigma = 1.0;
    raft::allocate(A, m * n);
    if (params.tolerance > 1) {  // Sanity check
      ASSERT(m == 3, "This test only supports mxn=3x2!");
      ASSERT(m * n == 6, "This test only supports mxn=3x2!");
      T data_h[] = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
      raft::update_device(A, data_h, m * n, stream);

      T left_eig_vectors_ref_h[] = {-0.308219, -0.906133, -0.289695};
      T right_eig_vectors_ref_h[] = {-0.638636, -0.769509};
      T sing_vals_ref_h[] = {7.065283};

      raft::allocate(left_eig_vectors_ref, m * 1);
      raft::allocate(right_eig_vectors_ref, n * 1);
      raft::allocate(sing_vals_ref, 1);

      raft::update_device(left_eig_vectors_ref, left_eig_vectors_ref_h, m * 1,
                          stream);
      raft::update_device(right_eig_vectors_ref, right_eig_vectors_ref_h, n * 1,
                          stream);
      raft::update_device(sing_vals_ref, sing_vals_ref_h, 1, stream);

    } else {  // Other normal tests
      r.normal(A, m * n, mu, sigma, stream);
    }
    A_backup_cpu = (T *)malloc(
      sizeof(T) * m *
      n);  // Backup A matrix as svdJacobi will destroy the content of A
    raft::update_host(A_backup_cpu, A, m * n, stream);

    // RSVD tests
    if (params.k == 0) {  // Test with PC and upsampling ratio
      params.k = max((int)(min(m, n) * params.PC_perc), 1);
      params.p = max((int)(min(m, n) * params.UpS_perc), 1);
      raft::allocate(U, m * params.k, true);
      raft::allocate(S, params.k, true);
      raft::allocate(V, n * params.k, true);
      rsvdPerc(handle, A, m, n, S, U, V, params.PC_perc, params.UpS_perc,
               params.use_bbt, true, true, false, eig_svd_tol, max_sweeps,
               stream);
    } else {  // Test with directly given fixed rank
      raft::allocate(U, m * params.k, true);
      raft::allocate(S, params.k, true);
      raft::allocate(V, n * params.k, true);
      rsvdFixedRank(handle, A, m, n, S, U, V, params.k, params.p,
                    params.use_bbt, true, true, true, eig_svd_tol, max_sweeps,
                    stream);
    }
    raft::update_device(A, A_backup_cpu, m * n, stream);

    free(A_backup_cpu);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(U));
    CUDA_CHECK(cudaFree(S));
    CUDA_CHECK(cudaFree(V));
    if (left_eig_vectors_ref) CUDA_CHECK(cudaFree(left_eig_vectors_ref));
    if (right_eig_vectors_ref) CUDA_CHECK(cudaFree(right_eig_vectors_ref));
    if (sing_vals_ref) CUDA_CHECK(cudaFree(sing_vals_ref));
  }

 protected:
  RsvdInputs<T> params;
  T *A, *A_backup_cpu,
    *U = nullptr, *S = nullptr, *V = nullptr, *left_eig_vectors_ref = nullptr,
    *right_eig_vectors_ref = nullptr, *sing_vals_ref = nullptr;

  cudaStream_t stream;
};

const std::vector<RsvdInputs<float>> inputs_fx = {
  // Test with ratios
  {0.20f, 256, 256, 0.2f, 0.05f, 0, 0, true, 4321ULL},     // Square + BBT
  {0.20f, 2048, 256, 0.2f, 0.05f, 0, 0, true, 4321ULL},    // Tall + BBT
  {0.20f, 256, 256, 0.2f, 0.05f, 0, 0, false, 4321ULL},    // Square + non-BBT
  {0.20f, 2048, 256, 0.2f, 0.05f, 0, 0, false, 4321ULL},   // Tall + non-BBT
  {0.20f, 2048, 2048, 0.2f, 0.05f, 0, 0, true, 4321ULL},   // Square + BBT
  {0.60f, 16384, 2048, 0.2f, 0.05f, 0, 0, true, 4321ULL},  // Tall + BBT
  {0.20f, 2048, 2048, 0.2f, 0.05f, 0, 0, false, 4321ULL},  // Square + non-BBT
  {0.60f, 16384, 2048, 0.2f, 0.05f, 0, 0, false, 4321ULL}  // Tall + non-BBT

  ,  // Test with fixed ranks
  {0.10f, 256, 256, 0.0f, 0.0f, 100, 5, true, 4321ULL},     // Square + BBT
  {0.12f, 2048, 256, 0.0f, 0.0f, 100, 5, true, 4321ULL},    // Tall + BBT
  {0.10f, 256, 256, 0.0f, 0.0f, 100, 5, false, 4321ULL},    // Square + non-BBT
  {0.12f, 2048, 256, 0.0f, 0.0f, 100, 5, false, 4321ULL},   // Tall + non-BBT
  {0.60f, 2048, 2048, 0.0f, 0.0f, 100, 5, true, 4321ULL},   // Square + BBT
  {1.00f, 16384, 2048, 0.0f, 0.0f, 100, 5, true, 4321ULL},  // Tall + BBT
  {0.60f, 2048, 2048, 0.0f, 0.0f, 100, 5, false, 4321ULL},  // Square + non-BBT
  {1.00f, 16384, 2048, 0.0f, 0.0f, 100, 5, false, 4321ULL}  // Tall + non-BBT
};

const std::vector<RsvdInputs<double>> inputs_dx = {
  // Test with ratios
  {0.20, 256, 256, 0.2, 0.05, 0, 0, true, 4321ULL},     // Square + BBT
  {0.20, 2048, 256, 0.2, 0.05, 0, 0, true, 4321ULL},    // Tall + BBT
  {0.20, 256, 256, 0.2, 0.05, 0, 0, false, 4321ULL},    // Square + non-BBT
  {0.20, 2048, 256, 0.2, 0.05, 0, 0, false, 4321ULL},   // Tall + non-BBT
  {0.20, 2048, 2048, 0.2, 0.05, 0, 0, true, 4321ULL},   // Square + BBT
  {0.60, 16384, 2048, 0.2, 0.05, 0, 0, true, 4321ULL},  // Tall + BBT
  {0.20, 2048, 2048, 0.2, 0.05, 0, 0, false, 4321ULL},  // Square + non-BBT
  {0.60, 16384, 2048, 0.2, 0.05, 0, 0, false, 4321ULL}  // Tall + non-BBT

  ,                                                     // Test with fixed ranks
  {0.10, 256, 256, 0.0, 0.0, 100, 5, true, 4321ULL},    // Square + BBT
  {0.12, 2048, 256, 0.0, 0.0, 100, 5, true, 4321ULL},   // Tall + BBT
  {0.10, 256, 256, 0.0, 0.0, 100, 5, false, 4321ULL},   // Square + non-BBT
  {0.12, 2048, 256, 0.0, 0.0, 100, 5, false, 4321ULL},  // Tall + non-BBT
  {0.60, 2048, 2048, 0.0, 0.0, 100, 5, true, 4321ULL},  // Square + BBT
  {1.00, 16384, 2048, 0.0, 0.0, 100, 5, true, 4321ULL},  // Tall + BBT
  {0.60, 2048, 2048, 0.0, 0.0, 100, 5, false, 4321ULL},  // Square + non-BBT
  {1.00, 16384, 2048, 0.0, 0.0, 100, 5, false, 4321ULL}  // Tall + non-BBT
};

const std::vector<RsvdInputs<float>> sanity_inputs_fx = {
  {100000000000000000.0f, 3, 2, 0.2f, 0.05f, 0, 0, true, 4321ULL},
  {100000000000000000.0f, 3, 2, 0.0f, 0.0f, 1, 1, true, 4321ULL},
  {100000000000000000.0f, 3, 2, 0.2f, 0.05f, 0, 0, false, 4321ULL},
  {100000000000000000.0f, 3, 2, 0.0f, 0.0f, 1, 1, false, 4321ULL}};

const std::vector<RsvdInputs<double>> sanity_inputs_dx = {
  {100000000000000000.0, 3, 2, 0.2, 0.05, 0, 0, true, 4321ULL},
  {100000000000000000.0, 3, 2, 0.0, 0.0, 1, 1, true, 4321ULL},
  {100000000000000000.0, 3, 2, 0.2, 0.05, 0, 0, false, 4321ULL},
  {100000000000000000.0, 3, 2, 0.0, 0.0, 1, 1, false, 4321ULL}};

typedef RsvdTest<float> RsvdSanityCheckValF;
TEST_P(RsvdSanityCheckValF, Result) {
  ASSERT_TRUE(devArrMatch(sing_vals_ref, S, params.k,
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef RsvdTest<double> RsvdSanityCheckValD;
TEST_P(RsvdSanityCheckValD, Result) {
  ASSERT_TRUE(devArrMatch(sing_vals_ref, S, params.k,
                          raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef RsvdTest<float> RsvdSanityCheckLeftVecF;
TEST_P(RsvdSanityCheckLeftVecF, Result) {
  ASSERT_TRUE(devArrMatch(left_eig_vectors_ref, U, params.n_row * params.k,
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef RsvdTest<double> RsvdSanityCheckLeftVecD;
TEST_P(RsvdSanityCheckLeftVecD, Result) {
  ASSERT_TRUE(devArrMatch(left_eig_vectors_ref, U, params.n_row * params.k,
                          raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef RsvdTest<float> RsvdSanityCheckRightVecF;
TEST_P(RsvdSanityCheckRightVecF, Result) {
  ASSERT_TRUE(devArrMatch(right_eig_vectors_ref, V, params.n_col * params.k,
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef RsvdTest<double> RsvdSanityCheckRightVecD;
TEST_P(RsvdSanityCheckRightVecD, Result) {
  ASSERT_TRUE(devArrMatch(right_eig_vectors_ref, V, params.n_col * params.k,
                          raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef RsvdTest<float> RsvdTestSquareMatrixNormF;
TEST_P(RsvdTestSquareMatrixNormF, Result) {
  raft::handle_t handle;

  ASSERT_TRUE(raft::linalg::evaluateSVDByL2Norm(
    handle, A, U, S, V, params.n_row, params.n_col, params.k,
    4 * params.tolerance, handle.get_stream()));
}

typedef RsvdTest<double> RsvdTestSquareMatrixNormD;
TEST_P(RsvdTestSquareMatrixNormD, Result) {
  raft::handle_t handle;

  ASSERT_TRUE(raft::linalg::evaluateSVDByL2Norm(
    handle, A, U, S, V, params.n_row, params.n_col, params.k,
    4 * params.tolerance, handle.get_stream()));
}

INSTANTIATE_TEST_CASE_P(RsvdTests, RsvdSanityCheckValF,
                        ::testing::ValuesIn(sanity_inputs_fx));

INSTANTIATE_TEST_CASE_P(RsvdTests, RsvdSanityCheckValD,
                        ::testing::ValuesIn(sanity_inputs_dx));

INSTANTIATE_TEST_CASE_P(RsvdTests, RsvdSanityCheckLeftVecF,
                        ::testing::ValuesIn(sanity_inputs_fx));

INSTANTIATE_TEST_CASE_P(RsvdTests, RsvdSanityCheckLeftVecD,
                        ::testing::ValuesIn(sanity_inputs_dx));

INSTANTIATE_TEST_CASE_P(RsvdTests, RsvdSanityCheckRightVecF,
                        ::testing::ValuesIn(sanity_inputs_fx));

INSTANTIATE_TEST_CASE_P(RsvdTests, RsvdSanityCheckRightVecD,
                        ::testing::ValuesIn(sanity_inputs_dx));

INSTANTIATE_TEST_CASE_P(RsvdTests, RsvdTestSquareMatrixNormF,
                        ::testing::ValuesIn(inputs_fx));

INSTANTIATE_TEST_CASE_P(RsvdTests, RsvdTestSquareMatrixNormD,
                        ::testing::ValuesIn(inputs_dx));

}  // end namespace LinAlg
}  // end namespace MLCommon
