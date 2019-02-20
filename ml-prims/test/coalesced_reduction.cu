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

#include <gtest/gtest.h>
#include "cuda_utils.h"
#include "linalg/coalesced_reduction.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace LinAlg {

template <typename Type>
__global__ void naiveReductionKernel(Type *dots, const Type *data, int D, int N) {
  Type acc = (Type)0;
  int rowStart = threadIdx.x + blockIdx.x * blockDim.x;
  if (rowStart < N) {
    for (int i = 0; i < D; ++i) {
        acc += data[rowStart * D + i] * data[rowStart * D + i];
    }
    dots[rowStart] = 2*acc;
  }
}

template <typename Type>
void naiveReduction(Type *dots, const Type *data, int D, int N) {
  static const int TPB = 64;
  int nblks = ceildiv(N, TPB);
  naiveReductionKernel<Type><<<nblks, TPB>>>(dots, data, D, N);
  CUDA_CHECK(cudaPeekAtLastError());
}


template <typename T>
struct coalescedReductionInputs {
  T tolerance;
  int rows, cols;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const coalescedReductionInputs<T> &dims) {
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename T>
void coalescedReductionLaunch(T *dots, const T *data, int cols, int rows,
                              bool inplace = false) {
  coalescedReduction(dots, data, cols, rows, (T)0,
                     inplace, 0,
                     [] __device__(T in) { return in * in; });
}

template <typename T>
class coalescedReductionTest : public ::testing::TestWithParam<coalescedReductionInputs<T>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<coalescedReductionInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int rows = params.rows, cols = params.cols;
    int len = rows * cols;
    allocate(data, len);
    allocate(dots_exp, rows);
    allocate(dots_act, rows);
    r.uniform(data, len, T(-1.0), T(1.0));
    naiveReduction(dots_exp, data, cols, rows);

    // Perform reduction with default inplace = false first
    coalescedReductionLaunch(dots_act, data, cols, rows);
    // Add to result with inplace = true next
    coalescedReductionLaunch(dots_act, data, cols, rows, true);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(dots_exp));
    CUDA_CHECK(cudaFree(dots_act));
  }

protected:
  coalescedReductionInputs<T> params;
  T *data, *dots_exp, *dots_act;
};

const std::vector<coalescedReductionInputs<float>> inputsf = {
  {0.000002f, 1024,  32, 1234ULL},
  {0.000002f, 1024,  64, 1234ULL},
  {0.000002f, 1024, 128, 1234ULL},
  {0.000002f, 1024, 256, 1234ULL},
  {0.000002f, 1024,  32, 1234ULL},
  {0.000002f, 1024,  64, 1234ULL},
  {0.000002f, 1024, 128, 1234ULL},
  {0.000002f, 1024, 256, 1234ULL}};

const std::vector<coalescedReductionInputs<double>> inputsd = {
  {0.000000001, 1024,  32, 1234ULL},
  {0.000000001, 1024,  64, 1234ULL},
  {0.000000001, 1024, 128, 1234ULL},
  {0.000000001, 1024, 256, 1234ULL},
  {0.000000001, 1024,  32, 1234ULL},
  {0.000000001, 1024,  64, 1234ULL},
  {0.000000001, 1024, 128, 1234ULL},
  {0.000000001, 1024, 256, 1234ULL}};

typedef coalescedReductionTest<float> coalescedReductionTestF;
TEST_P(coalescedReductionTestF, Result) {
  ASSERT_TRUE(devArrMatch(dots_exp, dots_act, params.rows,
                          CompareApprox<float>(params.tolerance)));
}

typedef coalescedReductionTest<double> coalescedReductionTestD;
TEST_P(coalescedReductionTestD, Result) {
  ASSERT_TRUE(devArrMatch(dots_exp, dots_act, params.rows,
                          CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(coalescedReductionTests, coalescedReductionTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(coalescedReductionTests, coalescedReductionTestD, ::testing::ValuesIn(inputsd));

} // end namespace LinAlg
} // end namespace MLCommon
