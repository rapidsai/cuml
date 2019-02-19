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
#include "linalg/norm.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace LinAlg {

template <typename Type>
__global__ void naiveNormKernel(Type *dots, const Type *data, int D, int N,
                                NormType type, bool do_sqrt) {
  Type acc = (Type)0;
  int rowStart = threadIdx.x + blockIdx.x * blockDim.x;
  if (rowStart < N) {
    for (int i = 0; i < D; ++i) {
      if (type == L2Norm) {
        acc += data[rowStart * D + i] * data[rowStart * D + i];
      } else {
        acc += myAbs(data[rowStart * D + i]);
      }
    }
    dots[rowStart] = do_sqrt ? mySqrt(acc) : acc;
  }
}

template <typename Type>
void naiveNorm(Type *dots, const Type *data, int D, int N, NormType type,
               bool do_sqrt) {
  static const int TPB = 64;
  int nblks = ceildiv(N, TPB);
  naiveNormKernel<Type><<<nblks, TPB>>>(dots, data, D, N, type, do_sqrt);
  CUDA_CHECK(cudaPeekAtLastError());
}


template <typename T>
struct NormInputs {
  T tolerance;
  int rows, cols;
  NormType type;
  bool do_sqrt;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const NormInputs<T> &dims) {
  return os;
}

template <typename T>
class NormTest : public ::testing::TestWithParam<NormInputs<T>> {
public:
  void SetUp() override {
    params = ::testing::TestWithParam<NormInputs<T>>::GetParam();
    Random::Rng<T> r(params.seed);
    int rows = params.rows, cols = params.cols;
    int len = rows * cols;
    allocate(data, len);
    allocate(dots_exp, rows);
    allocate(dots_act, rows);
    r.uniform(data, len, -1.f, 1.f);
    naiveNorm(dots_exp, data, cols, rows, params.type, params.do_sqrt);
    if (params.do_sqrt) {
      auto fin_op = [] __device__(T in) { return mySqrt(in); };
      norm(dots_act, data, cols, rows, params.type, fin_op);
    } else {
      norm(dots_act, data, cols, rows, params.type);
    }
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(dots_exp));
    CUDA_CHECK(cudaFree(dots_act));
  }

protected:
  NormInputs<T> params;
  T *data, *dots_exp, *dots_act;
};

const std::vector<NormInputs<float>> inputsf = {
  {0.00001f, 1024, 32, L1Norm, false, 1234ULL},
  {0.00001f, 1024, 64, L1Norm, false, 1234ULL},
  {0.00001f, 1024, 128, L1Norm, false, 1234ULL},
  {0.00001f, 1024, 256, L1Norm, false, 1234ULL},
  {0.00001f, 1024, 32, L2Norm, false, 1234ULL},
  {0.00001f, 1024, 64, L2Norm, false, 1234ULL},
  {0.00001f, 1024, 128, L2Norm, false, 1234ULL},
  {0.00001f, 1024, 256, L2Norm, false, 1234ULL},

  {0.00001f, 1024, 32, L1Norm, true, 1234ULL},
  {0.00001f, 1024, 64, L1Norm, true, 1234ULL},
  {0.00001f, 1024, 128, L1Norm, true, 1234ULL},
  {0.00001f, 1024, 256, L1Norm, true, 1234ULL},
  {0.00001f, 1024, 32, L2Norm, true, 1234ULL},
  {0.00001f, 1024, 64, L2Norm, true, 1234ULL},
  {0.00001f, 1024, 128, L2Norm, true, 1234ULL},
  {0.00001f, 1024, 256, L2Norm, true, 1234ULL}};

const std::vector<NormInputs<double>> inputsd = {
  {0.00000001, 1024, 32, L1Norm, false, 1234ULL},
  {0.00000001, 1024, 64, L1Norm, false, 1234ULL},
  {0.00000001, 1024, 128, L1Norm, false, 1234ULL},
  {0.00000001, 1024, 256, L1Norm, false, 1234ULL},
  {0.00000001, 1024, 32, L2Norm, false, 1234ULL},
  {0.00000001, 1024, 64, L2Norm, false, 1234ULL},
  {0.00000001, 1024, 128, L2Norm, false, 1234ULL},
  {0.00000001, 1024, 256, L2Norm, false, 1234ULL},

  {0.00000001, 1024, 32, L1Norm, true, 1234ULL},
  {0.00000001, 1024, 64, L1Norm, true, 1234ULL},
  {0.00000001, 1024, 128, L1Norm, true, 1234ULL},
  {0.00000001, 1024, 256, L1Norm, true, 1234ULL},
  {0.00000001, 1024, 32, L2Norm, true, 1234ULL},
  {0.00000001, 1024, 64, L2Norm, true, 1234ULL},
  {0.00000001, 1024, 128, L2Norm, true, 1234ULL},
  {0.00000001, 1024, 256, L2Norm, true, 1234ULL}};

typedef NormTest<float> NormTestF;
TEST_P(NormTestF, Result) {
  ASSERT_TRUE(devArrMatch(dots_exp, dots_act, params.rows,
                          CompareApprox<float>(params.tolerance)));
}

typedef NormTest<double> NormTestD;
TEST_P(NormTestD, Result) {
  ASSERT_TRUE(devArrMatch(dots_exp, dots_act, params.rows,
                          CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(NormTests, NormTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(NormTests, NormTestD, ::testing::ValuesIn(inputsd));

} // end namespace LinAlg
} // end namespace MLCommon
