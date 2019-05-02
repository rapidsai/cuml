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
#include "matrix/math.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace Matrix {

template <typename Type>
__global__ void nativePowerKernel(Type *in, Type *out, int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    out[idx] = in[idx] * in[idx];
  }
}

template <typename Type>
void naivePower(Type *in, Type *out, int len, cudaStream_t stream) {
  static const int TPB = 64;
  int nblks = ceildiv(len, TPB);
  nativePowerKernel<Type><<<nblks, TPB, 0, stream>>>(in, out, len);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename Type>
__global__ void nativeSqrtKernel(Type *in, Type *out, int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    out[idx] = sqrt(in[idx]);
  }
}

template <typename Type>
void naiveSqrt(Type *in, Type *out, int len) {
  static const int TPB = 64;
  int nblks = ceildiv(len, TPB);
  nativeSqrtKernel<Type><<<nblks, TPB>>>(in, out, len);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename Type>
__global__ void naiveSignFlipKernel(Type *in, Type *out, int rowCount,
                                     int colCount) {
  int d_i = blockIdx.x * rowCount;
  int end = d_i + rowCount;

  if (blockIdx.x < colCount) {
    Type max = 0.0;
    int max_index = 0;
    for (int i = d_i; i < end; i++) {
      Type val = in[i];
      if (val < 0.0) {
        val = -val;
      }
      if (val > max) {
        max = val;
        max_index = i;
      }
    }


    for (int i = d_i; i < end; i++) {
      if (in[max_index] < 0.0) {
        out[i] = -in[i];
      } else {
        out[i] = in[i];
      }
    }
  }

  __syncthreads();
}

template <typename Type>
void naiveSignFlip(Type *in, Type *out, int rowCount, int colCount) {
  naiveSignFlipKernel<Type><<<colCount, 1>>>(in, out, rowCount, colCount);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
struct MathInputs {
  T tolerance;
  int n_row;
  int n_col;
  int len;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const MathInputs<T> &dims) {
  return os;
}

template <typename T>
class MathTest : public ::testing::TestWithParam<MathInputs<T>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<MathInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int len = params.len;

    allocate(in_power, len);
    allocate(out_power_ref, len);
    allocate(in_sqrt, len);
    allocate(out_sqrt_ref, len);
    allocate(in_sign_flip, len);
    allocate(out_sign_flip_ref, len);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    allocate(in_ratio, 4);
    T in_ratio_h[4] = {1.0, 2.0, 2.0, 3.0};
    updateDevice(in_ratio, in_ratio_h, 4, stream);

    allocate(out_ratio_ref, 4);
    T out_ratio_ref_h[4] = {0.125, 0.25, 0.25, 0.375};
    updateDevice(out_ratio_ref, out_ratio_ref_h, 4, stream);

    r.uniform(in_power, len, T(-1.0), T(1.0), stream);
    r.uniform(in_sqrt, len, T(0.0), T(1.0), stream);
    // r.uniform(in_ratio, len, T(0.0), T(1.0));
    r.uniform(in_sign_flip, len, T(-100.0), T(100.0), stream);

    naivePower(in_power, out_power_ref, len, stream);
    power(in_power, len, stream);

    naiveSqrt(in_sqrt, out_sqrt_ref, len);
    seqRoot(in_sqrt, len, stream);

    auto mgr = makeDefaultAllocator();
    ratio(in_ratio, in_ratio, 4, mgr, stream);

    naiveSignFlip(in_sign_flip, out_sign_flip_ref, params.n_row, params.n_col);
    signFlip(in_sign_flip, params.n_row, params.n_col, stream);

    allocate(in_recip, 4);
    allocate(in_recip_ref, 4);
    allocate(out_recip, 4);
    // default threshold is 1e-15
    std::vector<T> in_recip_h = {0.1, 0.01, -0.01, 0.1e-16};
    std::vector<T> in_recip_ref_h = {10.0, 100.0, -100.0, 0.0};
    updateDevice(in_recip, in_recip_h.data(), 4, stream);
    updateDevice(in_recip_ref, in_recip_ref_h.data(), 4, stream);
    T recip_scalar = T(1.0);

    // this `reciprocal()` has to go first bc next one modifies its input
    reciprocal(in_recip, out_recip, recip_scalar, 4, stream);

    reciprocal(in_recip, recip_scalar, 4, stream, true);

    std::vector<T> in_small_val_zero_h = {0.1, 1e-16, -1e-16, -0.1};
    std::vector<T> in_small_val_zero_ref_h = {0.1, 0.0, 0.0, -0.1};
    allocate(in_smallzero, 4);
    allocate(out_smallzero, 4);
    allocate(out_smallzero_ref, 4);
    updateDevice(in_smallzero, in_small_val_zero_h.data(), 4, stream);
    updateDevice(out_smallzero_ref, in_small_val_zero_ref_h.data(), 4, stream);
    setSmallValuesZero(out_smallzero, in_smallzero, 4, stream);
    setSmallValuesZero(in_smallzero, 4, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));

  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(in_power));
    CUDA_CHECK(cudaFree(out_power_ref));
    CUDA_CHECK(cudaFree(in_sqrt));
    CUDA_CHECK(cudaFree(out_sqrt_ref));
    CUDA_CHECK(cudaFree(in_ratio));
    CUDA_CHECK(cudaFree(out_ratio_ref));
    CUDA_CHECK(cudaFree(in_sign_flip));
    CUDA_CHECK(cudaFree(out_sign_flip_ref));
    CUDA_CHECK(cudaFree(in_recip));
    CUDA_CHECK(cudaFree(in_recip_ref));
    CUDA_CHECK(cudaFree(out_recip));
    CUDA_CHECK(cudaFree(in_smallzero));
    CUDA_CHECK(cudaFree(out_smallzero));
    CUDA_CHECK(cudaFree(out_smallzero_ref));

  }

protected:
  MathInputs<T> params;
  T *in_power, *out_power_ref, *in_sqrt, *out_sqrt_ref, *in_ratio,
    *out_ratio_ref, *in_sign_flip, *out_sign_flip_ref, *in_recip,
    *in_recip_ref, *out_recip, *in_smallzero, *out_smallzero, *out_smallzero_ref;
};

const std::vector<MathInputs<float>> inputsf = {
  {0.00001f, 1024, 1024, 1024 * 1024, 1234ULL}};

const std::vector<MathInputs<double>> inputsd = {
  {0.00001, 1024, 1024, 1024 * 1024, 1234ULL}};

typedef MathTest<float> MathPowerTestF;
TEST_P(MathPowerTestF, Result) {
  ASSERT_TRUE(devArrMatch(in_power, out_power_ref, params.len,
                          CompareApprox<float>(params.tolerance)));
}

typedef MathTest<double> MathPowerTestD;
TEST_P(MathPowerTestD, Result) {
  ASSERT_TRUE(devArrMatch(in_power, out_power_ref, params.len,
                          CompareApprox<double>(params.tolerance)));
}

typedef MathTest<float> MathSqrtTestF;
TEST_P(MathSqrtTestF, Result) {
  ASSERT_TRUE(devArrMatch(in_sqrt, out_sqrt_ref, params.len,
                          CompareApprox<float>(params.tolerance)));
}

typedef MathTest<double> MathSqrtTestD;
TEST_P(MathSqrtTestD, Result) {
  ASSERT_TRUE(devArrMatch(in_sqrt, out_sqrt_ref, params.len,
                          CompareApprox<double>(params.tolerance)));
}

typedef MathTest<float> MathRatioTestF;
TEST_P(MathRatioTestF, Result) {
  ASSERT_TRUE(devArrMatch(in_ratio, out_ratio_ref, 4,
                          CompareApprox<float>(params.tolerance)));
}

typedef MathTest<double> MathRatioTestD;
TEST_P(MathRatioTestD, Result) {
  ASSERT_TRUE(devArrMatch(in_ratio, out_ratio_ref, 4,
                          CompareApprox<double>(params.tolerance)));
}

typedef MathTest<float> MathSignFlipTestF;
TEST_P(MathSignFlipTestF, Result) {
  ASSERT_TRUE(devArrMatch(in_sign_flip, out_sign_flip_ref, params.len,
                          CompareApprox<float>(params.tolerance)));
}

typedef MathTest<double> MathSignFlipTestD;
TEST_P(MathSignFlipTestD, Result) {
  ASSERT_TRUE(devArrMatch(in_sign_flip, out_sign_flip_ref, params.len,
                          CompareApprox<double>(params.tolerance)));
}

typedef MathTest<float> MathReciprocalTestF;
TEST_P(MathReciprocalTestF, Result) {
  ASSERT_TRUE(devArrMatch(in_recip, in_recip_ref, 4,
                          CompareApprox<float>(params.tolerance)));

  // 4-th term tests `setzero=true` functionality, not present in this version of `reciprocal`.
  ASSERT_TRUE(devArrMatch(out_recip, in_recip_ref, 3,
                          CompareApprox<float>(params.tolerance)));
}

typedef MathTest<double> MathReciprocalTestD;
TEST_P(MathReciprocalTestD, Result) {
  ASSERT_TRUE(devArrMatch(in_recip, in_recip_ref, 4,
                          CompareApprox<double>(params.tolerance)));

  // 4-th term tests `setzero=true` functionality, not present in this version of `reciprocal`.
  ASSERT_TRUE(devArrMatch(out_recip, in_recip_ref, 3,
                          CompareApprox<double>(params.tolerance)));
}

typedef MathTest<float> MathSetSmallZeroTestF;
TEST_P(MathSetSmallZeroTestF, Result) {
  ASSERT_TRUE(devArrMatch(in_smallzero, out_smallzero_ref, 4,
                          CompareApprox<float>(params.tolerance)));


  ASSERT_TRUE(devArrMatch(out_smallzero, out_smallzero_ref, 4,
                          CompareApprox<float>(params.tolerance)));
}

typedef MathTest<double> MathSetSmallZeroTestD;
TEST_P(MathSetSmallZeroTestD, Result) {
  ASSERT_TRUE(devArrMatch(in_smallzero, out_smallzero_ref, 4,
                          CompareApprox<double>(params.tolerance)));


  ASSERT_TRUE(devArrMatch(out_smallzero, out_smallzero_ref, 4,
                          CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(MathTests, MathPowerTestF,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(MathTests, MathPowerTestD,
                        ::testing::ValuesIn(inputsd));


INSTANTIATE_TEST_CASE_P(MathTests, MathSqrtTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(MathTests, MathSqrtTestD, ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_CASE_P(MathTests, MathRatioTestF,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(MathTests, MathRatioTestD,
                        ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_CASE_P(MathTests, MathSignFlipTestF,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(MathTests, MathSignFlipTestD,
                        ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_CASE_P(MathTests, MathReciprocalTestF,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(MathTests, MathReciprocalTestD,
                        ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_CASE_P(MathTests, MathSetSmallZeroTestF,
                        ::testing::ValuesIn(inputsf));
INSTANTIATE_TEST_CASE_P(MathTests, MathSetSmallZeroTestD,
                        ::testing::ValuesIn(inputsd));


} // end namespace Matrix
} // end namespace MLCommon
