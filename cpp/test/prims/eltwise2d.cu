/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include "test_utils.h"

#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>
#include <linalg/eltwise2d.cuh>

namespace MLCommon {
namespace LinAlg {

template <typename Type>
CUML_KERNEL void naiveEltwise2DAddKernel(int rows,
                                         int cols,
                                         const Type* aPtr,
                                         const Type* bPtr,
                                         const Type* cPtr,
                                         Type* dPtr,
                                         Type alpha,
                                         Type beta)
{
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < cols * rows) {
    const auto x = tid % cols;
    const auto y = tid / cols;
    const auto d = dPtr[tid];
    const auto a = aPtr[y];
    const auto b = bPtr[x];
    Type accm    = alpha * (a + b + d);

    if (beta) { accm += beta * cPtr[tid]; }
    dPtr[tid] = accm;
  }
}

template <typename Type>
void naiveEltwise2DAdd(int rows,
                       int cols,
                       const Type* aPtr,
                       const Type* bPtr,
                       const Type* cPtr,
                       Type* dPtr,
                       Type alpha,
                       Type beta,
                       cudaStream_t stream)
{
  static const int TPB = 64;
  int nblks            = raft::ceildiv(rows * cols, TPB);
  naiveEltwise2DAddKernel<Type>
    <<<nblks, TPB, 0, stream>>>(rows, cols, aPtr, bPtr, cPtr, dPtr, alpha, beta);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T>
struct Eltwise2dInputs {
  T tolerance;
  int w;
  int h;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const Eltwise2dInputs<T>& dims)
{
  return os;
}

template <typename Type>
void WrapperEltwise2d(int rows,
                      int cols,
                      const Type* aPtr,
                      const Type* bPtr,
                      const Type* cPtr,
                      Type* dPtr,
                      Type alpha,
                      Type beta)
{
  auto op_ = [] __device__(Type a, Type b, Type c) { return a + b + c; };
  eltwise2D<Type>(rows, cols, aPtr, bPtr, cPtr, dPtr, alpha, beta, op_, 0);
}

template <typename T>
class Eltwise2dTest : public ::testing::TestWithParam<Eltwise2dInputs<T>> {
 protected:
  Eltwise2dTest() : out_ref(0, stream), out(0, stream) {}

  void SetUp() override
  {
    params = ::testing::TestWithParam<Eltwise2dInputs<T>>::GetParam();
    raft::random::Rng r(params.seed);
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));
    auto w   = params.w;
    auto h   = params.h;
    auto len = w * h;
    rmm::device_uvector<T> in1(h, stream);
    rmm::device_uvector<T> in2(w, stream);
    out_ref.resize(len, stream);
    out.resize(len, stream);
    r.uniform(in1.data(), h, T(-1.0), T(1.0), stream);
    r.uniform(in2.data(), w, T(-1.0), T(1.0), stream);

    naiveEltwise2DAdd(
      h, w, in1.data(), in2.data(), out_ref.data(), out_ref.data(), (T)1, (T)1, stream);
    WrapperEltwise2d<T>(h, w, in1.data(), in2.data(), out.data(), out.data(), (T)1, (T)1);
    RAFT_CUDA_TRY(cudaStreamDestroy(stream));
  }

 protected:
  cudaStream_t stream = 0;
  Eltwise2dInputs<T> params;
  rmm::device_uvector<T> out_ref, out;
};

const std::vector<Eltwise2dInputs<float>> inputsf2 = {{0.000001f, 1024, 1024, 1234ULL}};

const std::vector<Eltwise2dInputs<double>> inputsd2 = {{0.00000001, 1024, 1024, 1234ULL}};

typedef Eltwise2dTest<float> Eltwise2dTestF;
TEST_P(Eltwise2dTestF, Result)
{
  ASSERT_TRUE(MLCommon::devArrMatch(out_ref.data(),
                                    out.data(),
                                    params.w * params.h,
                                    MLCommon::CompareApprox<float>(params.tolerance)));
}

typedef Eltwise2dTest<double> Eltwise2dTestD;
TEST_P(Eltwise2dTestD, Result)
{
  ASSERT_TRUE(MLCommon::devArrMatch(out_ref.data(),
                                    out.data(),
                                    params.w * params.h,
                                    MLCommon::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(Eltwise2dTests, Eltwise2dTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(Eltwise2dTests, Eltwise2dTestD, ::testing::ValuesIn(inputsd2));

}  // end namespace LinAlg
}  // end namespace MLCommon
