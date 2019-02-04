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
#include "linalg/strided_reduction.h"
#include "linalg/unary_op.h"
#include "random/rng.h"
#include "test_utils.h"

#include <thrust/device_vector.h>
#include <cublas_v2.h>

namespace MLCommon {
namespace LinAlg {

template <typename T>
struct stridedReductionInputs {
    T tolerance;
    int rows, cols;
    unsigned long long int seed;
};

template <typename T>
void stridedReductionLaunch(T *dots, const T *data, int cols, int rows) {
  stridedReduction(dots, data, cols, rows, (T)0, false, 0,
                   [] __device__(T in) { return in * in; });
}


template <typename T, typename GEMV_t>
void unaryAndGemv(T *dots, const T *data, int cols, int rows, GEMV_t gemv){
    //computes a MLCommon unary op on data (squares it), then computes Ax
    //(A input matrix and x column vector) to sum columns
    thrust::device_vector<T> sq(cols*rows);
    unaryOp(thrust::raw_pointer_cast(sq.data()), data, cols*rows,
            [] __device__(T v) { return v*v; });

    cublasHandle_t handle;
    ASSERT_TRUE(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);

    thrust::device_vector<T> ones(rows, 1); //column vector [1...1]
    T alpha = 1, beta = 0;
    ASSERT_TRUE(gemv(handle, CUBLAS_OP_N, cols, rows,
                &alpha, thrust::raw_pointer_cast(sq.data()), cols,
                thrust::raw_pointer_cast(ones.data()), 1, &beta, 
                dots, 1) == CUBLAS_STATUS_SUCCESS);
}

void unaryAndGemv(float *dots, const float *data, int cols, int rows){
    unaryAndGemv(dots, data, cols, rows, cublasSgemv);
}

void unaryAndGemv(double *dots, const double *data, int cols, int rows){
    unaryAndGemv(dots, data, cols, rows, cublasDgemv);
}


template <typename T>
class stridedReductionTest : public ::testing::TestWithParam<stridedReductionInputs<T>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<stridedReductionInputs<T>>::GetParam();
    Random::Rng<T> r(params.seed);
    int rows = params.rows, cols = params.cols;
    int len = rows*cols;

    allocate(data, len);
    allocate(dots_exp, cols); //expected dot products (from test)
    allocate(dots_act, cols); //actual dot products (from prim)
    r.uniform(data, len, -1.f, 1.f); //initialize matrix to random

    unaryAndGemv(dots_exp, data, cols, rows);
    stridedReductionLaunch(dots_act, data, cols, rows);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(dots_exp));
    CUDA_CHECK(cudaFree(dots_act));
  }

protected:
  stridedReductionInputs<T> params;
  T *data, *dots_exp, *dots_act;
};


const std::vector<stridedReductionInputs<float>> inputsf = {
  {0.00001f, 1024,  32, 1234ULL},
  {0.00001f, 1024,  64, 1234ULL},
  {0.00001f, 1024, 128, 1234ULL},
  {0.00001f, 1024, 256, 1234ULL},
  {0.00001f, 1024,  32, 1234ULL},
  {0.00001f, 1024,  64, 1234ULL},
  {0.00001f, 1024, 128, 1234ULL},
  {0.00001f, 1024, 256, 1234ULL}
};

const std::vector<stridedReductionInputs<double>> inputsd = {
  {0.000000001, 1024,  32, 1234ULL},
  {0.000000001, 1024,  64, 1234ULL},
  {0.000000001, 1024, 128, 1234ULL},
  {0.000000001, 1024, 256, 1234ULL},
  {0.000000001, 1024,  32, 1234ULL},
  {0.000000001, 1024,  64, 1234ULL},
  {0.000000001, 1024, 128, 1234ULL},
  {0.000000001, 1024, 256, 1234ULL}
};

typedef stridedReductionTest<float> stridedReductionTestF;
TEST_P(stridedReductionTestF, Result) {
  ASSERT_TRUE(devArrMatch(dots_exp, dots_act, params.cols,
                          CompareApprox<float>(params.tolerance)));
}

typedef stridedReductionTest<double> stridedReductionTestD;
TEST_P(stridedReductionTestD, Result) {
  ASSERT_TRUE(devArrMatch(dots_exp, dots_act, params.cols,
                          CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(stridedReductionTests, stridedReductionTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(stridedReductionTests, stridedReductionTestD, ::testing::ValuesIn(inputsd));

} // end namespace LinAlg
} // end namespace MLCommon
