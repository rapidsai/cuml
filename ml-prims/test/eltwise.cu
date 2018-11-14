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
#include "linalg/eltwise.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace LinAlg {

//// Testing unary ops

template <typename Type>
__global__ void naiveScaleKernel(Type* out, const Type* in, Type scalar, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len) {
        out[idx] = scalar * in[idx];
    }
}

template <typename Type>
void naiveScale(Type* out, const Type* in, Type scalar, int len) {
    static const int TPB = 64;
    int nblks = ceildiv(len, TPB);
    naiveScaleKernel<Type><<<nblks,TPB>>>(out, in, scalar, len);
    CUDA_CHECK(cudaPeekAtLastError());
}


template <typename T>
struct ScalarMultiplyInputs {
    T tolerance;
    int len;
    T scalar;
    unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os,
                           const ScalarMultiplyInputs<T>& dims) {
    return os;
}

template <typename T>
class ScalarMultiplyTest: public ::testing::TestWithParam<ScalarMultiplyInputs<T> > {
protected:
    void SetUp() override {
        params = ::testing::TestWithParam<ScalarMultiplyInputs<T>>::GetParam();
        Random::Rng<T> r(params.seed);
        int len = params.len;
        T scalar = params.scalar;
        allocate(in, len);
        allocate(out_ref, len);
        allocate(out, len);
        r.uniform(in, len, T(-1.0), T(1.0));
        naiveScale(out_ref, in, scalar, len);
        scalarMultiply(out, in, scalar, len);
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(in));
        CUDA_CHECK(cudaFree(out_ref));
        CUDA_CHECK(cudaFree(out));
    }

protected:
    ScalarMultiplyInputs<T> params;
    T *in, *out_ref, *out;
};

const std::vector<ScalarMultiplyInputs<float> > inputsf1 = {
    {0.000001f, 1024*1024, 2.f, 1234ULL}
};

const std::vector<ScalarMultiplyInputs<double> > inputsd1 = {
    {0.00000001, 1024*1024, 2.0, 1234ULL}
};

typedef ScalarMultiplyTest<float> ScalarMultiplyTestF;
TEST_P(ScalarMultiplyTestF, Result) {
    ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                            CompareApprox<float>(params.tolerance)));
}

typedef ScalarMultiplyTest<double> ScalarMultiplyTestD;
TEST_P(ScalarMultiplyTestD, Result){
    ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                            CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(ScalarMultiplyTests, ScalarMultiplyTestF,
                        ::testing::ValuesIn(inputsf1));

INSTANTIATE_TEST_CASE_P(ScalarMultiplyTests, ScalarMultiplyTestD,
                        ::testing::ValuesIn(inputsd1));


//// Testing binary ops

template <typename Type>
__global__ void naiveAddKernel(Type* out, const Type* in1, const Type* in2,
                               int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len) {
        out[idx] = in1[idx] + in2[idx];
    }
}

template <typename Type>
void naiveAdd(Type* out, const Type* in1, const Type* in2, int len) {
    static const int TPB = 64;
    int nblks = ceildiv(len, TPB);
    naiveAddKernel<Type><<<nblks,TPB>>>(out, in1, in2, len);
    CUDA_CHECK(cudaPeekAtLastError());
}


template <typename T>
struct EltwiseAddInputs {
    T tolerance;
    int len;
    unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const EltwiseAddInputs<T>& dims) {
    return os;
}

template <typename T>
class EltwiseAddTest: public ::testing::TestWithParam<EltwiseAddInputs<T> > {
protected:
    void SetUp() override {
        params = ::testing::TestWithParam<EltwiseAddInputs<T>>::GetParam();
        Random::Rng<T> r(params.seed);
        int len = params.len;
        allocate(in1, len);
        allocate(in2, len);
        allocate(out_ref, len);
        allocate(out, len);
        r.uniform(in1, len, T(-1.0), T(1.0));
        r.uniform(in2, len, T(-1.0), T(1.0));
        naiveAdd(out_ref, in1, in2, len);
        eltwiseAdd(out, in1, in2, len);
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(in1));
        CUDA_CHECK(cudaFree(in2));
        CUDA_CHECK(cudaFree(out_ref));
        CUDA_CHECK(cudaFree(out));
    }

protected:
    EltwiseAddInputs<T> params;
    T *in1, *in2, *out_ref, *out;
};

const std::vector<EltwiseAddInputs<float> > inputsf2 = {
    {0.000001f, 1024*1024, 1234ULL}
};

const std::vector<EltwiseAddInputs<double> > inputsd2 = {
    {0.00000001, 1024*1024, 1234ULL}
};

typedef EltwiseAddTest<float> EltwiseAddTestF;
TEST_P(EltwiseAddTestF, Result) {
    ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                            CompareApprox<float>(params.tolerance)));
}

typedef EltwiseAddTest<double> EltwiseAddTestD;
TEST_P(EltwiseAddTestD, Result){
    ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                            CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(EltwiseAddTests, EltwiseAddTestF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(EltwiseAddTests, EltwiseAddTestD,
                        ::testing::ValuesIn(inputsd2));

} // end namespace LinAlg
} // end namespace MLCommon
