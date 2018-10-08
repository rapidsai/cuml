#include <gtest/gtest.h>
#include "linalg/norm.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace LinAlg {

template <typename Type>
__global__ void naiveNormKernel(Type* dots, const Type* data, int D, int N,
                                NormType type) {
    Type acc = (Type)0;
    int rowStart = threadIdx.x + blockIdx.x * blockDim.x;
    if(rowStart < N) {
        for(int i=0;i<D;++i) {
            if(type == L2Norm) {
                acc += data[rowStart*D+i] * data[rowStart*D+i];
            } else {
                acc += myAbs(data[rowStart*D+i]);
            }
        }
        dots[rowStart] = acc;
    }
}

template <typename Type>
void naiveNorm(Type* dots, const Type* data, int D, int N, NormType type) {
    static const int TPB = 64;
    int nblks = ceildiv(N, TPB);
    naiveNormKernel<Type><<<nblks,TPB>>>(dots, data, D, N, type);
    CUDA_CHECK(cudaPeekAtLastError());
}


template <typename T>
struct NormInputs {
    T tolerance;
    int rows, cols;
    NormType type;
    unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const NormInputs<T>& dims) {
    return os;
}

template <typename T>
class NormTest: public ::testing::TestWithParam<NormInputs<T> > {
protected:
    void SetUp() override {
        params = ::testing::TestWithParam<NormInputs<T>>::GetParam();
        Random::Rng<T> r(params.seed);
        int rows = params.rows, cols = params.cols;
        int len = rows * cols;
        allocate(data, len);
        allocate(dots_exp, rows);
        allocate(dots_act, rows);
        r.uniform(data, len, -1.f, 1.f);
        naiveNorm(dots_exp, data, cols, rows, params.type);
        norm(dots_act, data, cols, rows, params.type);
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

const std::vector<NormInputs<float> > inputsf = {
    {0.000001f, 1024,  32, L1Norm, 1234ULL},
    {0.000001f, 1024,  64, L1Norm, 1234ULL},
    {0.000001f, 1024, 128, L1Norm, 1234ULL},
    {0.000001f, 1024, 256, L1Norm, 1234ULL},
    {0.000001f, 1024,  32, L2Norm, 1234ULL},
    {0.000001f, 1024,  64, L2Norm, 1234ULL},
    {0.000001f, 1024, 128, L2Norm, 1234ULL},
    {0.000001f, 1024, 256, L2Norm, 1234ULL}
};

const std::vector<NormInputs<double> > inputsd = {
    {0.000000001, 1024,  32, L1Norm, 1234ULL},
    {0.000000001, 1024,  64, L1Norm, 1234ULL},
    {0.000000001, 1024, 128, L1Norm, 1234ULL},
    {0.000000001, 1024, 256, L1Norm, 1234ULL},
    {0.000000001, 1024,  32, L2Norm, 1234ULL},
    {0.000000001, 1024,  64, L2Norm, 1234ULL},
    {0.000000001, 1024, 128, L2Norm, 1234ULL},
    {0.000000001, 1024, 256, L2Norm, 1234ULL}
};

typedef NormTest<float> NormTestF;
TEST_P(NormTestF, Result) {
    ASSERT_TRUE(devArrMatch(dots_exp, dots_act, params.rows,
                            CompareApprox<float>(params.tolerance)));
}

typedef NormTest<double> NormTestD;
TEST_P(NormTestD, Result){
    ASSERT_TRUE(devArrMatch(dots_exp, dots_act, params.rows,
                            CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(NormTests, NormTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(NormTests, NormTestD, ::testing::ValuesIn(inputsd));

} // end namespace LinAlg
} // end namespace MLCommon
