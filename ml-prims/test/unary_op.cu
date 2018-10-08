#include <gtest/gtest.h>
#include "linalg/unary_op.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace LinAlg {

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
struct UnaryOpInputs {
    T tolerance;
    int len;
    T scalar;
    unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const UnaryOpInputs<T>& dims) {
    return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access within its class
template <typename T>
void unaryOpLaunch(T* out, const T* in, T scalar, int len) {
    unaryOp(out, in, scalar, len,
            [] __device__ (T in, T scalar) {
                return in * scalar;
            });
}

template <typename T>
class UnaryOpTest: public ::testing::TestWithParam<UnaryOpInputs<T> > {
protected:
    void SetUp() override {
        params = ::testing::TestWithParam<UnaryOpInputs<T>>::GetParam();
        Random::Rng<T> r(params.seed);
        int len = params.len;
        T scalar = params.scalar;
        allocate(in, len);
        allocate(out_ref, len);
        allocate(out, len);
        r.uniform(in, len, T(-1.0), T(1.0));
        naiveScale(out_ref, in, scalar, len);
        unaryOpLaunch(out, in, scalar, len);
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(in));
        CUDA_CHECK(cudaFree(out_ref));
        CUDA_CHECK(cudaFree(out));
    }

protected:
    UnaryOpInputs<T> params;
    T *in, *out_ref, *out;
};

const std::vector<UnaryOpInputs<float> > inputsf = {
    {0.000001f, 1024*1024, 2.f, 1234ULL}
};

const std::vector<UnaryOpInputs<double> > inputsd = {
    {0.00000001, 1024*1024, 2.0, 1234ULL}
};

typedef UnaryOpTest<float> UnaryOpTestF;
TEST_P(UnaryOpTestF, Result) {
    ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                            CompareApprox<float>(params.tolerance)));
}

typedef UnaryOpTest<double> UnaryOpTestD;
TEST_P(UnaryOpTestD, Result){
    ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                            CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(UnaryOpTests, UnaryOpTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(UnaryOpTests, UnaryOpTestD, ::testing::ValuesIn(inputsd));

} // end namespace LinAlg
} // end namespace MLCommon
