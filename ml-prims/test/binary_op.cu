#include <gtest/gtest.h>
#include "linalg/binary_op.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace LinAlg {

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
struct BinaryOpInputs {
    T tolerance;
    int len;
    unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const BinaryOpInputs<T>& dims) {
    return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access within its class
template <typename T>
void binaryOpLaunch(T* out, const T* in1, const T* in2, int len) {
    binaryOp(out, in1, in2, len,
             [] __device__ (T a, T b) {
                 return a + b;
            });
}

template <typename T>
class BinaryOpTest: public ::testing::TestWithParam<BinaryOpInputs<T> > {
protected:
    void SetUp() override {
        params = ::testing::TestWithParam<BinaryOpInputs<T>>::GetParam();
        Random::Rng<T> r(params.seed);
        int len = params.len;
        allocate(in1, len);
        allocate(in2, len);
        allocate(out_ref, len);
        allocate(out, len);
        r.uniform(in1, len, T(-1.0), T(1.0));
        r.uniform(in2, len, T(-1.0), T(1.0));
        naiveAdd(out_ref, in1, in2, len);
        binaryOpLaunch(out, in1, in2, len);
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(in1));
        CUDA_CHECK(cudaFree(in2));
        CUDA_CHECK(cudaFree(out_ref));
        CUDA_CHECK(cudaFree(out));
    }

protected:
    BinaryOpInputs<T> params;
    T *in1, *in2, *out_ref, *out;
};

const std::vector<BinaryOpInputs<float> > inputsf = {
    {0.000001f, 1024*1024, 1234ULL}
};

const std::vector<BinaryOpInputs<double> > inputsd = {
    {0.00000001, 1024*1024, 1234ULL}
};

typedef BinaryOpTest<float> BinaryOpTestF;
TEST_P(BinaryOpTestF, Result) {
    ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                            CompareApprox<float>(params.tolerance)));
}

typedef BinaryOpTest<double> BinaryOpTestD;
TEST_P(BinaryOpTestD, Result){
    ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                            CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(BinaryOpTests, BinaryOpTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(BinaryOpTests, BinaryOpTestD, ::testing::ValuesIn(inputsd));

} // end namespace LinAlg
} // end namespace MLCommon
