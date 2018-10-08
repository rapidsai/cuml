#include <gtest/gtest.h>
#include "linalg/vector_broadcast.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace Broadcast {

template <typename Type>
__global__ void naiveAddKernel(Type* out, const Type* mat, const Type* vec,
                               int rows, int cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int len = rows * cols;
    int col = idx % cols;
    if(idx < len) {
        out[idx] = mat[idx] + vec[col];
    }
}

template <typename Type>
void naiveAdd(Type* out, const Type* mat, const Type* vec, int rows, int cols) {
    static const int TPB = 64;
    int len = rows * cols;
    int nblks = ceildiv(len, TPB);
    naiveAddKernel<Type><<<nblks,TPB>>>(out, mat, vec, rows, cols);
    CUDA_CHECK(cudaPeekAtLastError());
}


template <typename T>
struct VecBcastInputs {
    T tolerance;
    int rows, cols;
    unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const VecBcastInputs<T>& dims) {
    return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access within its class
template <typename T>
void vectorBroadcastLaunch(T* out, const T* mat, const T* vec, int rows, int cols) {
    vectorBroadcast(out, mat, vec, rows, cols,
                    [] __device__ (T a, T b) {
                        return a + b;
                    });
}

template <typename T>
class VecBcastTest: public ::testing::TestWithParam<VecBcastInputs<T> > {
protected:
    void SetUp() override {
        params = ::testing::TestWithParam<VecBcastInputs<T>>::GetParam();
        Random::Rng<T> r(params.seed);
        int rows = params.rows;
        int cols = params.cols;
        int len = rows * cols;
        allocate(mat, len);
        allocate(vec, cols);
        allocate(out_ref, len);
        allocate(out, len);
        r.uniform(mat, len, T(-1.0), T(1.0));
        r.uniform(vec, cols, T(-1.0), T(1.0));
        naiveAdd(out_ref, mat, vec, rows, cols);
        vectorBroadcastLaunch(out, mat, vec, rows, cols);
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(mat));
        CUDA_CHECK(cudaFree(vec));
        CUDA_CHECK(cudaFree(out_ref));
        CUDA_CHECK(cudaFree(out));
    }

protected:
    VecBcastInputs<T> params;
    T *mat, *vec, *out_ref, *out;
};

const std::vector<VecBcastInputs<float> > inputsf = {
    {0.000001f, 1024, 1024, 1234ULL},
    {0.000001f, 1024,  512, 1234ULL},
    {0.000001f, 1024,  256, 1234ULL},
    {0.000001f, 1024,  128, 1234ULL},
    {0.000001f, 1024,   64, 1234ULL}
};

const std::vector<VecBcastInputs<double> > inputsd = {
    {0.000001, 1024, 1024, 1234ULL},
    {0.000001, 1024,  512, 1234ULL},
    {0.000001, 1024,  256, 1234ULL},
    {0.000001, 1024,  128, 1234ULL},
    {0.000001, 1024,   64, 1234ULL}
};

typedef VecBcastTest<float> VecBcastTestF;
TEST_P(VecBcastTestF, Result) {
    ASSERT_TRUE(devArrMatch(out_ref, out, params.rows*params.cols,
                            CompareApprox<float>(params.tolerance)));
}

typedef VecBcastTest<double> VecBcastTestD;
TEST_P(VecBcastTestD, Result){
    ASSERT_TRUE(devArrMatch(out_ref, out, params.rows*params.cols,
                            CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(VecBcastTests, VecBcastTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(VecBcastTests, VecBcastTestD, ::testing::ValuesIn(inputsd));

} // end namespace Broadcast
} // end namespace MLCommon
