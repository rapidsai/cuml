#include <gtest/gtest.h>
#include "stats/mean.h"
#include "linalg/matrix_vector_op.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace LinAlg {

template <typename T>
struct MVInputs {
    T tolerance, mean;
    int rows, cols;
    bool sample, rowMajor;
    unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const MVInputs<T>& dims) {
    return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access within its class
template <typename T>
void matrixVectorOpLaunch(T* data, const T* mu, int cols, int rows, bool rowMajor) {
	matrixVectorOp(data, mu, cols, rows, rowMajor,
	        		       [] __device__ (T a, T b) {
	        		                 return a - b;
	        		            });
}

template <typename T>
class MVOpTest: public ::testing::TestWithParam<MVInputs<T> > {
protected:
    void SetUp() override {
        params = ::testing::TestWithParam<MVInputs<T>>::GetParam();
        Random::Rng<T> r(params.seed);
        int rows = params.rows, cols = params.cols;
        int len = rows * cols;
        allocate(data, len);
        allocate(mean_act, cols);
        r.normal(data, len, params.mean, (T)1.0);
        Stats::mean(mean_act, data, cols, rows, params.sample, params.rowMajor);
        matrixVectorOpLaunch(data, mean_act, cols, rows, params.rowMajor);
        Stats::mean(mean_act, data, cols, rows, params.sample, params.rowMajor);
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(data));
        CUDA_CHECK(cudaFree(mean_act));
    }

protected:
    MVInputs<T> params;
    T *data, *mean_act;
};

const std::vector<MVInputs<float> > inputsf = {
    {0.05f,  1.f, 1024,  32,  true, false, 1234ULL},
    {0.05f,  1.f, 1024,  64,  true, false, 1234ULL},
    {0.05f,  1.f, 1024, 128,  true, false, 1234ULL},
    {0.05f,  1.f, 1024, 256,  true, false, 1234ULL},
    {0.05f, -1.f, 1024,  32, false, false, 1234ULL},
    {0.05f, -1.f, 1024,  64, false, false, 1234ULL},
    {0.05f, -1.f, 1024, 128, false, false, 1234ULL},
    {0.05f, -1.f, 1024, 256, false, false, 1234ULL},
    {0.05f,  1.f, 1024,  32,  true,  true, 1234ULL},
    {0.05f,  1.f, 1024,  64,  true,  true, 1234ULL},
    {0.05f,  1.f, 1024, 128,  true,  true, 1234ULL},
    {0.05f,  1.f, 1024, 256,  true,  true, 1234ULL},
    {0.05f, -1.f, 1024,  32, false,  true, 1234ULL},
    {0.05f, -1.f, 1024,  64, false,  true, 1234ULL},
    {0.05f, -1.f, 1024, 128, false,  true, 1234ULL},
    {0.05f, -1.f, 1024, 256, false,  true, 1234ULL}
};

const std::vector<MVInputs<double> > inputsd = {
    {0.05,  1.0, 1024,  32,  true, false, 1234ULL},
    {0.05,  1.0, 1024,  64,  true, false, 1234ULL},
    {0.05,  1.0, 1024, 128,  true, false, 1234ULL},
    {0.05,  1.0, 1024, 256,  true, false, 1234ULL},
    {0.05, -1.0, 1024,  32, false, false, 1234ULL},
    {0.05, -1.0, 1024,  64, false, false, 1234ULL},
    {0.05, -1.0, 1024, 128, false, false, 1234ULL},
    {0.05, -1.0, 1024, 256, false, false, 1234ULL},
    {0.05,  1.0, 1024,  32,  true,  true, 1234ULL},
    {0.05,  1.0, 1024,  64,  true,  true, 1234ULL},
    {0.05,  1.0, 1024, 128,  true,  true, 1234ULL},
    {0.05,  1.0, 1024, 256,  true,  true, 1234ULL},
    {0.05, -1.0, 1024,  32, false,  true, 1234ULL},
    {0.05, -1.0, 1024,  64, false,  true, 1234ULL},
    {0.05, -1.0, 1024, 128, false,  true, 1234ULL},
    {0.05, -1.0, 1024, 256, false,  true, 1234ULL}
};

typedef MVOpTest<float> MVOpTestF;
TEST_P(MVOpTestF, Result) {
    ASSERT_TRUE(devArrMatch(0.f, mean_act, params.cols,
                            CompareApprox<float>(params.tolerance)));
}

typedef MVOpTest<double> MVOpTestD;
TEST_P(MVOpTestD, Result){
    ASSERT_TRUE(devArrMatch(0.0, mean_act, params.cols,
                            CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(MVOpTests, MVOpTestF,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(MVOpTests, MVOpTestD,
                        ::testing::ValuesIn(inputsd));

} // end namespace Stats
} // end namespace MLCommon
