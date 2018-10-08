#include <gtest/gtest.h>
#include "stats/mean.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace Stats {

template <typename T>
struct MeanInputs {
    T tolerance, mean;
    int rows, cols;
    bool sample, rowMajor;
    unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const MeanInputs<T>& dims) {
    return os;
}

template <typename T>
class MeanTest: public ::testing::TestWithParam<MeanInputs<T> > {
protected:
    void SetUp() override {
        params = ::testing::TestWithParam<MeanInputs<T>>::GetParam();
        Random::Rng<T> r(params.seed);
        int rows = params.rows, cols = params.cols;
        int len = rows * cols;
        allocate(data, len);
        allocate(mean_act, cols);
        r.normal(data, len, params.mean, (T)1.0);
        mean(mean_act, data, cols, rows, params.sample, params.rowMajor);
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(data));
        CUDA_CHECK(cudaFree(mean_act));
    }

protected:
    MeanInputs<T> params;
    T *data, *mean_act;
};

const std::vector<MeanInputs<float> > inputsf = {
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

const std::vector<MeanInputs<double> > inputsd = {
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

typedef MeanTest<float> MeanTestF;
TEST_P(MeanTestF, Result) {
    ASSERT_TRUE(devArrMatch(params.mean, mean_act, params.cols,
                            CompareApprox<float>(params.tolerance)));
}

typedef MeanTest<double> MeanTestD;
TEST_P(MeanTestD, Result){
    ASSERT_TRUE(devArrMatch(params.mean, mean_act, params.cols,
                            CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(MeanTests, MeanTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(MeanTests, MeanTestD, ::testing::ValuesIn(inputsd));

} // end namespace Stats
} // end namespace MLCommon
