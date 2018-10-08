#include <gtest/gtest.h>
#include "stats/mean.h"
#include "stats/mean_center.h"
#include "random/rng.h"
#include "test_utils.h"
#include "matrix/math.h"


namespace MLCommon {
namespace Stats {

template <typename T>
struct MeanCenterInputs {
    T tolerance, mean;
    int rows, cols;
    bool sample, rowMajor;
    unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const MeanCenterInputs<T>& dims) {
    return os;
}

template <typename T>
class MeanCenterTest: public ::testing::TestWithParam<MeanCenterInputs<T> > {
protected:
    void SetUp() override {
        params = ::testing::TestWithParam<MeanCenterInputs<T>>::GetParam();
        Random::Rng<T> r(params.seed);
        int rows = params.rows, cols = params.cols;
        int len = rows * cols;
        allocate(data, len);
        allocate(mean_act, cols);
        r.normal(data, len, params.mean, (T)1.0);
        mean(mean_act, data, cols, rows, params.sample, params.rowMajor);
        meanCenter(data, mean_act, cols, rows, params.rowMajor);

        mean(mean_act, data, cols, rows, params.sample, params.rowMajor);
        Matrix::matrixVectorBinarySub(data, mean_act, rows, cols, false);
        mean(mean_act, data, cols, rows, params.sample, params.rowMajor);
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(data));
        CUDA_CHECK(cudaFree(mean_act));
    }

protected:
    MeanCenterInputs<T> params;
    T *data, *mean_act;
};

const std::vector<MeanCenterInputs<float> > inputsf = {
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

const std::vector<MeanCenterInputs<double> > inputsd = {
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

typedef MeanCenterTest<float> MeanCenterTestF;
TEST_P(MeanCenterTestF, Result) {
    ASSERT_TRUE(devArrMatch(0.f, mean_act, params.cols,
                            CompareApprox<float>(params.tolerance)));
}

typedef MeanCenterTest<double> MeanCenterTestD;
TEST_P(MeanCenterTestD, Result){
    ASSERT_TRUE(devArrMatch(0.0, mean_act, params.cols,
                            CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(MeanCenterTests, MeanCenterTestF,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(MeanCenterTests, MeanCenterTestD,
                        ::testing::ValuesIn(inputsd));

} // end namespace Stats
} // end namespace MLCommon
