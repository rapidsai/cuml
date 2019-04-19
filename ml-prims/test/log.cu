#include <gtest/gtest.h>
#include "functions/log.h"
#include "test_utils.h"
#include "cuda_utils.h"


namespace MLCommon {
namespace Functions {

template <typename T>
struct LogInputs {
    T tolerance;
    int len;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const LogInputs<T>& dims) {
    return os;
}

template <typename T>
class LogTest: public ::testing::TestWithParam<LogInputs<T> > {
protected:
    void SetUp() override {
        params = ::testing::TestWithParam<LogInputs<T>>::GetParam();
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        int len = params.len;

        allocate(data, len);
        T data_h[params.len] = { 2.1, 4.5, 0.34, 10.0 };
        updateDevice(data, data_h, len);

        allocate(result, len);
        allocate(result_ref, len);
        T result_ref_h[params.len] = { 0.74193734, 1.5040774, -1.07880966, 2.30258509 };
        updateDevice(result_ref, result_ref_h, len);

        f_log(result, data, T(1), len, stream);
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(data));
        CUDA_CHECK(cudaFree(result));
        CUDA_CHECK(cudaFree(result_ref));
    }

protected:
    LogInputs<T> params;
    T *data, *result, *result_ref;
};

const std::vector<LogInputs<float> > inputsf2 = {
    {0.001f, 4}
};

const std::vector<LogInputs<double> > inputsd2 = {
    {0.001, 4}
};

typedef LogTest<float> LogTestValF;
TEST_P(LogTestValF, Result) {
	ASSERT_TRUE(devArrMatch(result_ref, result, params.len,
			                CompareApproxAbs<float>(params.tolerance)));
}

typedef LogTest<double> LogTestValD;
TEST_P(LogTestValD, Result){
	ASSERT_TRUE(devArrMatch(result_ref, result, params.len,
				            CompareApproxAbs<double>(params.tolerance)));
}


INSTANTIATE_TEST_CASE_P(LogTests, LogTestValF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(LogTests, LogTestValD,
                        ::testing::ValuesIn(inputsd2));



} // end namespace Functions
} // end namespace MLCommon
