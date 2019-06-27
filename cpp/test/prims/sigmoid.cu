#include <gtest/gtest.h>
#include "cuda_utils.h"
#include "functions/sigmoid.h"
#include "test_utils.h"

namespace MLCommon {
namespace Functions {

template <typename T>
struct SigmoidInputs {
  T tolerance;
  int len;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const SigmoidInputs<T>& dims) {
  return os;
}

template <typename T>
class SigmoidTest : public ::testing::TestWithParam<SigmoidInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<SigmoidInputs<T>>::GetParam();

    int len = params.len;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    allocate(data, len);
    T data_h[params.len] = {2.1, -4.5, -0.34, 10.0};
    updateDevice(data, data_h, len, stream);

    allocate(result, len);
    allocate(result_ref, len);
    T result_ref_h[params.len] = {0.89090318, 0.01098694, 0.41580948,
                                  0.9999546};
    updateDevice(result_ref, result_ref_h, len, stream);

    sigmoid(result, data, len, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(result));
    CUDA_CHECK(cudaFree(result_ref));
  }

 protected:
  SigmoidInputs<T> params;
  T *data, *result, *result_ref;
};

const std::vector<SigmoidInputs<float>> inputsf2 = {{0.001f, 4}};

const std::vector<SigmoidInputs<double>> inputsd2 = {{0.001, 4}};

typedef SigmoidTest<float> SigmoidTestValF;
TEST_P(SigmoidTestValF, Result) {
  ASSERT_TRUE(devArrMatch(result_ref, result, params.len,
                          CompareApproxAbs<float>(params.tolerance)));
}

typedef SigmoidTest<double> SigmoidTestValD;
TEST_P(SigmoidTestValD, Result) {
  ASSERT_TRUE(devArrMatch(result_ref, result, params.len,
                          CompareApproxAbs<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(SigmoidTests, SigmoidTestValF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(SigmoidTests, SigmoidTestValD,
                        ::testing::ValuesIn(inputsd2));

}  // end namespace Functions
}  // end namespace MLCommon
