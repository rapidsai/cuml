/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <cuml/common/functional.hpp>
#include <cuml/common/utils.hpp>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/datasets/make_regression.hpp>
#include <cuml/svm/linear.hpp>

#include <raft/core/handle.hpp>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/rng.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>
#include <test_utils.h>

#include <cmath>

namespace ML {
namespace SVM {

struct LinearSVMTestParams {
  int nRowsTrain;
  int nRowsTest;
  int nCols;
  /** nClasses == 1 implies regression. */
  int nClasses;
  /** Standard deviation of clusters or noise. */
  double errStd;
  double bias;
  double tolerance;
  uint64_t seed;
  LinearSVMParams modelParams;
};

template <typename T, typename ParamsReader>
struct LinearSVMTest : public ::testing::TestWithParam<typename ParamsReader::Params> {
  const LinearSVMTestParams params;
  const raft::handle_t handle;
  cudaStream_t stream;

  LinearSVMTest()
    : testing::TestWithParam<typename ParamsReader::Params>(),
      params(
        ParamsReader::read(::testing::TestWithParam<typename ParamsReader::Params>::GetParam())),
      handle(rmm::cuda_stream_per_thread, std::make_shared<rmm::cuda_stream_pool>(8)),
      stream(handle.get_stream())
  {
  }

  bool isInputValid() const
  {
    /* Fail to fit data with bias. */
    if (params.nClasses == 1 && params.bias != 0 && !params.modelParams.fit_intercept) return false;

    /* This means we don't have enough dimensions to linearly separate every cluster
       from the rest.
       In such case, the error is always huge (fitting is impossible).
     */
    if (params.nClasses > 1 && params.nClasses > (1 << min(30, params.nCols))) return false;

    return true;
  }

  testing::AssertionResult errorRate()
  {
    auto [XBuf, yBuf]    = genData(params.nRowsTrain + params.nRowsTest);
    auto [XTrain, XTest] = splitData(XBuf, params.nRowsTrain, params.nCols);
    auto [yTrain, yTest] = splitData(yBuf, params.nRowsTrain, 1);
    auto model           = LinearSVMModel<T>::fit(handle,
                                        params.modelParams,
                                        XTrain.data(),
                                        params.nRowsTrain,
                                        params.nCols,
                                        yTrain.data(),
                                        (const T*)nullptr);

    rmm::device_uvector<T> yOut(yTest.size(), stream);
    LinearSVMModel<T>::predict(
      handle, params.modelParams, model, XTest.data(), params.nRowsTest, params.nCols, yOut.data());

    rmm::device_scalar<T> errorBuf(stream);
    if (params.nClasses == 1)  // regression
      raft::linalg::mapThenSumReduce(
        errorBuf.data(),
        params.nRowsTest,
        [] __device__(const T yRef, const T yOut) {
          T s = yRef * yRef + yOut * yOut;
          T d = yRef - yOut;
          return d * d / s;
        },
        stream,
        yTest.data(),
        yOut.data());
    else  // classification
      raft::linalg::mapThenSumReduce(
        errorBuf.data(),
        params.nRowsTest,
        [] __device__(const T yRef, const T yOut) { return T(yRef != yOut); },
        stream,
        yTest.data(),
        yOut.data());
    // getting the error value forces the stream synchronization
    T error = errorBuf.value(stream) / T(params.nRowsTest);

    LinearSVMModel<T>::free(handle, model);
    if (error <= params.tolerance)
      return testing::AssertionSuccess();
    else
      return testing::AssertionFailure()
             << "Error rate = " << error << " > tolerance = " << params.tolerance;
  }

  testing::AssertionResult probabilitySumsToOne()
  {
    if (!params.modelParams.probability)
      return testing::AssertionFailure() << "Non-probabolistic model does not support this test.";
    if (params.nClasses < 2)
      return testing::AssertionFailure() << "Regression model does not support this test.";

    auto [XBuf, yBuf]    = genData(params.nRowsTrain + params.nRowsTest);
    auto [XTrain, XTest] = splitData(XBuf, params.nRowsTrain, params.nCols);
    auto [yTrain, yTest] = splitData(yBuf, params.nRowsTrain, 1);
    auto model           = LinearSVMModel<T>::fit(handle,
                                        params.modelParams,
                                        XTrain.data(),
                                        params.nRowsTrain,
                                        params.nCols,
                                        yTrain.data(),
                                        (const T*)nullptr);

    rmm::device_scalar<T> errorBuf(stream);
    rmm::device_uvector<T> yProbs(yTest.size() * params.nClasses, stream);
    LinearSVMModel<T>::predictProba(handle,
                                    params.modelParams,
                                    model,
                                    XTest.data(),
                                    params.nRowsTest,
                                    params.nCols,
                                    false,
                                    yProbs.data());

    rmm::device_uvector<T> yOut(yTest.size(), stream);
    raft::linalg::reduce<T, T, int>(
      yOut.data(), yProbs.data(), params.nClasses, params.nRowsTest, 0, true, true, stream);
    raft::linalg::mapThenReduce(
      errorBuf.data(),
      params.nRowsTest,
      T(0),
      [] __device__(const T yOut) { return raft::abs<T>(1.0 - yOut); },
      ML::detail::maximum{},
      stream,
      yOut.data());
    T error = errorBuf.value(stream);

    LinearSVMModel<T>::free(handle, model);
    if (error <= params.tolerance)
      return testing::AssertionSuccess();
    else
      return testing::AssertionFailure()
             << "Sum of probabilities deviated from zero (error = " << error << ")";
  }

  testing::AssertionResult probabilityErrorRate()
  {
    if (!params.modelParams.probability)
      return testing::AssertionFailure() << "Non-probabolistic model does not support this test.";
    if (params.nClasses < 2)
      return testing::AssertionFailure() << "Regression model does not support this test.";

    auto [XBuf, yBuf]    = genData(params.nRowsTrain + params.nRowsTest);
    auto [XTrain, XTest] = splitData(XBuf, params.nRowsTrain, params.nCols);
    auto [yTrain, yTest] = splitData(yBuf, params.nRowsTrain, 1);
    auto model           = LinearSVMModel<T>::fit(handle,
                                        params.modelParams,
                                        XTrain.data(),
                                        params.nRowsTrain,
                                        params.nCols,
                                        yTrain.data(),
                                        (const T*)nullptr);

    rmm::device_scalar<T> errorBuf(stream);
    rmm::device_uvector<T> yProbs(yTest.size() * params.nClasses, stream);
    rmm::device_uvector<T> yOut(yTest.size(), stream);
    LinearSVMModel<T>::predictProba(handle,
                                    params.modelParams,
                                    model,
                                    XTest.data(),
                                    params.nRowsTest,
                                    params.nCols,
                                    false,
                                    yProbs.data());

    raft::linalg::reduce<T, T, int>(
      yOut.data(),
      yProbs.data(),
      params.nClasses,
      params.nRowsTest,
      0,
      true,
      true,
      stream,
      false,
      [] __device__(const T p, const int i) { return T(i * 2) + p + 0.5; },
      [] __device__(const T a, const T b) { return fmod(a, 2) >= fmod(b, 2) ? a : b; });
    raft::linalg::mapThenSumReduce(
      errorBuf.data(),
      params.nRowsTest,
      [] __device__(const T yRef, const T yOut) {
        T p = yOut - 2 * yRef;
        return T(p <= 0 || p >= 2);
      },
      stream,
      yTest.data(),
      yOut.data());
    // getting the error value forces the stream synchronization
    T error = errorBuf.value(stream) / T(params.nRowsTest);

    LinearSVMModel<T>::free(handle, model);
    if (error <= params.tolerance)
      return testing::AssertionSuccess();
    else
      return testing::AssertionFailure()
             << "Error rate = " << error << " > tolerance = " << params.tolerance;
  }

  /** Generate a required amount of (X, y) data at once. */
  std::tuple<rmm::device_uvector<T>, rmm::device_uvector<T>> genData(const int nRows)
  {
    rmm::device_uvector<T> X(nRows * params.nCols, stream);
    rmm::device_uvector<T> y(nRows * params.nClasses, stream);
    if (params.nClasses == 1)  // regression
    {
      int nInformative = max(params.nCols / 3, min(params.nCols, 5));
      rmm::device_uvector<T> Xt(nRows * params.nCols, stream);
      ML::Datasets::make_regression(handle,
                                    Xt.data(),
                                    y.data(),
                                    nRows,
                                    params.nCols,
                                    nInformative,
                                    nullptr,
                                    1,
                                    params.bias,
                                    -1,
                                    T(0),
                                    T(params.errStd),
                                    true,
                                    params.seed);
      raft::linalg::transpose(handle, Xt.data(), X.data(), params.nCols, nRows, stream);
    } else  // classification
    {
      rmm::device_uvector<int> labels(nRows * params.nClasses, stream);
      raft::random::Rng r(params.seed);
      rmm::device_uvector<T> centers(params.nCols * params.nClasses, stream);
      r.uniform(centers.data(), params.nCols * params.nClasses, T(0), T(1), stream);
      // override manually some of the cluster coordinates to ensure
      // the distance between any of them is large enough.
      int d = max(2, int(std::ceil(std::pow(double(params.nClasses), 1.0 / double(params.nCols)))));
      int modCols = int(std::ceil(std::log2(double(params.nClasses)) / std::log2(double(d))));
      for (int i = 0; i < params.nClasses; i++) {
        int r = i;
        for (int j = 0; j < modCols; j++) {
          T value = T((r % d) * params.nClasses) + T(params.bias);
          centers.set_element_async(j * params.nClasses + i, value, stream);
          r /= d;
        }
      }
      ML::Datasets::make_blobs(handle,
                               X.data(),
                               labels.data(),
                               nRows,
                               params.nCols,
                               params.nClasses,
                               false,
                               centers.data(),
                               nullptr,
                               T(params.errStd),
                               true,
                               0,
                               0,
                               params.seed);
      raft::linalg::unaryOp(
        y.data(), labels.data(), labels.size(), [] __device__(int x) { return T(x); }, stream);
    }
    return std::make_tuple(std::move(X), std::move(y));
  }

  /** Split a column-major matrix in two along the rows. */
  std::tuple<rmm::device_uvector<T>, rmm::device_uvector<T>> splitData(rmm::device_uvector<T>& x,
                                                                       const int takeNRows,
                                                                       const int nCols)
  {
    const int nRows     = x.size() / nCols;
    const int dropNRows = nRows - takeNRows;
    rmm::device_uvector<T> x1(takeNRows * nCols, stream);
    rmm::device_uvector<T> x2(dropNRows * nCols, stream);
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(x1.data(),
                                    sizeof(T) * takeNRows,
                                    x.data(),
                                    sizeof(T) * nRows,
                                    sizeof(T) * takeNRows,
                                    nCols,
                                    cudaMemcpyDeviceToDevice,
                                    stream));
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(x2.data(),
                                    sizeof(T) * dropNRows,
                                    x.data() + takeNRows,
                                    sizeof(T) * nRows,
                                    sizeof(T) * dropNRows,
                                    nCols,
                                    cudaMemcpyDeviceToDevice,
                                    stream));
    return std::make_tuple(std::move(x1), std::move(x2));
  }
};

#define TEST_SVM(fun, TestClass, ElemType)                           \
  typedef LinearSVMTest<ElemType, TestClass> TestClass##_##ElemType; \
  TEST_P(TestClass##_##ElemType, fun)                                \
  {                                                                  \
    if (!isInputValid()) GTEST_SKIP();                               \
    ASSERT_TRUE(fun());                                              \
  }                                                                  \
  INSTANTIATE_TEST_SUITE_P(LinearSVM, TestClass##_##ElemType, TestClass##Params)

auto TestClasTargetsParams =
  ::testing::Combine(::testing::Values(LinearSVMParams::HINGE, LinearSVMParams::SQUARED_HINGE),
                     ::testing::Values(LinearSVMParams::L1, LinearSVMParams::L2),
                     ::testing::Values(2, 3, 8),
                     ::testing::Values(1, 50));

struct TestClasTargets {
  typedef std::tuple<LinearSVMParams::Loss, LinearSVMParams::Penalty, int, int> Params;
  static LinearSVMTestParams read(Params ps)
  {
    LinearSVMParams mp;
    mp.penalty = std::get<1>(ps);
    mp.loss    = std::get<0>(ps);
    return {/* .nRowsTrain   */ 100,
            /* .nRowsTest    */ 100,
            /* .nCols        */ std::get<3>(ps),
            /* .nClasses     */ std::get<2>(ps),
            /* .errStd       */ 0.4,
            /* .bias         */ 0.0,
            /* .tolerance    */ 0.05,
            /* .seed         */ 42ULL,
            /* .modelParams  */ mp};
  }
};

auto TestClasBiasParams = ::testing::Combine(::testing::Bool(),
                                             ::testing::Bool(),
                                             ::testing::Values(2, 3),
                                             ::testing::Values(10, 50),
                                             ::testing::Values(0.0, -10.0));

struct TestClasBias {
  typedef std::tuple<bool, bool, int, int, double> Params;
  static LinearSVMTestParams read(Params ps)
  {
    LinearSVMParams mp;
    mp.fit_intercept       = std::get<0>(ps);
    mp.penalized_intercept = std::get<1>(ps);
    return {/* .nRowsTrain   */ 1000,
            /* .nRowsTest    */ 100,
            /* .nCols        */ std::get<3>(ps),
            /* .nClasses     */ std::get<2>(ps),
            /* .errStd       */ 0.2,
            /* .bias         */ std::get<4>(ps),
            /* .tolerance    */ 0.05,
            /* .seed         */ 42ULL,
            /* .modelParams  */ mp};
  }
};

auto TestClasManyClassesParams = ::testing::Values(2, 3, 16, 31, 32, 33, 67);

struct TestClasManyClasses {
  typedef int Params;
  static LinearSVMTestParams read(Params ps)
  {
    LinearSVMParams mp;
    return {/* .nRowsTrain   */ 1000,
            /* .nRowsTest    */ 1000,
            /* .nCols        */ 200,
            /* .nClasses     */ ps,
            /* .errStd       */ 1.0,
            /* .bias         */ 0,
            /* .tolerance    */ 0.01,
            /* .seed         */ 42ULL,
            /* .modelParams  */ mp};
  }
};

auto TestClasProbsSumParams = ::testing::Values(2, 3, 16, 31, 32, 33, 67);

struct TestClasProbsSum {
  typedef int Params;
  static LinearSVMTestParams read(Params ps)
  {
    LinearSVMParams mp;
    mp.probability = true;
    mp.max_iter    = 100;
    return {/* .nRowsTrain   */ 100,
            /* .nRowsTest    */ 100,
            /* .nCols        */ 80,
            /* .nClasses     */ ps,
            /* .errStd       */ 1.0,
            /* .bias         */ 0,
            /* .tolerance    */ 1e-5,
            /* .seed         */ 42ULL,
            /* .modelParams  */ mp};
  }
};

auto TestClasProbsParams = ::testing::Values(2, 3, 16, 31, 32, 33, 67);

struct TestClasProbs {
  typedef int Params;
  static LinearSVMTestParams read(Params ps)
  {
    LinearSVMParams mp;
    mp.probability = true;
    return {/* .nRowsTrain   */ 1000,
            /* .nRowsTest    */ 1000,
            /* .nCols        */ 200,
            /* .nClasses     */ ps,
            /* .errStd       */ 0.9,
            /* .bias         */ 0,
            /* .tolerance    */ 0.01,
            /* .seed         */ 42ULL,
            /* .modelParams  */ mp};
  }
};

auto TestRegTargetsParams =
  ::testing::Combine(::testing::Values(LinearSVMParams::EPSILON_INSENSITIVE,
                                       LinearSVMParams::SQUARED_EPSILON_INSENSITIVE),
                     ::testing::Values(LinearSVMParams::L1, LinearSVMParams::L2),
                     ::testing::Bool(),
                     ::testing::Values(1, 50),
                     ::testing::Values(0.0, -10.0),
                     ::testing::Values(0.0, 0.01));
struct TestRegTargets {
  typedef std::tuple<LinearSVMParams::Loss, LinearSVMParams::Penalty, bool, int, double, double>
    Params;
  static LinearSVMTestParams read(Params ps)
  {
    LinearSVMParams mp;
    mp.loss          = std::get<0>(ps);
    mp.penalty       = std::get<1>(ps);
    mp.fit_intercept = std::get<2>(ps);
    // The regularization parameter strongly affects the model performance in some cases,
    // a larger-than-default value of C seems to always yield better scores on this generated
    // dataset.
    mp.C       = 100.0;
    mp.epsilon = std::get<5>(ps);
    mp.verbose = 2;
    return {/* .nRowsTrain   */ 1000,
            /* .nRowsTest    */ 100,
            /* .nCols        */ std::get<3>(ps),
            /* .nClasses     */ 1,
            /* .errStd       */ 0.02,
            /* .bias         */ std::get<4>(ps),
            /* .tolerance    */ 0.05,
            /* .seed         */ 42ULL,
            /* .modelParams  */ mp};
  }
};

TEST_SVM(errorRate, TestClasTargets, float);
TEST_SVM(errorRate, TestClasTargets, double);
TEST_SVM(errorRate, TestClasBias, float);
TEST_SVM(errorRate, TestClasManyClasses, float);
TEST_SVM(errorRate, TestClasManyClasses, double);
TEST_SVM(errorRate, TestRegTargets, float);
TEST_SVM(errorRate, TestRegTargets, double);
TEST_SVM(probabilitySumsToOne, TestClasProbsSum, float);
TEST_SVM(probabilitySumsToOne, TestClasProbsSum, double);
TEST_SVM(probabilityErrorRate, TestClasProbs, float);
TEST_SVM(probabilityErrorRate, TestClasProbs, double);

}  // namespace SVM
}  // namespace ML
