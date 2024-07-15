/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <raft/util/cudart_utils.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <test_utils.h>
#include <cuml/decomposition/params.hpp>
#include <kpca/kpca.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/random/rng.cuh>
#include <vector>

namespace ML {

using namespace MLCommon;

template <typename T>
struct KPcaInputs {
  T tolerance;
  int n_rows;
  int n_cols;
  int n_components;
  int algo;
  raft::distance::kernels::KernelParams kernel;
  std::vector<T> data_h;
  std::vector<T> eigenvectors_ref_h;
  std::vector<T> eigenvalues_ref_h;
  std::vector<T> trans_data_ref_h;
};

template <typename T>
class KPcaTest : public ::testing::TestWithParam<KPcaInputs<T>> {
 public:
    KPcaTest()
    : params(::testing::TestWithParam<KPcaInputs<T>>::GetParam()),
      stream(handle.get_stream()),
      data(params.n_rows * params.n_cols, stream),
      trans_data(params.n_rows * params.n_rows, stream),
      eigenvectors(params.n_rows * params.n_rows, stream),
      eigenvalues(params.n_rows, stream),
      trans_data_ref(params.n_rows * params.n_rows, stream),
      eigenvectors_ref(params.n_rows * params.n_rows, stream),
      eigenvalues_ref(params.n_rows, stream)
      {
        basicTest();
        fitTransformTest();
      }
 protected:
  void basicTest() {
    raft::update_device(data.data(), params.data_h.data(), params.n_rows * params.n_cols, stream);
    raft::update_device(trans_data_ref.data(), params.trans_data_ref_h.data(), params.n_rows * params.n_rows, stream);
    raft::update_device(eigenvalues_ref.data(), params.eigenvalues_ref_h.data(), params.n_rows, stream);
    raft::update_device(eigenvectors_ref.data(), params.eigenvectors_ref_h.data(), params.n_rows * params.n_rows, stream);

    paramsKPCA prms;
    prms.n_rows = params.n_rows;
    prms.n_training_samples = params.n_rows;
    prms.n_cols = params.n_cols;
    prms.n_components = params.n_components;
    prms.kernel = params.kernel;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else
      prms.algorithm = solver::COV_EIG_JACOBI;

    kpcaFit(handle, data.data(), eigenvectors.data(), eigenvalues.data(), prms, stream);
    kpcaTransform(handle, data.data(), data.data(), eigenvectors.data(), eigenvalues.data(), trans_data.data(), prms, stream);
  }

  void fitTransformTest() {
    raft::update_device(data.data(), params.data_h.data(), params.n_rows * params.n_cols, stream);
    raft::update_device(trans_data_ref.data(), params.trans_data_ref_h.data(), params.n_rows * params.n_rows, stream);
    raft::update_device(eigenvalues_ref.data(), params.eigenvalues_ref_h.data(), params.n_rows, stream);
    raft::update_device(eigenvectors_ref.data(), params.eigenvectors_ref_h.data(), params.n_rows * params.n_rows, stream);

    paramsKPCA prms;
    prms.n_rows = params.n_rows;
    prms.n_training_samples = params.n_rows;
    prms.n_cols = params.n_cols;
    prms.n_components = params.n_components;
    prms.kernel = params.kernel;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else
      prms.algorithm = solver::COV_EIG_JACOBI;

    kpcaFitTransform(handle, data.data(), eigenvectors.data(), eigenvalues.data(), trans_data.data(), prms, stream);
  }

 protected:
  KPcaInputs<T> params;

  rmm::device_uvector<T> data, trans_data, eigenvectors, eigenvalues, trans_data_ref, eigenvectors_ref, eigenvalues_ref;
    

  raft::handle_t handle;
  cudaStream_t stream;
};


float tolerance = 0.01f;
int n_rows = 3;
int n_cols = 2;
int n_components = 2;
int algo = 1;
std::vector<float> data_h = {1.0, 2.0, 5.0, 4.0, 2.0, 1.0};

raft::distance::kernels::KernelParams lin_kern = {raft::distance::kernels::LINEAR, 0, 0, 0};
std::vector<float> lin_eigenvectors_ref_h = {-0.6525, -0.0987, 0.7513, -0.4907, 0.8105, -0.3197};
std::vector<float> lin_eigenvalues_ref_h = {12.6759, 0.6574};
std::vector<float> lin_trans_data_ref_h = {-2.32318647,-0.35170213, 2.6748886, -0.39794495, 0.65716145,-0.25921649};
KPcaInputs<float> linear_inputs = {tolerance, n_rows, n_cols, n_components, algo, lin_kern
                                  , data_h, lin_eigenvectors_ref_h, lin_eigenvalues_ref_h, lin_trans_data_ref_h};

raft::distance::kernels::KernelParams poly_kern = {raft::distance::kernels::POLYNOMIAL, 3, 1.0/2.0f, 1};
std::vector<float> poly_eigenvectors_ref_h = {-0.5430, -0.2565, 0.7995, -0.6097, 0.7751, -0.1653};
std::vector<float> poly_eigenvalues_ref_h = {1790.3207, 210.3639};
std::vector<float> poly_trans_data_ref_h = {-22.9760, -10.8554, 33.8314, -8.8438, 11.2426, -2.3987};
KPcaInputs<float> poly_inputs = {tolerance, n_rows, n_cols, n_components, algo, poly_kern
                              , data_h, poly_eigenvectors_ref_h, poly_eigenvalues_ref_h, poly_trans_data_ref_h};

raft::distance::kernels::KernelParams rbf_kern = {raft::distance::kernels::RBF, 0, 1.0/2.0f, 0};
std::vector<float> rbf_eigenvectors_ref_h = {-0.4341, -0.3818, 0.8159, -0.6915, 0.7217, -0.0301};
std::vector<float> rbf_eigenvalues_ref_h = {1.0230, 0.9177};
std::vector<float> rbf_trans_data_ref_h = {-0.4391, -0.3862, 0.8253, -0.6624, 0.6914, -0.0289};
KPcaInputs<float> rbf_inputs = {tolerance, n_rows, n_cols, n_components, algo, rbf_kern
                              , data_h, rbf_eigenvectors_ref_h, rbf_eigenvalues_ref_h, rbf_trans_data_ref_h};

const std::vector<KPcaInputs<float>> inputs_f = {linear_inputs, poly_inputs, rbf_inputs};

typedef KPcaTest<float> KPcaTestEigenvaluesF;
TEST_P(KPcaTestEigenvaluesF, Result) {
  ASSERT_TRUE(MLCommon::devArrMatch(eigenvalues.data(), eigenvalues_ref.data(), params.n_cols,
                          MLCommon::CompareApproxAbs<float>(params.tolerance)));
}

typedef KPcaTest<float> KPcaTestEigenvectorsF;
TEST_P(KPcaTestEigenvectorsF, Result) {
  ASSERT_TRUE(MLCommon::devArrMatch(eigenvectors.data(), eigenvectors_ref.data(),
                          (params.n_rows * params.n_cols),
                          MLCommon::CompareApproxAbs<float>(params.tolerance)));
}

typedef KPcaTest<float> KPcaTestTransDataF;
TEST_P(KPcaTestTransDataF, Result) {
  ASSERT_TRUE(MLCommon::devArrMatch(trans_data.data(), trans_data_ref.data(),
                          (params.n_rows * params.n_cols),
                          MLCommon::CompareApproxAbs<float>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestEigenvaluesF, ::testing::ValuesIn(inputs_f));

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestEigenvectorsF,
                        ::testing::ValuesIn(inputs_f));

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestTransDataF,
                        ::testing::ValuesIn(inputs_f));

// Adding double type test cases

double tolerance_d = 0.01;
std::vector<double> data_hd = {1.0, 2.0, 5.0, 4.0, 2.0, 1.0};

std::vector<double> lin_eigenvectors_ref_hd = {-0.6525, -0.0987, 0.7513, -0.4907, 0.8105, -0.3197};
std::vector<double> lin_eigenvalues_ref_hd = {12.6759, 0.6574};
std::vector<double> lin_trans_data_ref_hd = {-2.32318647,-0.35170213, 2.6748886, -0.39794495, 0.65716145,-0.25921649};
KPcaInputs<double> linear_inputs_d = {tolerance_d, n_rows, n_cols, n_components, algo, lin_kern
                                  , data_hd, lin_eigenvectors_ref_hd, lin_eigenvalues_ref_hd, lin_trans_data_ref_hd};

std::vector<double> poly_eigenvectors_ref_hd = {-0.5430, -0.2565, 0.7995, -0.6097, 0.7751, -0.1653};
std::vector<double> poly_eigenvalues_ref_hd = {1790.3207, 210.3639};
std::vector<double> poly_trans_data_ref_hd = {-22.9760, -10.8554, 33.8314, -8.8438, 11.2426, -2.3987};
KPcaInputs<double> poly_inputs_d = {tolerance_d, n_rows, n_cols, n_components, algo, poly_kern
                              , data_hd, poly_eigenvectors_ref_hd, poly_eigenvalues_ref_hd, poly_trans_data_ref_hd};

std::vector<double> rbf_eigenvectors_ref_hd = {-0.4341, -0.3818, 0.8159, -0.6915, 0.7217, -0.0301};
std::vector<double> rbf_eigenvalues_ref_hd = {1.0230, 0.9177};
std::vector<double> rbf_trans_data_ref_hd = {-0.4391, -0.3862, 0.8253, -0.6624, 0.6914, -0.0289};
KPcaInputs<double> rbf_inputs_d = {tolerance_d, n_rows, n_cols, n_components, algo, rbf_kern
                              , data_hd, rbf_eigenvectors_ref_hd, rbf_eigenvalues_ref_hd, rbf_trans_data_ref_hd};

const std::vector<KPcaInputs<double>> inputs_d = {linear_inputs_d, poly_inputs_d, rbf_inputs_d};

typedef KPcaTest<double> KPcaTestEigenvaluesD;
TEST_P(KPcaTestEigenvaluesD, Result) {
  ASSERT_TRUE(MLCommon::devArrMatch(eigenvalues.data(), eigenvalues_ref.data(), params.n_cols,
                          MLCommon::CompareApproxAbs<double>(params.tolerance)));
}

typedef KPcaTest<double> KPcaTestEigenvectorsD;
TEST_P(KPcaTestEigenvectorsD, Result) {
  ASSERT_TRUE(MLCommon::devArrMatch(eigenvectors.data(), eigenvectors_ref.data(),
                          (params.n_rows * params.n_cols),
                          MLCommon::CompareApproxAbs<double>(params.tolerance)));
}

typedef KPcaTest<double> KPcaTestTransDataD;
TEST_P(KPcaTestTransDataD, Result) {
  ASSERT_TRUE(MLCommon::devArrMatch(trans_data.data(), trans_data_ref.data(),
                          (params.n_rows * params.n_cols),
                          MLCommon::CompareApproxAbs<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestEigenvaluesD, ::testing::ValuesIn(inputs_d));

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestEigenvectorsD,
                        ::testing::ValuesIn(inputs_d));

INSTANTIATE_TEST_CASE_P(KPcaTests, KPcaTestTransDataD,
                        ::testing::ValuesIn(inputs_d));

}  // end namespace ML
