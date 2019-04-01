/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

// #include <gtest/gtest.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
//
// #include "other/inverse.h"
//
// #include "linalg/mean_squared_error.h"
//
// #include "cuda_utils.h"
//
//
//
// using namespace MLCommon::LinAlg;
// using namespace MLCommon;
//
// namespace MLCommon {
// namespace HMM {
//
//
// template <typename T>
// struct InverseInputs {
//         T tolerance;
// };
//
// template <typename T>
// ::std::ostream& operator<<(::std::ostream& os, const InverseInputs<T>& dims) {
//         return os;
// }
//
// template <typename T>
// class InverseTest : public ::testing::TestWithParam<InverseInputs<T> > {
// protected:
// void SetUp() override {
//         params = ::testing::TestWithParam<InverseInputs<T> >::GetParam();
//
//         initialize_parameters(params);
//         allocate_memory();
//         initialize();
//         copy_to_device();
//         compute_error_inv();
// }
//
//
// void initialize_parameters(InverseInputs<T> params){
//         tolerance = params.tolerance;
//         nDim = 2;
// }
//
// void initialize(){
//
//         M_h[0] = 16.0;
//         M_h[1] = 5.0;
//         M_h[2] = 5.0;
//         M_h[3] = 9;
//
//         compute_true_inv();
// }
//
// void compute_true_inv(){
//         T det = (M_h[0] * M_h[3] - M_h[1] * M_h[2]);
//         true_inv_h[0] = M_h[3] / det;
//         true_inv_h[1] = -M_h[2] / det;
//         true_inv_h[2] = -M_h[1] / det;
//         true_inv_h[3] = M_h[0] / det;
// }
//
// void allocate_memory(){
//         M_h = (T *)malloc(nDim * nDim * sizeof(T));
//         allocate(M_d, nDim * nDim);
//         CUDA_CHECK(cudaMemset(M_d, (T)0, nDim * nDim));
//
//         true_inv_h = (T *)malloc(nDim * nDim * sizeof(T));
//         allocate(true_inv_d, nDim * nDim);
//         CUDA_CHECK(cudaMemset(true_inv_d, (T)0, nDim * nDim));
//
//         est_inv_h = (T *)malloc(nDim * nDim * sizeof(T));
//         allocate(est_inv_d, nDim * nDim);
//         CUDA_CHECK(cudaMemset(est_inv_d, (T)0, nDim * nDim));
//
//         allocate(error_d, 1);
//
//         CUSOLVER_CHECK(cusolverDnCreate(&cusolverHandle));
//
//         this->Inv = new Inverse<T>(nDim, &cusolverHandle);
// }
//
//
// void copy_to_device(){
//         updateDevice(M_d, M_h, nDim * nDim);
//         updateDevice(true_inv_d, true_inv_h, nDim * nDim);
// }
//
//
// void compute_error_inv(){
//         Inv->compute(M_d, est_inv_d);
//         meanSquaredError(error_d, est_inv_d, true_inv_d, nDim*nDim);
//         updateHost(&error, error_d, 1);
// }
//
//
// void TearDown() override {
//         free(true_inv_h);
//         free(est_inv_h);
//         free(M_h);
//
//         CUDA_CHECK(cudaFree(M_d));
//         CUDA_CHECK(cudaFree(true_inv_d));
//         CUDA_CHECK(cudaFree(est_inv_d));
//         CUDA_CHECK(cudaFree(error_d));
//
//         CUSOLVER_CHECK(cusolverDnDestroy(cusolverHandle));
//         Inv->TearDown();
// }
//
// protected:
// InverseInputs<T> params;
// T error, tolerance;
// int nDim;
//
// Inverse<T> *Inv;
//
// T *error_d;
// T *true_inv_d, *true_inv_h;
// T *est_inv_d, *est_inv_h;
// T *M_h, *M_d;
//
// cusolverDnHandle_t cusolverHandle;
// };
//
//
// const std::vector<InverseInputs<float> > inputsf2 = {
//         {0.00001f}
// };
//
// const std::vector<InverseInputs<double> > inputsd2 = {
//         {0.00001}
// };
//
//
// typedef InverseTest<float> InverseTestF;
// TEST_P(InverseTestF, Result){
//         EXPECT_LT(error, tolerance) << " error out of tol.";
// }
//
// typedef InverseTest<double> InverseTestD;
// TEST_P(InverseTestD, Result){
//         EXPECT_LT(error, tolerance) << " error out of tol.";
// }
//
// INSTANTIATE_TEST_CASE_P(InverseTests, InverseTestF,
//                         ::testing::ValuesIn(inputsf2));
//
// INSTANTIATE_TEST_CASE_P(InverseTests, InverseTestD,
//                         ::testing::ValuesIn(inputsd2));
//
// } // end namespace HMM
// } // end namespace MLCommon
