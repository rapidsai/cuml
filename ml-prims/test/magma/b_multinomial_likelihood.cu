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
//
// #include "magma/b_multinomial_likelihood.h"
//
// using namespace MLCommon;
//
//
// template <typename T>
// T run_multinomial_likelihood(magma_int_t nStates,
//                              magma_int_t nObs,
//                              magma_int_t batchCount)
// {
// // declaration:
//         T *dX=NULL, **dPb_array=NULL, *dO_naive=NULL, *dO_magma=NULL;
//         magma_int_t lddLlhd = magma_roundup(batchCount, RUP_SIZE);
//         T *error_d, error = 0;
//
// // Allocation
//         allocate(dX, nObs);
//         allocate_pointer_array(dPb_array, nStates, batchCount);
//         allocate(dO_magma, batchCount);
//         allocate(dO_naive, batchCount);
//         allocate(error_d, 1);
//
// // filling:
//         fill_matrix_gpu_batched(m, n, batchCount, dA_array, ldda );
//         fill_matrix_gpu_batched(m, 1, batchCount, dX_array, lddx );
//
// // computation:
//         naive_multinomial_likelihood_batched(nStates, nObs, batchCount,
//                                              dX, dPb_array,
//                                              dLlhd, lddLlhd,
//                                              isLog);
//
//         multinomial_likelihood_batched(nStates, nObs, batchCount,
//                                        dX, dPb_array,
//                                        dLlhd, lddLlhd,
//                                        isLog);
//
//         // Error
//         meanSquaredError(error_d, dO_naive, dO_magma, batchCount);
//         updateHost(&error, error_d, 1);
//
// // cleanup:
//         free_pointer_array(dA_array, batchCount);
//         free_pointer_array(dX_array, batchCount);
//         free_pointer_array(dY_array, batchCount);
//         CUDA_CHECK(cudaFree(dO_naive));
//         CUDA_CHECK(cudaFree(dO_magma));
//
//         return error;
// }
//
// template <typename T>
// struct MultinomialLikelihoodInputs {
//         T tolerance;
//         magma_int_t nStates, nObs, batchCount;
// };
//
// template <typename T>
// ::std::ostream& operator<<(::std::ostream& os, const MultinomialLikelihoodInputs<T>& dims) {
//         return os;
// }
//
// template <typename T>
// class MultinomialLikelihoodTest : public ::testing::TestWithParam<MultinomialLikelihoodInputs<T> > {
// protected:
// void SetUp() override {
//         params = ::testing::TestWithParam<MultinomialLikelihoodInputs<T> >::GetParam();
//         tolerance = params.tolerance;
//
//         magma_init();
//         error = run_multinomial_likelihood<T>(params.m, params.n, params.batchCount);
//         magma_finalize();
// }
//
// protected:
// MultinomialLikelihoodInputs<T> params;
// T error, tolerance;
// };
//
// const std::vector<MultinomialLikelihoodInputs<float> > MultinomialLikelihoodInputsf2 = {
//         {0.000001f, 5, 20, 4}
// };
//
// const std::vector<MultinomialLikelihoodInputs<double> > MultinomialLikelihoodInputsd2 = {
//         {0.000001, 5, 20, 4}
// };
//
//
// typedef MultinomialLikelihoodTest<float> MultinomialLikelihoodTestF;
// TEST_P(MultinomialLikelihoodTestF, Result){
//         EXPECT_LT(error, tolerance) << " error out of tol.";
// }
//
// typedef MultinomialLikelihoodTest<double> MultinomialLikelihoodTestD;
// TEST_P(MultinomialLikelihoodTestD, Result){
//         EXPECT_LT(error, tolerance) << " error out of tol.";
// }
//
// INSTANTIATE_TEST_CASE_P(MultinomialLikelihoodTests, MultinomialLikelihoodTestF,
//                         ::testing::ValuesIn(MultinomialLikelihoodInputsf2));
//
// INSTANTIATE_TEST_CASE_P(MultinomialLikelihoodTests, MultinomialLikelihoodTestD,
//                         ::testing::ValuesIn(MultinomialLikelihoodInputsd2));
