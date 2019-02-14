// #include <gtest/gtest.h>
//
// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
//
// #include "hmm/gmm.h"
// #include "hmm/structs.h"
// #include "hmm/utils.h"
// #include "hmm/random.h"
//
// #include "cuda_utils.h"
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
// struct GMMInputs {
//         T tolerance;
// };
//
// template <typename T>
// ::std::ostream& operator<<(::std::ostream& os, const GMMInputs<T>& dims) {
//         return os;
// }
//
// template <typename T>
// class GMMTest : public ::testing::TestWithParam<GMMInputs<T> > {
// protected:
// void SetUp() override {
//         params = ::testing::TestWithParam<GMMInputs<T> >::GetParam();
//         tolerance = params.tolerance;
//
//         setup_gmm();
//         allocate_memory();
//         initialize_data();
//         copy_to_device();
//         ML::GMM::fit(gmm, data_d);
//         error = compute_error(gmm);
// }
//
// T compute_error(GMM<T> gmm){
//         error = 100.0;
// }
//
// void setup_gmm(){
//         nDim = 2;
//         nCl = 2;
//         nObs = 2;
//
//         paramsEM paramsEM(100);
//         paramsRandom paramsRd(0, 1, 1234ULL);
//         ML::GMM::set_gmm(gmm, nCl, nDim, paramsRd, paramsEM);
//         ML::GMM::initialize(gmm);
// }
//
//
// void allocate_memory(){
//         allocate(data_d, nDim * nObs);
//         CUDA_CHECK(cudaMemset(data_d, (T)0, nDim * nObs ));
//         data_h = (T *)malloc(nDim * nObs * sizeof(T));
//
//         allocate(mus_d, nDim * nCl);
//         CUDA_CHECK(cudaMemset(mus_d, (T)0,nDim * nCl ));
//         mus_h = (T *)malloc(nDim * nCl * sizeof(T));
//
//         allocate(sigmas_d, nDim *nDim * nCl);
//         CUDA_CHECK(cudaMemset(sigmas_d, (T) 0, nDim *nDim * nCl));
//         sigmas_h = (T *)malloc(nDim * nDim * nCl * sizeof(T));
//
//         allocate(rhos_d, nCl);
//         CUDA_CHECK(cudaMemset(rhos_d, (T) 0, nCl));
//         rhos_h = (T *)malloc(nCl * sizeof(T));
//
//         CUBLAS_CHECK(cublasCreate(&handle));
// }
//
//
// void initialize_data(){
//
//         data_h[0] = 0.0;
//         data_h[1] = -1.0;
//         data_h[2] = 1.0;
//         data_h[3] = 5.0;
//
//         mus_h[0] = -1.0;
//         mus_h[1] = 2.0;
//         mus_h[2] = 1.0;
//         mus_h[3] = 1.0;
//
//         rhos_h[0] = 0.3;
//         rhos_h[1] = 0.7;
//
//         sigmas_h[0] = 1;
//         sigmas_h[1] = 0.0;
//         sigmas_h[2] = 0.0;
//         sigmas_h[3] = 1.0;
//         sigmas_h[4] = 1.0;
//         sigmas_h[5] = 0.5;
//         sigmas_h[6] = 0.7;
//         sigmas_h[7] = -2.0;
// }
//
// void copy_to_device(){
//         updateDevice(data_d, data_h, nDim * nObs);
//         updateDevice(mus_d, mus_h, nDim * nCl);
//         updateDevice(sigmas_d, sigmas_h, nDim *nDim * nCl);
//         updateDevice(rhos_d, rhos_h, nCl);
// }
//
// void TearDown() override {
//         free(data_h);
//         free(mus_h);
//         free(sigmas_h);
//         free(rhos_h);
//
//         CUDA_CHECK(cudaFree(data_d));
//         CUDA_CHECK(cudaFree(mus_d));
//         CUDA_CHECK(cudaFree(sigmas_d));
//         CUDA_CHECK(cudaFree(rhos_d));
// }
//
// protected:
// GMM gmm;
//
// GMMInputs<T> params;
// T error, tolerance;
// int nDim, nCl, nObs;
//
// T *data_h, *mus_h, *sigmas_h, *rhos_h;
// T *data_d, *mus_d, *sigmas_d, *rhos_d;
//
// T *est_mus_h, *est_sigmas_h, *est_rhos_h;
// T *est_mus_d, *est_sigmas_d, *est_rhos_d;
// };
//
// const std::vector<GMMInputs<float> > inputsf2 = {
//         {0.00001f}
// };
//
// const std::vector<GMMInputs<double> > inputsd2 = {
//         {0.00001}
// };
//
//
// typedef GMMTest<float> GMMTestF;
// TEST_P(GMMTestF, Result){
//         EXPECT_LT(error, tolerance) << " error out of tol.";
// }
//
// typedef GMMTest<double> GMMTestD;
// TEST_P(GMMTestD, Result){
//         EXPECT_LT(error, tolerance) << " error out of tol.";
// }
//
// INSTANTIATE_TEST_CASE_P(GMMTests, GMMTestF,
//                         ::testing::ValuesIn(inputsf2));
//
// INSTANTIATE_TEST_CASE_P(GMMTests, GMMTestD,
//                         ::testing::ValuesIn(inputsd2));
//
// } // end namespace HMM
// } // end namespace MLCommon
