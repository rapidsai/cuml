// #include <gtest/gtest.h>
//
// // #include <magma_v2.h>
// #include "gmm/gmm.h"
// #include <magma/magma_utils.h>
//
// using namespace MLCommon;
// using namespace gmm;
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
// void run(magma_int_t nCl, magma_int_t nDim, magma_int_t nObs, int n_iter)
// {
//         T *dX;
//         T* workspace;
//
//         // declaration:
//         T *dmu, *dsigma, *dPis, *dPis_inv, *dLlhd, *cur_llhd;
//         magma_int_t lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd;
//         lddx = magma_roundup(nDim, RUP_SIZE);
//         lddmu = magma_roundup(nDim, RUP_SIZE);
//         lddsigma = magma_roundup(nDim, RUP_SIZE);
//         lddsigma_full = nDim * lddsigma;
//         lddLlhd = magma_roundup(nCl, RUP_SIZE);
//         lddPis = lddLlhd;
//
//         T reg_covar = 0;
//         size_t workspaceSize=0;
//
//         // Random parameters
//         cublasHandle_t cublasHandle;
//         CUBLAS_CHECK(cublasCreate(&cublasHandle));
//
//         int device = 0;
//         magma_queue_t queue;
//         magma_queue_create(device, &queue);
//
//         allocate(dX, lddx * nObs);
//         allocate(dmu, lddmu * nCl);
//         allocate(dsigma, lddsigma_full * nCl);
//         allocate(dLlhd, lddLlhd * nObs);
//         allocate(dPis, lddPis);
//         allocate(dPis_inv, lddPis);
//         allocate(cur_llhd, 1);
//
//         fill_matrix_gpu(nDim, nObs, dX, lddx);
//         fill_matrix_gpu(nDim, nCl, dmu, lddmu);
//         fill_matrix_gpu(nDim * nCl, nDim, dsigma, lddsigma);
//         fill_matrix_gpu(nDim, nObs, dLlhd, lddLlhd);
//         fill_matrix_gpu(nDim, 1, dPis, lddPis);
//         fill_matrix_gpu(nDim, 1, dPis_inv, lddPis);
//
//
// // computation:
//         GMM<T> gmm;
//         init(gmm,
//              dmu, dsigma, dPis, dPis_inv, dLlhd,
//              lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd,
//              cur_llhd, reg_covar,
//              nCl, nDim, nObs);
//
//         workspaceSize = gmm_bufferSize(gmm);
//         CUDA_CHECK(cudaMalloc((void **)&workspace, workspaceSize));
//         printf("%d\n", (int) workspaceSize);
//         create_GMMHandle_new(gmm, workspace);
//
//         // setup(gmm);
//         // create_GMMHandle(gmm);
//         fit(dX, n_iter, gmm, cublasHandle, queue);
//
// // cleanup:
//         // CUDA_CHECK(cudaFree(dX));
// }
//
//
// template <typename T>
// class GMMTest : public ::testing::TestWithParam<GMMInputs<T> > {
// protected:
// void SetUp() override {
//         params = ::testing::TestWithParam<GMMInputs<T> >::GetParam();
//         tolerance = params.tolerance;
//
//         magma_int_t nCl = 30;
//         magma_int_t nDim = 10;
//         magma_int_t nObs = 300;
//         int n_iter = 5;
//
//         run<T>(nCl, nDim, nObs, n_iter);
//
//         error = 0;
//
//         magma_finalize();
// }
//
// void TearDown() override {
// }
//
// protected:
// GMMInputs<T> params;
// T error, tolerance;
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
// // INSTANTIATE_TEST_CASE_P(GMMTests, GMMTestF,
// // ::testing::ValuesIn(inputsf2));
//
// INSTANTIATE_TEST_CASE_P(GMMTests, GMMTestD,
//                         ::testing::ValuesIn(inputsd2));
