// #include <gtest/gtest.h>
//
// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
//
// #include "hmm/utils.h"
// #include "hmm/gmm.cu"
//
// #include "linalg/cusolver_wrappers.h"
// #include "linalg/mean_squared_error.h"
// #include "cuda_utils.h"
//
//
// using namespace MLCommon::LinAlg;
// using namespace MLCommon;
// using namespace ML::HMM;
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
//         initialize(gmm);
//         initialize_data();
//         copy_to_device();
//         fit(gmm, data_d);
//         // compute_error(gmm);
// }
//
// // void compute_error(GMM<T>& gmm){
// //         error = 0.0;
// //
// //         print_matrix( gmm.mus, nDim, 1, "gmm mus");
// //         print_matrix( mus_d, nDim, 1, "mus");
// //
// //         meanSquaredError(error_d, sigmas_d, gmm.sigmas, nDim * nDim * nCl);
// //         updateHost(&temp_err, error_d, 1);
// //         error += temp_err;
// //
// //         meanSquaredError(error_d, ps_d, gmm.rhos, nCl);
// //         updateHost(&temp_err, error_d, 1);
// //         error += temp_err;
// // }
//
// void setup_gmm(){
//         nDim = 1;
//         nCl = 2;
//         nObs = 110;
//
//         n_iter = 5;
//
//         this->params_em = new paramsEM(n_iter);
//         this->params_rd = new paramsRandom<T>((int) 0, (int) 1,
//                                               (unsigned long long) 1234ULL);
//
//         set_gmm(gmm, nCl, nDim, nObs, params_rd, params_em,
//                 &cusolverHandle, &cublasHandle);
// }
//
//
// void allocate_memory(){
//         allocate(data_d, nDim * nObs);
//         allocate(mus_d, nDim * nCl);
//         allocate(sigmas_d, nDim *nDim * nCl);
//         allocate(ps_d, nCl);
//
//         data_h = (T *)malloc(nDim * nObs * sizeof(T));
//         mus_h = (T *)malloc(nDim * nCl * sizeof(T));
//         sigmas_h = (T *)malloc(nDim * nDim * nCl * sizeof(T));
//         ps_h = (T *)malloc(nCl * sizeof(T));
//
//         allocate(error_d, 1);
//
//         CUBLAS_CHECK(cublasCreate(&cublasHandle));
//         CUSOLVER_CHECK(cusolverDnCreate(&cusolverHandle));
// }
//
//
// void initialize_data(){
//
//
//         T data[] = {-1.00244764, -0.99180406, -1.01255938, -1.01529848, -0.99818674,
//                     -1.0063647, -0.99772352, -1.01088834, -1.00519181, -1.01321278};
//         T data2[] = {5.12011711, 5.0542267, 4.90866965, 4.80215899, 5.07620704,
//                      5.07082148, 5.11350811, 5.060283, 4.97280651, 5.10266182,
//                      5.01221184, 5.16219858, 5.00311359, 4.9516878, 5.00995328,
//                      4.91233788, 5.1927453, 4.98450732, 5.05447048, 5.04727952,
//                      5.05325614, 5.18202476, 5.02159417, 4.96978688, 5.04032078,
//                      5.15322225, 5.09355578, 4.87507089, 5.15266876, 5.02414508,
//                      4.97050659, 5.10162411, 4.94815264, 5.01037141, 5.02655141,
//                      4.97076149, 5.099396, 5.05406731, 5.06445034, 5.02351308,
//                      4.80839546, 5.01836235, 4.85482882, 4.71953188, 4.98732387,
//                      4.93715448, 4.87173272, 4.96534476, 4.9798542, 4.94890233,
//                      4.96081575, 5.05475121, 5.10087884, 5.00677598, 4.97584568,
//                      4.9876678, 4.90430707, 5.05282888, 5.05456455, 4.97341481,
//                      4.96079878, 4.89705711, 5.11486365, 4.95939813, 4.97461712,
//                      5.10901276, 4.96503741, 5.13764613, 5.11478684, 4.98587892,
//                      4.84740951, 5.07641578, 4.92305138, 5.04894517, 5.07326954,
//                      4.99805807, 5.02216814, 5.05096778, 5.0518506, 5.07835277,
//                      4.96328265, 5.01868385, 5.00992409, 5.04187129, 5.16899261,
//                      4.96513055, 5.05736929, 4.96854749, 4.93183931, 5.05763236,
//                      5.00418924, 4.87041479, 5.08546032, 4.94536309, 5.02689104,
//                      5.08068905, 4.89939977, 5.04180028, 5.17622495, 4.94689816};
//
//         for (size_t i = 0; i < 10; i++) {
//                 data_h[i] = (T) data[i];
//         }
//         for (size_t i = 0; i < 100; i++) {
//                 data_h[i + 10] = (T) data2[i];
//         }
//
//         mus_h[0] = -1.0;
//         mus_h[1] = 5;
//
//         ps_h[0] = (T) 1/11;
//         ps_h[1] = (T) 10 /11;
//
//         sigmas_h[0] = 0.01;
//         sigmas_h[1] = 0.1;
//
// }
//
// void copy_to_device(){
//         updateDevice(data_d, data_h, nDim * nObs);
//         updateDevice(mus_d, mus_h, nDim * nCl);
//         updateDevice(sigmas_d, sigmas_h, nDim *nDim * nCl);
//         updateDevice(ps_d, ps_h, nCl);
// }
//
// void TearDown() override {
//         free(data_h);
//         free(mus_h);
//         free(sigmas_h);
//         free(ps_h);
//
//         CUDA_CHECK(cudaFree(data_d));
//         CUDA_CHECK(cudaFree(mus_d));
//         CUDA_CHECK(cudaFree(sigmas_d));
//         CUDA_CHECK(cudaFree(ps_d));
//
//         CUDA_CHECK(cudaFree(error_d));
//
//         CUSOLVER_CHECK(cusolverDnDestroy(cusolverHandle));
//         CUBLAS_CHECK(cublasDestroy(cublasHandle));
//
//         delete params_em;
//         delete params_rd;
//
//         TearDownGMM(gmm);
// }
//
// protected:
// GMM<T> gmm;
//
// GMMInputs<T> params;
// T error, tolerance;
// int nDim, nCl, nObs;
//
// int n_iter;
//
// T *data_h, *mus_h, *sigmas_h, *ps_h;
// T *data_d, *mus_d, *sigmas_d, *ps_d;
//
// cusolverDnHandle_t cusolverHandle;
// cublasHandle_t cublasHandle;
//
// paramsEM *params_em;
// paramsRandom<T> *params_rd;
//
//
// T *error_d, temp_err;
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
// // INSTANTIATE_TEST_CASE_P(GMMTests, GMMTestD,
// //                         ::testing::ValuesIn(inputsd2));
//
// } // end namespace HMM
// } // end namespace MLCommon
