// #pragma once
//
// #include <cublas_v2.h>
//
// #include "cuda_utils.h"
// #include "linalg/cublas_wrappers.h"
// #include "linalg/cusolver_wrappers.h"
//
//
// #include <thrust/transform.h>
// #include <thrust/reduce.h>
// #include <thrust/iterator/counting_iterator.h>
// #include <thrust/iterator/transform_iterator.h>
//
// namespace MLCommon {
//
//
// template <typename T>
// __device__
// T sign(T x){
//         if (x > 0)
//                 return (T) 1;
//         else if (x < 0)
//                 return (T) -1;
//         else
//                 return 0;
// }
//
// template <typename T>
// __global__
// void diag_kernel(int n, T* dU, int lddu){
//         int idxThread = threadIdx.x + blockDim.x * blockIdx.x;
//         if(idxThread == 0) {
//                 T sg = 1;
//                 T det = 0;
//                 for (size_t j = 0; j < n; j++) {
//                         det += std::log(std::abs(dU[IDX(j, j, lddu)]));
//                         sg *= sign(dU[IDX(j, j, lddu)]);
//                 }
//                 return sg * std::exp(det);
//         }
// }
//
// }
//
// template <typename T>
// struct Determinant
// {
//         int n, ldda;
//         T *tempM;
//         bool is_hermitian;
//
//         int *devIpiv;
//
//         int *info, info_h;
//         cusolverDnHandle_t *handle;
//         cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
//         int WsSize;
//         T *Ws = NULL;
//
//         Determinant(int _n, int _ldda, cusolverDnHandle_t *_handle,
//                     bool _is_hermitian){
//                 n = _n;
//                 ldda = _ldda;
//                 is_hermitian = _is_hermitian;
//                 handle = _handle;
//
//
//                 if (is_hermitian) {
//                         CUSOLVER_CHECK(LinAlg::cusolverDnpotrf_bufferSize(*handle, uplo, n, tempM, ldda, &WsSize));
//
//                 }
//                 else
//                 {
//                         CUSOLVER_CHECK(LinAlg::cusolverDngetrf_bufferSize(*handle,  n, n, tempM, ldda, &WsSize));
//                         allocate(devIpiv, n);
//                 }
//
//                 allocate(Ws, WsSize);
//                 allocate(info, 1);
//                 allocate(tempM, ldda * n);
//         }
//
//         T compute(T* M){
//                 copy(tempM, M, ldda * n);
//
//                 if(is_hermitian) {
//                         // CUSOLVER_CHECK(LinAlg::cusolverDnpotrf(*handle,
//                         //                                        uplo,
//                         //                                        n,
//                         //                                        tempM,
//                         //                                        ldda,
//                         //                                        Ws,
//                         //                                        WsSize,
//                         //                                        info));
//                         printf("Hermitian not yet tested\n");
//                 }
//                 else{
//                         CUSOLVER_CHECK(LinAlg::cusolverDngetrf(*handle,
//                                                                n,
//                                                                n,
//                                                                tempM,
//                                                                ldda,
//                                                                Ws,
//                                                                devIpiv,
//                                                                info));
//
//                 }
//                 updateHost(&info_h, info, 1);
//                 ASSERT(info_h == 0,
//                        "sigma: error in determinant, info=%d | expected=0", info_h);
//
//                 return diag_kernel<<< 1, 1 >>>(n, tempM, ldda);
//         }
//
//         void TearDown() {
//                 CUDA_CHECK(cudaFree(Ws));
//                 CUDA_CHECK(cudaFree(tempM));
//                 CUDA_CHECK(cudaFree(info));
//
//                 if (is_hermitian) {
//                         CUDA_CHECK(cudaFree(devIpiv));
//                 }
//         }
//
// };
//
// }
