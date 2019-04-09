// /*
//  * Copyright (c) 2019, NVIDIA CORPORATION.
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  *     http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */
//
// #pragma once
//
// #include <stdlib.h>
// #include <vector>
//
//
//
// namespace hmm {
//

//
// template <typename T>
// __global__
// void TestWrite(T* dA,
//                int m, int n, int ldda,
//                int nThreads_x, int nThreads_y){
//         int i_start = threadIdx.x + blockDim.x * blockIdx.x;
//         int j_start = threadIdx.y + blockDim.y * blockIdx.y;
//
//         for (size_t i = i_start; i < m; i+=nThreads_x) {
//                 for (size_t j = j_start; j < n; j+=nThreads_y) {
//                         // dA[IDX(i, j, ldda)] = (unsigned short int) 10;
//                         printf("%f\n", (float) dA[IDX(i, j, ldda)]);
//                 }
//
//         }
// }
// template <typename T>
// __global__
// void createSigmasBatchesKernel(int nCl,
//                                T **dX_batches, T **dmu_batches, T **dsigma_batches,
//                                T *dX, magma_int_t lddx,
//                                T *dmu,  magma_int_t lddmu,
//                                T *dsigma,  magma_int_t lddsigma, magma_int_t lddsigma_full,
//                                int nThreads_x){
//         int i_start = threadIdx.x + blockDim.x * blockIdx.x;
//
//         for (size_t clId = i_start; clId < nCl; clId+=nThreads_x) {
//                 dX_batches[clId] = dX;
//                 dmu_batches[clId] = dmu + IDX(0, clId, lddmu);
//                 dsigma_batches[clId] = dsigma + IDX(0, clId, lddsigma_full);
//         }
// }
//
// template <typename T, typename D>
// void copy_pis(T* dPi_array, std::vector<D> dists){
//         dim3 block(32, 1, 1);
//         dim3 grid(ceildiv((int) nCl, (int) block.x), 1, 1);
//
//         int nThreads_x = grid.x * block.x;
//
//         copy_pis_kernel<T> <<< grid, block >>>(nCl,
//                                                dX_batches, dmu_batches, dsigma_batches,
//                                                dX, lddx,
//                                                dmu,  lddmu,
//                                                dsigma, lddsigma, lddsigma_full,
//                                                nThreads_x);
//
//         CUDA_CHECK(cudaPeekAtLastError());
// }
//
//
// }
