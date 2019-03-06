#pragma once

#include "hmm/magma/b_bilinear.h"
#include "hmm/magma/b_inverse.h"
#include "hmm/magma/b_determinant.h"

using namespace MLCommon;
using namespace MLCommon::LinAlg;


// TODO : ADD cudaFree

template <typename T>
__global__
void createLlhdBatchesKernel(int nObs, int nCl,
                             T **dX_batches, T **dmu_batches, T **dInvSigma_batches,
                             T **dX_array, T **dmu_array, T **dInvSigma_array,
                             int nThreads_x, int nThreads_y){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;

        for (size_t obsId = j_start; obsId < nObs; obsId+=nThreads_y) {
                for (size_t clId = i_start; clId < nCl; clId+=nThreads_x) {

                        // if(j_start == 0 && i_start == 0) {
                        //         dX_array[0][0] = 0.;
                        // }
                        // dX_batches[IDX(clId, obsId, nCl)] = NULL;
                        size_t idx = IDX(clId, obsId, nCl);
                        dX_batches[idx] = dX_array[obsId];
                        dmu_batches[idx] = dmu_array[clId];
                        dInvSigma_batches[idx] = dInvSigma_array[clId];
                }
        }
}

template <typename T>
void create_llhd_batches(int nObs, int nCl,
                         T **&dX_batches, T **&dmu_batches, T **&dInvSigma_batches,
                         T **&dX_array, T **&dmu_array, T **&dInvSigma_array){
        dim3 block(32, 32, 1);
        dim3 grid(ceildiv((int) nCl, (int) block.x),
                  ceildiv((int) nObs, (int) block.y),
                  1);

        int nThreads_x = grid.x * block.x;
        int nThreads_y = grid.y * block.y;

        createLlhdBatchesKernel<T> <<< grid, block >>>(nObs, nCl,
                                                       dX_batches, dmu_batches, dInvSigma_batches,
                                                       dX_array, dmu_array, dInvSigma_array,
                                                       nThreads_x, nThreads_y);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}


template <typename T>
__global__
void subtractBatchedKernel(magma_int_t m, magma_int_t n, magma_int_t batchCount,
                           T** dO_array, magma_int_t lddO,
                           T** dX_array, magma_int_t lddx,
                           T** dY_array, magma_int_t lddy,
                           int nThreads_x, int nThreads_y, int nThreads_z){

        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;
        int k_start = threadIdx.z + blockDim.z * blockIdx.z;

        int idxO, idxX, idxY;
        // TODO : Check the difference

        for (size_t bId = k_start; bId < batchCount; bId+=nThreads_z) {
                for (size_t j = j_start; j < n; j+=nThreads_x) {
                        for (size_t i = i_start; i < m; i+=nThreads_y) {
                                idxO = IDX(i, j, lddO);
                                idxX = IDX(i, j, lddx);
                                idxY = IDX(i, 0, lddy);
                                dO_array[bId][idxO] = dX_array[bId][idxX] - dY_array[bId][idxY];
                        }
                }
        }
}

template <typename T>
void subtract_batched(magma_int_t m, magma_int_t n, magma_int_t batchCount,
                      T** dO_array, magma_int_t lddO,
                      T** dX_array, magma_int_t lddx,
                      T** dY_array, magma_int_t lddy){
        dim3 block(32, 32, 1);
        dim3 grid(ceildiv((int)n, (int)block.y),
                  ceildiv((int)m, (int)block.z),
                  1);

        int nThreads_x = grid.x * block.x;
        int nThreads_y = grid.y * block.y;
        int nThreads_z = grid.z * block.z;

        subtractBatchedKernel<T> <<< grid, block >>>(m, n, batchCount,
                                                     dO_array, lddO,
                                                     dX_array, lddx,
                                                     dY_array, lddy,
                                                     nThreads_x, nThreads_y, nThreads_z);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__host__ __device__
T lol_llhd_atomic(T det, T bil, int nDim){
        // T EPS = 1e-6;
        return -0.5 * (std::log(det) + nDim * std::log(2 * M_PI) + bil);
}

template <typename T>
__global__
void LogLikelihoodKernel(int nObs, int nCl, int nDim,
                         T* dDet_array, T* dBil_batches,
                         T* dLlhd, int lddLlhd, bool isLog,
                         int nThreads_x, int nThreads_y){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;
        int idx;
        for (size_t clId = i_start; clId < nCl; clId+=nThreads_x) {
                for (size_t oId = j_start; oId < nObs; oId+=nThreads_y) {
                        idx = IDX(clId, oId, lddLlhd);
                        // printf("%f \n", lol_llhd_atomic(dDet_array[clId],
                        //                                 dBil_batches[idx],
                        //                                 nDim));
                        dLlhd[idx] = lol_llhd_atomic(dDet_array[clId],
                                                     dBil_batches[idx],
                                                     nDim);
                        if (!isLog) {
                                dLlhd[idx] = std::exp(dLlhd[idx]);
                        }
                }
        }
}


template <typename T>
void _likelihood_batched(int nObs, int nCl, int nDim,
                         T* dDet_array, T* dBil_batches,
                         T* dLlhd, int lddLlhd, bool isLog){
        dim3 block(32, 32, 1);
        dim3 grid(ceildiv(nCl, (int)block.x),
                  ceildiv(nObs, (int)block.y),
                  1);

        int nThreads_x = grid.x * block.x;
        int nThreads_y = grid.y * block.y;

        LogLikelihoodKernel<T> <<< grid, block >>>(nObs, nCl, nDim,
                                                   dDet_array, dBil_batches,
                                                   dLlhd, lddLlhd, isLog,
                                                   nThreads_x, nThreads_y);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());

}

template <typename T>
void likelihood_batched(magma_int_t nCl, magma_int_t nDim,
                        magma_int_t nObs,
                        T** &dX_array, int lddx,
                        T** &dmu_array, int lddmu,
                        T** &dsigma_array, int lddsigma_full, int lddsigma,
                        T* dLlhd, int lddLlhd,
                        bool isLog){

// Allocate
        T **dInvSigma_array=NULL, *dDet_array=NULL;
        T **dX_batches=NULL, **dmu_batches=NULL,
        **dInvSigma_batches=NULL, **dDiff_batches=NULL;
        T *dBil_batches=NULL;


        magma_int_t batchCount = nObs * nCl;

        allocate_pointer_array(dInvSigma_array, lddsigma_full, nCl);
        allocate(dDet_array, nCl);

        allocate(dBil_batches, batchCount);
        allocate(dX_batches, batchCount);
        allocate(dmu_batches, batchCount);
        allocate(dInvSigma_batches, batchCount);
        allocate_pointer_array(dDiff_batches, lddx, batchCount);


        int device = 0;    // CUDA device ID
        magma_queue_t queue;
        magma_queue_create(device, &queue);

// Compute sigma inverses
        // print_matrix_batched(nDim, nDim, nCl, dsigma_array, lddsigma, "dSigma matrix");

        // print_matrix_batched(nDim, nDim, nCl, dInvSigma_array, lddsigma, "dInvSigma_array before");

        inverse_batched(nDim, dsigma_array, lddsigma, dInvSigma_array, nCl, queue);

        // print_matrix_batched(nDim, nDim, nCl, dInvSigma_array, lddsigma, "dInvSigma_array");
// Compute sigma inv dets
        det_batched(nDim, dInvSigma_array, lddsigma, dDet_array, nCl, queue);
        // print_matrix_device(nCl, 1, dDet_array, nCl, "dDet_array");

// Create batches
        // print_matrix_batched(nDim, nDim, nCl*nObs, dInvSigma_batches, lddsigma, "dInvSigma_batches before");
        // print_matrix_batched(nDim, 1, nObs, dX_array, lddx, "dX_array");

        create_llhd_batches(nObs, nCl,
                            dX_batches, dmu_batches, dInvSigma_batches,
                            dX_array, dmu_array, dInvSigma_array);

        // print_matrix_batched(nDim, 1, nObs * nCl, dX_batches, lddx, "dX_batches");
        //
        //
        // print_matrix_batched(nDim, nDim, nCl*nObs, dInvSigma_batches, lddsigma, "dInvSigma_batches");


// Compute diffs
        subtract_batched(nDim, 1, batchCount,
                         dDiff_batches, lddx,
                         dX_batches, lddx,
                         dmu_batches, lddmu);

        // Compute bilinears
        bilinear_batched(nDim, nDim,
                         dDiff_batches, dInvSigma_batches, lddsigma,
                         dDiff_batches, dBil_batches, batchCount, queue);

        // print_matrix_device(batchCount, 1, dBil_batches, batchCount, "dBil_batches");
        // print_matrix_device(nCl, 1, dDet_array, nCl, "dDet_array");


        // Compute log likelihoods
        _likelihood_batched(nObs, nCl, nDim,
                            dDet_array, dBil_batches, dLlhd, lddLlhd, isLog);
        // print_matrix_device(nCl, nObs, dLlhd, lddLlhd, "dLlhd");

        // print_matrix_batched(nDim, nDim, nCl, dsigma_array, lddsigma, "dSigma matrix");



// free
        // free_pointer_array(dInvSigma_array);
        // CUDA_CHECK(cudaFree(dBil_batches));
        // CUDA_CHECK(cudaFree(dDet_array));
}
