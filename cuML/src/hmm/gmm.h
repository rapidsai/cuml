#pragma once

// #include <hmm/structs.h>
#include <hmm/magma/b_likelihood.h>
// #include <hmm/cuda/stats.h>
#include <hmm/cuda/random.h>

#include <stats/sum.h>
#include <linalg/sqrt.h>
#include <ml_utils.h>

using namespace MLCommon::HMM;
using namespace MLCommon;


template <typename T>
__global__
void dgmmBatchedKernel(magma_int_t m, magma_int_t n, magma_int_t batchCount,
                       T** dO_array, magma_int_t lddO,
                       T** dX_array, magma_int_t lddx,
                       T* dD_array, magma_int_t lddd,
                       int nThreads_x, int nThreads_y, int nThreads_z){

        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;
        int k_start = threadIdx.z + blockDim.z * blockIdx.z;

        int idxO, idxX, idxD;

        for (size_t bId = k_start; bId < batchCount; bId+=nThreads_z) {
                for (size_t j = j_start; j < n; j+=nThreads_x) {
                        for (size_t i = i_start; i < m; i+=nThreads_y) {
                                idxO = IDX(i, j, lddO);
                                idxX = IDX(i, j, lddx);
                                idxD = IDX(bId, i, lddd);
                                dO_array[bId][idxO] = dX_array[bId][idxX] * dD_array[idxD];
                        }
                }
        }
}

template <typename T>
void dgmm_batched(magma_int_t m, magma_int_t n, magma_int_t batchCount,
                  T** dO_array, magma_int_t lddO,
                  T** dX_array, magma_int_t lddx,
                  T* dD_array, magma_int_t lddd){
        dim3 block(32, 32, 1);
        dim3 grid(ceildiv((int)n, (int)block.y),
                  ceildiv((int)m, (int)block.z),
                  1);

        int nThreads_x = grid.x * block.x;
        int nThreads_y = grid.y * block.y;
        int nThreads_z = grid.z * block.z;

        dgmmBatchedKernel<T> <<< grid, block >>>(m, n, batchCount,
                                                 dO_array, lddO,
                                                 dX_array, lddx,
                                                 dD_array, lddd,
                                                 nThreads_x, nThreads_y, nThreads_z);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}
//
template <typename math_t>
void square(math_t *out, const math_t *in, int len,
            cudaStream_t stream = 0) {
        unaryOp(out, in, len,
                [] __device__(math_t in) {
                        return in * in;
                },
                stream);
}

template <typename T>
__global__
void createSigmasBatchesKernel(int nCl,
                               T **dX_batches, T **dmu_batches, T **dsigma_batches,
                               T *dX, magma_int_t lddx,
                               T *dmu,  magma_int_t lddmu,
                               T *dsigma,  magma_int_t lddsigma, magma_int_t lddsigma_full,
                               int nThreads_x){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;

        for (size_t clId = i_start; clId < nCl; clId+=nThreads_x) {
                dX_batches[clId] = dX;
                dmu_batches[clId] = dmu + IDX(0, clId, lddmu);
                dsigma_batches[clId] = dsigma + IDX(0, clId, lddsigma_full);
        }
}

template <typename T>
void create_sigmas_batches(int nCl,
                           T **&dX_batches, T **&dmu_batches, T **&dsigma_batches,
                           T *&dX, magma_int_t lddx,
                           T *&dmu,  magma_int_t lddmu,
                           T *&dsigma,  magma_int_t lddsigma, magma_int_t lddsigma_full){
        dim3 block(32, 1, 1);
        dim3 grid(ceildiv((int) nCl, (int) block.x), 1, 1);

        int nThreads_x = grid.x * block.x;

        createSigmasBatchesKernel<T> <<< grid, block >>>(nCl,
                                                         dX_batches, dmu_batches, dsigma_batches,
                                                         dX, lddx,
                                                         dmu,  lddmu,
                                                         dsigma, lddsigma, lddsigma_full,
                                                         nThreads_x);

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}

template <typename math_t>
void inverse(math_t *out, const math_t *in, int len,
             cudaStream_t stream = 0) {
        unaryOp(out, in, len,
                [] __device__(math_t in) {
                        return 1 / in;
                },
                stream);
}

template <typename T>
void generate_trans_matrix(magma_int_t m, magma_int_t n, T* dA, magma_int_t lda, bool colwise){
        fill_matrix_gpu(m, n, dA, lda, false);
        normalize_matrix(m, n, dA, lda, colwise);
}

template <typename T>
struct GMM {
        T *dX, *dmu, *dsigma, *dPis, *dPis_inv, *dLlhd;
        T **dX_array=NULL, **dmu_array=NULL, **dsigma_array=NULL;

        magma_int_t lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd;

        int nCl, nDim, nObs;

        T start, end;
        unsigned long long seed;

        bool initialized=false;
        cublasHandle_t cublasHandle;
        magma_queue_t queue;

        GMM(int _nCl, int _nDim, int _nObs) {
                nDim = _nDim;
                nCl = _nCl;
                nObs = _nObs;

                lddx = magma_roundup(nDim, RUP_SIZE);
                lddmu = magma_roundup(nDim, RUP_SIZE);
                lddsigma = magma_roundup(nDim, RUP_SIZE);
                lddsigma_full = nDim * lddsigma;
                lddLlhd = magma_roundup(nCl, RUP_SIZE);
                lddPis = nObs;

                // Random parameters
                start=0;
                end = 1;
                seed = 1234ULL;

                allocate(dX, lddx * nObs);
                allocate(dmu, lddmu * nCl);
                allocate(dsigma, lddsigma_full * nCl);
                allocate(dLlhd, lddLlhd * nObs);
                allocate(dPis, nObs);
                allocate(dPis_inv, nObs);

                // Batches allocations
                allocate(dX_array, nObs);
                allocate_pointer_array(dmu_array, lddmu, nCl);
                allocate_pointer_array(dsigma_array, lddsigma_full, nCl);

                CUBLAS_CHECK(cublasCreate(&cublasHandle));
                int device = 0;    // CUDA device ID
                magma_queue_create(device, &queue);
        }

        void initialize() {
                random_matrix_batched(nDim, 1, nCl, dmu, lddmu, false, seed, start, end);
                random_matrix_batched(nDim, nDim, nCl, dsigma, lddsigma, true, seed, start, end);
                generate_trans_matrix(nCl, nObs, dLlhd, lddLlhd, false);
                generate_trans_matrix(nObs, 1, dPis, nObs, false);
                initialized = true;
        }

        void free(){
                CUDA_CHECK(cudaFree(dmu_array));
                CUDA_CHECK(cudaFree(dsigma_array));
                CUDA_CHECK(cudaFree(dLlhd));
                CUDA_CHECK(cudaFree(dPis));
                CUDA_CHECK(cudaFree(dPis_inv));
                CUBLAS_CHECK(cublasDestroy(cublasHandle));
        }

        void fit(T* dX, int n_iter) {
                _em(dX, n_iter);
        }

        void update_rhos(T* dX){
                printf("*************** update rhos\n");

                bool isLog = false;

                // print_matrix_device(nDim, nObs, dX, lddx, "dx matrix");
                // // print_matrix_device(nDim, nDim, nCl, dsigma_batches, lddsigma, "dsigma matrix");
                // print_matrix_device(nDim, 1, dPis, lddPis, "dPis");
                // print_matrix_device(nDim, 1, dmu, lddmu, "dmu matrix");

                split_to_batches(nObs, dX_array, dX, lddx);
                split_to_batches(nCl, dmu_array, dmu, lddmu);
                split_to_batches(nCl, dsigma_array, dsigma, lddsigma_full);

                print_matrix_batched(nDim, 1, nObs, dX_array, lddx, "dx matrix");
                print_matrix_batched(nDim, 1, nCl, dmu_array, lddmu, "dmu matrix");
                print_matrix_batched(nDim, nDim, nCl, dsigma_array, lddsigma, "dSigma matrix");

                likelihood_batched(nCl, nDim, nObs,
                                   dX_array, lddx,
                                   dmu_array, lddmu,
                                   dsigma_array, lddsigma_full, lddsigma,
                                   dLlhd, lddLlhd,
                                   isLog);

                cublasdgmm(cublasHandle, CUBLAS_SIDE_RIGHT, nCl, nObs,
                           dLlhd, lddLlhd, dPis, lddPis, dLlhd, lddLlhd);

                normalize_matrix(nCl, nObs, dLlhd, lddLlhd, false);

                // print_matrix_batched(nDim, nDim, nCl, dsigma_array, lddsigma, "dSigma matrix");

                printf(" update rhos **********************\n");
        }

        void update_mus(T* dX){
                T alpha = (T)1.0 / nObs, beta = (T)0.0;
                // printf("  ********************** update mus\n");

                // print_matrix_batched(nDim, nDim, nCl, dsigma_array, lddsigma, "dSigma matrix");
                // print_matrix_device(nDim, nDim, dsigma, lddsigma, "dSigma matrix");

                // print_matrix_device(nDim, nCl, dmu, lddmu, "dmu matrix");
                // print_matrix_device(nDim, nObs, dX, lddx, "dx matrix");
                // print_matrix_device(nCl, nObs, dLlhd, lddLlhd, "dllhd matrix");

                CUBLAS_CHECK(cublasgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, nDim, nCl, nObs, &alpha, dX, lddx, dLlhd, lddLlhd, &beta, dmu, lddmu));
                // magmablas_gemm(MagmaNoTrans, MagmaTrans,
                //                nDim, nObs, nCl,
                //                alpha, dX, lddx, dLlhd, lddLlhd,
                //                beta, dmu, lddmu, queue);
                // print_matrix_device(nDim, nDim, dsigma, lddsigma, "dSigma matrix");

                // print_matrix_device(nDim, nCl, dmu, lddmu, "dmu matrix");

                inverse(dPis_inv, dPis, nCl);
                CUBLAS_CHECK(cublasdgmm(cublasHandle, CUBLAS_SIDE_RIGHT,
                                        nDim, nCl,
                                        dmu, lddmu,
                                        dPis_inv, 1,
                                        dmu, lddmu));
                // printf(" update mus **********************\n");

        }

        void update_sigmas(T* dX){
                T **dX_batches=NULL, **dmu_batches=NULL, **dsigma_batches=NULL,
                **dDiff_batches=NULL;

                int batchCount=nCl;
                int ldDiff= lddx;

                allocate(dX_batches, batchCount);
                allocate(dmu_batches, batchCount);
                allocate(dsigma_batches, batchCount);
                allocate_pointer_array(dDiff_batches, lddx, batchCount);

                create_sigmas_batches(nCl,
                                      dX_batches, dmu_batches, dsigma_batches,
                                      dX, lddx, dmu, lddmu, dsigma, lddsigma, lddsigma_full);

                // Compute diffs
                subtract_batched(nDim, 1, batchCount,
                                 dDiff_batches, ldDiff,
                                 dX_batches, lddx,
                                 dmu_batches, lddmu);

                // Compute sigmas
                sqrt(dLlhd, dLlhd, lddLlhd * nObs);

                // // Split to batches
                dgmm_batched(nDim, nObs, nCl,
                             dsigma_batches, lddsigma,
                             dDiff_batches, lddx,
                             dLlhd, lddLlhd);

                // get the sum of all the covs
                T alpha = (T)1.0 / nObs, beta = (T)0.0;
                magmablas_gemm_batched(MagmaNoTrans, MagmaNoTrans,
                                       nDim, nDim, nObs,
                                       alpha, dDiff_batches, ldDiff,
                                       dDiff_batches, ldDiff, beta,
                                       dsigma_batches, lddsigma, nCl, queue);

                square(dLlhd, dLlhd, lddLlhd * nObs);
        }

        void update_pis(){
                sum(dPis, dLlhd, nCl, nObs, false);
        }


        void _em(T* dX, int n_iter){
                assert(initialized);

                // Run the EM algorithm
                for (int it = 0; it < n_iter; it++) {
                        printf("\n -------------------------- \n");
                        printf(" iteration %d\n", it);

                        // E step
                        update_rhos(dX);


                        // M step
                        update_mus(dX);


                        update_sigmas(dX);
                        update_pis();

                }
        }
//
//         void predict(T* dX, T* doutRho, int nObs){
//                 likelihood_batched(nCl, nDim, nObs,
//                                    dX, lddx,
//                                    dmu, lddmu,
//                                    dsigma, lddsigma_full, lddsigma,
//                                    doutRho, lddLlhd, false);
//         }
//


};
