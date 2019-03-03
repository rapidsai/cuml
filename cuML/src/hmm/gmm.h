#pragma once

// #include <hmm/structs.h>
#include <hmm/magma/b_likelihood.h>
#include <hmm/cuda/stats.h>
#include <hmm/cuda/random.h>

#include <stats/sum.h>
#include <ml_utils.h>

using namespace MLCommon::HMM;
using namespace MLCommon;


template <typename T>
__global__
void dgmmBatchedKernel(magma_int_t m, magma_int_t n, magma_int_t batchCount,
                       T** dO_array, magma_int_t lddO,
                       T** dX_array, magma_int_t lddx,
                       T* dD, magma_int_t lddd,
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
                                idxD = IDX2(j, i, bId, lddd, lddd * n);
                                dO_array[bId][idxO] = dX_array[bId][idxX] * dD[bId][idxY];
                        }
                }
        }
}

template <typename T>
void dgmm_batched(magma_int_t m, magma_int_t n, magma_int_t batchCount,
                  T** dO_array, magma_int_t lddO,
                  T** dX_array, magma_int_t lddx,
                  T** dD_array, magma_int_t lddd){
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
                                                     dD_array, lddd,
                                                     nThreads_x, nThreads_y, nThreads_z);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}

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
void generate_trans_matrix(magma_int_t m, magma_int_t n, T* dA, magma_int_t lda, bool colwise){
        fill_matrix_gpu(m, n, dA, lda);
        normalize_matrix(m, n, dA, lda, colwise);
}

template <typename T>
void weighted_mus(int nDim, int nObs, int nCl,
                  T* dX, int lddx, T* dLlhd, int ldLlhd,
                  T* ps, int lddp, T* mus, int lddmu,
                  cublasHandle_t handle, magma_queue_t queue ){
        T alfa = (T)1.0 / nObs, beta = (T)0.0;

        magma_gemm( magma_trans_t transA, magma_trans_t transB, nDim, nObs, nCl, alpha, dX, lddx, dLlhd, ldLlhd, beta, dmu, ldmu, queue);
        CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nDim,
                                nCl, nObs, &alfa, dX, lddx, dW, lddw, &beta, mus, lddmu));

// TODO : Check the 1 in the next call !!!!!!!!!!!!!!!!
// TODO : Check this part
        CUBLAS_CHECK(cublasdgmm(handle, CUBLAS_SIDE_RIGHT,
                                nDim, nCl, mus, lddmu, ps, nCl, mus, lddmu));
}






template <typename T>
struct GMM {
        T *dX, *dmu, *dsigma, *dPis;
        T *dmu_array, *dsigma_array;
        T *dLlhd;

        magma_int_t ldx, ldmu, ldsigma, ldsigma_full, lddPis, ldLlhd;

        int nCl, nDim, nObs;

        // // Random parameters
        // paramsRandom<T> paramsRd = paramsRandom<T>((T) 0, (T) 1,
        //                                            (unsigned long long) 1234ULL);

        // all the workspace related pointers
        int *info;
        bool initialized=false;
        cublasHandle_t cublasHandle;
        magma_queue_t queue;

        GMM(int _nCl, int _nDim, int _nObs) {
                nDim = _nDim;
                nCl = _nCl;
                nObs = _nObs;

                ldx = magma_roundup(nDim, RUP_SIZE);
                ldmu = magma_roundup(nDim, RUP_SIZE);
                ldsigma = magma_roundup(nDim, RUP_SIZE);
                ldsigma_full = nDim * ldsigma;
                ldLlhd = magma_roundup(nCl, RUP_SIZE);

                allocate(dX, ldx * nObs);
                allocate(dmu, ldmu * nCl);
                allocate(dsigma, ldsigma_full * nCl);
                allocate(dLlhd, ldLlhd * nObs);

                CUBLAS_CHECK(cublasCreate(&cublasHandle));
                magma_queue_create(device, &queue);
        }

        void initialize() {
                random_matrix_batched(nDim, 1, nCl, dmu, ldmu, false);
                random_matrix_batched(nDim, nDim, nCl, dsigma, ldsigma, true);
                generate_trans_matrix(nCl, nObs, dLlhd, ldLlhd, false);
                generate_trans_matrix(nObs, 1, dPis, nObs, false);
                initialized = true;
        }

        void free(){
                CUDA_CHECK(cudaFree(dmu_array));
                CUDA_CHECK(cudaFree(dsigma_array));
                CUDA_CHECK(cudaFree(dLlhd));
                CUDA_CHECK(cudaFree(dPis));
                CUBLAS_CHECK(cublasDestroy(cublasHandle));
        }

        void fit(T* dX, int n_iter) {
                _em(dX, n_iter);
        }


        void compute_dPis(int m, int n, T* dPis, T* dLlhd, int lddLlhd){
                sum(dPis, dLlhd, n, m, false);
        }

        template <typename T>
        void compute_rhos(T* dX, T* dPis){
                likelihood_batched(nCl, nDim, nObs,
                                   dX, ldx,
                                   dmu, ldmu,
                                   dsigma, ldsigma_full, ldsigma,
                                   dLlhd, ldLlhd, isLog);

                cublasdgmm(cublasHandle, CUBLAS_SIDE_RIGHT, m, n,
                           dLlhd, ldLlhd, dPis, nObs, dLlhd,ldLlhd)

// TODO : Check the booleans everywhere
                normalize_matrix(nCl, nObs, dLlhd, lddLlhd, false);
        }

        void _em(T* dX, int n_iter){
                assert(initialized);
                bool isLog = false;

                // Run the EM algorithm
                for (int it = 0; it < n_iter; it++) {
                        printf("\n -------------------------- \n");
                        printf(" iteration %d\n", it);

                        // E step
                        compute_rhos(dX, dPis)

                        // M step
                        weighted_mu(nDim, nObs, nCl,
                                    dX, lddx, dLlhd, ldLlhd,
                                    ps, lddp, mus, lddmu,
                                    handle, queue);

                        _weighted_sigmas(dX);
                        compute_dPis(nCl, nObs, dPis, dLlhd, lddLlhd);

                }
        }

        void predict(T* dX, T* doutRho, int nObs){
                likelihood_batched(nCl, nDim, nObs,
                                   dX, ldx,
                                   dmu, ldmu,
                                   dsigma, ldsigma_full, ldsigma,
                                   doutRho, ldLlhd, false);
        }

        void _weighted_sigmas(T* dX){

                // Compute diffs
                subtract_batched(nDim, 1, batchCount,
                                 dDiff_batches, lddx,
                                 dX_batches, lddx,
                                 dmu_batches, lddmu);

                // Compute sigmas
                sqrt(dLlhd, dLlhd, lddLlhd * nObs);

                // Split to batches

                dgmm_batched(nDim, nObs, nCl,
                             dsigma_array, lddsigma,
                             dX_batches, lddx,
                             dLlhd, lddLlhd);

                // get the sum of all the covs
                T alfa = (T)1.0 / nPts, beta = (T)0.0;
                void magmablas_gemm_batched(MagmaNoTrans, MagmaNoTrans, nDim, nDim, nObs, alpha, dDiff_batches, ldDiff, dDiff_batches, ldDiff, beta, dSigma_batches, lddsigma, nCl, queue);

                square(dLlhd, dLlhd, lddLlhd * nObs);
        }
};
