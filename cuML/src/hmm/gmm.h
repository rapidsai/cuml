#pragma once


#include <cuda.h>
#include <cublas_v2.h>

// #include <hmm/structs.h>
#include <hmm/magma/b_likelihood.h>
#include <hmm/hmm_variables.h>
// #include <hmm/cuda/random.h>

#include <linalg/cublas_wrappers.h>
#include <hmm/cuda/cublas_wrappers.h>

#include <stats/sum.h>
#include <linalg/sqrt.h>
#include <ml_utils.h>

using namespace MLCommon::HMM;
using namespace MLCommon::LinAlg;
using namespace MLCommon;


template <typename T>
__global__
void normalizeMatrixKernel(size_t m, size_t n,
                           T *dA, size_t ldda,
                           T* x, bool colwise,
                           int numThreads_x, int numThreads_y){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;

        for (size_t j = j_start; j < n; j+=numThreads_y) {
                for (size_t i = i_start; i < m; i+=numThreads_x) {
                        if(colwise) {
                                dA[IDX(i, j, ldda)] = dA[IDX(i, j, ldda)] / x[j];
                        }
                        else{
                                dA[IDX(i, j, ldda)] = dA[IDX(i, j, ldda)] / x[i];
                        }
                }
        }
}

template <typename T>
void normalize_matrix(size_t m, size_t n,
                      T *dA, size_t ldda,
                      bool colwise)
{
        dim3 block(32, 32, 1);
        dim3 grid(ceildiv((int)m, (int)block.x),
                  ceildiv((int)n, (int)block.y),
                  1);

        int numThreads_x = grid.x * block.x;
        int numThreads_y = grid.y * block.y;

        T* sums;
        if(colwise) {
                allocate(sums, n);
                MLCommon::Stats::sum(sums, dA, n, ldda, false);
        }
        else{
                allocate(sums, ldda);
                MLCommon::Stats::sum(sums, dA, ldda, n, true);
        }

        normalizeMatrixKernel<T> <<< grid, block >>>(m, n, dA, ldda,
                                                     sums, colwise,
                                                     numThreads_x, numThreads_y);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());

        CUDA_CHECK(cudaFree(sums));
}

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
                for (size_t j = j_start; j < n; j+=nThreads_y) {
                        for (size_t i = i_start; i < m; i+=nThreads_x) {
                                idxO = IDX(i, j, lddO);
                                idxX = IDX(i, j, lddx);
                                idxD = IDX(bId, j, lddd);
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
        dim3 grid(ceildiv((int)m, (int)block.x),
                  ceildiv((int)n, (int)block.y),
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
void setup(GMM<T> &gmm) {
        allocate(gmm.dX_array, gmm.nObs);
        allocate_pointer_array(gmm.dmu_array, gmm.lddmu, gmm.nCl);
        allocate_pointer_array(gmm.dsigma_array, gmm.lddsigma_full, gmm.nCl);

        magma_init();
}

template <typename T>
void init(GMM<T> &gmm,
          T *dmu, T *dsigma, T *dPis, T *dPis_inv, T *dLlhd,
          int lddx, int lddmu, int lddsigma, int lddsigma_full, int lddPis, int lddLlhd,
          int nCl, int nDim, int nObs) {
        gmm.dmu = dmu;
        gmm.dsigma = dsigma;
        gmm.dPis = dPis;
        gmm.dPis_inv = dPis_inv;
        gmm.dLlhd=dLlhd;
        gmm.lddx=lddx;
        gmm.lddmu=lddmu;
        gmm.lddsigma=lddsigma;
        gmm.lddsigma_full=lddsigma_full;
        gmm.lddPis=lddPis;
        gmm.lddLlhd=lddLlhd;
        gmm.nCl=nCl;
        gmm.nDim=nDim;
        gmm.nObs=nObs;
}

template <typename T>
void update_rhos(T* dX, GMM<T>& gmm,
                 cublasHandle_t cublasHandle, magma_queue_t queue){
        printf("*************** update rhos\n");

        bool isLog = false;

        // print_matrix_device(gmm.nDim, gmm.nObs, dX, gmm.lddx, "dx matrix");
        // print_matrix_device(gmm.nCl, gmm.nObs, gmm.dLlhd, gmm.lddLlhd, "dllhd matrix");
        // print_matrix_device(gmm.nDim, gmm.nCl, gmm.dmu, gmm.lddmu, "dmu matrix");
        // print_matrix_device(gmm.nCl, 1, gmm.dPis, gmm.lddPis, "dPis matrix");

        split_to_batches(gmm.nObs, gmm.dX_array, dX, gmm.lddx);
        split_to_batches(gmm.nCl, gmm.dmu_array, gmm.dmu, gmm.lddmu);
        split_to_batches(gmm.nCl, gmm.dsigma_array, gmm.dsigma, gmm.lddsigma_full);

        // print_matrix_batched(gmm.nDim, 1, gmm.nObs, gmm.dX_array, gmm.lddx, "dx matrix");
        // print_matrix_batched(gmm.nDim, 1, gmm.nCl, gmm.dmu_array, gmm.lddmu, "dmu matrix");
        // print_matrix_batched(gmm.nDim, gmm.nDim, gmm.nCl, gmm.dsigma_array, gmm.lddsigma, "dSigma matrix");

        likelihood_batched(gmm.nCl, gmm.nDim, gmm.nObs,
                           gmm.dX_array, gmm.lddx,
                           gmm.dmu_array, gmm.lddmu,
                           gmm.dsigma_array, gmm.lddsigma_full, gmm.lddsigma,
                           gmm.dLlhd, gmm.lddLlhd,
                           isLog);

        // print_matrix_device(gmm.nCl, gmm.nObs, gmm.dLlhd, gmm.lddLlhd, "dllhd matrix before ");

        cublasdgmm(cublasHandle, CUBLAS_SIDE_LEFT, gmm.nCl, gmm.nObs,
                   gmm.dLlhd, gmm.lddLlhd, gmm.dPis, 1, gmm.dLlhd, gmm.lddLlhd);

        // print_matrix_device(gmm.nCl, gmm.nObs, gmm.dLlhd, gmm.lddLlhd, "dllhd matrix after");

        normalize_matrix(gmm.nCl, gmm.nObs, gmm.dLlhd, gmm.lddLlhd, true);

        // print_matrix_batched(gmm.nDim, gmm.nDim, gmm.nCl, gmm.dsigma_array, gmm.lddsigma, "dSigma matrix");
        // printf(" update rhos after ***** n");
        // print_matrix_device(gmm.nDim, gmm.nObs, dX, gmm.lddx, "dx matrix");
        // print_matrix_device(gmm.nCl, gmm.nObs, gmm.dLlhd, gmm.lddLlhd, "dllhd matrix");
        // print_matrix_device(gmm.nDim, gmm.nCl, gmm.dmu, gmm.lddmu, "dmu matrix");
        // print_matrix_device(gmm.nCl, 1, gmm.dPis, gmm.lddPis, "dPis matrix");
        // printf(" update rhos **********************\n");
}

template <typename T>
void update_mus(T* dX, GMM<T>& gmm,
                cublasHandle_t cublasHandle, magma_queue_t queue){
        T alpha = (T)1.0 / gmm.nObs, beta = (T)0.0;
        // printf("  ********************** update mus\n");

        // print_matrix_device(gmm.nDim, gmm.nDim, gmm.dsigma, gmm.lddsigma, "dSigma matrix");

        // print_matrix_device(gmm.nDim, gmm.nCl, gmm.dmu, gmm.lddmu, "dmu matrix");
        // print_matrix_device(gmm.nDim, gmm.nObs, dX, gmm.lddx, "dx matrix");
        // print_matrix_device(gmm.nDim, gmm.nDim, gmm.dsigma, gmm.lddsigma, "dSigma matrix");
        // print_matrix_device(gmm.nCl, gmm.nObs, gmm.dLlhd, gmm.lddLlhd, "dllhd matrix");

        CUBLAS_CHECK(cublasgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, gmm.nDim, gmm.nCl, gmm.nObs, &alpha, dX, gmm.lddx, gmm.dLlhd, gmm.lddLlhd, &beta, gmm.dmu, gmm.lddmu));
        // magmablas_gemm(MagmaNoTrans, MagmaTrans,
        //                gmm.nDim, gmm.nObs, gmm.nCl,
        //                alpha, dX, gmm.lddx, gmm.dLlhd, gmm.lddLlhd,
        //                beta, gmm.dmu, gmm.lddmu, queue);

        // print_matrix_device(gmm.nCl, 1, gmm.dPis_inv, gmm.lddPis, "dPis inv matrix");
        CUBLAS_CHECK(cublasdgmm(cublasHandle, CUBLAS_SIDE_RIGHT,
                                gmm.nDim, gmm.nCl,
                                gmm.dmu, gmm.lddmu,
                                gmm.dPis_inv, 1,
                                gmm.dmu, gmm.lddmu));
        // print_matrix_device(gmm.nDim, gmm.nCl, gmm.dmu, gmm.lddmu, "dmu matrix");

        // print_matrix_device(gmm.nCl, 1, gmm.dPis, gmm.lddPis, "dPis matrix");

        inverse(gmm.dPis_inv, gmm.dPis, gmm.nCl);

        // printf(" update mus **********************\n");

}

template <typename T>
void update_sigmas(T* dX, GMM<T>& gmm,
                   cublasHandle_t cublasHandle, magma_queue_t queue){
        // printf("\n\n update sigmas **********************\n");
        // print_matrix_device(gmm.nDim, gmm.nObs, dX, gmm.lddx, "dx matrix");
        // print_matrix_device(gmm.nCl, gmm.nObs, gmm.dLlhd, gmm.lddLlhd, "dllhd matrix");
        // print_matrix_device(gmm.nDim, gmm.nCl, gmm.dmu, gmm.lddmu, "dmu matrix");

        T **dX_batches=NULL, **dmu_batches=NULL, **dsigma_batches=NULL,
        **dDiff_batches=NULL;

        int batchCount=gmm.nCl;
        int ldDiff= gmm.lddx;

        allocate(dX_batches, batchCount);
        allocate(dmu_batches, batchCount);
        allocate(dsigma_batches, batchCount);
        allocate_pointer_array(dDiff_batches, gmm.lddx * gmm.nObs, batchCount);

        create_sigmas_batches(gmm.nCl,
                              dX_batches, dmu_batches, dsigma_batches,
                              dX, gmm.lddx, gmm.dmu, gmm.lddmu, gmm.dsigma, gmm.lddsigma, gmm.lddsigma_full);


        // print_matrix_batched(gmm.nDim, gmm.nObs, gmm.nCl, dDiff_batches, ldDiff, "dDiff batches");
        // print_matrix_batched(gmm.nDim, gmm.nObs, gmm.nCl, dX_batches, gmm.lddsigma, "dx batches");
        // print_matrix_batched(gmm.nDim, 1, gmm.nCl, dmu_batches, gmm.lddmu, "dmu batches");

        // Compute diffs
        // printf("-------- nObs %d\n", gmm.nObs);
        subtract_batched(gmm.nDim, gmm.nObs, batchCount,
                         dDiff_batches, ldDiff,
                         dX_batches, gmm.lddx,
                         dmu_batches, gmm.lddmu);


        // Compute sigmas
        sqrt(gmm.dLlhd, gmm.dLlhd, gmm.lddLlhd * gmm.nObs);

        // print_matrix_batched(gmm.nDim, gmm.nObs, gmm.nCl, dDiff_batches, ldDiff, "dDiff matrix");

        dgmm_batched(gmm.nDim, gmm.nObs, gmm.nCl,
                     dDiff_batches, ldDiff,
                     dDiff_batches, ldDiff,
                     gmm.dLlhd, gmm.lddLlhd);

        // print_matrix_batched(gmm.nDim, gmm.nObs, gmm.nCl, dDiff_batches, ldDiff, "dDiff matrix");

        // get the sum of all the covs
        T alpha = (T) 1.0 / gmm.nObs, beta = (T)0.0;
        magmablas_gemm_batched(MagmaNoTrans, MagmaTrans,
                               gmm.nDim, gmm.nDim, gmm.nObs,
                               alpha, dDiff_batches, ldDiff,
                               dDiff_batches, ldDiff, beta,
                               dsigma_batches, gmm.lddsigma, gmm.nCl, queue);


        // cublasgemmBatched(cublasHandle,
        //                   CUBLAS_OP_N,
        //                   CUBLAS_OP_T,
        //                   gmm.nDim, gmm.nDim, gmm.nObs,
        //                   &alpha,
        //                   dDiff_batches, ldDiff,
        //                   dDiff_batches, ldDiff,
        //                   &beta,
        //                   dsigma_batches, gmm.lddsigma,
        //                   gmm.nCl);

        // print_matrix_batched(gmm.nDim, gmm.nDim, gmm.nCl, dsigma_batches, gmm.lddsigma, "dSigma matrix after gemm");


        // Normalize with respect to N_k
        inverse(gmm.dPis_inv, gmm.dPis, gmm.nCl);
        // print_matrix_device(gmm.nCl, 1, gmm.dPis_inv, gmm.lddPis, "dPis inv matrix");

        CUBLAS_CHECK(cublasdgmm(cublasHandle, CUBLAS_SIDE_RIGHT,
                                gmm.lddsigma_full, gmm.nCl,
                                gmm.dsigma, gmm.lddsigma_full,
                                gmm.dPis_inv, 1,
                                gmm.dsigma, gmm.lddsigma_full));

        square(gmm.dLlhd, gmm.dLlhd, gmm.lddLlhd * gmm.nObs);

        // printf(" ************** after ***********\n");
        // print_matrix_device(gmm.nDim, gmm.nObs, dX, gmm.lddx, "dx matrix");
        // print_matrix_device(gmm.nCl, gmm.nObs, gmm.dLlhd, gmm.lddLlhd, "dllhd matrix");
        // print_matrix_batched(gmm.nDim, gmm.nDim, gmm.nCl, dsigma_batches, gmm.lddsigma, "dSigma matrix");
        //
        // printf(" update sigmas **********************\n\n\n");

}

template <typename T>
void update_pis(GMM<T>& gmm){
        // print_matrix_device(gmm.lddPis, 1, gmm.dPis, gmm.lddPis, "dPis matrix");
        // print_matrix_device(gmm.nCl, gmm.nObs, gmm.dLlhd, gmm.lddLlhd, "dllhd matrix");

        MLCommon::Stats::sum(gmm.dPis, gmm.dLlhd, gmm.lddLlhd, gmm.nObs, true);
        normalize_matrix(gmm.lddPis, 1, gmm.dPis, gmm.lddPis, true);
}


template <typename T>
void em(T* dX, int n_iter, GMM<T>& gmm,
        cublasHandle_t cublasHandle, magma_queue_t queue){
        // Run the EM algorithm
        for (int it = 0; it < n_iter; it++) {
                printf("\n -------------------------- \n");
                printf(" iteration %d\n", it);

                // E step
                update_rhos(dX, gmm, cublasHandle, queue);

                // M step
                update_mus(dX, gmm, cublasHandle, queue);
                update_sigmas(dX, gmm, cublasHandle, queue);
                update_pis(gmm);
        }
}

template <typename T>
void fit(T* dX, int n_iter, GMM<T>& gmm,
         cublasHandle_t cublasHandle, magma_queue_t queue) {
        em(dX, n_iter, gmm, cublasHandle, queue);
}

template <typename T>
void free(GMM<T>& gmm){
        CUDA_CHECK(cudaFree(gmm.dX_array));
        CUDA_CHECK(cudaFree(gmm.dsigma_array));
        CUDA_CHECK(cudaFree(gmm.dmu_array));
}
