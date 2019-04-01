#pragma once

#include <gmm/gmm_variables.h>

#include <hmm/magma/magma_test_utils.h>
#include <hmm/cuda/cublas_wrappers.h>
#include <hmm/magma/b_likelihood.h>

#include <cuda.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/sqrt.h>
#include <ml_utils.h>
#include <cublas_v2.h>

// using namespace MLCommon::HMM;
using namespace MLCommon::LinAlg;
using namespace MLCommon;

namespace gmm {

template <typename math_t>
void inverse(math_t *out, const math_t *in, int len,
             cudaStream_t stream = 0) {
        unaryOp(out, in, len,
                [] __device__(math_t in) {
                        return 1 / in;
                },
                stream);
}

template <typename math_t>
void exp(math_t *out, const math_t *in, int len,
         cudaStream_t stream = 0) {
        unaryOp(out, in, len,
                [] __device__(math_t in) {
                        return std::exp(in);
                },
                stream);
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

template <typename Type>
__global__ void naiveAddElemKernel(Type *out, const Type *in1, const Type in2,
                                   int len) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < len) {
                out[idx] = in1[idx] + in2;
        }
}

template <typename Type>
void naiveAddElem(Type *out, const Type *in1, const Type in2, int len) {
        static const int TPB = 64;
        int nblks = ceildiv(len, TPB);
        naiveAddElemKernel<Type><<<nblks, TPB>>>(out, in1, in2, len);
        CUDA_CHECK(cudaPeekAtLastError());
}

// template <typename T>
// __global__ void regularizeKernel (int n, int batchCount, T *A, int ldda, T reg, int nThreads_x, int nThreads_y, int nThreads_z) {
//         int i_start = threadIdx.x + blockDim.x * blockIdx.x;
//         int j_start = threadIdx.y + blockDim.y * blockIdx.y;
//         int k_start = threadIdx.z + blockDim.z * blockIdx.z;
//
//         for (size_t bId = k_start; bId < batchCount; bId+=nThreads_z) {
//                 for (size_t j = j_start; j < n; j+=nThreads_y) {
//                         for (size_t i = i_start; i <  n; i+=nThreads_x) {
//                                 if (i == j)
//                                         A[bId][IDX(i, j, ldda)] += eps;
//                         }
//                 }
//         }
// }


// template <typename T>
// void regularize_sigmas(int n, int batchCount, T** dA_array, int ldda, T reg) {
//         dim3 block(32,32);
//         dim3 grid(ceildiv(n, (int)block.x),
//                   ceildiv(n, (int)block.y),
//                   1);
//         int nThreads_x = grid.x * block.x;
//         int nThreads_y = grid.y * block.y;
//         int nThreads_z = grid.z * block.z;
//
//         regularizeKernel<T> <<< grid, block >>>(n, batchCount A, ldda, reg,
//                                                 nThreads_x, nThreads_y, nThreads_z);
//         cudaDeviceSynchronize();
//         CUDA_CHECK(cudaPeekAtLastError());
// }

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
                                dA[IDX(i, j, ldda)] /= x[j];
                        }
                        else{
                                dA[IDX(i, j, ldda)] /= x[i];
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

template <typename T>
void generate_trans_matrix(magma_int_t m, magma_int_t n, T* dA, magma_int_t lda, bool colwise){
        fill_matrix_gpu(m, n, dA, lda, false);
        normalize_matrix(m, n, dA, lda, colwise);
}


}
