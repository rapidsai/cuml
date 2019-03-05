#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <magma_v2.h>

#include "cuda_utils.h"
#include <random/rng.h>
#include "linalg/mean_squared_error.h"

#define IDX(i,j,lda) ((i)+(j)*(lda))
#define IDX2(i,j,k,lda,la) ((i)+(j)*(lda) + (k)*(la))
#define RUP_SIZE 32

namespace MLCommon {


template <typename T>
void allocate_pointer_array(T **&dA_array, magma_int_t n, magma_int_t batchCount){
        T **A_array;

        A_array = (T **)malloc(sizeof(T*) * batchCount);
        for(int i = 0; i < batchCount; i++) {
                allocate(A_array[i], n);
        }

        allocate(dA_array, batchCount);
        updateDevice(dA_array, A_array, batchCount);
        free(A_array);
}

template <typename T>
void free_pointer_array(T **&dA_array, magma_int_t batchCount){
        T **A_array;
        A_array = (T **)malloc(sizeof(T*) * batchCount);
        updateHost(A_array, dA_array, batchCount);
        for(int i = 0; i < batchCount; i++) {
                CUDA_CHECK(cudaFree(A_array[i]));
        }
        CUDA_CHECK(cudaFree(dA_array));
}

template <typename T>
void print_matrix_host(magma_int_t m, magma_int_t n, T *A, magma_int_t lda,
                       const std::string& msg){

          #define A(i_, j_) A[ (i_) + (j_)*lda ]

        printf("%s\n", msg.c_str());

        for (magma_int_t i=0; i < m; ++i) {
                for (magma_int_t j=0; j < n; ++j) {
                        printf("%f | ", (float) A(i,j));
                }
                printf("\n");
        }
        printf("\n");

          #undef A
}

template <typename T>
void print_matrix_device(magma_int_t m, magma_int_t n, T *dA, magma_int_t lda, const std::string& msg){
        T *A;
        A = (T *)malloc(sizeof(T) * lda * n);
        updateHost(A, dA, lda * n);
        print_matrix_host(m, n, A, lda, msg);
        free(A);
}

template <typename T>
void print_matrix_batched(magma_int_t m, magma_int_t n, magma_int_t batchCount, T **&dA_array, magma_int_t lda, const std::string& msg){

        printf("%s\n", msg.c_str());

        T **A_array;
        A_array = (T **)malloc(sizeof(T*) * batchCount);
        updateHost(A_array, dA_array, batchCount);

        for (magma_int_t l=0; l < batchCount; ++l) {
                printf("Matrix %d \n", l);
                print_matrix_device(m, n, A_array[l], lda, msg);
        }

        free(A_array);

}

template <typename T>
void fill_matrix(magma_int_t m, magma_int_t n, T *A, magma_int_t lda, bool isSpd)
{
      #define A(i_, j_) A[ (i_) + (j_)*lda ]

        for (magma_int_t j=0; j < n; ++j) {
                for (magma_int_t i=0; i < lda; ++i) {
                        if (i < m ) {
                                A(i,j) = (T) (rand() / ((T) RAND_MAX));
                        }
                        else{
                                A(i,j) = (T) 0;
                        }
                }
        }

        if (isSpd) {
                assert(m == n && "the matrix should be square");
                T temp;
                for (size_t i = 0; i < m; i++) {
                        for (size_t j = 0; j <= i; j++) {
                                if (i == j) {
                                        A(i, j) = A(i, j) + m;
                                }
                                else{
                                        temp = (A(i, j) + A(j, i)) / 2;
                                        A(i, j) = temp;
                                        A(j, i) = temp;
                                }
                        }
                        /* code */
                }
        }

      #undef A
}

// isSpd : Symetric Positive Definite
template <typename T>
void fill_matrix_gpu(
        magma_int_t m, magma_int_t n, T *dA, magma_int_t ldda, bool isSpd=false)
{
        T *A;
        int lda = ldda;

        A = (T *)malloc(n * lda * sizeof(T));

        if (A == NULL) {
                fprintf( stderr, "malloc failed at %d in file %s\n", __LINE__, __FILE__ );
                return;
        }

        fill_matrix(m, n, A, lda, isSpd);

        updateDevice(dA, A, n * lda);

        free(A);
}



template <typename T>
void fill_matrix_gpu_batched(
        magma_int_t m, magma_int_t n, magma_int_t batchCount, T **&dA_array, magma_int_t ldda, bool isSpd=false)
{
        T **A_array;
        A_array = (T **)malloc(sizeof(T*) * batchCount);
        updateHost(A_array, dA_array, batchCount);
        for (size_t i = 0; i < batchCount; i++) {
                fill_matrix_gpu(m, n, A_array[i], ldda, isSpd);
        }
        free(A_array);
}



template <typename T>
__global__
void zeroOutValuesKernel(T * A, size_t m, size_t n, size_t batchCount, size_t ldda, int numThreads_x, int numThreads_y, int numThreads_z){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;
        int k_start = threadIdx.z + blockDim.z * blockIdx.z;

        int la = n * ldda;

        for (size_t bId = k_start; bId < batchCount; bId+=numThreads_z) {
                for (size_t j = j_start; j < n; j+=numThreads_y) {
                        for (size_t i = i_start; i < ldda - m; i+=numThreads_x) {
                                A[IDX2(i + m, j, bId, ldda, la)] = 0.;
                        }
                }
        }
}

template <typename T>
void zeroOutValues(T * A, size_t m, size_t n, size_t N, size_t ldda)
{
        dim3 block(32, 32, 1);
        dim3 grid(ceildiv((int) (ldda - m), (int)block.x),
                  ceildiv((int)n, (int)block.y),
                  1);

        int numThreads_x = grid.x * block.x;
        int numThreads_y = grid.y * block.y;
        int numThreads_z = grid.z * block.z;

        zeroOutValuesKernel<T> <<< grid, block >>>(A, m, n, N, ldda,
                                                   numThreads_x,
                                                   numThreads_y,
                                                   numThreads_z);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
T array_mse_batched(
        magma_int_t m, magma_int_t n, magma_int_t batchCount, T **&dA_array,
        magma_int_t ldda, T **&dB_array, magma_int_t lddb)
{
        // ASSERT(ldda == lldb, "the two arrays much have the same leading dimensions");

        T **A_array;
        A_array = (T **)malloc(sizeof(T*) * batchCount);
        updateHost(A_array, dA_array, batchCount);

        T **B_array;
        B_array = (T **)malloc(sizeof(T*) * batchCount);
        updateHost(B_array, dB_array, batchCount);

        T* error_d;
        T error_h=0, error=0;
        allocate(error_d, 1);



        for (size_t bId = 0; bId < batchCount; bId++) {
                zeroOutValues(A_array[bId], m, n, 1, ldda);
                zeroOutValues(B_array[bId], m, n, 1, lddb);
                MLCommon::LinAlg::meanSquaredError(error_d, A_array[bId], B_array[bId], ldda * n);
                updateHost(&error_h, error_d, 1);
                error += error_h;
        }
        error /= batchCount;

        free(A_array);
        free(B_array);

        return error;

}

template <typename T>
void fill_pointer_array(magma_int_t batchCount, T **&dA_array, T *dB){
        T **A_array;
        A_array = (T **)malloc(sizeof(T*) * batchCount);
        updateHost(A_array, dA_array, batchCount);
        for (size_t bId = 0; bId < batchCount; bId++) {
                A_array[bId] = dB;
        }
        updateDevice(dA_array, A_array, batchCount);
        free(A_array);
}

template <typename T>
void split_to_batches(magma_int_t n, T **&dA_array, T *&dA, magma_int_t ldda){
        T **A_array;
        A_array = (T **)malloc(sizeof(T*) * n);
        for (size_t bId = 0; bId < n; bId++) {
                A_array[bId] = dA + IDX(0, bId, ldda);
        }

        updateDevice(dA_array, A_array, n);
        free(A_array);
}

template <typename T>
void copy_batched(magma_int_t batchCount, T **dA_dest_array, T **dA_src_array,
                  size_t len){
        T **A_dest_array, **A_src_array;

        A_src_array = (T **)malloc(sizeof(T*) * batchCount);
        A_dest_array = (T **)malloc(sizeof(T*) * batchCount);

        updateHost(A_src_array, dA_src_array, batchCount);
        updateHost(A_dest_array, dA_dest_array, batchCount);

        for (size_t bId = 0; bId < batchCount; bId++) {
                copy(A_dest_array[bId], A_src_array[bId], len);
        }

        free(A_src_array);
        free(A_dest_array);
}


// void assert_batched(int batchCount, int *d_info_array){
//         int *h_info_array;
//         h_info_array = (int *)malloc(sizeof(int) * batchCount);
//         updateHost(h_info_array, d_info_array, batchCount);
//         for (size_t i = 0; i < batchCount; i++) {
//                 ASSERT(h_info_array[i] == 0, "info returned val=%d", h_info_array[i]);
//         }
//         free(h_info_array);
// }


template <typename T>
__global__
void atomicRandom(T* x){

}

template <typename T>
__global__
void symetrizeBatchedKernel(magma_int_t m, magma_int_t n, magma_int_t batchCount,
                            T* dA, magma_int_t ldda,
                            int nThreads_x, int nThreads_y, int nThreads_z){

        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;
        int k_start = threadIdx.z + blockDim.z * blockIdx.z;

        int idxA;
        int la = ldda * n;

        T temp;
        for (size_t bId = k_start; bId < batchCount; bId+=nThreads_z) {
                for (size_t j = j_start; j < n; j+=nThreads_x) {
                        for (size_t i = i_start; i <= j; i+=nThreads_y) {
                                if (i == j) {
                                        idxA = IDX2(i, j, bId, ldda, la);
                                        dA[idxA] += m;
                                }
                                else{
                                        temp = (dA[IDX2(i, j, bId, ldda, la)] + dA[IDX2(i, j, bId, ldda, la)]) / 2;
                                        dA[IDX2(i, j, bId, ldda, la)] = temp;
                                        dA[IDX2(j, i, bId, ldda, la)] = temp;
                                }
                        }
                }

        }
}

template <typename T>
void symetrize_batched(magma_int_t m, magma_int_t batchCount,
                       T* dA, magma_int_t ldda){
        // Symetrize
        dim3 block(32, 32, 1);
        dim3 grid(ceildiv((int)m, (int)block.x),
                  ceildiv((int)m, (int)block.y),
                  1);

        int nThreads_x = grid.x * block.x;
        int nThreads_y = grid.y * block.y;
        int nThreads_z = grid.z * block.z;

        symetrizeBatchedKernel<T> <<< grid, block >>>(m, m, batchCount,
                                                      dA, ldda,
                                                      nThreads_x, nThreads_y, nThreads_z);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
void random_matrix_batched(magma_int_t m, magma_int_t n, magma_int_t batchCount,
                           T* dA, magma_int_t ldda, bool isSpd,
                           uint64_t seed, T start, T end){
        if (isSpd) {
                assert(m == n && "The matrix should be square");
        }

// Random generation
        int dim = ldda * n * batchCount;
        MLCommon::Random::Rng rng(seed);
        rng.uniform(dA, dim, start, end);

// Symetrize
        if (isSpd) {
                symetrize_batched(m, batchCount, dA, ldda);
        }

// Fill out zeros regrding ldd
        zeroOutValues(dA, m, n, batchCount, ldda);

}

}
