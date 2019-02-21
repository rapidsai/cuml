#pragma once

#include <cuda_utils.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define IDX2C(i,j,ld) (j*ld + i)

using namespace MLCommon;

// printf("line number %d in file %s\n", __LINE__, __FILE__);

template <typename T>
void print_matrix_host(T* cpu, int rows, int cols, const std::string& msg){
        printf("\n\n");
        printf("%s\n", msg.c_str());
        for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++)
                        printf("%f | ", cpu[IDX2C(i, j, rows)]);
                printf("\n");
        }
}


template <typename T>
void print_matrix(T* gpu, int rows, int cols, const std::string& msg){
        T* cpu;
        cpu = (T *)malloc(sizeof(T)*rows*cols);
        updateHost(cpu, gpu, rows*cols);
        print_matrix_host(cpu, rows, cols, msg);
}

// TODO : Remove duplicates with KF

template <typename T>
__global__ void Linear_KF_ID_kernel (T *w, int dim) {
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        if (i < dim && j < dim)
        {
                if (i == j)
                        w[IDX2C(i, j, dim)] = 1.0;
                else
                        w[IDX2C(i, j, dim)] = 0.0;
        }
}


/**
 * @brief identity matrix generating function
 * @tparam T the data type
 * @param I the out matrix
 * @param dim dimension of I
 */
template <typename T>
void make_ID_matrix(T *I, int dim) {
        dim3 block(32,32);
        dim3 grid(ceildiv(dim, (int)block.x), ceildiv(dim, (int)block.y));
        Linear_KF_ID_kernel<T> <<< grid, block >>>(I, dim);
        // cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}
