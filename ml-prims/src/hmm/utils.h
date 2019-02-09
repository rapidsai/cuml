#pragma once

#include <cuda_utils.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define IDX2C(i,j,ld) (j*ld + i)

using namespace MLCommon;


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
