#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>

#include <linalg/cublas_wrappers.h>
#include <random/rng.h>
#include <hmm/cublas_wrappers.h>


using namespace MLCommon::LinAlg;
using namespace MLCommon;

namespace MLCommon {
namespace HMM {

template <typename T>
struct paramsRandom {
        T start, end;
        unsigned long long seed;
        paramsRandom(T _start, T _end, unsigned long long _seed) : start(_start),
                end(_end), seed(_seed){
        };
};

template <typename T>
struct Inv_functor
{
        __host__ __device__
        T operator()(T& x)
        {
                return (T) 1.0 / x;
        }
};




template <typename T>
void gen_matrix(int m, int n, T* dA, int ldda, paramsRandom<T> *paramsRd){
        MLCommon::Random::Rng<T> rng(paramsRd->seed);
        rng.uniform(dA, ldda * n, paramsRd->start, paramsRd->end);
        // TODO : Fill with zeros
}


template <typename T>
void normalize_matrix(int m, int n, T* dA, int ldda, bool colwise){
        // cublas handles
        cublasHandle_t cublas_handle;
        cublasCreate(&cublas_handle);
        T *sums, *ones;


        if(colwise) {
                // initializations
                T *sums;
                allocate(sums, n);
                allocate(ones, ldda);

                thrust::device_ptr<T> sums_th(sums);
                thrust::device_ptr<T> ones_th(ones);

                const T alpha = (T) 1;
                const T beta = (T) 0;

                thrust::fill(sums_th, sums_th + ldda, beta);
                thrust::fill(ones_th, ones_th + n, alpha);

                // Compute the sum of each row
                sum(sums, dA, n, llda, false);

                // Inverse the sums
                thrust::transform(sums_th, sums_th + m, sums_th, Inv_functor<T>());

                // Multiply by the inverse
                CUBLAS_CHECK(cublasdgmm(cublas_handle, CUBLAS_SIDE_LEFT, ldda, n, dA, m, sums, 1, dA, m));
        }
        else{
                // initializations
                T *sums, *ones;
                cudaMalloc(&sums, n * sizeof(T));
                cudaMalloc(&ones, m * sizeof(T));

                thrust::device_ptr<T> sums_th(sums);
                thrust::device_ptr<T> ones_th(ones);

                const T alpha = (T) 1;
                const T beta = (T) 0;

                thrust::fill(ones_th, ones_th + m, alpha);
                thrust::fill(sums_th, sums_th + n, beta);

                // Compute the sum of each col
                CUBLAS_CHECK(cublasgemv(cublas_handle, CUBLAS_OP_T, m, n, &alpha, dA, m, ones, 1, &beta, sums, 1));

                // Inverse the sums
                thrust::transform(sums_th, sums_th + n, sums_th, Inv_functor<T>());

                // Multiply by the inverse
                CUBLAS_CHECK(cublasdgmm(cublas_handle, CUBLAS_SIDE_RIGHT, m, n, dA, m, sums, 1, dA, m));

        }

}

template <typename T>
void gen_trans_matrix(int m, int n, T* dA,  paramsRandom<T> *paramsRd, bool colwise){
        gen_A(m, n, dA, ldda, paramsRd);
        normalize_A(m, n, dA, ldda, colwise);
}

}
}
