#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cublas_v2.h>

#include <random/rng.h>
#include "random/mvg.h"

#include <linalg/cublas_wrappers.h>
#include <linalg/sqrt.h>
#include <linalg/transpose.h>

#include <hmm/cublas_wrappers.h>

// #include "cuda_utils.h"
// #include "utils.h"

namespace MLCommon {
namespace HMM {

using namespace MLCommon::LinAlg;
using namespace MLCommon;
using MLCommon::Random::matVecAdd;

template <typename T>
void weighted_means(T* weights, T* data, T* means, int dim, int n_samples, int n_classes, cublasHandle_t handle){
        // X (dim, n_samples)
        // rhos (n_classes, n_samples)
        // mu (dim, n_classes)
        // return data * rhos^T

        T alfa = (T)1.0, beta = (T)0.0;
        CUBLAS_CHECK(cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, dim, n_classes,
                                n_samples, &alfa, data, dim, weights, n_classes, &beta, means, dim));
}



template <typename T>
struct _w_cov_functor
{
        T *data, *mus, *weights;
        T *sigmas, *s_weights;
        int dim, nPts, nCl;
        cublasHandle_t* handle;


        _w_cov_functor(T *data, T *weights, T *mus, T *sigmas,
                       int _dim, int _nPts, int _nCl,
                       cublasHandle_t* handle){
                dim = _dim;
                nPts = _nPts;
                nCl = _nCl;

                this->data = data;
                this->weights = weights;

                allocate(s_weights, nCl * nPts);
                transpose(weights, s_weights, nCl, nPts, *handle);
                sqrt(s_weights, s_weights, nCl * nPts);

                this->mus = mus;
                this->sigmas = sigmas;
                this->handle = handle;

        }


        // Compute sigma_k
        // __host__ __device__
        // void operator() (const int cluster_id)
        void run (int cluster_id)
        {
                // find the offsets of points
                T* matrix_diff;
                allocate(matrix_diff, nPts*dim);
                CUDA_CHECK(cudaMemset(matrix_diff, 0, nPts*dim));
                matVecAdd(matrix_diff, data, mus + IDX2C(0, cluster_id, dim),
                          T(-1.0), nPts, dim);

                // multiply by sqrt(weights)
                CUBLAS_CHECK(cublasdgmm(*handle, CUBLAS_SIDE_RIGHT,
                                        dim, nPts, matrix_diff, dim,
                                        s_weights + IDX2C(0, cluster_id, nPts),
                                        1, matrix_diff, dim));

                // get the sum of all the covs
                T alfa = (T)1.0, beta = (T)0.0;
                CUBLAS_CHECK(cublasgemm(*handle, CUBLAS_OP_N,
                                        CUBLAS_OP_T, dim, dim, nPts,
                                        &alfa, matrix_diff,
                                        dim, matrix_diff, dim,
                                        &beta, sigmas + IDX2C(0, cluster_id, dim*dim),
                                        dim));
        }
};



template <typename T>
void weighted_covs(T *data, T *weights, T *mus, T *sigmas,
                   int dim, int nPts, int n_classes, cublasHandle_t* handle){

        _w_cov_functor<T> w_cov(data, weights, mus, sigmas, dim, nPts, n_classes,
                                handle);

        for (size_t i = 0; i < n_classes; i++) {
                w_cov.run(i);
        }
        // thrust::counting_iterator<int> first(0);
        // thrust::counting_iterator<int> last = first + n_classes;

        // thrust::for_each(thrust::device, first, last, w_cov);
}
}

}
