#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <cublas_v2.h>

#include <random/rng.h>
#include "random/mvg.h"

#include <linalg/eltwise.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/sqrt.h>
#include <linalg/transpose.h>

#include <hmm/cublas_wrappers.h>
#include <hmm/utils.h>

// #include "cuda_utils.h"
// #include "utils.h"

namespace MLCommon {
namespace HMM {

using namespace MLCommon::LinAlg;
using namespace MLCommon;
using MLCommon::Random::matVecAdd;



template <typename T>
struct _w_cov_functor
{
        T *s_weights;
        T* matrix_diff;
        int dim, nPts, nCl;
        cublasHandle_t* handle;


        _w_cov_functor(int _dim, int _nPts, int _nCl, cublasHandle_t* handle){
                dim = _dim;
                nPts = _nPts;
                nCl = _nCl;
                allocate(s_weights, nCl * nPts);
                allocate(matrix_diff, nPts*dim);

                this->handle = handle;
        }

        void run (int cluster_id, T *data, T *weights, T *mus, T *sigmas)
        {
                // find the offsets of points
                sqrt(s_weights, weights, nCl * nPts);
                CUDA_CHECK(cudaMemset(matrix_diff, 0, nPts*dim));
                matVecAdd(matrix_diff, data, mus + IDX2C(0, cluster_id, dim),
                          T(-1.0), nPts, dim);

                // print_matrix(data, dim, nPts, "data");
                // print_matrix(mus, dim, nCl, "mus");
                // print_matrix(matrix_diff, dim, nPts, "matrix_diff");

                // multiply by sqrt(weights)
                CUBLAS_CHECK(cublasdgmm(*handle, CUBLAS_SIDE_RIGHT,
                                        dim, nPts, matrix_diff, dim,
                                        s_weights + IDX2C(0, cluster_id, nPts),
                                        1, matrix_diff, dim));

                // print_matrix(weights, nPts, nCl, "weights");
                // print_matrix(s_weights, nPts, nCl, "s_weights");
                // print_matrix(matrix_diff, dim, nPts, "matrix_diff_weighted");

                // get the sum of all the covs
                T alfa = (T)1.0 / nPts, beta = (T)0.0;
                CUBLAS_CHECK(cublasgemm(*handle, CUBLAS_OP_N,
                                        CUBLAS_OP_T, dim, dim, nPts,
                                        &alfa, matrix_diff,
                                        dim, matrix_diff, dim,
                                        &beta, sigmas + IDX2C(0, cluster_id, dim*dim),
                                        dim));

        }
};



template <typename T>
void weighted_covs(T *data, T *weights, T *mus, T *sigmas, T* ps,
                   int dim, int nPts, int n_classes, cublasHandle_t* cublasHandle){

        _w_cov_functor<T> w_cov(dim, nPts, n_classes, cublasHandle);

        for (int i = 0; i < n_classes; i++) {
                w_cov.run(i, data, weights, mus, sigmas);
        }
        print_matrix(ps, 1, n_classes, "ps");

        CUBLAS_CHECK(cublasdgmm(*cublasHandle, CUBLAS_SIDE_RIGHT, dim * dim, n_classes, sigmas, dim * dim, ps, 1, sigmas, dim * dim));

}
}

}
