#pragma once

#include "cublas_wrappers.h"
#include "cusolver_wrappers.h"
#include "../matrix/matrix.h"

namespace MLCommon {
namespace LinAlg {

/**
 * @defgroup QR decomposition, return the Q matrix
 * @param M: input matrix
 * @param Q: Q matrix to be returned (on GPU)
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @{
 */
template <typename math_t>
void qrGetQ(math_t* &M, math_t* &Q, int n_rows, int n_cols, cusolverDnHandle_t cusolverH){
    int m = n_rows, n = n_cols;
    int k = min(m, n);
    cudaMemcpy(Q, M, sizeof(math_t) * m * n, cudaMemcpyDeviceToDevice);

    math_t *tau;
    allocate<math_t>(tau, k, true);

    int Lwork, *devInfo;
    CUDA_CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
    math_t *workspace;

        CUSOLVER_CHECK(cusolverDngeqrf_bufferSize(cusolverH, m, n, Q, m, &Lwork));
        CUDA_CHECK(cudaMalloc((void**)&workspace, sizeof(math_t)*Lwork));
        CUSOLVER_CHECK(cusolverDngeqrf(cusolverH, m, n, Q, m, tau, workspace, Lwork, devInfo));
        CUDA_CHECK(cudaFree(workspace));
        CUSOLVER_CHECK(cusolverDnorgqr_bufferSize(cusolverH, m, n, k, Q, m, tau, &Lwork));
        CUDA_CHECK(cudaMalloc((void**)&workspace, sizeof(math_t)*Lwork));
        CUSOLVER_CHECK(cusolverDnorgqr(cusolverH, m, n, k, Q, m, tau, workspace, Lwork, devInfo));
        CUDA_CHECK(cudaFree(workspace));
        CUDA_CHECK(cudaFree(devInfo));

    // clean up
    CUDA_CHECK(cudaFree(tau));
}

/**
 * @defgroup QR decomposition, return the Q matrix
 * @param M: input matrix
 * @param Q: Q matrix to be returned (on GPU)
 * @param R: R matrix to be returned (on GPU)
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @{
 */
template <typename math_t>
void qrGetQR(math_t* &M, math_t* &Q, math_t* &R, int n_rows, int n_cols, cusolverDnHandle_t cusolverH){
    int m = n_rows, n = n_cols;
    math_t *R_full, *tau;
    allocate<math_t>(R_full, m * n, true);
    allocate<math_t>(tau, min(m, n), true);
    int R_full_nrows = m, R_full_ncols = n;
    CUDA_CHECK(cudaMemcpy(R_full, M, sizeof(math_t)*m*n, cudaMemcpyDeviceToDevice));

    int Lwork, *devInfo;
    CUDA_CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
    math_t *workspace;

        CUSOLVER_CHECK(cusolverDngeqrf_bufferSize(cusolverH, R_full_nrows, R_full_ncols, R_full, R_full_nrows, &Lwork));
        CUDA_CHECK(cudaMalloc((void**)&workspace, sizeof(math_t)*Lwork));
        CUSOLVER_CHECK(cusolverDngeqrf(cusolverH, R_full_nrows, R_full_ncols, R_full, R_full_nrows, tau, workspace, Lwork, devInfo));
        CUDA_CHECK(cudaFree(workspace));

        Matrix::copyUpperTriangular(R_full, R, m, n);

        CUDA_CHECK(cudaMemcpy(Q, R_full, sizeof(math_t)*m*n, cudaMemcpyDeviceToDevice));
        int Q_nrows = m, Q_ncols = n;

        CUSOLVER_CHECK(cusolverDnorgqr_bufferSize(cusolverH, Q_nrows, Q_ncols, min(Q_ncols,Q_nrows), Q, Q_nrows, tau, &Lwork));
        CUDA_CHECK(cudaMalloc((void**)&workspace, sizeof(math_t)*Lwork));
        CUSOLVER_CHECK(cusolverDnorgqr(cusolverH, Q_nrows, Q_ncols, min(Q_ncols,Q_nrows), Q, Q_nrows, tau, workspace, Lwork, devInfo));
        CUDA_CHECK(cudaFree(workspace));
        CUDA_CHECK(cudaFree(devInfo));

    // clean up
    CUDA_CHECK(cudaFree(R_full));
    CUDA_CHECK(cudaFree(tau));
}

}; // end namespace LinAlg
}; // end namespace MLCommon