#pragma once

#include "cusolver_wrappers.h"
#include "cuda_utils.h"
#include "matrix/matrix.h"

namespace MLCommon {
namespace LinAlg {

/**
 * @defgroup eig decomp with divide and conquer method for the column-major symmetric matrices
 * @param in the input buffer (symmetric matrix that has real eig values and vectors.
 * @param n_rows: number of rows of the input
 * @param n_cols: number of cols of the input
 * @param eig_vectors: eigenvectors
 * @param eig_vals: eigen values
 * @{
 */
template <typename math_t>
void eigDC(const math_t* in, int n_rows, int n_cols, math_t* eig_vectors, math_t* eig_vals, cusolverDnHandle_t cusolverH) {

	int lwork;
	CUSOLVER_CHECK(
			cusolverDnsyevd_bufferSize(cusolverH,
					CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
					n_rows, in, n_cols,
					eig_vals, &lwork));

	math_t *d_work;
	CUDA_CHECK(cudaMalloc(&d_work, sizeof(math_t) * lwork));

	int *dev_info = NULL;
	CUDA_CHECK(cudaMalloc((void** )&dev_info, sizeof(int)));

	MLCommon::Matrix::copy(in, eig_vectors, n_rows, n_cols);

	CUSOLVER_CHECK(
			cusolverDnsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
					CUBLAS_FILL_MODE_UPPER, n_rows,
					eig_vectors, n_cols, eig_vals,
					d_work, lwork, dev_info));

	if (d_work)
        CUDA_CHECK(cudaFree(d_work));

	if (dev_info)
	    CUDA_CHECK(cudaFree(dev_info));

	CUDA_CHECK(cudaGetLastError());
}


/**
 * @defgroup overloaded function for eig decomp with Jacobi method for the column-major symmetric matrices (in parameter)
 * @param n_rows: number of rows of the input
 * @param n_cols: number of cols of the input
 * @param eig_vectors: eigenvectors
 * @param eig_vals: eigen values
 * @{
 */
template <typename math_t>
void eigJacobi(const math_t* in, int n_rows, int n_cols, math_t* eig_vectors, math_t* eig_vals, cusolverDnHandle_t cusolverH) {
	math_t tol = 1.e-7;
	int sweeps = 15;
	eigJacobi(in, eig_vectors, eig_vals, tol, sweeps, n_rows, n_cols, cusolverH);
}

/**
 * @defgroup overloaded function for eig decomp with Jacobi method for the column-major symmetric matrices (in parameter)
 * @param n_rows: number of rows of the input
 * @param n_cols: number of cols of the input
 * @param eig_vectors: eigenvectors
 * @param eig_vals: eigen values
 * @param tol: error tolerance for the jacobi method. Algorithm stops when the error is below tol
 * @param sweeps: number of sweeps in the Jacobi algorithm. The more the better accuracy.
 * @{
 */
template <typename math_t>
void eigJacobi(const math_t* in, int n_rows, int n_cols, math_t* eig_vectors, math_t* eig_vals, math_t tol, int sweeps, cusolverDnHandle_t cusolverH) {

	syevjInfo_t syevj_params = NULL;
	CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));
	CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params, tol));
	CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params, sweeps));

	int lwork;
	CUSOLVER_CHECK(
			cusolverDnsyevj_bufferSize(cusolverH,
					CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
					n_rows, eig_vectors,
					n_cols, eig_vals, &lwork,
					syevj_params));

	math_t *d_work;
	CUDA_CHECK(cudaMalloc(&d_work, sizeof(math_t) * lwork));

	int *dev_info = NULL;
	CUDA_CHECK(cudaMalloc((void** )&dev_info, sizeof(int)));

	MLCommon::Matrix::copy(in, eig_vectors, n_rows, n_cols);

	CUSOLVER_CHECK(
			cusolverDnsyevj(cusolverH, CUSOLVER_EIG_MODE_VECTOR,
					CUBLAS_FILL_MODE_UPPER, n_rows,
					eig_vectors, n_cols, eig_vals,
					d_work, lwork, dev_info, syevj_params));

	int executed_sweeps;
	CUSOLVER_CHECK(
			cusolverDnXsyevjGetSweeps(cusolverH, syevj_params,
					&executed_sweeps));

	if (d_work)
	    CUDA_CHECK(cudaFree(d_work));

	if (dev_info)
	    CUDA_CHECK(cudaFree(dev_info));

	CUDA_CHECK(cudaGetLastError());
	CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params));
}

}; // end namespace LinAlg
}; // end namespace MLCommon
