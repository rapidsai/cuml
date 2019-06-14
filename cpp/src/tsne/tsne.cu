
#pragma once

#include "common/cumlHandle.hpp"

#include "tsne/tsne.h"

#include "cublas_v2.h"
#include "distances.h"
//#include "kernels.h"
#include "fast_kernels.h"
#include "utils.h"

#define TEST_NNZ 12021


namespace ML {
using namespace MLCommon;

void TSNE(const cumlHandle &handle, const float *X, float *Y, const int n,
			const int p, const int n_components = 2, int n_neighbors = 90,
			const float perplexity = 30.0f, const int perplexity_max_iter = 100,
			const int perplexity_tol = 1e-5,
			const float early_exaggeration = 12.0f,
			const int exaggeration_iter = 250, const float min_gain = 0.01f,
			const float eta = 500.0f, const int max_iter = 500,
			const float pre_momentum = 0.8, const float post_momentum = 0.5,
			const long long seed = -1, const bool initialize_embeddings = true,
			const bool verbose = true, const char *method = "Fast")
{
	assert(n > 0 && p > 0 && n_components > 0 && n_neighbors > 0 && X != NULL && Y != NULL);
	if (verbose)
		printf("[Info]	Data size = (%d, %d) with n_components = %d\n", n, p, n_components);

	auto d_alloc = handle.getDeviceAllocator();
	cudaStream_t stream = handle.getStream();

	if (n_neighbors > n) n_neighbors = n;

	// Some preliminary intializations for cuBLAS and cuML
	const int k = n_components;
	cublasHandle_t BLAS = handle.getImpl().getCublasHandle();


	// Get distances
	if (verbose) printf("[Info]	Getting distances.\n");
	float *distances = (float *)d_alloc->allocate(n * n_neighbors * sizeof(float), stream);
	long *indices = (long *)d_alloc->allocate(sizeof(long) * n * n_neighbors, stream);

	get_distances(X, n, p, indices, distances, n_neighbors, stream);


	// Normalize distances
	if (verbose) printf("[Info]	Now normalizing distances so exp(D) doesn't explode.\n");
	normalize_distances(n, distances, n_neighbors, stream);


	// Optimal perplexity
	if (verbose) printf("[Info]	Searching for optimal perplexity via bisection search.\n");
	float *P = (float *)d_alloc->allocate(sizeof(float) * n * n_neighbors, stream);

	const float P_sum = determine_sigmas(distances, P, perplexity, perplexity_max_iter,
										perplexity_tol, n, n_neighbors, stream);
	d_alloc->deallocate(distances, n * n_neighbors * sizeof(float), stream);
	if (verbose) printf("[Info]	Perplexity sum = %f\n", P_sum);


	// Convert data to COO layout
	COO_t<float> P_PT;
	symmetrize_perplexity(P, indices, &P_PT, n, n_neighbors, P_sum, early_exaggeration, stream);
		
	const int NNZ = P_PT.nnz;
	float *VAL = P_PT.vals;
	const int *COL = P_PT.rows;
	const int *ROW = P_PT.cols;


	// Allocate data [NOTICE Fortran Contiguous for method = Naive and C-Contiguous for fast]
	float *noise = (float *)d_alloc->allocate(sizeof(float) * n, stream);
	random_vector(noise, -0.003f, 0.003f, n, stream, seed);

	if (initialize_embeddings)
		random_vector(Y, -0.1f, 0.1f, n * k, stream, seed);


	// Allocate space
	if (verbose) printf("[Info]	Now allocating memory for TSNE.\n");
	float *norm = (float *)d_alloc->allocate(sizeof(float) * n, stream);
	float *Q_sum = (float *)d_alloc->allocate(sizeof(float) * n, stream);
	double *sum = (double *)d_alloc->allocate(sizeof(double), stream);

	float *attract = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);
	float *repel = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);

	float *iY = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);
	float *gains = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);


	// Do gradient updates

	float momentum = pre_momentum;
	double Z;
	int error;

	if (verbose) printf("[Info]	Start gradient updates!\n");
	if (method == "Fast") {
		/*
		Fast algorithm uses 0 matrices, and does all computations within
		the GPU registers. This guarantees no memory movement and so can
		be fast in practice.
		*/
		for (int iter = 0; iter < max_iter; iter++) {
			if (iter == exaggeration_iter) {
				momentum = post_momentum;
				// Divide perplexities
				const float div = 1.0f / early_exaggeration;
				thrust<float> begin = to_thrust(VAL);
				thrust::transform(__STREAM__, begin, begin + NNZ, begin, div * _1);
			}
			// Get norm(Y)
			get_norm(Y, norm, n, k, stream);

			// Fast compute attractive forces from COO matrix
			attractive_fast(VAL, COL, ROW, Y, norm, attract, NNZ, n, n_components, stream);

			// Fast compute repulsive forces
			Z = repulsive_fast(Y, repel, norm, Q_sum, n, n_components, stream);
			if (verbose) printf("[Info]	Z at iter = %d is %lf", iter, Z);

			// Integrate forces with momentum
			apply_forces(attract, repel, Y, iY, noise, gains, n, k, Z, min_gain,
						momentum, eta, stream);
		}
	}

	else if (method == "Naive") {
		/*
		Naive algorithm uses cuBLAS to compute the full Y @ Y.T matrix.
		Code flow follows closely to Maaten's original TSNE code.
		*/
		const float neg2 = -2.0f, beta = 0.0f, one = 1.0f;
		float *Q = (float *)d_alloc->allocate(sizeof(float) * n * n, stream);

		for (int iter = 0; iter < max_iter; iter++) {
			if (iter == exaggeration_iter) {
				momentum = post_momentum;
				// Divide perplexities
				const float div = 1.0f / early_exaggeration;
				thrust<float> begin = to_thrust(VAL);
				thrust::transform(__STREAM__, begin, begin + NNZ, begin, div * _1);
			}
			// Get norm(Y)
			get_norm(Y, norm, n, k, stream);

			// Find Y @ Y.T
			if (error = cublasSsyrk(BLAS, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k,
									&neg2, Y, n, &beta, Q, n)) {
				if (verbose) printf("[ERROR]	BLAS failed. Terminating TSNE\n");
				break;
			}

			// Form T = 1 / (1 + d) = 1 / (1 + -2*Y@Y.T )
			Z = form_t_distribution(Q, norm, n, Q_sum, sum, stream);
			if (verbose) printf("[Info]	Z at iter = %d is %lf", iter, Z);

			// Compute attractive forces from COO matrix
			attractive_forces(VAL, COL, ROW, Q, Y, attract, NNZ, n, k, stream);

			// Change Q to Q**2
			postprocess_Q(Q, Q_sum, n, stream);

			// Do Q**2 @ Y
			if (error = cublasSsymm(BLAS, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, n,
			                        k, &one, Q, n, Y, n, &beta, repel, n)) {
				if (verbose) printf("[ERROR]	BLAS failed. Terminating TSNE\n");
				break;
			}

			// Compute repel - Q**2 @ mean_Y
			repel_minus_QY(repel, Q_sum, Y, n, k, stream);

			// Integrate forces with momentum
			apply_forces(attract, repel, Y, iY, noise, gains, n, k, Z, min_gain,
						momentum, eta, stream);
		}

		d_alloc->deallocate(Q, sizeof(float) * n * n, stream);
	}

	P_PT.destroy();

	d_alloc->deallocate(noise, sizeof(float) * n, stream);

	d_alloc->deallocate(norm, sizeof(float) * n, stream);
	d_alloc->deallocate(Q_sum, sizeof(float) * n, stream);
	d_alloc->deallocate(sum, sizeof(double), stream);

	d_alloc->deallocate(attract, sizeof(float) * n * k, stream);
	d_alloc->deallocate(repel, sizeof(float) * n * k, stream);

	d_alloc->deallocate(iY, sizeof(float) * n * k, stream);
	d_alloc->deallocate(gains, sizeof(float) * n * k, stream);
}


}  // namespace ML
