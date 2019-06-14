
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
			const float *distances_vector = NULL, const long *indices_vector = NULL,
			float *VAL_vector = NULL, const int *COL_vector = NULL, const int *ROW_vector = NULL,
			const float perplexity = 30.0f, const int perplexity_max_iter = 100,
			const int perplexity_tol = 1e-5,
			const float early_exaggeration = 12.0f,
			const int exaggeration_iter = 500, const float min_gain = 0.01f,
			const float eta = 500.0f, const int max_iter = 10000,
			const float pre_momentum = 0.8, const float post_momentum = 0.5,
			const long long seed = -1, const bool initialize_embeddings = false,
			const bool verbose = false, const char *method = "Fast") {

	auto d_alloc = handle.getDeviceAllocator();
	cudaStream_t stream = handle.getStream();

	assert(n > 0 && p > 0 && n_components > 0 && n_neighbors > 0);
	if (n_neighbors > n) n_neighbors = n;

	// Some preliminary intializations for cuBLAS and cuML
	DEBUG("[Info] Create cuBLAS and cuML handles.\n");
	const int k = n_components;
	cublasHandle_t BLAS = handle.getImpl().getCublasHandle();

	const float neg2 = -2.0f, beta = 0.0f, one = 1.0f;


	// Get distances
	DEBUG("[Info] Get distances\n");
	float *distances =
			(float *)d_alloc->allocate(n * n_neighbors * sizeof(float), stream);
	long *indices =
		(long *)d_alloc->allocate(sizeof(long) * n * n_neighbors, stream);

	if (distances == NULL && indices == NULL) {
		get_distances(X, n, p, indices, distances, n_neighbors, stream);
	}
	else {
		MLCommon::updateDevice(distances, distances_vector, n * n_neighbors, stream);
		MLCommon::updateDevice(indices, indices_vector, n * n_neighbors, stream);

		std::cout << MLCommon::arr2Str(distances, 20, "Distances", stream) << std::endl;
		std::cout << MLCommon::arr2Str(indices, 20, "indices", stream) << std::endl;
	}



	normalize_distances(n, distances, n_neighbors, stream);
#if IF_DEBUG
		printf("[Info]  Normalized distances\n\n");
		std::cout << MLCommon::arr2Str(distances, 20, "Distances", stream) << std::endl;
#endif


	// Get perplexity
	DEBUG("[Info] Get perplexity\n");
	float *P = (float *)d_alloc->allocate(sizeof(float) * n * n_neighbors, stream);
	float P_sum = determine_sigmas(distances, P, perplexity, perplexity_max_iter,
									perplexity_tol, n, n_neighbors, stream);
	d_alloc->deallocate(distances, n * n_neighbors * sizeof(float), stream);
	DEBUG("[Info] P_sum = %f\n", P_sum);
#if IF_DEBUG
		printf("[Info]  Perplexity results\n\n");
		std::cout << MLCommon::arr2Str(P, 20, "Perplexity", stream) << std::endl;
#endif



	// Convert data to COO layout
	float *VAL;
	int *COL, *ROW;
	int NNZ;

	if (VAL_vector == NULL) {
		DEBUG("[Info] Convert to COO layout and symmetrize\n");
		COO_t<float> P_PT;
		symmetrize_perplexity(P, indices, &P_PT, n, n_neighbors, P_sum,
													early_exaggeration, stream);
		CUDA_CHECK(cudaPeekAtLastError());
		NNZ = P_PT.nnz;
		VAL = P_PT.vals;
		COL = P_PT.rows;
		ROW = P_PT.cols;
	}
	else {
		NNZ = TEST_NNZ;
		VAL = (float *)d_alloc->allocate(sizeof(float) * NNZ, stream);
		COL = (int *)d_alloc->allocate(sizeof(int) * NNZ, stream);
		ROW = (int *)d_alloc->allocate(sizeof(int) * NNZ, stream);
		MLCommon::updateDevice(VAL, VAL_vector, NNZ, stream);
		MLCommon::updateDevice(COL, COL_vector, NNZ, stream);
		MLCommon::updateDevice(ROW, ROW_vector, NNZ, stream);
	}

#if IF_DEBUG
		printf("[Info]  Symmetrized Perplexity results\n\n");
		std::cout << MLCommon::arr2Str(VAL, 20, "Perplexity", stream) << std::endl;

		printf("[Info]  COL\n\n");
		std::cout << MLCommon::arr2Str(COL, 20, "Perplexity", stream) << std::endl;

		printf("[Info]  RWW\n\n");
		std::cout << MLCommon::arr2Str(ROW, 20, "Perplexity", stream) << std::endl;
#endif



	// Allocate data [NOTICE all Fortran Contiguous]
	DEBUG("[Info] Malloc data and space\n");
	float *noise = (float *)d_alloc->allocate(sizeof(float) * n, stream);
	cudaMemset(noise, 0, sizeof(float) * n);
	//random_vector(noise, -0.003f, 0.003f, n, stream, seed);
	CUDA_CHECK(cudaPeekAtLastError());

	if (initialize_embeddings) {
		random_vector(Y, -0.1f, 0.1f, n * k, stream, seed);
		CUDA_CHECK(cudaPeekAtLastError());
	}


	float *norm = (float *)d_alloc->allocate(sizeof(float) * n, stream);
	float *Q_sum = (float *)d_alloc->allocate(sizeof(float) * n, stream);
	double *sum = (double *)d_alloc->allocate(sizeof(double), stream);

	float *attract = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);
	float *repel = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);

	float *iY = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);
	float *gains = (float *)d_alloc->allocate(sizeof(float) * n * k, stream);

	// Do gradient updates
	float momentum = pre_momentum;
	float Z;
	int error;

	if (method == "Fast") {

		for (int iter = 0; iter < max_iter; iter++) {
			if (iter == exaggeration_iter) {
				momentum = post_momentum;
				// Divide perplexities
				float div = 1.0f / early_exaggeration;
				thrust::transform(__STREAM__, VAL, VAL + NNZ, VAL, div * _1);
			}
			// Get norm(Y)
			get_norm(Y, norm, n, k, stream);

			// Fast compute attractive forces from COO matrix
			attractive_fast(VAL, COL, ROW, Y, norm, attract, NNZ, n, n_components, stream);

			// Fast compute repulsive forces
			Z = repulsive_fast(Y, repel, norm, Q_sum, n, n_components, stream);
			if (verbose) printf("[INFO]	Z at iter = %d is %lf", iter, Z);

			// Integrate forces with momentum
			apply_forces(attract, repel, Y, iY, noise, gains, n, k, Z, min_gain,
						momentum, eta, stream);
		}
	}

	else if (method == "Naive") {

		float *Q = (float *)d_alloc->allocate(sizeof(float) * n * n, stream);

		for (int iter = 0; iter < max_iter; iter++) {
			if (iter == exaggeration_iter) {
				momentum = post_momentum;
				// Divide perplexities
				float div = 1.0f / early_exaggeration;
				thrust::transform(__STREAM__, VAL, VAL + NNZ, VAL, div * _1);
			}
			// Get norm(Y)
			get_norm(Y, norm, n, k, stream);

			// Find Y @ Y.T
			if (error = cublasSsyrk(BLAS, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k,
									&neg2, Y, n, &beta, Q, n)) {
				if (verbose) printf("[ERROR]	BLAS failed. Terminating TSNE\n");
				break;
			}
			CUDA_CHECK(cudaPeekAtLastError());

			// Form T = 1 / (1 + d) = 1 / (1 + -2*Y@Y.T )
			Z = form_t_distribution(Q, norm, n, Q_sum, sum, stream);
			if (verbose) printf("[INFO]	Z at iter = %d is %lf", iter, Z);

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
			CUDA_CHECK(cudaPeekAtLastError());

			// Compute repel - Q**2 @ mean_Y
			repel_minus_QY(repel, Q_sum, Y, n, k, stream);

			// Integrate forces with momentum
			apply_forces(attract, repel, Y, iY, noise, gains, n, k, Z, min_gain,
						momentum, eta, stream);
		}

		d_alloc->deallocate(Q, sizeof(float) * n * n, stream);
	}


#if not IF_DEBUG
	P_PT.destroy();
#else
	d_alloc->deallocate(VAL, sizeof(float) * NNZ, stream);
	d_alloc->deallocate(COL, sizeof(int) * NNZ, stream);
	d_alloc->deallocate(ROW, sizeof(int) * NNZ, stream);
#endif

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
