

using namespace ML;
#include "utils.h"
#include <assert.h>

#include "distances.h"
#include "perplexity_search.h"

#pragma once

template <typename Type>
void runTsne(	const Type * __restrict__ X,
				const int n,
				const int p,
				const int n_components = 2,
				const float perplexity = 50f,
				const float perplexity_epsilon = 1e-3,
				const float early_exaggeration = 2f,
				int n_neighbors = 100,

				// Learning rates and momentum
				const float learning_rate = 200,
				const float pre_momentum = 0.5f,
				const float post_momentum = 0.8f,

				// Barnes Hut parameters
				const float theta = 0.5f,
				const float epsilon_squared = 0.0025f,

				// Iterations, termination crtierion
				const int exaggeration_iter = 250,
				const int max_iter = 1000,
				const float min_grad_norm = 1e-7)
{	
	// Currently only allows n_components = 2
	assert(n_components == 2);
	if (n_neighbors > n) n_neighbors = n;

	// Nodes needed for BH
	const int n_nodes = 2*n;
	const int SIZE = n*n_neighbors;

	// From glm.cu
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));


	// Allocate distances
	Type *distances;	cuda_malloc(distances, SIZE);
	long *indices;		cuda_malloc(indices, SIZE);
	// Use FAISS for nearest neighbors [returns squared norm]
	// Divide by max(abs(D)) to not cause exp(D) to explode
	getDistances(X, n, p, indices, distances, n_neighbors, stream);


	// Allocate Pij
	float *Pij;			cuda_malloc(Pij, SIZE);
	// Search Perplexity
	searchPerplexity(Pij, distances, perplexity, perplexity_epsilon, n, n_neighbors, SIZE, stream);
	// Free distances
	cuda_free(indices);
	cuda_free(distances);



	// Free Pij
	cuda_free(Pij);


	// Destory CUDA stream
	CUDA_CHECK(cudaStreamDestroy(stream));
}
