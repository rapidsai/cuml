
#include "utils.h"

using namespace ML;
using namespace MLCommon::Sparse;

#include <assert.h>

#include "distances.h"
#include "perplexity_search.h"
#include "gpu_info.h"
#include "intialization.h"

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
				const float min_grad_norm = 1e-7,

                // Seed for random data
                const long long seed = -1)
{	
	// Currently only allows n_components = 2
	assert(n_components == 2);
	if (n_neighbors > n) n_neighbors = n;


	// Get all device info and properties
	int BLOCKS;
	int TPB_X; 		// Notice only 32 is supported
    int integration_kernel_threads;
    int integration_kernel_factor;
    int repulsive_kernel_threads;
    int repulsive_kernel_factor;
    int bounding_kernel_threads;
    int bounding_kernel_factor; 
    int tree_kernel_threads;
    int tree_kernel_factor;
    int sort_kernel_threads;
    int sort_kernel_factor;
    int summary_kernel_threads;
    int summary_kernel_factor;

    GPU_Info_::gpuInfo(&BLOCKS, &TPB_X, &integration_kernel_threads, 
    	&integration_kernel_factor, &repulsive_kernel_threads, &repulsive_kernel_factor, 
    	&bounding_kernel_threads, &bounding_kernel_factor, &tree_kernel_threads, 
    	&tree_kernel_factor, &sort_kernel_threads, &sort_kernel_factor, 
    	&summary_kernel_threads, &summary_kernel_factor);

    // Intialize cache levels and errors
    int *err;       cuda_malloc(err, 1);
    Intialization_::Initialize(err);
    //


	// Nodes needed for BH
	int nnodes = 2*n;
	if (nnodes < 1024 * BLOCKS) nnodes = 1024 * BLOCKS;
	while ((nnodes & (TPB_X - 1)) != 0)
		nnodes++;
    nnodes--;

    const int N_NODES = nnodes;
	const int SIZE = n*n_neighbors;


	// From glm.cu
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));


	// Allocate distances
	float *distances;	cuda_malloc(distances, SIZE);
	long *indices;		cuda_malloc(indices, SIZE);

	// Use FAISS for nearest neighbors [returns squared norm]
	// Divide by max(abs(D)) to not cause exp(D) to explode
	Distances_::getDistances(X, n, p, indices, distances, n_neighbors, SIZE, stream);


	// Allocate Pij
	float *Pij;			cuda_malloc(Pij, SIZE);
	// Search Perplexity
	Perplexity_Search_::searchPerplexity(Pij, distances, perplexity, perplexity_epsilon,
		n, n_neighbors, SIZE, stream);
	cuda_free(distances);


	// Change P to COO matrix
	COO<float> P;
	MLCommon::Sparse::from_knn(indices, Pij, n, n_neighbors, &P);
	cuda_free(Pij);


	// Perform P + P.T
	COO<float> P_PT;	// P and P + P.T
	Perplexity_Search_::P_add_PT(indices, n, n_neighbors, &P, &P_PT, stream);
	P.destroy();
	cuda_free(indices);
	const int NNZ = P_PT.nnz;	// Get total NNZ
	

	// Allocate space
	float *P_x_Q;			cuda_malloc(P_x_Q, NNZ);
	float *repulsion;		cuda_calloc(repulsion, (N_NODES+1)*2, 0.0f);
	float *attraction;		cuda_calloc(attraction, n*2, 0.0f);
	float *normalization;	cuda_malloc(normalization, N_NODES+1);

	float *gains;			cuda_calloc(gains, n*2, 1.0f);
	float *old_forces;		cuda_calloc(prev_forces, n*2, 0.0f);

	int *cell_starts;		cuda_malloc(cell_starts, N_NODES+1);
	int *children;			cuda_malloc(children, (N_NODES+1)*4);
	float *cell_mass;		cuda_calloc(cell_mass, N_NODES+1, 1.0f);
	int *cell_counts;		cuda_malloc(cell_counts, N_NODES+1);
	int *cell_sorted;		cuda_malloc(cell_sorted, N_NODES+1);
	
	float *x_max;			cuda_malloc(x_max, BLOCKS*bounding_kernel_factor);
	float *y_max;			cuda_malloc(y_max, BLOCKS*bounding_kernel_factor);
	float *x_min;			cuda_malloc(x_min, BLOCKS*bounding_kernel_factor);
	float *y_min;			cuda_malloc(y_min, BLOCKS*bounding_kernel_factor);


    // Intialize embedding
    float *embedding = Intialization_::randomVector(-5, 5, (N_NODES+1)*2, seed, stream);

    // Make a random vector to add noise to the embeddings
    float *noise = Intialization_::randomVector(-0.05, 0.05, (N_NODES+1)*2, seed, stream);



    // Gradient updates
    float exaggeration = early_exaggeration;
    float momentum = pre_momentum;

    for (size_t i = 0; i < max_iter; i++) {
        if (i == exaggeration_iter) {
            exaggeration = 1.0f;
            momentum = post_momentum;
        }


    }
    //


	// Free everything
	P_PT.destroy();

    cuda_free(noise);
    cuda_free(embedding);

	cuda_free(y_min);
	cuda_free(x_min);
	cuda_free(y_max);
	cuda_free(x_max);

	cuda_free(cell_sorted);
	cuda_free(cell_mass);
	cuda_free(children);
	cuda_free(cell_starts);

	cuda_free(old_forces);
	cuda_free(gains);

	cuda_free(normalization);
	cuda_free(attraction);
	cuda_free(repulsion);
	cuda_free(P_x_Q);

    cuda_free(err);

	// Destory CUDA stream
	CUDA_CHECK(cudaStreamDestroy(stream));
}
