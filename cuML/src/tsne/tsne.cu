
#include "tsne.h"

namespace TSNE {

	void fit(const float * __restrict__ X,
            const int n,
            const int p,
            const int n_components = 2,
            const float perplexity = 50.0f,
            const float perplexity_epsilon = 1e-3f,
            const float early_exaggeration = 2.0f,
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
            const float min_grad_norm = 1e-7f,

            // Seed for random data
            const long long seed = -1

            // resulting embedding
            float *embeddings)
    {
    	*embeddings = runTsne(
    		X, n, p, n_components, 
    		perplexity, perplexity_epsilon, early_exaggeration, n_neighbors,
    		learning_rate, pre_momentum, post_momentum,
    		theta, epsilon_squared,
    		exaggeration_iter, max_iter, min_grad_norm,
    		seed);
    }



    void fit(const double * __restrict__ X,
            const int n,
            const int p,
            const int n_components = 2,
            const float perplexity = 50.0f,
            const float perplexity_epsilon = 1e-3f,
            const float early_exaggeration = 2.0f,
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
            const float min_grad_norm = 1e-7f,

            // Seed for random data
            const long long seed = -1

            // resulting embedding
            float *embeddings)
    {
    	*embeddings = runTsne(
    		X, n, p, n_components, 
    		perplexity, perplexity_epsilon, early_exaggeration, n_neighbors,
    		learning_rate, pre_momentum, post_momentum,
    		theta, epsilon_squared,
    		exaggeration_iter, max_iter, min_grad_norm,
    		seed);
    }
}