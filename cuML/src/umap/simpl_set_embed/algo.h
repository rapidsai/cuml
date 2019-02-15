/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "umap/umap.h"
#include "solver/solver_c.h"
#include "solver/learning_rate.h"
#include "functions/penalty.h"
#include "functions/linearReg.h"
#include "random/rng.h"
#include <math.h>

namespace UMAPAlgo {

	namespace SimplSetEmbed {

	    using namespace ML;

        static unsigned int g_seed;

	    template<typename T>
	    float rdist(T *X, T *Y, int n) {
	        float result = 0.0;
	        for(int i = 0; i < n; i++)
	            result += pow(X[i]-Y[i], 2);
	        return result;
	    }

	    inline void fast_srand(int seed) { g_seed = seed; }

	    inline int fastrand() {
	      g_seed = (214013*g_seed+2531011);
	      return (g_seed>>16)&0x7FFF;
	    }

	    /**
	     * Given a set of weights and number of epochs, generate
	     * the number of epochs per sample for each weight.
	     *
	     * @param weights: The weights of how much we wish to sample each 1-simplex
	     * @param weights_n: the size of the weights array
	     * @param n_epochs: the total number of epochs we want to train for
	     * @returns an array of number of epochs per sample, one for each 1-simplex
	     */
	    template<typename T>
	    void make_epochs_per_sample(const T *weights, int weights_n, int n_epochs, T *result) {
	        T weights_max = -1.0;
	        for(int i = 0; i < weights_n; i++)  {
	            if(weights[i] > weights_max)
	                weights_max = weights[i];
                result[i] = -1;
	        }
	    }


	    template<typename T>
	    void optimize_layout(
	            T *head_embedding, int head_n,
	            T *tail_embedding, int tail_n,
	            T *head, T *tail, int nnz,
	            T *epochs_per_sample,
	            int n_vertices,
	            UMAPParams *params) {

	        int dim = params->n_neighbors;
	        bool move_other = head_n == tail_n;
	        T alpha = params->initial_alpha;

	        //TODO: Parallelize this!
	        T *epochs_per_negative_sample = (T*)malloc(nnz * sizeof(T));
	        for(int i = 0; i < nnz; i++) {
	            epochs_per_sample[i] / params->negative_sample_rate;
	        }

	        T *epoch_of_next_negative_sample = (T*)malloc(nnz*sizeof(T));
	        memcpy(epoch_of_next_negative_sample, epochs_per_negative_sample, nnz);

	        T *epoch_of_next_sample = (T*)malloc(nnz*sizeof(T));
	        memcpy(epoch_of_next_sample, epochs_per_sample, nnz);

	        for(int n = 0; n < params->n_epochs; n++) {
	            for(int i = 0; i < nnz; i++) {

	                if(epoch_of_next_sample[i] <= n) {
	                    T j = head[i];
	                    T k = tail[i];

	                    T *current = head_embedding[j];
	                    T *other = tail_embedding[k];

	                    float dist_squared = rdist(current, other);

	                    float grad_coeff = 0.0;
	                    if(dist_squared > 0.0) {
	                        grad_coeff = -2.0 * params->a * params->b *
	                                pow(dist_squared, params->b - 1.0);
	                        grad_coeff /= params->a * pow(dist_squared, params->b) + 1.0;
	                    }

	                    for(int d = 0; d < dim; d++) {
	                        float grad_d = clip(grad_coeff * (current[d]-other[d]));
	                        current[d] += grad_d * alpha;
	                        if(move_other)
	                            other[d] += grad_d * alpha;
	                    }

	                    epoch_of_next_sample[i] += epochs_per_sample[i];

	                    int n_neg_samples = int(
	                        (n - epoch_of_next_negative_sample[i]) /
	                        epochs_per_negative_sample[i]
	                    );

	                    for(int p = 0; p < n_neg_samples; p++) {
	                        int k = fastrand() % n_vertices;

	                        other = tail_embedding[k];

	                        dist_squared = rdist(current, other);

	                        if(dist_squared > 0.0) {
	                            grad_coeff = 2.0 * params->gamma * params->b;
	                            grad_coeff /= (0.001 + dist_squared) * (
	                                params->a * pow(dist_squared, params->b) + 1
	                            );
	                        } else if(j == k) {
	                            continue;
	                        } else {
	                            grad_coeff = 0.0;
	                        }

	                        for(int d = 0; d < dim; d++) {

	                            T grad_d = 0.0;
	                            if(grad_coeff > 0.0) {
	                                grad_d = clip(grad_coeff * (current[d] - other[d]));
	                            } else {
	                                grad_d = 4.0;
	                            }

	                            current[d] += grad_d * alpha;
	                        }

	                        epoch_of_next_negative_sample[i] += (
	                            n_neg_samples * epochs_per_negative_sample[i]
	                        );
	                    }
	                }

	                alpha = params->initial_alpha * (1.0 - (float(n) / float(params->n_epochs)));
	            }
	        }
	    }



	    /**
	     * Perform a fuzzy simplicial set embedding, using a specified
	     * initialization method and then minimizing the fuzzy set
	     * cross entropy between the 1-skeleton of the high and low
	     * dimensional fuzzy simplicial sets.
	     */
	    template<typename T>
		void launcher(const T *X, int m, int n,
		        const int *rows, const int *cols, const T *vals, int nnz,
		        UMAPParams *params, T* embedding) {

	        /**
	         * Sum duplicates
	         */

	        /**
	         * Find vals.max()
	         */

	        /**
	         * Go thorugh data and set everything that's less than
	         * vals.max() / params->n_epochs to 0.0
	         */

	        /**
	         * n_vertices = n_neighbors?
	         */





            T *epochs_per_sample = (T*)malloc(nnz * sizeof(T));
            make_epochs_per_sample(vals, nnz, params->n_epochs, epochs_per_sample);


	        T *embedding;
	        MLCommon::allocate(embedding, m * params->n_components);

	        // Doing a random initialization for now
	        MLCommon::Random::Rng<T>::uniform(embedding, m*params->n_components, -10, 10);

	        optimize_layout(embedding, embedding, rows, cols, nnz,
	                          epochs_per_sample, params->n_neighbors, params);
		}
	}
}
