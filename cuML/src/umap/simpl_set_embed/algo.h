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
#include "umap/umapparams.h"

#include "solver/solver_c.h"
#include "solver/learning_rate.h"
#include "functions/penalty.h"
#include "functions/linearReg.h"

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#include <math.h>
#include <string>

namespace UMAPAlgo {

	namespace SimplSetEmbed {

	    namespace Algo {


	        using namespace ML;

	        static unsigned int g_seed;

	        template<typename T>
	        float rdist(const T *X, const T *Y, int n) {
	            float result = 0.0;

	            //TODO: Parallelize
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

	        /**
	         * This could be parallelized
	         */
	        template<typename T>
	        void make_epochs_per_sample(const T *weights, int weights_n, int n_epochs, T *result) {
	            T weights_max = -1.0;

	            // TODO: Parallelize
	            for(int i = 0; i < weights_n; i++)  {
	                if(weights[i] > weights_max)
	                    weights_max = weights[i];
	                result[i] = -1;
	            }

	            // TODO: Parallelize
	            for(int i = 0; i < weights_n; i++) {
	                T v = weights[i] / weights_max;
	                if(v*n_epochs > 0)
	                    result[i] = v;

	            }
	        }

	        template<typename T>
	        T clip(T val, T lb, T ub) {
	            if(val > ub)
	                return ub;
	            else if(val < lb)
	                return lb;
	            else
	                return val;
	        }

	        template<typename T>
	        void print_arr(T *arr, int len, std::string name) {
	            std::cout << name << " = [";
	            for(int i = 0; i < len; i++)
	                std::cout << arr[i] << " ";
	            std::cout << "]" << std::endl;
	        }

	        template<typename T>
	        T repulsive_grad(T dist_squared, UMAPParams *params) {
                T grad_coeff = 2.0 * params->gamma * params->b;
                grad_coeff /= (0.001 + dist_squared) * (
                    params->a * pow(dist_squared, params->b) + 1
                );
                return grad_coeff;
	        }

	        template<typename T>
	        T attractive_grad(T dist_squared, UMAPParams *params) {
                T grad_coeff = -2.0 * params->a * params->b *
                        pow(dist_squared, params->b - 1.0);
                grad_coeff /= params->a * pow(dist_squared, params->b) + 1.0;
                return grad_coeff;
	        }

	        /**
	         * Runs a stochastic gradient descent using sampling weights defined on
	         * both the attraction and repulsion vectors.
	         *
	         * The python version is not mini-batched, but this would be ideal
	         * in order to improve the parallelism.
	         *
	         * In this SGD implementation, the weights being tuned are actually the
	         * embeddings themselves, as the objective function is attracting
	         * positive weights (neighbors in the 1-skeleton) and repelling
	         * negative weights (non-neighbors in the 1-skeleton). It's important
	         * to think through the implications of this in the batching, as
	         * it means threads will need to be synchronized when updating
	         * the same embeddings.
	         */
	        template<typename T>
	        void optimize_layout(
	                T *head_embedding, int head_n,
	                T *tail_embedding, int tail_n,
	                const int *head, const int *tail, int nnz,
	                T *epochs_per_sample,
	                int n_vertices,
	                UMAPParams *params) {

	            // unsupervised training?
	            bool move_other = head_n == tail_n;

	            T alpha = params->initial_alpha;
	            T *epochs_per_negative_sample = (T*)malloc(nnz * sizeof(T));

	            print_arr(epochs_per_negative_sample, nnz, "epochs_per_negative_sample");

	            //TODO: Parallelize
	            for(int i = 0; i < nnz; i++)
	                epochs_per_negative_sample[i] = epochs_per_sample[i] / params->negative_sample_rate;

	            print_arr(epochs_per_sample, nnz, "epochs_per_sample");

	            T *epoch_of_next_negative_sample = (T*)malloc(nnz*sizeof(T));
	            memcpy(epoch_of_next_negative_sample, epochs_per_negative_sample, nnz);

	            print_arr(epoch_of_next_negative_sample, nnz, "epoch_of_next_negative_sample");

	            T *epoch_of_next_sample = (T*)malloc(nnz*sizeof(T));
	            memcpy(epoch_of_next_sample, epochs_per_sample, nnz);

	            print_arr(epoch_of_next_sample, nnz, "epoch_of_next_sample");

	            // TODO: Need to think about this so that it can be done in parallel because
	            // Leland wrote this in a way where sequential iterations are required.
	            // There are better ways we could guarantee certain neighbors in the 1-skeleton
	            // are chosen some pre-determined number of times without the need for a
	            // dependence across iterations.

	            for(int n = 0; n < params->n_epochs; n++) {

	                /**
	                 * TODO: Do this on GPU with the following SGD design:
	                 * 1) A pluggable batching strategy that is able to sample embeddings based on
	                 *    a set of weights.
	                 * 2) A pluggable strategy for providing coefficients to the loss function
	                 *    (in UMAP, the embeddings themselves ARE the parameters)
	                 * 3) A negative sampling strategy, also making use of possible weighted
	                 *    sampling.
	                 */
	                for(int i = 0; i < nnz; i++) {

	                    /**
	                     * Positive sample stage (attractive forces)
	                     */
	                    if(epoch_of_next_sample[i] <= n) {

	                        int j = head[i];
	                        int k = tail[i];

	                        T *current = head_embedding+(j*params->n_components);
	                        T *other = tail_embedding+(k*params->n_components);

	                        float dist_squared = rdist(current, other, params->n_components);

                            // Aatractive force between the two vertices
	                        T attractive_grad_coeff = 0.0;
	                        if(dist_squared > 0.0) {
	                            attractive_grad_coeff = attractive_grad(dist_squared, params);
	                        }

	                        /**
	                         * Apply attractive force between `current` and `other`
	                         * by updating their 'weights' to put them closer in
	                         * Euclidean space.
	                         * (update other embedding only if we are
	                         * performing unsupervised training).
	                         */
	                        for(int d = 0; d < params->n_components; d++) {
	                            T grad_d = clip(attractive_grad_coeff * (current[d]-other[d]), -4.0f, 4.0f);
	                            current[d] += grad_d * alpha;

	                            // happens only during unsupervised training
	                            if(move_other)
	                                other[d] += -grad_d * alpha;
	                        }

	                        epoch_of_next_sample[i] += epochs_per_sample[i];

	                        // choose negative samples
	                        int n_neg_samples = int(
	                            (n - epoch_of_next_negative_sample[i]) /
	                            epochs_per_negative_sample[i]
	                        );

	                        /**
	                         * Negative sampling stage (repulsive forces)
	                         */
	                        for(int p = 0; p < n_neg_samples; p++) {

	                            int rand = fastrand() % n_vertices;

	                            T *negative_sample = tail_embedding+(rand*params->n_components);
	                            dist_squared = rdist(current, negative_sample, params->n_components);

	                            // repulsive force between two vertices
	                            T repulsive_grad_coeff = 0.0;
	                            if(dist_squared > 0.0) {
	                                repulsive_grad_coeff = repulsive_grad(dist_squared, params);
	                            } else if(j == rand)
	                                continue;

	                            /**
	                             * Apply repulsive force between `current` and `other`
	                             * (which is has been negatively sampled) by updating
	                             * their 'weights' to push them farther in Euclidean space.
	                             */
	                            for(int d = 0; d < params->n_components; d++) {
	                                T grad_d = 0.0;
	                                if(repulsive_grad_coeff > 0.0)
	                                    grad_d = clip(repulsive_grad_coeff * (current[d] - negative_sample[d]), -4.0f, 4.0f);
	                                else
	                                    grad_d = 4.0;
	                                current[d] += grad_d * alpha;
	                            }

	                            epoch_of_next_negative_sample[i] +=
	                                n_neg_samples * epochs_per_negative_sample[i];
	                        }
	                    }

	                    alpha = params->initial_alpha * (1.0 - (float(n) / float(params->n_epochs)));
	                }
	            }

	            free(epochs_per_negative_sample);
	            free(epoch_of_next_negative_sample);
	            free(epoch_of_next_sample);
	        }

	        /**
	         * Perform a fuzzy simplicial set embedding, using a specified
	         * initialization method and then minimizing the fuzzy set
	         * cross entropy between the 1-skeleton of the high and low
	         * dimensional fuzzy simplicial sets.
	         */
	        template<typename T, int TPB_X>
	        void launcher(int m, int n,
	                const int *rows, const int *cols, T *vals, int nnz,
	                UMAPParams *params, T* embedding) {

	            dim3 grid(MLCommon::ceildiv(m, TPB_X), 1, 1);
	            dim3 blk(TPB_X, 1, 1);
	            srand(50);

	            /**
	             * Find vals.max()
	             */
	            thrust::device_ptr<const T> d_ptr = thrust::device_pointer_cast(vals);
	            T max = *(thrust::max_element(d_ptr, d_ptr+nnz));

	            /**
	             * Go through COO values and set everything that's less than
	             * vals.max() / params->n_epochs to 0.0
	             */
	            auto adjust_vals_op = [] __device__(T input, T scalar) {
	                if (input < scalar)
	                    return 0.0f;
	                else
	                    return input;
	            };

	            MLCommon::LinAlg::unaryOp<T>(vals, vals, (max / params->n_epochs), nnz, adjust_vals_op);

	            T *vals_h = (T*)malloc(nnz * sizeof(T));
	            MLCommon::updateHost(vals_h, vals, nnz);

	            std::cout << "nnz=" << nnz << std::endl;

	            T *epochs_per_sample = (T*)malloc(nnz * sizeof(T));
	            make_epochs_per_sample(vals_h, nnz, params->n_epochs, epochs_per_sample);

	            int *head_h = (int*)malloc(nnz * sizeof(int));
	            int *tail_h = (int*)malloc(nnz * sizeof(int));

	            MLCommon::updateHost(head_h, rows, nnz);
	            MLCommon::updateHost(tail_h, cols, nnz);

	            T *embedding_h = (T*)malloc(m*params->n_components*sizeof(T));
	            MLCommon::updateHost(embedding_h, embedding, m*params->n_components);

	            std::cout << "Calling optimize layout" << std::endl;

	            optimize_layout(embedding_h, m,
	                            embedding_h, m,
	                            head_h, tail_h, nnz,
	                            epochs_per_sample,
	                            m,
	                            params);

	            MLCommon::updateDevice(embedding, embedding_h, m*params->n_components);

                print_arr(embedding_h, m*params->n_components, "embeddings");

                std::cout << "Freeing vecs" << std::endl;
	            free(head_h);
	            free(tail_h);
	            free(vals_h);

                free(epochs_per_sample);

                std::cout << "Freeing embedding_h" << std::endl;
                free(embedding_h);
	        }
		}
	}
}
