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

#include "umap/umapparams.h"

#include "random/rng.h"

#include <cstdlib>


#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#include <math.h>
#include <string>

#pragma once

namespace UMAPAlgo {

	namespace SimplSetEmbed {

	    namespace Algo {

            using namespace ML;

            unsigned long long nextSeed() {
	            unsigned long long t1 = (unsigned long long) (rand() + rand());
	            unsigned long long t2 = (unsigned long long) (rand() + rand());
	            return ((t2 << 32) | t1);
	        }

	        template <typename Type>
	        DI Type randVal(unsigned long long& state) {
	            Type res;
	            for (int i=0;i<128;i++)
	                state = (state >> 1) ^ (-(state & 1ULL) & TAPS);
	            res = static_cast<Type>(state);
	            res /= static_cast<Type>(1.8446744073709551614e19);
	            return res;
	        }



	        template<typename T>
	        __device__ __host__ float rdist(const T *X, const T *Y, int n) {
	            float result = 0.0;
	            //TODO: Parallelize
	            for(int i = 0; i < n; i++)
	                result += pow(X[i]-Y[i], 2);
	            return result;
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
	        void make_epochs_per_sample(T *weights, int weights_n, int n_epochs, T *result) {
	            thrust::device_ptr<T> d_weights = thrust::device_pointer_cast(weights);
	            T weights_max = *(thrust::max_element(d_weights, d_weights+weights_n));

                MLCommon::LinAlg::unaryOp<T>(result, weights, weights_max, weights_n,
                    [n_epochs] __device__(T input, T scalar) {
                        T v = input / scalar;
                        if(v*n_epochs > 0)
                            return v;
                        else
                            return T(-1.0);
                    }
                );
	        }

	        template<typename T>
	        __device__ __host__ T clip(T val, T lb, T ub) {
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
	        __device__ __host__ T repulsive_grad(T dist_squared, float gamma, UMAPParams params) {
                T grad_coeff = 2.0 * gamma * params.b;
                grad_coeff /= (0.001 + dist_squared) * (
                    params.a * pow(dist_squared, params.b) + 1
                );
                return grad_coeff;
	        }

	        template<typename T>
	        __device__ __host__ T attractive_grad(T dist_squared, UMAPParams params) {
                T grad_coeff = -2.0 * params.a * params.b *
                        pow(dist_squared, params.b - 1.0);
                grad_coeff /= params.a * pow(dist_squared, params.b) + 1.0;
                return grad_coeff;
	        }

	        template <typename T, int TPB_X>
	        __global__ void optimize_batch_kernel(
                    T *head_embedding, int head_n,
                    T *tail_embedding, int tail_n,
                    const int *head, const int *tail, int nnz,
                    T *epochs_per_sample,
                    int n_vertices,
                    bool move_other,
                    T *epochs_per_negative_sample,
                    T *epoch_of_next_negative_sample,
                    T *epoch_of_next_sample,
                    float alpha,
                    int epoch,
                    float gamma,
                    unsigned long long seed,
                    UMAPParams params) {

                int row = (blockIdx.x * TPB_X) + threadIdx.x;
                unsigned long long state = seed + row + 1;

                if(row < nnz) {
                    /**
                     * Positive sample stage (attractive forces)
                     */
                    if(epoch_of_next_sample[row] <= epoch) {

                        int j = head[row];
                        int k = tail[row];

                        T *current = head_embedding+(j*params.n_components);
                        T *other = tail_embedding+(k*params.n_components);

                        float dist_squared = rdist(current, other, params.n_components);

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
                        for(int d = 0; d < params.n_components; d++) {
                            T grad_d = clip(attractive_grad_coeff * (current[d]-other[d]), -4.0f, 4.0f);
                            atomicAdd(current+d, grad_d * alpha);

                            // happens only during unsupervised training
                            if(move_other)
                                atomicAdd(other+d, -grad_d * alpha);
                        }

                        epoch_of_next_sample[row] += epochs_per_sample[row];

                        // choose negative samples
                        int n_neg_samples = int(
                            (epoch - epoch_of_next_negative_sample[row]) /
                            epochs_per_negative_sample[row]
                        );

                        for(int p = 0; p < n_neg_samples; p++) {

                            int t = int(randVal<float>(state) * tail_n);

                            T *negative_sample = tail_embedding+(t*params.n_components);
                            dist_squared = rdist(current, negative_sample, params.n_components);

                            // repulsive force between two vertices
                            T repulsive_grad_coeff = 0.0;
                            if(dist_squared > 0.0) {
                                repulsive_grad_coeff = repulsive_grad(dist_squared, gamma, params);
                            } else if(j == t)
                                continue;

                            /**
                             * Apply repulsive force between `current` and `other`
                             * (which is has been negatively sampled) by updating
                             * their 'weights' to push them farther in Euclidean space.
                             */
                            for(int d = 0; d < params.n_components; d++) {
                                T grad_d = 0.0;
                                if(repulsive_grad_coeff > 0.0)
                                    grad_d = clip(repulsive_grad_coeff * (current[d] - negative_sample[d]), -4.0f, 4.0f);
                                else
                                    grad_d = 4.0;
                                atomicAdd(current+d, grad_d * alpha);
                            }

                            epoch_of_next_negative_sample[row] +=
                                n_neg_samples * epochs_per_negative_sample[row];
                        }
                    }
                }
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
	        template<typename T, int TPB_X>
	        void optimize_layout(
	                T *head_embedding, int head_n,
	                T *tail_embedding, int tail_n,
	                const int *head, const int *tail, int nnz,
	                T *epochs_per_sample,
	                int n_vertices,
	                float gamma,
	                UMAPParams *params) {

	            // have we been given y-values?
	            bool move_other = head_n == tail_n;

	            T alpha = params->initial_alpha;

	            T *epochs_per_negative_sample;
	            MLCommon::allocate(epochs_per_negative_sample, nnz);

	            MLCommon::LinAlg::unaryOp<T>(epochs_per_negative_sample, epochs_per_sample,
	                    params->negative_sample_rate, nnz,
	                    [] __device__(T input, T scalar) { return input / scalar; }
	            );

	            T *epoch_of_next_negative_sample;
                MLCommon::allocate(epoch_of_next_negative_sample, nnz);
                MLCommon::copy(epoch_of_next_negative_sample, epochs_per_negative_sample, nnz);

	            T *epoch_of_next_sample;
                MLCommon::allocate(epoch_of_next_sample, nnz);
                MLCommon::copy(epoch_of_next_sample, epochs_per_sample, nnz);

                dim3 grid(MLCommon::ceildiv(head_n, TPB_X), 1, 1);
                dim3 blk(TPB_X, 1, 1);
	            for(int n = 0; n < params->n_epochs; n++) {

	                // TODO: Might need to batch this further when nnz > TPB * N_BLOCKS
	                optimize_batch_kernel<T, TPB_X><<<grid,blk>>>(
	                    head_embedding, head_n,
	                    tail_embedding, tail_n,
	                    head, tail, nnz,
	                    epochs_per_sample,
	                    n_vertices,
	                    move_other,
	                    epochs_per_negative_sample,
	                    epoch_of_next_negative_sample,
	                    epoch_of_next_sample,
	                    alpha,
	                    n,
	                    gamma,
	                    nextSeed(),
	                    *params
	                );

                    alpha = params->initial_alpha * (1.0 - (float(n) / float(params->n_epochs)));
	            }

	            cudaFree(epochs_per_negative_sample);
	            cudaFree(epoch_of_next_negative_sample);
	            cudaFree(epoch_of_next_sample);
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
                MLCommon::LinAlg::unaryOp<T>(vals, vals, (max / params->n_epochs), nnz,
                    [] __device__(T input, T scalar) {
                        if (input < scalar)
                            return 0.0f;
                        else
                            return input;
                    }
                );

	            T *epochs_per_sample;
	            MLCommon::allocate(epochs_per_sample, nnz);

	            make_epochs_per_sample(vals, nnz, params->n_epochs, epochs_per_sample);

	            std::cout << "Calling optimize layout" << std::endl;

	            optimize_layout<T, TPB_X>(embedding, m,
	                            embedding, m,
	                            rows, cols, nnz,
	                            epochs_per_sample,
	                            m,
	                            params->gamma,
	                            params);

                std::cout << MLCommon::arr2Str(embedding, m*params->n_components, "embeddings") << std::endl;

                CUDA_CHECK(cudaFree(epochs_per_sample));
	        }
		}
	}
}
