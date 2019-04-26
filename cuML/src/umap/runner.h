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

#pragma once

#include "umapparams.h"
#include "optimize.h"
#include "supervised.h"

#include "fuzzy_simpl_set/runner.h"
#include "knn_graph/runner.h"
#include "simpl_set_embed/runner.h"
#include "init_embed/runner.h"


#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>

#include "sparse/csr.h"
#include "sparse/coo.h"

#include "knn/knn.h"

#include "cuda_utils.h"

#include <iostream>
#include <cuda_runtime.h>



namespace UMAPAlgo {

    // Swap this as impls change for now.
    namespace FuzzySimplSetImpl = FuzzySimplSet::Naive;
    namespace SimplSetEmbedImpl = SimplSetEmbed::Algo;

	using namespace ML;
	using namespace MLCommon::Sparse;


    template<int TPB_X, typename T>
    __global__ void init_transform(int *indices, T *weights, int n,
                        const T *embeddings, int embeddings_n, int n_components,
                        T *result, int n_neighbors) {

        // row-based matrix 1 thread per row
        int row = (blockIdx.x * TPB_X) + threadIdx.x;
        int i = row * n_neighbors; // each thread processes one row of the dist matrix

        if(row < n) {
            for(int j = 0; j < n_neighbors; j++) {
                for(int d = 0; d < n_components; d++) {
                    result[row*n_components+d] += weights[i+j] * embeddings[indices[i+j]*n_components+d];
                }
            }
        }
    }

    /**
     * Simple helper function to set the values of an array to zero.
     *
     * @param vals array of values
     * @param nnz size of array of values
     */
    template<int TPB_X>
    __global__ void reset_vals(int *vals, int nnz) {
        int row = (blockIdx.x * TPB_X) + threadIdx.x;
        if(row < nnz)
            vals[row] = 0.0;
    }



    /**
     * Firt exponential decay curve to find the parameters
     * a and b, which are based on min_dist and spread
     * parameters.
     */
    void find_ab(UMAPParams *params, cudaStream_t stream) {
        Optimize::find_params_ab(params, stream);
    }

    /**
     * Fit
     */
	template<typename T, int TPB_X>
	size_t _fit(T *X,       // input matrix
	            int n,      // rows
	            int d,      // cols
	            kNN *knn,
	            UMAPParams *params,
	            T *embeddings,
              cudaStream_t stream) {

        int k = params->n_neighbors;

	    find_ab(params, stream);

		/**
		 * Allocate workspace for kNN graph
		 */
		long *knn_indices;
		T *knn_dists;

        MLCommon::allocate(knn_indices, n*k);
		MLCommon::allocate(knn_dists, n*k);

		kNNGraph::run(X, n,d, knn_indices, knn_dists, knn, params, stream);
		CUDA_CHECK(cudaPeekAtLastError());

		COO<T> rgraph_coo(n*k*2, n, n);

		FuzzySimplSet::run<TPB_X, T>(rgraph_coo.n_rows,
		                   knn_indices, knn_dists,
		                   k,
                           &rgraph_coo,
						   params, 0);
		CUDA_CHECK(cudaPeekAtLastError());

		/**
		 * Remove zeros from simplicial set
		 */
        int *row_count_nz, *row_count;
        MLCommon::allocate(row_count_nz, n, true);
        MLCommon::allocate(row_count, n, true);

        COO<T> cgraph_coo;
        MLCommon::Sparse::coo_remove_zeros<TPB_X, T>(&rgraph_coo, &cgraph_coo, stream);

        /**
         * Run initialization method
         */
		InitEmbed::run(X, n, d,
		        knn_indices, knn_dists,
		        &cgraph_coo,
		        params, embeddings, stream,
		        params->init);

		/**
		 * Run simplicial set embedding to approximate low-dimensional representation
		 */
		SimplSetEmbed::run<TPB_X, T>(
		        X, n, d,
		        &cgraph_coo,
		        params, embeddings, stream);

        CUDA_CHECK(cudaPeekAtLastError());

		CUDA_CHECK(cudaFree(knn_dists));
        CUDA_CHECK(cudaFree(knn_indices));

		return 0;
	}

    template<typename T, int TPB_X>
	size_t _fit(T *X,    // input matrix
	            T *y,    // labels
                int n,
                int d,
                kNN *knn,
                UMAPParams *params,
                T *embeddings, cudaStream_t stream) {

        int k = params->n_neighbors;

        std::cout << params->verbose << std::endl;

	    if(params->target_n_neighbors == -1)
	        params->target_n_neighbors = params->n_neighbors;

        find_ab(params, stream);

        /**
         * Allocate workspace for kNN graph
         */
        long *knn_indices;
        T *knn_dists;

        MLCommon::allocate(knn_indices, n*k, true);
        MLCommon::allocate(knn_dists, n*k, true);

        kNNGraph::run(X, n,d, knn_indices, knn_dists, knn, params, stream);
        CUDA_CHECK(cudaPeekAtLastError());

        /**
         * Allocate workspace for fuzzy simplicial set.
         */
        COO<T> rgraph_coo(n*k*2, n, n);

        /**
         * Run Fuzzy simplicial set
         */
       //int nnz = n*k*2;
        FuzzySimplSet::run<TPB_X, T>(n,
                           knn_indices, knn_dists,
                           params->n_neighbors,
                           &rgraph_coo,
                           params, 0);
        CUDA_CHECK(cudaPeekAtLastError());

        COO<T> final_coo;

        /**
         * If target metric is 'categorical', apply a
         * categorical simplicial set intersection.
         */
        if(params->target_metric == ML::UMAPParams::MetricType::CATEGORICAL) {
            Supervised::perform_categorical_intersection<TPB_X, T>(
                    y,
                    &rgraph_coo, &final_coo,
                    params, stream);

        /**
         * Otherwise, perform general simplicial set intersection
         */
        } else {
            Supervised::perform_general_intersection<TPB_X, T>(
                    y,
                    &rgraph_coo, &final_coo,
                    params, stream);
        }

        /**
         * Remove zeros
         */
        int *final_row_count, *final_row_count_nz;
        MLCommon::allocate(final_row_count, rgraph_coo.n_rows, true);
        MLCommon::allocate(final_row_count_nz, rgraph_coo.n_rows, true);

        MLCommon::Sparse::coo_sort<T>(&final_coo);

        COO<T> ocoo;
        MLCommon::Sparse::coo_remove_zeros<TPB_X, T>(&final_coo, &ocoo, stream);

        if(params->verbose) {
            std::cout << "Reset Local connectivity" << std::endl;
            std::cout << "result_nnz=" << ocoo.nnz << std::endl;
            std::cout << MLCommon::arr2Str(ocoo.rows, ocoo.nnz, "final_rows") << std::endl;
            std::cout << MLCommon::arr2Str(ocoo.cols, ocoo.nnz, "final_cols") << std::endl;
            std::cout << MLCommon::arr2Str(ocoo.vals, ocoo.nnz, "final_vals") << std::endl;
        }


        /**
         * Initialize embeddings
         */
        InitEmbed::run(X, n, d,
                knn_indices, knn_dists, &ocoo,
                params, embeddings, stream, params->init);

        /**
         * Run simplicial set embedding to approximate low-dimensional representation
         */
        SimplSetEmbed::run<TPB_X, T>(
                X, n, d,
                &ocoo,
                params, embeddings, stream);

        if(params->verbose)
            std::cout << MLCommon::arr2Str(embeddings, n*params->n_components, "embeddings") << std::endl;

        CUDA_CHECK(cudaPeekAtLastError());

        CUDA_CHECK(cudaFree(knn_dists));
        CUDA_CHECK(cudaFree(knn_indices));

	    return 0;
	}

	/**
	 *
	 */
	template<typename T, int TPB_X>
	size_t _transform(const float *X,
	                  int n,
	                  int d,
	                  T *embedding,
	                  int embedding_n,
                      kNN *knn,
	                  UMAPParams *params,
	                  T *transformed,
                    cudaStream_t stream) {

	    /**
	     * Perform kNN of X
	     */
        long *knn_indices;
        float *knn_dists;
        MLCommon::allocate(knn_indices, n*params->n_neighbors);
        MLCommon::allocate(knn_dists, n*params->n_neighbors);

        knn->search(X, n, knn_indices, knn_dists, params->n_neighbors);
        CUDA_CHECK(cudaPeekAtLastError());

        MLCommon::LinAlg::unaryOp<T>(knn_dists, knn_dists, n*params->n_neighbors,
            [] __device__(T input) { return sqrt(input); },
        stream);

	    float adjusted_local_connectivity = max(0.0, params->local_connectivity - 1.0);

	    if(params->verbose)
	        std::cout << "adjusted_local_connectivity=" << adjusted_local_connectivity << std::endl;

	    /**
	     * Perform smooth_knn_dist
	     */
        T *sigmas;
        T *rhos;
        MLCommon::allocate(sigmas, n, true);
        MLCommon::allocate(rhos, n, true);

        dim3 grid_n(MLCommon::ceildiv(n, TPB_X), 1, 1);
        dim3 blk(TPB_X, 1, 1);

        FuzzySimplSetImpl::smooth_knn_dist<TPB_X, T>(n, knn_indices, knn_dists,
                rhos, sigmas, params, params->n_neighbors, adjusted_local_connectivity, stream
        );

        /**
         * Compute graph of membership strengths
         */

        int nnz = n*params->n_neighbors;

        dim3 grid_nnz(MLCommon::ceildiv(nnz, TPB_X), 1, 1);

        /**
         * Allocate workspace for fuzzy simplicial set.
         */

        COO<T> graph_coo(nnz, n, n);


        FuzzySimplSetImpl::compute_membership_strength_kernel<TPB_X><<<grid_n, blk>>>(
                knn_indices, knn_dists,
                sigmas, rhos,
                graph_coo.vals, graph_coo.rows, graph_coo.cols, graph_coo.n_rows,
                params->n_neighbors);
        CUDA_CHECK(cudaPeekAtLastError());

        if(params->verbose) {
            std::cout << MLCommon::arr2Str(sigmas, n, "sigmas") << std::endl;
            std::cout << MLCommon::arr2Str(rhos, n, "rhos") << std::endl;
        }

        int *row_ind, *ia;
        MLCommon::allocate(row_ind, n);
        MLCommon::allocate(ia, n);

        MLCommon::Sparse::sorted_coo_to_csr(&graph_coo, row_ind);
        MLCommon::Sparse::coo_row_count<TPB_X>(&graph_coo, ia);

        T *vals_normed;
        MLCommon::allocate(vals_normed, graph_coo.nnz, true);

         MLCommon::Sparse::csr_row_normalize_l1<TPB_X, T><<<grid_n, blk>>>(row_ind, graph_coo.vals, graph_coo.nnz,
                 graph_coo.n_rows, vals_normed);

        init_transform<TPB_X, T><<<grid_n,blk>>>(graph_coo.cols, vals_normed, graph_coo.n_rows,
                embedding, embedding_n, params->n_components,
                transformed, params->n_neighbors);

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaFree(vals_normed));

        reset_vals<TPB_X><<<grid_n,blk>>>(ia, n);

        /**
         * Go through COO values and set everything that's less than
         * vals.max() / params->n_epochs to 0.0
         */
        thrust::device_ptr<T> d_ptr = thrust::device_pointer_cast(graph_coo.vals);
        T max = *(thrust::max_element(d_ptr, d_ptr+nnz));

        int n_epochs = 1000;//params->n_epochs;

        MLCommon::LinAlg::unaryOp<T>(graph_coo.vals, graph_coo.vals, graph_coo.nnz,
            [=] __device__(T input) {
                if (input < (max / float(n_epochs)))
                    return 0.0f;
                else
                    return input;
            },
            stream);

        CUDA_CHECK(cudaPeekAtLastError());

        /**
         * Remove zeros
         */
        MLCommon::Sparse::COO<T> comp_coo;
        MLCommon::Sparse::coo_remove_zeros<TPB_X, T>(&graph_coo, &comp_coo, stream);

        CUDA_CHECK(cudaPeekAtLastError());

        T *epochs_per_sample;
        MLCommon::allocate(epochs_per_sample, nnz);

        SimplSetEmbedImpl::make_epochs_per_sample(comp_coo.vals, comp_coo.nnz, params->n_epochs,
                                                    epochs_per_sample, stream);
        CUDA_CHECK(cudaPeekAtLastError());

        SimplSetEmbedImpl::optimize_layout<TPB_X, T>(
            transformed, n,
            embedding, embedding_n,
            comp_coo.rows, comp_coo.cols, comp_coo.nnz,
            epochs_per_sample,
            n,
            params->repulsion_strength,
            params,
            n_epochs,
            stream
        );
        CUDA_CHECK(cudaPeekAtLastError());

        CUDA_CHECK(cudaFree(knn_dists));
        CUDA_CHECK(cudaFree(knn_indices));


        CUDA_CHECK(cudaFree(sigmas));
        CUDA_CHECK(cudaFree(rhos));


        CUDA_CHECK(cudaFree(ia));
        CUDA_CHECK(cudaFree(row_ind));

        CUDA_CHECK(cudaFree(epochs_per_sample));

        return 0;

	}

}
