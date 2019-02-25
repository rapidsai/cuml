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
#include "umapparams.h"
#include "optimize.h"

#include "fuzzy_simpl_set/runner.h"
#include "knn_graph/runner.h"
#include "simpl_set_embed/runner.h"
#include "init_embed/runner.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/count.h>
#include <thrust/extrema.h>

#include "sparse/csr.h"
#include "sparse/coo.h"

#include "knn/knn.h"

#include "cuda_utils.h"

#include <iostream>
#include <cuda_runtime.h>

#pragma once

namespace UMAPAlgo {

    // Swap this as impls change for now.
    namespace FuzzySimplSetImpl = FuzzySimplSet::Naive;
    namespace SimplSetEmbedImpl = SimplSetEmbed::Algo;

	using namespace ML;

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

                    printf("INDICES: %d\n", indices[i+j]);
                    result[row*n_components+d] += weights[i+j] * embeddings[indices[i+j]*n_components+d];
                }
            }
        }
	}

    template<int TPB_X, typename T>
    __global__ void coo_row_counts(int *rows, T *vals, int nnz,
            int *results, int n) {
        int row = (blockIdx.x * TPB_X) + threadIdx.x;
        if(row < n && vals[row] != 0.0)
            atomicAdd(results+row, 1);
    }


    void find_ab(UMAPParams *params) {
        Optimize::find_params_ab(params);
    }


	template<typename T>
	size_t _fit(const T *X,       // input matrix
	            int n,      // rows
	            int d,      // cols
	            kNN *knn,
	            UMAPParams *params,
	            T *embeddings) {

		/**
		 * Allocate workspace for kNN graph
		 */
		long *knn_indices;
		T *knn_dists;

        MLCommon::allocate(knn_indices, n*params->n_neighbors);
		MLCommon::allocate(knn_dists, n*params->n_neighbors);

        std::cout << "Running knnGraph" << std::endl;

        kNNGraph::run(X, n,d, knn_indices, knn_dists, knn, params);
		CUDA_CHECK(cudaPeekAtLastError());

		std::cout << MLCommon::arr2Str(knn_indices, n*params->n_neighbors, "knn_indices");
        std::cout << MLCommon::arr2Str(knn_dists, n*params->n_neighbors, "knn_dists");

		std::cout << "Finished knnGraph" << std::endl;

		int *graph_rows, *graph_cols;
		T *graph_vals;

		/**
		 * Allocate workspace for fuzzy simplicial set.
		 */
		MLCommon::allocate(graph_rows, n*params->n_neighbors);
		MLCommon::allocate(graph_cols, n*params->n_neighbors);
		MLCommon::allocate(graph_vals, n*params->n_neighbors);

		/**
		 * Run Fuzzy simplicial set
		 */
        std::cout << "Running FuzzySimplSet knnGraph" << std::endl;
		int nnz = 0;

		FuzzySimplSet::run(n, knn_indices, knn_dists,
						   graph_rows,
						   graph_cols,
						   graph_vals,
						   params, &nnz,0);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::cout << "ROWS=" << MLCommon::arr2Str<int>(graph_rows, nnz, "rows") << std::endl;
        std::cout << "COLS=" << MLCommon::arr2Str<int>(graph_cols, nnz, "cols") << std::endl;
        std::cout << "VALS=" << MLCommon::arr2Str<float>(graph_vals, nnz, "vals") << std::endl;


		CUDA_CHECK(cudaPeekAtLastError());
        std::cout << "Finished FuzzySimplSet" << std::endl;

		T *embedding;
		MLCommon::allocate(embedding, n * params->n_components);

		InitEmbed::run(X, n, d,
		        knn_indices, knn_dists,
		        params, embedding);
		CUDA_CHECK(cudaDeviceSynchronize());

		/**
		 * Run simplicial set embedding to approximate low-dimensional representation
		 */
        std::cout << "Running SimplSetEmbed" << std::endl;
		SimplSetEmbed::run<T, 256>(
		        X, n, d,
		        graph_rows, graph_cols, graph_vals, nnz,
		        params, embedding);

        CUDA_CHECK(cudaPeekAtLastError());
		std::cout << "Finished SimplSetEmbed" << std::endl;

		MLCommon::updateHost(embeddings, embedding, n*params->n_components);

		CUDA_CHECK(cudaFree(embedding));
		CUDA_CHECK(cudaFree(knn_dists));
        CUDA_CHECK(cudaFree(knn_indices));
        CUDA_CHECK(cudaFree(graph_rows));
        CUDA_CHECK(cudaFree(graph_cols));
        CUDA_CHECK(cudaFree(graph_vals));

		return 0;
	}

	template<typename T, int TPB_X>
	size_t _transform(const float *X,
	                  int n,
	                  int d,
	                  T *embedding,
	                  int embedding_n,
                      kNN *knn,
	                  UMAPParams *params,
	                  T *transformed) {

        dim3 grid(MLCommon::ceildiv(n, TPB_X), 1, 1);
        dim3 blk(TPB_X, 1, 1);

        std::cout << "Inside transform" << std::endl;

	    /**
	     * Perform kNN of X
	     */
        long *knn_indices;
        float *knn_dists;
        MLCommon::allocate(knn_indices, n*params->n_neighbors);
        MLCommon::allocate(knn_dists, n*params->n_neighbors);

        std::cout << "Running knnGraph" << std::endl;


        knn->search(X, n, knn_indices, knn_dists, params->n_neighbors);
        CUDA_CHECK(cudaPeekAtLastError());

        auto sqrt_vals_op = [] __device__(T input, T scalar) {
            return sqrt(input);
        };
        MLCommon::LinAlg::unaryOp<T>(knn_dists, knn_dists, 1.0, n*params->n_neighbors, sqrt_vals_op);

	    float adjusted_local_connectivity = max(0.0, params->local_connectivity - 1.0);

	    /**
	     * Perform smooth_knn_dist
	     */
        T *sigmas;
        T *rhos;
        MLCommon::allocate(sigmas, n);
        MLCommon::allocate(rhos, n);

        FuzzySimplSetImpl::smooth_knn_dist<TPB_X, T>(n, knn_indices, knn_dists,
                rhos, sigmas, params
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        std::cout << MLCommon::arr2Str(rhos, n, "rhos") << std::endl;
        std::cout << MLCommon::arr2Str(sigmas, n, "sigmas") << std::endl;

        /**
         * Compute graph of membership strengths
         */
        int *graph_rows, *graph_cols;
        T *graph_vals;

        int nnz = n*params->n_neighbors;

        /**
         * Allocate workspace for fuzzy simplicial set.
         */
        MLCommon::allocate(graph_rows, nnz);
        MLCommon::allocate(graph_cols, nnz);
        MLCommon::allocate(graph_vals, nnz);

        std::cout << "Running fuzzySimplSetImpl." << std::endl;

        FuzzySimplSetImpl::compute_membership_strength_kernel<TPB_X><<<grid, blk>>>(
                knn_indices, knn_dists,
                sigmas, rhos,
                graph_vals, graph_rows, graph_cols, n,
                params->n_neighbors);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::cout << MLCommon::arr2Str(graph_vals, nnz, "graph_cals") << std::endl;

        std::cout << "FuzzySimplSetImpl done." << std::endl;

        int *ia, *ex_scan;
        MLCommon::allocate(ia, n, true);
        MLCommon::allocate(ex_scan, n, true);

        std::cout << MLCommon::arr2Str(graph_vals, nnz, "graph_cals") << std::endl;

        std::cout << "Runing coo_row_counts" << std::endl;

        // COO should be sorted by row at this point- we get the counts and then normalize
        coo_row_counts<TPB_X, T><<<grid, blk>>>(graph_rows, graph_vals, nnz, ia, n);

        std::cout << "Done." << std::endl;

        std::cout << "Runing ex_scan" << std::endl;

        thrust::device_ptr<int> dev_ia = thrust::device_pointer_cast(ia);
        thrust::device_ptr<int> dev_ex_scan = thrust::device_pointer_cast(ex_scan);
        exclusive_scan(dev_ia, dev_ia + n, dev_ex_scan);

        std::cout << MLCommon::arr2Str(graph_vals, nnz, "graph_cals") << std::endl;


        std::cout << "Done." << std::endl;

        std::cout << "Runing csr_row_normalize_l1" << std::endl;

         MLCommon::csr_row_normalize_l1<TPB_X, T><<<grid, blk>>>(dev_ex_scan.get(), graph_vals, nnz,
                 n, params->n_neighbors, graph_vals);

         std::cout << "Done." << std::endl;
         std::cout << MLCommon::arr2Str(graph_vals, nnz, "graph_cals") << std::endl;


        /**
         * Init_transform()
         *
         */
        // cols.shape = (X.shape[0], self._n_neighbors)
        // vas.shape = (X.shape[0], self._n_neighbors)

        std::cout << "Runing init_transform" << std::endl;

        T *embeddings_d;
        MLCommon::allocate(embeddings_d, n*params->n_components);

        T *result;
        MLCommon::allocate(result, n*params->n_components);

        init_transform<TPB_X, T><<<grid,blk>>>(graph_cols, graph_vals, n,
                embeddings_d, embedding_n, params->n_components,
                result, params->n_neighbors);
        CUDA_CHECK(cudaPeekAtLastError());

        std::cout << "Done." << std::endl;

        std::cout << MLCommon::arr2Str(graph_vals, nnz, "graph_cals") << std::endl;



        std::cout << "Finding max" << std::endl;

        std::cout << MLCommon::arr2Str(graph_vals, nnz, "graph_cals") << std::endl;

        /**
         * Find max of data
         */
        thrust::device_ptr<T> d_ptr = thrust::device_pointer_cast(graph_vals);

        std::cout << "Wrapped" << std::endl;

        T max = *(thrust::max_element(d_ptr, d_ptr+nnz));

        std::cout << "Done." << std::endl;


        std::cout << "Runing unary_op" << std::endl;

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

        MLCommon::LinAlg::unaryOp<T>(graph_vals, graph_vals, (max / params->n_epochs), nnz, adjust_vals_op);
        CUDA_CHECK(cudaPeekAtLastError());

        std::cout << "Done." << std::endl;

        std::cout << "Runing count" << std::endl;

        thrust::device_ptr<T> dev_gvals = thrust::device_pointer_cast(graph_vals);
        int non_zero_vals = nnz-thrust::count(dev_gvals, dev_gvals+nnz, 0.0);
        CUDA_CHECK(cudaPeekAtLastError());

        std::cout << "Done." << std::endl;

        std::cout << "Runing coo_remove_zeros" << std::endl;

        /**
         * Remove zeros
         */
        int *crows, *ccols;
        T *cvals;
        MLCommon::allocate(crows, non_zero_vals, true);
        MLCommon::allocate(ccols, non_zero_vals, true);
        MLCommon::allocate(cvals, non_zero_vals, true);

        MLCommon::coo_remove_zeros<TPB_X, T>(nnz,
                graph_rows, graph_cols, graph_vals,
                crows, ccols, cvals,
                ia, n);
        CUDA_CHECK(cudaPeekAtLastError());

        std::cout << "Done." << std::endl;

        T *epochs_per_sample = (T*)malloc(nnz*sizeof(T));
        T *cvals_h = (T*)malloc(non_zero_vals*sizeof(T));
        MLCommon::updateHost(cvals_h, cvals, non_zero_vals);

        std::cout << "Runing make_epochs_per_sample" << std::endl;

        SimplSetEmbedImpl::make_epochs_per_sample(cvals_h, non_zero_vals, params->n_epochs, epochs_per_sample);
        CUDA_CHECK(cudaPeekAtLastError());

        std::cout << "Done." << std::endl;

        std::cout << "Runing optimize_layout" << std::endl;

        int *head = (int*)malloc(non_zero_vals*sizeof(int));
        int *tail = (int*)malloc(non_zero_vals*sizeof(int));

        MLCommon::updateHost(head, crows, non_zero_vals);
        MLCommon::updateHost(tail, ccols, non_zero_vals);
        MLCommon::updateHost(embedding, embeddings_d, n*params->n_components);

        T *result_h = (T*)malloc(n*params->n_components*sizeof(T));
        MLCommon::updateHost(result_h, result, n*params->n_components);

        SimplSetEmbedImpl::print_arr(result_h, n*params->n_components, "embedding");

        SimplSetEmbedImpl::optimize_layout(
            result_h, embedding_n,
            embedding, n,
            head, tail, non_zero_vals,
            epochs_per_sample,
            n,
            params
        );
        CUDA_CHECK(cudaPeekAtLastError());

        SimplSetEmbedImpl::print_arr(embedding, n*params->n_components, "embedding");

        std::cout << "Done." << std::endl;


        CUDA_CHECK(cudaFree(knn_dists));
        CUDA_CHECK(cudaFree(knn_indices));


        CUDA_CHECK(cudaFree(sigmas));
        CUDA_CHECK(cudaFree(rhos));

        CUDA_CHECK(cudaFree(graph_rows));
        CUDA_CHECK(cudaFree(graph_cols));
        CUDA_CHECK(cudaFree(graph_vals));


        CUDA_CHECK(cudaFree(ia));
        CUDA_CHECK(cudaFree(ex_scan));

        CUDA_CHECK(cudaFree(result));


        free(epochs_per_sample);
        free(cvals_h);

        CUDA_CHECK(cudaFree(crows));
        CUDA_CHECK(cudaFree(ccols));
        CUDA_CHECK(cudaFree(cvals));

        return 0;

	}

}
