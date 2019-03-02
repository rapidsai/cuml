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
                    result[row*n_components+d] += weights[i+j] * embeddings[indices[i+j]*n_components+d];
                }
            }
        }
	}

    template<int TPB_X, typename T>
    __global__ void exact_coo_row_counts(int *rows, T *vals, int nnz,
            int *results, int n) {
        int row = (blockIdx.x * TPB_X) + threadIdx.x;
//        if(row < nnz && vals[row] > 0.0)
        if(row < nnz)
            atomicAdd(results+rows[row], 1);
    }

    template<int TPB_X, typename T>
    __global__ void nonzero_coo_row_counts(int *rows, T *vals, int nnz,
            int *results, int n) {
        int row = (blockIdx.x * TPB_X) + threadIdx.x;
        if(row < nnz && vals[row] > 0.0) {
            printf("increasing row=%d for val=%f\n", rows[row], vals[row]);
            atomicAdd(results+rows[row], 1);
        }
    }

    template<int TPB_X>
    __global__ void reset_vals(int *vals, int nnz) {
        int row = (blockIdx.x * TPB_X) + threadIdx.x;
        if(row < nnz)
            vals[row] = 0.0;
    }

    void find_ab(UMAPParams *params) {
        Optimize::find_params_ab(params);
    }


    /**
     *
     */
	template<typename T>
	size_t _fit(T *X,       // input matrix
	            int n,      // rows
	            int d,      // cols
	            kNN *knn,
	            UMAPParams *params,
	            T *embeddings) {


	    // TODO: Allocate workspace up front


		/**
		 * Allocate workspace for kNN graph
		 */
		long *knn_indices;
		T *knn_dists;

        MLCommon::allocate(knn_indices, n*params->n_neighbors);
		MLCommon::allocate(knn_dists, n*params->n_neighbors);

        kNNGraph::run(X, n,d, knn_indices, knn_dists, knn, params);
		CUDA_CHECK(cudaPeekAtLastError());

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
		int nnz = 0;

		FuzzySimplSet::run(n, knn_indices, knn_dists,
						   graph_rows,
						   graph_cols,
						   graph_vals,
						   params, &nnz,0);
		CUDA_CHECK(cudaPeekAtLastError());

		std::cout << "nnz=" << nnz << std::endl;

		InitEmbed::run(X, n, d,
		        knn_indices, knn_dists,
		        graph_rows, graph_cols, graph_vals,
		        nnz,
		        params, embeddings, 1);

		std::cout << "Running simplsetembed" << std::endl;

		/**
		 * Run simplicial set embedding to approximate low-dimensional representation
		 */
		SimplSetEmbed::run<T, 256>(
		        X, n, d,
		        graph_rows, graph_cols, graph_vals, nnz,
		        params, embeddings);

        std::cout << "Running simplsetembed" << std::endl;

        CUDA_CHECK(cudaPeekAtLastError());

		CUDA_CHECK(cudaFree(knn_dists));
        CUDA_CHECK(cudaFree(knn_indices));
        CUDA_CHECK(cudaFree(graph_rows));
        CUDA_CHECK(cudaFree(graph_cols));
        CUDA_CHECK(cudaFree(graph_vals));

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
	                  T *transformed) {


	    // TODO: Allocate workspace up front.

        dim3 grid(MLCommon::ceildiv(n, TPB_X), 1, 1);
        dim3 blk(TPB_X, 1, 1);


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
            [] __device__(T input) { return sqrt(input); }
        );

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

        FuzzySimplSetImpl::compute_membership_strength_kernel<TPB_X><<<grid, blk>>>(
                knn_indices, knn_dists,
                sigmas, rhos,
                graph_vals, graph_rows, graph_cols, n,
                params->n_neighbors);
        CUDA_CHECK(cudaPeekAtLastError());

        int *ia, *ex_scan;
        MLCommon::allocate(ia, n, true);
        MLCommon::allocate(ex_scan, n, true);

        // COO should be sorted by row at this point- we get the counts and then normalize
        exact_coo_row_counts<TPB_X, T><<<grid, blk>>>(graph_rows, graph_vals, nnz, ia, n);

        std::cout << MLCommon::arr2Str(ia, n, "ia") << std::endl;

        thrust::device_ptr<int> dev_ia = thrust::device_pointer_cast(ia);
        thrust::device_ptr<int> dev_ex_scan = thrust::device_pointer_cast(ex_scan);
        exclusive_scan(dev_ia, dev_ia + n, dev_ex_scan);

        std::cout << MLCommon::arr2Str(ex_scan, n, "ex_scan") << std::endl;

        std::cout << MLCommon::arr2Str(graph_rows, nnz, "graph_rows") << std::endl;
        std::cout << MLCommon::arr2Str(graph_cols, nnz, "graph_cols") << std::endl;
        std::cout << MLCommon::arr2Str(graph_vals, nnz, "graph_cvals") << std::endl;

         MLCommon::csr_row_normalize_l1<TPB_X, T><<<grid, blk>>>(dev_ex_scan.get(), graph_vals, nnz,
                 n, params->n_neighbors, graph_vals);

         reset_vals<TPB_X><<<grid,blk>>>(ia, n);
         nonzero_coo_row_counts<TPB_X, T><<<grid,blk>>>(graph_rows, graph_vals, nnz, ia, n);

         std::cout << MLCommon::arr2Str(ia, n, "ia") << std::endl;

        init_transform<TPB_X, T><<<grid,blk>>>(graph_cols, graph_vals, n,
                embedding, embedding_n, params->n_components,
                transformed, params->n_neighbors);
        CUDA_CHECK(cudaPeekAtLastError());

        std::cout << MLCommon::arr2Str(transformed, n*params->n_components, "transformed") << std::endl;

        /**
         * Find max of data
         */
        thrust::device_ptr<T> d_ptr = thrust::device_pointer_cast(graph_vals);
        T max = *(thrust::max_element(d_ptr, d_ptr+nnz));

        /**
         * Go through COO values and set everything that's less than
         * vals.max() / params->n_epochs to 0.0
         */

        int n_epochs = params->n_epochs;
        MLCommon::LinAlg::unaryOp<T>(graph_vals, graph_vals, nnz,
            [=] __device__(T input) {
                if (input < (max / n_epochs))
                    return 0.0f;
                else
                    return input;
            }
        );

        CUDA_CHECK(cudaPeekAtLastError());

        std::cout << MLCommon::arr2Str(graph_vals, nnz, "graph_cvals") << std::endl;

        thrust::device_ptr<T> dev_gvals = thrust::device_pointer_cast(graph_vals);
        int non_zero_vals = nnz-thrust::count(dev_gvals, dev_gvals+nnz, 0.0);
        CUDA_CHECK(cudaPeekAtLastError());

        std::cout << "non_zero_vals=" << non_zero_vals << std::endl;

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

        T *epochs_per_sample;
        MLCommon::allocate(epochs_per_sample, nnz);

        std::cout << MLCommon::arr2Str(crows, non_zero_vals, "crows") << std::endl;
        std::cout << MLCommon::arr2Str(ccols, non_zero_vals, "ccols") << std::endl;
        std::cout << MLCommon::arr2Str(cvals, non_zero_vals, "cvals") << std::endl;

        SimplSetEmbedImpl::make_epochs_per_sample(cvals, non_zero_vals, params->n_epochs, epochs_per_sample);
        CUDA_CHECK(cudaPeekAtLastError());

        std::cout << MLCommon::arr2Str(epochs_per_sample, non_zero_vals, "epochs_per_sample") << std::endl;


        SimplSetEmbedImpl::optimize_layout<T, TPB_X>(
            transformed, embedding_n,
            embedding, n,
            crows, ccols, non_zero_vals,
            epochs_per_sample,
            n,
            params->repulsion_strength,
            params
        );
        CUDA_CHECK(cudaPeekAtLastError());

        std::cout << MLCommon::arr2Str(transformed, n*params->n_components, "embeddings") << std::endl;

        CUDA_CHECK(cudaFree(knn_dists));
        CUDA_CHECK(cudaFree(knn_indices));


        CUDA_CHECK(cudaFree(sigmas));
        CUDA_CHECK(cudaFree(rhos));

        CUDA_CHECK(cudaFree(graph_rows));
        CUDA_CHECK(cudaFree(graph_cols));
        CUDA_CHECK(cudaFree(graph_vals));


        CUDA_CHECK(cudaFree(ia));
        CUDA_CHECK(cudaFree(ex_scan));

        CUDA_CHECK(cudaFree(epochs_per_sample));
        CUDA_CHECK(cudaFree(crows));
        CUDA_CHECK(cudaFree(ccols));

        return 0;

	}

}
