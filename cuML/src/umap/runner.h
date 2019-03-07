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

    template<int TPB_X, typename T>
    __global__ void fast_intersection(
        int * rows, int *cols, T *vals, int nnz,
        T *target,
        float unknown_dist = 1.0,
        float far_dist = 5.0
    ) {
        int row = (blockIdx.x * TPB_X) + threadIdx.x;
        if(row < nnz) {
            int i = rows[row];
            int j = cols[row];
            if(target[i] == -1 || target[j] == -1)
                vals[row] *= exp(-unknown_dist);
            else
                vals[row] *= exp(-far_dist);
        }
    }


    /**
     * Firt exponential decay curve to find the parameters
     * a and b, which are based on min_dist and spread
     * parameters.
     */
    void find_ab(UMAPParams *params) {
        Optimize::find_params_ab(params);
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
	            T *embeddings) {

	    find_ab(params);

		/**
		 * Allocate workspace for kNN graph
		 */
		long *knn_indices;
		T *knn_dists;

        MLCommon::allocate(knn_indices, n*params->n_neighbors);
		MLCommon::allocate(knn_dists, n*params->n_neighbors);

        kNNGraph::run(X, n,d, knn_indices, knn_dists, knn, params);
		CUDA_CHECK(cudaPeekAtLastError());

        int *rgraph_rows, *rgraph_cols;
        T *rgraph_vals;

		/**
		 * Allocate workspace for fuzzy simplicial set.
		 */
        MLCommon::allocate(rgraph_rows, n*params->n_neighbors*2);
        MLCommon::allocate(rgraph_cols, n*params->n_neighbors*2);
        MLCommon::allocate(rgraph_vals, n*params->n_neighbors*2);


		/**
		 * Run Fuzzy simplicial set
		 */
		int nnz = 0;

		FuzzySimplSet::run<TPB_X, T>(n, knn_indices, knn_dists,
                           rgraph_rows,
                           rgraph_cols,
                           rgraph_vals,
						   params, &nnz,0);
		CUDA_CHECK(cudaPeekAtLastError());

		InitEmbed::run(X, n, d,
		        knn_indices, knn_dists,
		        rgraph_rows, rgraph_cols, rgraph_vals,
		        nnz,
		        params, embeddings, params->init);

		/**
		 * Run simplicial set embedding to approximate low-dimensional representation
		 */

		SimplSetEmbed::run<TPB_X, T>(
		        X, n, d,
		        rgraph_rows, rgraph_cols, rgraph_vals, nnz,
		        params, embeddings);

        CUDA_CHECK(cudaPeekAtLastError());

		CUDA_CHECK(cudaFree(knn_dists));
        CUDA_CHECK(cudaFree(knn_indices));
        CUDA_CHECK(cudaFree(rgraph_rows));
        CUDA_CHECK(cudaFree(rgraph_cols));
        CUDA_CHECK(cudaFree(rgraph_vals));

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
                rhos, sigmas, params, adjusted_local_connectivity
        );

        /**
         * Compute graph of membership strengths
         */

        int nnz = n*params->n_neighbors;

        dim3 grid_nnz(MLCommon::ceildiv(nnz, TPB_X), 1, 1);

        /**
         * Allocate workspace for fuzzy simplicial set.
         */
        int *graph_rows, *graph_cols;
        T *graph_vals;
        MLCommon::allocate(graph_rows, nnz);
        MLCommon::allocate(graph_cols, nnz);
        MLCommon::allocate(graph_vals, nnz);

        if(params->verbose)
            std::cout << "n_neighbors=" << params->n_neighbors << std::endl;

        FuzzySimplSetImpl::compute_membership_strength_kernel<TPB_X><<<grid_n, blk>>>(
                knn_indices, knn_dists,
                sigmas, rhos,
                graph_vals, graph_rows, graph_cols, n,
                params->n_neighbors);
        CUDA_CHECK(cudaPeekAtLastError());

        if(params->verbose) {
            std::cout << MLCommon::arr2Str(sigmas, n, "sigmas") << std::endl;
            std::cout << MLCommon::arr2Str(rhos, n, "rhos") << std::endl;
        }

        int *ia, *ex_scan;
        MLCommon::allocate(ia, n, true);
        MLCommon::allocate(ex_scan, n, true);

        // COO should be sorted by row at this point- we get the counts and then normalize
        MLCommon::coo_row_count<TPB_X, T><<<grid_nnz, blk>>>(graph_rows, nnz, ia, n);

        thrust::device_ptr<int> dev_ia = thrust::device_pointer_cast(ia);
        thrust::device_ptr<int> dev_ex_scan = thrust::device_pointer_cast(ex_scan);
        exclusive_scan(dev_ia, dev_ia + n, dev_ex_scan);

        T *vals_normed;
        MLCommon::allocate(vals_normed, nnz, true);

         MLCommon::csr_row_normalize_l1<TPB_X, T><<<grid_n, blk>>>(dev_ex_scan.get(), graph_vals, nnz,
                 n, vals_normed);

        init_transform<TPB_X, T><<<grid_n,blk>>>(graph_cols, vals_normed, n,
                embedding, embedding_n, params->n_components,
                transformed, params->n_neighbors);

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaFree(vals_normed));

        reset_vals<TPB_X><<<grid_n,blk>>>(ia, n);

        MLCommon::coo_row_count_nz<TPB_X, T><<<grid_nnz,blk>>>(graph_rows, graph_vals, nnz, ia, n);

        /**
         * Go through COO values and set everything that's less than
         * vals.max() / params->n_epochs to 0.0
         */
        thrust::device_ptr<T> d_ptr = thrust::device_pointer_cast(graph_vals);
        T max = *(thrust::max_element(d_ptr, d_ptr+nnz));

        int n_epochs = 1000;//params->n_epochs;

        MLCommon::LinAlg::unaryOp<T>(graph_vals, graph_vals, nnz,
            [=] __device__(T input) {
                if (input < (max / float(n_epochs)))
                    return 0.0f;
                else
                    return input;
            }
        );

        CUDA_CHECK(cudaPeekAtLastError());

        if(params->verbose)
            std::cout << "nnz=" << nnz << std::endl;

        thrust::device_ptr<T> dev_gvals = thrust::device_pointer_cast(graph_vals);
        int non_zero_vals = nnz-thrust::count(dev_gvals, dev_gvals+nnz, T(0));
        CUDA_CHECK(cudaPeekAtLastError());


        if(params->verbose)
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

        SimplSetEmbedImpl::make_epochs_per_sample(cvals, non_zero_vals, params->n_epochs, epochs_per_sample);
        CUDA_CHECK(cudaPeekAtLastError());

        SimplSetEmbedImpl::optimize_layout<TPB_X, T>(
            transformed, n,
            embedding, embedding_n,
            crows, ccols, non_zero_vals,
            epochs_per_sample,
            n,
            params->repulsion_strength,
            params,
            n_epochs
        );
        CUDA_CHECK(cudaPeekAtLastError());

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
        CUDA_CHECK(cudaFree(cvals));

        return 0;

	}

}
