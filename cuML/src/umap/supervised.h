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

    namespace Supervised {

        using namespace ML;

        using namespace MLCommon::Sparse;


        template<int TPB_X, typename T>
        __global__ void fast_intersection_kernel(
            int * rows, int *cols, T *vals, int nnz,
            T *target,
            float unknown_dist = 1.0,
            float far_dist = 5.0
        ) {
            int row = (blockIdx.x * TPB_X) + threadIdx.x;
            if(row < nnz) {
                int i = rows[row];
                int j = cols[row];
                if(target[i] == -1.0 || target[j] == -1.0)
                    vals[row] *= exp(-unknown_dist);
                else
                    vals[row] *= exp(-far_dist);
            }
        }

        /**
         * In order to reset membership strengths, we need to perform a
         * P+P.T, which can be done with the COO matrices by creating
         * the transposed elements only if they are not already in P,
         * and adding them.
         */
        template< typename T, int TPB_X>
        __global__ void reset_membership_strengths_kernel(
                int *row_ind,
                int *rows, int *cols, T *vals,
                int *orows, int *ocols, T *ovals, int *rnnz,
                int n, int cnnz) {

            int row = (blockIdx.x * TPB_X) + threadIdx.x;

            if (row < n) {

                int start_idx = row_ind[row]; // each thread processes one row
                int stop_idx = MLCommon::Sparse::get_stop_idx(row, n, cnnz, row_ind);

                int nnz = 0;
                for (int idx = 0; idx < stop_idx-start_idx; idx++) {

                    int out_idx = start_idx*2+nnz;
                    int row_lookup = cols[idx+start_idx];
                    int t_start = row_ind[row_lookup]; // Start at
                    int t_stop = MLCommon::Sparse::get_stop_idx(row_lookup, n, cnnz, row_ind);

                    T transpose = 0.0;
                    bool found_match = false;
                    for (int t_idx = t_start; t_idx < t_stop; t_idx++) {

                        // If we find a match, let's get out of the loop
                        if (cols[t_idx] == rows[idx+start_idx]
                                && rows[t_idx] == cols[idx+start_idx]
                                && vals[t_idx] != 0.0) {
                            transpose = vals[t_idx];
                            found_match = true;
                            break;
                        }
                    }


                    // if we didn't find an exact match, we need to add
                    // the transposed value into our current matrix.
                    if (!found_match && vals[idx] != 0.0) {
                        orows[out_idx + nnz] = cols[idx+start_idx];
                        ocols[out_idx + nnz] = rows[idx+start_idx];
                        ovals[out_idx + nnz] = vals[idx+start_idx];
                        ++nnz;
                    }

                    printf("row=%d, start_idx=%d, out_idx=%d, nnz=%d\n", row, start_idx, out_idx, nnz);

                    T result = vals[idx+start_idx];
                    T prod_matrix = result * transpose;

                    // @todo: This line is the only difference between
                    // this function and compute_result from the fuzzy
                    // simplicial set. Should combine these to do
                    // transposed eltwise ops.
                    T res = result + transpose - prod_matrix;

                    if (res != 0.0) {
                        orows[out_idx + nnz] = rows[idx+start_idx];
                        ocols[out_idx + nnz] = cols[idx+start_idx];
                        ovals[out_idx + nnz] = T(res);
                        ++nnz;
                    }

                    printf("row=%d, start_idx=%d, out_idx=%d, nnz=%d\n", row, start_idx, out_idx, nnz);
                }


                rnnz[row] = nnz;
                atomicAdd(rnnz + n, nnz); // fused operation to count number of nonzeros
            }
        }

        template<typename T, int TPB_X>
        void reset_local_connectivity(
            int *row_ind,
            COO<T> *in_coo,
            COO<T> *out_coo,
            int *rnnz // size = nnz*2
        ) {

            dim3 grid_n(MLCommon::ceildiv(in_coo->n_rows, TPB_X), 1, 1);
            dim3 blk_n(TPB_X, 1, 1);

            // Perform l_inf normalization
            MLCommon::Sparse::csr_row_normalize_max<TPB_X, T>(
                    row_ind,
                    in_coo->vals,
                    in_coo->nnz,
                    in_coo->n_rows,
                    in_coo->vals
            );

            CUDA_CHECK(cudaPeekAtLastError());

            std::cout << MLCommon::arr2Str(in_coo->rows, in_coo->nnz, "graph_rows") << std::endl;
            std::cout << MLCommon::arr2Str(in_coo->cols, in_coo->nnz, "graph_cols") << std::endl;
            std::cout << MLCommon::arr2Str(in_coo->vals, in_coo->nnz, "graph_vals") << std::endl;

            out_coo->allocate(in_coo->nnz*2, in_coo->n_rows, in_coo->n_cols);

            // reset membership strengths
            reset_membership_strengths_kernel<T, TPB_X><<<grid_n, blk_n>>>(
                row_ind,
                in_coo->rows, in_coo->cols, in_coo->vals,
                out_coo->rows, out_coo->cols, out_coo->vals, rnnz,
                in_coo->n_rows, in_coo->nnz
            );

            std::cout << MLCommon::arr2Str(out_coo->rows, out_coo->nnz, "orows") << std::endl;
            std::cout << MLCommon::arr2Str(out_coo->cols, out_coo->nnz, "ocols") << std::endl;
            std::cout << MLCommon::arr2Str(out_coo->vals, out_coo->nnz, "ovals") << std::endl;

            CUDA_CHECK(cudaPeekAtLastError());
        }

        template<typename T, int TPB_X>
        void categorical_simplicial_set_intersection(
             COO<T> *graph_coo, T *target,
             float far_dist = 5.0,
             float unknown_dist = 1.0) {

            dim3 grid(MLCommon::ceildiv(graph_coo->nnz, TPB_X), 1, 1);
            dim3 blk(TPB_X, 1, 1);
            fast_intersection_kernel<TPB_X, T><<<grid,blk>>>(
                    graph_coo->rows,
                    graph_coo->cols,
                    graph_coo->vals,
                    graph_coo->nnz,
                    target,
                    unknown_dist,
                    far_dist
            );
        }


        template<typename T, int TPB_X>
        __global__ void sset_intersection_kernel(
            int *row_ind1, int *cols1, T *vals1, int nnz1,
            int *row_ind2, int *cols2, T *vals2, int nnz2,
            int *result_ind, int *result_cols, T *result_vals, int nnz,
            T left_min, T right_min,
            int m, float mix_weight = 0.5
        ) {

            int row = (blockIdx.x * TPB_X) + threadIdx.x;

            if(row < m) {
                int start_idx_res = result_ind[row];
                int stop_idx_res = MLCommon::Sparse::get_stop_idx(row, m, nnz, result_ind);

                int start_idx1 = row_ind1[row];
                int stop_idx1 = MLCommon::Sparse::get_stop_idx(row, m, nnz1, row_ind1);

                int start_idx2 = row_ind2[row];
                int stop_idx2 = MLCommon::Sparse::get_stop_idx(row, m, nnz2, row_ind2);

                for(int j = start_idx_res; j < stop_idx_res; j++) {
                    int col = result_cols[j];

                    T left_val = left_min;
                    for(int k = start_idx1; k < stop_idx2; k++) {
                        if(cols1[k] == col) {
                            left_val = vals1[k];
                        }
                    }

                    T right_val = right_min;
                    for(int k = start_idx2; k < stop_idx2; k++) {
                        if(cols2[k] == col) {
                            right_val = vals2[k];
                        }
                    }

                    if(left_val > left_min || right_val > right_min) {
                        if(mix_weight < 0.5) {
                            result_vals[row] = left_val *
                                    powf(right_val, mix_weight / (1.0 - mix_weight));
                        } else {
                            result_vals[row] = powf(left_val, (1.0 - mix_weight) / mix_weight)
                                    * right_val;
                        }
                    }
                }
            }
        }


        /**
         * Computes the CSR column index pointer and values
         * for the general simplicial set intersecftion.
         */
        template<typename T, int TPB_X>
        void general_simplicial_set_intersection(
            int *row1_ind, COO<T> *in1,
            int *row2_ind, COO<T> *in2,
            COO<T> *result,
            float weight
        ) {

            int *result_ind;
            MLCommon::allocate(result_ind, in1->n_rows, true);


            int result_nnz = MLCommon::Sparse::csr_add_calc_inds<float, 32>(
                row1_ind, in1->cols, in1->vals, in1->nnz,
                row2_ind, in2->cols, in2->vals, in2->nnz,
                in1->n_rows, result_ind
            );

            result->allocate(result_nnz, in1->n_rows);

            MLCommon::Sparse::csr_add_finalize<float, 32>(
                row1_ind, in1->cols, in1->vals, in1->nnz,
                row2_ind, in2->cols, in2->vals, in2->nnz,
                in1->n_rows, result_ind, result->cols, result->vals
            );

            thrust::device_ptr<const T> d_ptr1 = thrust::device_pointer_cast(in1->vals);
            T min1 = *(thrust::min_element(d_ptr1, d_ptr1+in1->nnz));

            thrust::device_ptr<const T> d_ptr2 = thrust::device_pointer_cast(in2->vals);
            T min2 = *(thrust::min_element(d_ptr2, d_ptr2+in2->nnz));

            T left_min = min(min1 / 2.0, 1e-8);
            T right_min = min(min2 / 2.0, 1e-8);

            dim3 grid(MLCommon::ceildiv(in1->nnz, TPB_X), 1, 1);
            dim3 blk(TPB_X, 1, 1);

            //@todo: Write a wrapper function for this
            sset_intersection_kernel<T, TPB_X><<<grid, blk>>>(
                row1_ind, in1->cols, in1->vals, in1->nnz,
                row2_ind, in2->cols, in2->vals, in2->nnz,
                result_ind, result->cols, result->vals, result->nnz,
                left_min, right_min,
                in1->n_rows, weight
            );

            dim3 grid_n(MLCommon::ceildiv(result->nnz, TPB_X), 1, 1);

            //@todo: Write a wrapper function for this
            MLCommon::Sparse::csr_to_coo<TPB_X><<<grid_n, blk>>>(
                    result_ind, result->n_rows, result->rows, result->nnz);

        }


        template<int TPB_X, typename T>
        void perform_categorical_intersection(
                T *y,
                COO<T> *rgraph_coo, COO<T> *final_coo,
                UMAPParams *params,
                cudaStream_t stream) {

            float far_dist = 1.0e12;  // target weight
            if(params->target_weights < 1.0)
                far_dist = 2.5 * (1.0 / (1.0 - params->target_weights));

            categorical_simplicial_set_intersection<T, TPB_X>(
                    rgraph_coo, y, far_dist);



            if(params->verbose) {
                std::cout << MLCommon::arr2Str(rgraph_coo->rows, rgraph_coo->nnz, "rgraph_rows") << std::endl;
                std::cout << MLCommon::arr2Str(rgraph_coo->cols, rgraph_coo->nnz, "rgraph_cols") << std::endl;
                std::cout << MLCommon::arr2Str(rgraph_coo->vals, rgraph_coo->nnz, "rgraph_vals") << std::endl;
            }

            // reset local connectivity
            int *final_nnz;
            MLCommon::allocate(final_nnz, rgraph_coo->n_rows+1, true);

            COO<T> comp_coo;
            coo_remove_zeros<TPB_X, T>(rgraph_coo, &comp_coo, stream);

            int *result_ind;
            MLCommon::allocate(result_ind, rgraph_coo->n_rows, true);

            MLCommon::Sparse::sorted_coo_to_csr(&comp_coo, result_ind);

            std::cout << MLCommon::arr2Str(result_ind, comp_coo.n_rows, "result_ind") << std::endl;

            reset_local_connectivity<T, TPB_X>(
                result_ind,
                &comp_coo,
                final_coo,
                final_nnz
            );

            CUDA_CHECK(cudaPeekAtLastError());

            CUDA_CHECK(cudaFree(result_ind));
            CUDA_CHECK(cudaFree(final_nnz));
        }


        template<int TPB_X, typename T>
        void perform_general_intersection(
                T *y,
                COO<T> *rgraph_coo, COO<T> *final_coo,
                UMAPParams *params,
                cudaStream_t stream) {

            /**
             * Calculate kNN for Y
             */
            if(params->verbose)
                std::cout << "Runnning knn_Graph on Y" << std::endl;

            kNN y_knn(1);
            long *y_knn_indices;
            T *y_knn_dists;

            int knn_dims = rgraph_coo->n_rows*params->target_n_neighbors;

            MLCommon::allocate(y_knn_indices, knn_dims, true);
            MLCommon::allocate(y_knn_dists, knn_dims, true);

            kNNGraph::run(y, rgraph_coo->n_rows, 1, y_knn_indices, y_knn_dists, &y_knn, params, stream);
            CUDA_CHECK(cudaPeekAtLastError());

            /**
             * Compute fuzzy simplicial set
             */
            COO<T> ygraph_coo(knn_dims*2, rgraph_coo->n_rows, rgraph_coo->n_rows);

            FuzzySimplSet::run<TPB_X, T>(rgraph_coo->n_rows,
                               y_knn_indices, y_knn_dists,
                               params->target_n_neighbors,
                               &ygraph_coo,
                               params, 0);
            CUDA_CHECK(cudaPeekAtLastError());


            CUDA_CHECK(cudaFree(y_knn_indices));
            CUDA_CHECK(cudaFree(y_knn_dists));

            /**
             * Compute general simplicial set intersection.
             */
            int *xrow_ind, *yrow_ind;
            MLCommon::allocate(xrow_ind, rgraph_coo->n_rows, true);
            MLCommon::allocate(yrow_ind, rgraph_coo->n_rows, true);

            MLCommon::Sparse::sorted_coo_to_csr(&ygraph_coo, yrow_ind);
            MLCommon::Sparse::sorted_coo_to_csr(rgraph_coo, xrow_ind);

            if(params->verbose)
                std::cout << "Running general simpl set intersection" << std::endl;

            COO<T> result_coo;
            general_simplicial_set_intersection<T, TPB_X>(
                xrow_ind, rgraph_coo,
                yrow_ind, &ygraph_coo,
                &result_coo,
                params->target_weights
            );

            CUDA_CHECK(cudaFree(xrow_ind));
            CUDA_CHECK(cudaFree(yrow_ind));

            /**
             * Remove zeros
             */

            COO<T> out;

            coo_remove_zeros<TPB_X, T>(&result_coo, &out, stream);

            result_coo.free();

            int *out_row_ind;
            MLCommon::allocate(out_row_ind, out.n_rows);

            MLCommon::Sparse::sorted_coo_to_csr(&out, out_row_ind);

            if(params->verbose) {
                std::cout << "Reset Local connectivity" << std::endl;
                std::cout << "result_nnz=" << out.nnz << std::endl;
                std::cout << MLCommon::arr2Str(out.rows, out.nnz, "final_rows") << std::endl;
                std::cout << MLCommon::arr2Str(out.cols, out.nnz, "final_cols") << std::endl;
                std::cout << MLCommon::arr2Str(out.vals, out.nnz, "final_vals") << std::endl;
            }

            int *final_nnz;
            MLCommon::allocate(final_nnz, rgraph_coo->n_rows+1, true);

            reset_local_connectivity<T, TPB_X>(
                out_row_ind,
                &out,
                final_coo,
                final_nnz
            );
            CUDA_CHECK(cudaPeekAtLastError());

            CUDA_CHECK(cudaFree(out_row_ind));
        }
    }
}
