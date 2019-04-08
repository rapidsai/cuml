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

#include "cuda_utils.h"


#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#pragma once

namespace MLCommon {
    namespace Sparse {
        template<int TPB_X, typename T>
        __global__ void csr_row_normalize_l1(
                int *ia,    // csr row ex_scan (sorted by row)
                T *vals, int nnz,  // array of values and number of non-zeros
                int m,          // num rows in csr
                T *result) {    // output array

            // row-based matrix 1 thread per row
            int row = (blockIdx.x * TPB_X) + threadIdx.x;

            // sum all vals for row and divide each val by sum
            if(row < m) {
                int start_idx = ia[row];
                int stop_idx = 0;
                if(row < m-1) {
                    stop_idx = ia[row+1];
                } else
                    stop_idx = nnz;

                T sum = T(0.0);
                for(int j = start_idx; j < stop_idx; j++) {
                    sum = sum + vals[j];
                }

                for(int j = start_idx; j < stop_idx; j++) {
                    if(sum > 0.0) {
                        T val = vals[j];
                        result[j] = val / sum;
                    }
                    else {
                        result[j] = 0.0;
                    }
                }
            }
        }

        template<int TPB_X, typename T>
        __global__ void csr_row_normalize_max(
                int *ia,    // csr row ex_scan (sorted by row)
                T *vals, int nnz,  // array of values and number of non-zeros
                int m,          // num rows in csr
                T *result) {    // output array

            // row-based matrix 1 thread per row
            int row = (blockIdx.x * TPB_X) + threadIdx.x;

            // sum all vals for row and divide each val by sum
            if(row < m) {
                int start_idx = ia[row];
                int stop_idx = 0;
                if(row < m-1) {
                    stop_idx = ia[row+1];
                } else
                    stop_idx = nnz;

                T max = 0.0;  // todo: Make this min possible T
                for(int j = start_idx; j < stop_idx; j++) {
                    if(vals[j] > max)
                        max = vals[j];
                }

                for(int j = start_idx; j < stop_idx; j++) {
                    if(max > 0.0) {
                        T val = vals[j];
                        result[j] = val / max;
                    }
                    else {
                        result[j] = 0.0;
                    }
                }
            }
        }



        __device__ int get_stop_idx(int row, int m, int *ind) {
            int stop_idx = 0;
            if(row < (m-1))
                stop_idx = ind[row+1];
            else
                stop_idx = m;

            return stop_idx;
        }

        /**
         * Calculate how many unique columns per row
         */
        template<typename T, int TPB_X>
        __global__ void csr_add_calc_row_counts_kernel(
                int *a_ind, int *a_indptr, T *a_val,
                int *b_ind, int *b_indptr, T *b_val,
                int nnz, int m,
                int *out_rowcounts) {

            // loop through columns in each set of rows and
            // calculate number of unique cols across both rows
            int row = (blockIdx.x * TPB_X) + threadIdx.x;

            if(row < m) {
                int a_start_idx = a_ind[row];
                int a_stop_idx = get_stop_idx(row, m, a_ind);

                int b_start_idx = b_ind[row];
                int b_stop_idx = get_stop_idx(row, m, b_ind);

                int max_size = (a_stop_idx - a_start_idx) +
                        (b_stop_idx - b_start_idx);

                int *arr  = new int[max_size];
                int cur_arr_idx = 0;
                for(int j = a_start_idx; j < a_stop_idx; j++) {
                    arr[cur_arr_idx] = a_indptr[j];
                    cur_arr_idx++;
                }

                int arr_size = cur_arr_idx;

                int final_size = arr_size;

                for(int j = b_start_idx; j < b_stop_idx; j++) {

                    int cur_col = b_indptr[j];
                    bool found = false;
                    for(int k = 0; k < arr_size; k++) {
                        if(arr[k] == cur_col)
                            found = true;
                    }

                    if(!found)
                        final_size++;
                }

                out_rowcounts[row] = final_size;
                atomicAdd(out_rowcounts+m, final_size);

                delete arr;
            }
        }


        template<typename T, int TPB_X>
        __global__ void csr_add_kernel(
               int *a_ind, int *a_indptr, T *a_val,
               int *b_ind, int *b_indptr, T *b_val,
               int nnz, int m,
               int *out_ind, int *out_indptr, T *out_val) {

            // 1 thread per row
            int row = (blockIdx.x * TPB_X) + threadIdx.x;

            if(row < m) {
                int a_start_idx = a_ind[row];
                int a_stop_idx = get_stop_idx(row, m, a_ind);

                int b_start_idx = b_ind[row];
                int b_stop_idx = get_stop_idx(row, m, b_ind);

                int o_idx = out_ind[row];

                printf("row=%d, a_start_idx=%d, a_stop_idx=%d, b_start_idx=%d, b_stop_idx=%d\n",
                        row, a_start_idx, a_stop_idx, b_start_idx, b_stop_idx);

                printf("row=%d, o_idx=%d\n", o_idx);

                int cur_o_idx = o_idx;
                for(int j = a_start_idx; j < a_stop_idx; j++) {
                    out_indptr[cur_o_idx] = a_indptr[j];
                    out_val[cur_o_idx] = a_val[j];
                    cur_o_idx++;
                }

                int arr_size = cur_o_idx;
                for(int j = b_start_idx; j < b_stop_idx; j++) {
                    int cur_col = b_indptr[j];
                    bool found = false;
                    for(int k = 0; k < arr_size; k++) {
                        // If we found a match, sum the two values
                        if(out_indptr[k] == cur_col) {
                            out_val[k] += b_val[j];
                            found = true;
                        }
                    }

                    // if we didn't find a match, add the value for b
                    if(!found) {
                        out_indptr[arr_size] = cur_col;
                        out_val[arr_size] = b_val[j];
                        arr_size++;

                        printf("row=%d, col=%d, j=%d, out_val[%d]=%f\n", row, cur_col, j, arr_size, b_val[j]);
                    }
                }
            }
        }

        template<typename T, int TPB_X>
        void csr_add_calc_inds(
            int *a_ind, int *a_indptr, T *a_val,
            int *b_ind, int *b_indptr, T *b_val,
            int nnz, int m,
            int *out_nnz, int *out_ind
        ) {

            dim3 grid(ceildiv(m, TPB_X), 1, 1);
            dim3 blk(TPB_X, 1, 1);

            int *row_counts;
            MLCommon::allocate(row_counts, m+1, true);

            std::cout << "About to run calc_row_counts_kernel" << std::endl;

            csr_add_calc_row_counts_kernel<T,TPB_X><<<grid, blk>>>(
                a_ind, a_indptr, a_val,
                b_ind, b_indptr, b_val,
                nnz, m,
                row_counts
            );
            CUDA_CHECK(cudaPeekAtLastError());

            std::cout << "Done. " << std::endl;

            std::cout << MLCommon::arr2Str(row_counts, m+1, "row_counts") << std::endl;

            int cnnz = 0;
            MLCommon::updateHost(&cnnz, row_counts+m, 1);

            std::cout << "cnnz=" << cnnz << std::endl;

            std::cout << "Setting now..." << std::endl;

            out_nnz[0] = cnnz;

            std::cout << "Done setting." << std::endl;

//            memset(out_nnz, cnnz, sizeof(int));

//            std::cout << MLCommon::arr2Str(out_nnz, 1, "out_nnz") << std::endl;


            // create csr compressed row index from row counts
            thrust::device_ptr<int> row_counts_d = thrust::device_pointer_cast(row_counts);
            thrust::device_ptr<int> c_ind_d = thrust::device_pointer_cast(out_ind);
            exclusive_scan(row_counts_d, row_counts_d + m, c_ind_d);

            std::cout << "Done ex_scan" << std::endl;

            CUDA_CHECK(cudaFree(row_counts));
        }

        template<typename T, int TPB_X>
        void csr_add_finalize(
            int *a_ind, int *a_indptr, T *a_val,
            int *b_ind, int *b_indptr, T *b_val,
            int nnz, int m,
            int *c_ind, int *c_indptr, T *c_val
        ) {
            dim3 grid(MLCommon::ceildiv(m, TPB_X), 1, 1);
            dim3 blk(TPB_X, 1, 1);

            csr_add_kernel<T, TPB_X><<<grid,blk>>>(
                a_ind, a_indptr, a_val,
                b_ind, b_indptr, b_val,
                nnz, m,
                c_ind, c_indptr, c_val
            );
           CUDA_CHECK(cudaPeekAtLastError());
        }

    }



}
