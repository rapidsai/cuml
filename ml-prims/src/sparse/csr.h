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

#include "cuda_utils.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>


namespace MLCommon {
namespace Sparse {

static const float MIN_FLOAT = std::numeric_limits<float>::min();


template <typename T>
class CSR {

public:
    int *row_ind;
    int *row_ind_ptr;
    T *vals;
    int nnz;
    int n_rows;
    int n_cols;

    CSR(): row_ind(nullptr), row_ind_ptr(nullptr), vals(nullptr), nnz(-1), n_rows(-1), n_cols(-1){}

    CSR(int *row_ind, int *row_ind_ptr, T *vals, int nnz, int n_rows = -1, int n_cols = -1) {
        this->row_ind = row_ind;
        this->row_ind_ptr = row_ind_ptr;
        this->vals = vals;
        this->nnz = nnz;
        this->n_rows = n_rows;
        this->n_cols = n_cols;
    }

    CSR(int nnz, int n_rows = -1, int n_cols = -1, bool init = true):
        row_ind(nullptr), row_ind_ptr(nullptr), vals(nullptr), nnz(nnz),
        n_rows(n_rows), n_cols(n_cols) {
        this->allocate(nnz, n_rows, n_cols, init);
    }

    ~CSR() {
        this->free();
    }

    bool validate_size() {
        if(this->nnz < 0 || n_rows < 0 || n_cols < 0)
            return false;
        return true;
    }

    bool validate_mem() {
        if(this->row_ind == nullptr ||
                this->row_ind_ptr == nullptr ||
                this->vals == nullptr) {
            return false;
        }

        return true;
    }

   friend std::ostream & operator << (std::ostream &out, const CSR &c) {
       out << arr2Str(c->row_ind, c->nnz, "row_ind") << std::endl;
       out << arr2Str(c->row_ind_ptr, c->nnz, "cols") << std::endl;
       out << arr2Str(c->vals, c->nnz, "vals") << std::endl;
       out << c->nnz << std::endl;
   }

    void setSize(int n_rows, int n_cols) {
        this->n_rows = n_rows;
        this->n_cols = n_cols;
    }

    void setSize(int n) {
        this->n_rows = n;
        this->n_cols = n;
    }

    void allocate(int nnz, bool init = true) {
        this->allocate(nnz, -1, init);
    }

    void allocate(int nnz, int size, bool init = true) {
        this->allocate(nnz, size, size, init);
    }

    void allocate(int nnz, int n_rows, int n_cols, bool init = true) {
        this->n_rows = n_rows;
        this->n_cols = n_cols;
        this->nnz = nnz;
        MLCommon::allocate(this->row_ind, this->nnz, init);
        MLCommon::allocate(this->row_ind_ptr, this->nnz, init);
        MLCommon::allocate(this->vals, this->nnz, init);
    }

    void free() {

        try {
            if(row_ind != nullptr)
                CUDA_CHECK(cudaFree(row_ind));

            if(row_ind_ptr != nullptr)
                CUDA_CHECK(cudaFree(row_ind_ptr));

            if(vals != nullptr)
                CUDA_CHECK(cudaFree(vals));

            row_ind = nullptr;
            row_ind_ptr = nullptr;
            vals = nullptr;

        } catch(Exception &e) {
            std::cout << "An exception occurred freeing COO memory" << std::endl;
        }
    }

};

/**
 * Row-normalizes a CSR matrix using the sum of
 * each row as the normalizer.
 *
 */
template<int TPB_X, typename T>
__global__ void csr_row_normalize_l1_kernel(
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
            if(sum != 0.0) {
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
void csr_row_normalize_l1(
        int *ia,    // csr row ex_scan (sorted by row)
        T *vals, int nnz,  // array of values and number of non-zeros
        int m,          // num rows in csr
        T *result) {    // output array

    dim3 grid(MLCommon::ceildiv(m, TPB_X), 1, 1);
    dim3 blk(TPB_X, 1, 1);

    csr_row_normalize_l1_kernel<TPB_X, T><<<grid, blk>>>(ia, vals, nnz,
                     m, result);
}

/**
 * Row-normalizes a CSR matrix using the max of each
 * row as the normalizer.
 */
template<int TPB_X, typename T>
__global__ void csr_row_normalize_max_kernel(
        int *ia,    // csr row ind array (sorted by row)
        T *vals, int nnz,  // array of values and number of non-zeros
        int m,          // num total rows in csr
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

        T max = MIN_FLOAT;  // todo: Make this min possible T
        for(int j = start_idx; j < stop_idx; j++) {
            if(vals[j] > max)
                max = vals[j];
        }

        for(int j = start_idx; j < stop_idx; j++) {
            if(max != 0.0) {
                T val = vals[j];
                result[j] = val / max;
            }
            else {
                result[j] = 0.0;
            }
        }
    }
}

template<int TPB_X, typename T>
void csr_row_normalize_max(
        int *ia,    // csr row ind array (sorted by row)
        T *vals, int nnz,  // array of values and number of non-zeros
        int m,          // num total rows in csr
        T *result) {

    dim3 grid(MLCommon::ceildiv(m, TPB_X), 1, 1);
    dim3 blk(TPB_X, 1, 1);

    csr_row_normalize_max_kernel<TPB_X, T><<<grid, blk>>>(ia, vals, nnz,
                     m, result);

}

template<typename T>
__device__ int get_stop_idx(T row, int m, int nnz, T *ind) {
    int stop_idx = 0;
    if(row < (m-1))
        stop_idx = ind[row+1];
    else
        stop_idx = nnz;

    return stop_idx;
}

template<int TPB_X>
__global__ void csr_to_coo_kernel(int *row_ind, int m, int *coo_rows, int nnz) {

    // row-based matrix 1 thread per row
    int row = (blockIdx.x * TPB_X) + threadIdx.x;
    if(row < m) {
        int start_idx = row_ind[row];
        int stop_idx = get_stop_idx(row, m, nnz, row_ind);
        for(int i = start_idx; i < stop_idx; i++)
            coo_rows[i] = row;
    }
}

template<int TPB_X>
void csr_to_coo(int *row_ind, int m, int *coo_rows, int nnz,
        cudaStream_t stream) {
    dim3 grid(MLCommon::ceildiv(m, TPB_X), 1, 1);
    dim3 blk(TPB_X, 1, 1);

    csr_to_coo_kernel<TPB_X><<<grid,blk, 0, stream>>>(row_ind, m, coo_rows, nnz);
}


/**
 * Calculate how many unique columns per row
 */
template<typename T, int TPB_X>
__global__ void csr_add_calc_row_counts_kernel(
        int *a_ind, int *a_indptr, T *a_val, int nnz1,
        int *b_ind, int *b_indptr, T *b_val, int nnz2,
        int m, int *out_rowcounts) {

    // loop through columns in each set of rows and
    // calculate number of unique cols across both rows
    int row = (blockIdx.x * TPB_X) + threadIdx.x;

    if(row < m) {
        int a_start_idx = a_ind[row];
        int a_stop_idx = get_stop_idx(row, m, nnz1, a_ind);

        int b_start_idx = b_ind[row];
        int b_stop_idx = get_stop_idx(row, m, nnz2, b_ind);

        /**
         * Union of columns within each row of A and B so that we can scan through
         * them, adding their values together.
         */
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
                if(arr[k] == cur_col) {
                    found = true;
                    break;
                }
            }

            if(!found) {
                final_size++;
            }
        }

        out_rowcounts[row] = final_size;
        atomicAdd(out_rowcounts+m, final_size);

        delete arr;
    }
}


template<typename T, int TPB_X>
__global__ void csr_add_kernel(
       int *a_ind, int *a_indptr, T *a_val, int nnz1,
       int *b_ind, int *b_indptr, T *b_val, int nnz2,
       int m,
       int *out_ind, int *out_indptr, T *out_val) {

    // 1 thread per row
    int row = (blockIdx.x * TPB_X) + threadIdx.x;


    if(row < m) {
        int a_start_idx = a_ind[row];
        int a_stop_idx = get_stop_idx(row, m, nnz1, a_ind);

        int b_start_idx = b_ind[row];
        int b_stop_idx = get_stop_idx(row, m, nnz2, b_ind);

        int o_idx = out_ind[row];

        int cur_o_idx = o_idx;
        for(int j = a_start_idx; j < a_stop_idx; j++) {
            out_indptr[cur_o_idx] = a_indptr[j];
            out_val[cur_o_idx] = a_val[j];
            cur_o_idx++;
        }

        int arr_size = cur_o_idx-o_idx;
        for(int j = b_start_idx; j < b_stop_idx; j++) {
            int cur_col = b_indptr[j];
            bool found = false;
            for(int k = o_idx; k < o_idx+arr_size; k++) {
                // If we found a match, sum the two values
                if(out_indptr[k] == cur_col) {
                    out_val[k] += b_val[j];
                    found = true;
                    break;
                }
            }

            // if we didn't find a match, add the value for b
            if(!found) {
                out_indptr[o_idx+arr_size] = cur_col;
                out_val[o_idx+arr_size] = b_val[j];
                arr_size++;
            }
        }
    }
}

template<typename T, int TPB_X>
size_t csr_add_calc_inds(
    int *a_ind, int *a_indptr, T *a_val, int nnz1,
    int *b_ind, int *b_indptr, T *b_val, int nnz2,
    int m, int *out_ind
) {
    dim3 grid(ceildiv(m, TPB_X), 1, 1);
    dim3 blk(TPB_X, 1, 1);

    int *row_counts;
    MLCommon::allocate(row_counts, m+1, true);

    csr_add_calc_row_counts_kernel<T,TPB_X><<<grid, blk>>>(
        a_ind, a_indptr, a_val, nnz1,
        b_ind, b_indptr, b_val, nnz2,
        m, row_counts
    );
    CUDA_CHECK(cudaPeekAtLastError());

    int cnnz = 0;
    MLCommon::updateHost(&cnnz, row_counts+m, 1);

    // create csr compressed row index from row counts
    thrust::device_ptr<int> row_counts_d = thrust::device_pointer_cast(row_counts);
    thrust::device_ptr<int> c_ind_d = thrust::device_pointer_cast(out_ind);
    exclusive_scan(row_counts_d, row_counts_d + m, c_ind_d);
    CUDA_CHECK(cudaFree(row_counts));

    return cnnz;

}

template<typename T, int TPB_X>
void csr_add_finalize(
    int *a_ind, int *a_indptr, T *a_val, int nnz1,
    int *b_ind, int *b_indptr, T *b_val, int nnz2,
    int m, int *c_ind, int *c_indptr, T *c_val
) {
    dim3 grid(MLCommon::ceildiv(m, TPB_X), 1, 1);
    dim3 blk(TPB_X, 1, 1);

    csr_add_kernel<T, TPB_X><<<grid,blk>>>(
        a_ind, a_indptr, a_val, nnz1,
        b_ind, b_indptr, b_val, nnz2,
        m, c_ind, c_indptr, c_val
    );
   CUDA_CHECK(cudaPeekAtLastError());
}
};
};
