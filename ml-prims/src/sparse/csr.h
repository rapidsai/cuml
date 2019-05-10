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

#include "array/array.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuda_runtime.h>
#include <stdio.h>

#include <iostream>


namespace MLCommon {
namespace Sparse {

static const float MIN_FLOAT = std::numeric_limits<float>::min();

/**
 * @brief a container object for sparse CSR formatted matrices
 */
template <typename T>
class CSR {

public:
    int *row_ind;
    int *row_ind_ptr;
    T *vals;
    int nnz;
    int n_rows;
    int n_cols;

    /**
     * @brief default constructor
     */
    CSR(): row_ind(nullptr), row_ind_ptr(nullptr), vals(nullptr), nnz(-1), n_rows(-1), n_cols(-1){}

    /*
     * @brief construct a CSR object with arrays
     *
     * @param row_ind: the array of row_indices
     * @param row_ind_ptr: array of row_index_ptr
     * @param vals: array of data
     * @param nnz: size of data and row_ind_ptr arrays
     * @param n_rows: number of rows in the dense matrix
     * @param n_cols: number of cols in the dense matrix
     */
    CSR(int* const row_ind, int* const row_ind_ptr, T* const vals, int nnz, int n_rows = -1, int n_cols = -1) {
        this->row_ind = row_ind;
        this->row_ind_ptr = row_ind_ptr;
        this->vals = vals;
        this->nnz = nnz;
        this->n_rows = n_rows;
        this->n_cols = n_cols;
    }
    /*
     * @brief construct an empty allocated CSR given its size
     *
     * @param nnz: size of data and row_ind_ptr arrays
     * @param n_rows: number of rows in the dense matrix
     * @param n_cols: number of cols in the dense matrix
     * @param init: initialize arrays to zeros?
     */

    CSR(int nnz, int n_rows = -1, int n_cols = -1, bool init = true):
        row_ind(nullptr), row_ind_ptr(nullptr), vals(nullptr), nnz(nnz),
        n_rows(n_rows), n_cols(n_cols) {
        this->allocate(nnz, n_rows, n_cols, init);
    }

    ~CSR() {
        this->free();
    }

    /**
     * @brief validate size of CSR object is >0 and that
     * number of rows/cols of dense matrix are also >0.
     */
    bool validate_size() {
        if(this->nnz < 0 || n_rows < 0 || n_cols < 0)
            return false;
        return true;
    }

    /**
     * @brief Return true if underlying arrays have been allocated, false otherwise.
     */
    bool validate_mem() {
        if(this->row_ind == nullptr ||
                this->row_ind_ptr == nullptr ||
                this->vals == nullptr) {
            return false;
        }

        return true;
    }

    /**
     * @brief Send human-readable object state to the given output stream
     */
   friend std::ostream & operator << (std::ostream &out, const CSR &c) {
       out << arr2Str(c->row_ind, c->nnz, "row_ind") << std::endl;
       out << arr2Str(c->row_ind_ptr, c->nnz, "cols") << std::endl;
       out << arr2Str(c->vals, c->nnz, "vals") << std::endl;
       out << c->nnz << std::endl;
   }

   /**
    * @brief Sets the size of a non-square dense matrix
    * @param n_rows: number of rows in dense matrix
    * @param n_cols: number of cols in dense matrix
    */
    void setSize(int n_rows, int n_cols) {
        this->n_rows = n_rows;
        this->n_cols = n_cols;
    }

    /**
     * @brief Sets the size of a square dense matrix
     * @param n: number of rows & cols in dense matrix
     */
    void setSize(int n) {
        this->n_rows = n;
        this->n_cols = n;
    }

    /**
     * @brief Allocate underlying arrays
     * @param nnz: sets the size of the underlying arrays
     * @param init: should arrays be initialized to zeros?
     */
    void allocate(int nnz, bool init = true) {
        this->allocate(nnz, -1, init);
    }

    /**
     * @brief Allocate underlying arrays and the size of the square dense matrix
     * @param nnz: sets the size of the underlying arrays
     * @param size: number of rows and cols in the square dense matrix
     * @param init: should arrays be initialized to zeros?
     */
    void allocate(int nnz, int size, bool init = true) {
        this->allocate(nnz, size, size, init);
    }

    /**
     * @brief Allocate underlying arrays and the size of the non-square dense matrix
     * @param nnz: sets the size of the underlying arrays
     * @param n_rows: number of rows in the dense matrix
     * @param n_cols: number of cols in the dense matrix
     * @param init: should arrays be initialized to zeros?
     */
    void allocate(int nnz, int n_rows, int n_cols, bool init = true) {
        this->n_rows = n_rows;
        this->n_cols = n_cols;
        this->nnz = nnz;
        MLCommon::allocate(this->row_ind, this->nnz, init);
        MLCommon::allocate(this->row_ind_ptr, this->nnz, init);
        MLCommon::allocate(this->vals, this->nnz, init);
    }

    /**
     * @brief Frees the memory from the underlying arrays
     */
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

/**
 * @brief Perform L1 normalization on the rows of a given CSR-formatted sparse matrix
 *
 * @param ia: row_ind array
 * @param vals: data array
 * @param nnz: size of data array
 * @param m: size of row_ind array
 * @param result: l1 normalized data array
 * @param stream: cuda stream to use
 */
template<int TPB_X = 32, typename T>
void csr_row_normalize_l1(
        int* const ia,    // csr row ex_scan (sorted by row)
        T* const vals, int nnz,  // array of values and number of non-zeros
        int m,          // num rows in csr
        T *result,
        cudaStream_t stream) {    // output array

    dim3 grid(MLCommon::ceildiv(m, TPB_X), 1, 1);
    dim3 blk(TPB_X, 1, 1);

    csr_row_normalize_l1_kernel<TPB_X, T><<<grid, blk, 0, stream>>>(ia, vals, nnz,
                     m, result);
}

template<int TPB_X = 32, typename T>
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


/**
 * @brief Perform L_inf normalization on a given CSR-formatted sparse matrix
 *
 * @param ia: row_ind array
 * @param vals: data array
 * @param nnz: size of data array
 * @param m: size of row_ind array
 * @param result: l1 normalized data array
 * @param stream: cuda stream to use
 */

template<int TPB_X = 32, typename T>
void csr_row_normalize_max(
        int* const ia,    // csr row ind array (sorted by row)
        T* const vals, int nnz,  // array of values and number of non-zeros
        int m,          // num total rows in csr
        T *result,
        cudaStream_t stream) {

    dim3 grid(MLCommon::ceildiv(m, TPB_X), 1, 1);
    dim3 blk(TPB_X, 1, 1);

    csr_row_normalize_max_kernel<TPB_X, T><<<grid, blk, 0, stream>>>(ia, vals, nnz,
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

template<int TPB_X = 32>
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

/**
 * @brief Convert a CSR row_ind array to a COO rows array
 * @param row_ind: Input CSR row_ind array
 * @param m: size of row_ind array
 * @param coo_rows: Output COO row array
 * @param nnz: size of output COO row array
 * @param stream: cuda stream to use
 */
template<int TPB_X>
void csr_to_coo(int *row_ind, int m, int *coo_rows, int nnz,
        cudaStream_t stream) {
    dim3 grid(MLCommon::ceildiv(m, TPB_X), 1, 1);
    dim3 blk(TPB_X, 1, 1);

    csr_to_coo_kernel<TPB_X><<<grid,blk, 0, stream>>>(row_ind, m, coo_rows, nnz);
}


template<typename T, int TPB_X = 32>
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


template<typename T, int TPB_X = 32>
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

/**
 * @brief Calculate the CSR row_ind array that would result
 * from summing together two CSR matrices
 * @param a_ind: left hand row_ind array
 * @param a_indptr: left hand index_ptr array
 * @param a_val: left hand data array
 * @param nnz1: size of left hand index_ptr and val arrays
 * @param b_ind: right hand row_ind array
 * @param b_indptr: right hand index_ptr array
 * @param b_val: right hand data array
 * @param nnz2: size of right hand index_ptr and val arrays
 * @param m: size of output array (number of rows in final matrix)
 * @param out_ind: output row_ind array
 * @param stream: cuda stream to use
 */
template<typename T, int TPB_X = 32>
size_t csr_add_calc_inds(
    int*  const a_ind, int* const a_indptr, T* const a_val, int nnz1,
    int* const b_ind, int* const b_indptr, T* const b_val, int nnz2,
    int m, int *out_ind,
    cudaStream_t stream
) {
    dim3 grid(ceildiv(m, TPB_X), 1, 1);
    dim3 blk(TPB_X, 1, 1);

    int *row_counts;
    MLCommon::allocate(row_counts, m+1, true);

    csr_add_calc_row_counts_kernel<T,TPB_X><<<grid, blk, 0, stream>>>(
        a_ind, a_indptr, a_val, nnz1,
        b_ind, b_indptr, b_val, nnz2,
        m, row_counts
    );
    CUDA_CHECK(cudaPeekAtLastError());

    int cnnz = 0;
    MLCommon::updateHost(&cnnz, row_counts+m, 1, stream);

    // create csr compressed row index from row counts
    thrust::device_ptr<int> row_counts_d = thrust::device_pointer_cast(row_counts);
    thrust::device_ptr<int> c_ind_d = thrust::device_pointer_cast(out_ind);
    exclusive_scan(thrust::cuda::par.on(stream),row_counts_d, row_counts_d + m, c_ind_d);
    CUDA_CHECK(cudaFree(row_counts));

    return cnnz;

}

/**
 * @brief Calculate the CSR row_ind array that would result
 * from summing together two CSR matrices
 * @param a_ind: left hand row_ind array
 * @param a_indptr: left hand index_ptr array
 * @param a_val: left hand data array
 * @param nnz1: size of left hand index_ptr and val arrays
 * @param b_ind: right hand row_ind array
 * @param b_indptr: right hand index_ptr array
 * @param b_val: right hand data array
 * @param nnz2: size of right hand index_ptr and val arrays
 * @param m: size of output array (number of rows in final matrix)
 * @param c_ind: output row_ind array
 * @param c_indptr: output ind_ptr array
 * @param c_val: output data array
 * @param stream: cuda stream to use
 */
template<typename T, int TPB_X = 32>
void csr_add_finalize(
    int* const a_ind, int* const a_indptr, T* const a_val, int nnz1,
    int* const b_ind, int* const b_indptr, T* const b_val, int nnz2,
    int m, int* const c_ind, int *c_indptr, T *c_val,
    cudaStream_t stream
) {
    dim3 grid(MLCommon::ceildiv(m, TPB_X), 1, 1);
    dim3 blk(TPB_X, 1, 1);

    csr_add_kernel<T, TPB_X><<<grid,blk, 0, stream>>>(
        a_ind, a_indptr, a_val, nnz1,
        b_ind, b_indptr, b_val, nnz2,
        m, c_ind, c_indptr, c_val
    );
   CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T, int TPB_X = 32, typename Lambda = auto(T, T, T)->void>
__global__ void csr_row_op_kernel(T* const row_ind, T n_rows,
        T nnz, Lambda op) {
    T row = blockIdx.x*TPB_X + threadIdx.x;
    if(row < n_rows) {
        T start_idx = row_ind[row];
        T stop_idx = row < n_rows-1 ? row_ind[row+1] : nnz;
        op(row, start_idx, stop_idx);
    }
}

/**
 * @brief Perform a custom row operation on a CSR matrix in batches.
 * @tparam T numerical type of row_ind array
 * @tparam TPB_X number of threads per block to use for underlying kernel
 * @tparam Lambda type of custom operation function
 * @param row_ind the CSR row_ind array to perform parallel operations over
 * @param total_rows total number vertices in graph
 * @param batchSize size of row_ind
 * @param op custom row operation functor accepting the row and beginning index.
 * @param stream cuda stream to use
 */
template<typename T, int TPB_X = 32, typename Lambda = auto(T, T, T)->void>
void csr_row_op(T* const row_ind, T n_rows, T nnz,
        Lambda op, cudaStream_t stream) {

    dim3 grid(MLCommon::ceildiv(n_rows, TPB_X), 1, 1);
    dim3 blk(TPB_X, 1, 1);
    csr_row_op_kernel<T, TPB_X><<<grid, blk, 0, stream>>>
            (row_ind, n_rows, nnz, op);

    CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Constructs an adjacency graph CSR row_ind_ptr array from
 * a row_ind array and adjacency array.
 * @tparam T the numeric type of the index arrays
 * @tparam TPB_X the number of threads to use per block for kernels
 * @tparam Lambda function for fused operation in the adj_graph construction
 * @param row_ind the input CSR row_ind array
 * @param total_rows number of vertices in graph
 * @param batchSize number of vertices in current batch
 * @param adj an adjacency array (size batchSize * total_rows)
 * @param row_ind_ptr output CSR row_ind_ptr for adjacency graph
 * @param stream cuda stream to use
 */
template<typename T, int TPB_X = 32, typename Lambda = auto(T, T, T)->void>
void csr_adj_graph_batched(T* const row_ind, T total_rows, T nnz, T batchSize,
        bool* const adj, T *row_ind_ptr, cudaStream_t stream, Lambda fused_op) {
    csr_row_op<T, TPB_X>(row_ind, batchSize, nnz,
            [fused_op, adj, total_rows, row_ind_ptr, batchSize] __device__
                (T row, T start_idx, T stop_idx) {

            fused_op(row, start_idx, stop_idx);
            int k = 0;
            for(T i=0; i<total_rows; i++) {
                // @todo: uncoalesced mem accesses!
                if(adj[batchSize * i + row]) {
                    row_ind_ptr[start_idx + k] = i;
                    k += 1;
                }
            }
    }, stream);
}

template<typename T, int TPB_X = 32, typename Lambda = auto (T, T, T)->void>
void csr_adj_graph_batched(T* const row_ind, T total_rows, T nnz, T batchSize,
        bool* const adj, T *row_ind_ptr, cudaStream_t stream) {
    csr_adj_graph_batched(row_ind, total_rows, nnz, batchSize, adj,
    row_ind_ptr, stream, [] __device__ (T row, T start_idx, T stop_idx) {});
}

/**
 * @brief Constructs an adjacency graph CSR row_ind_ptr array from a
 * a row_ind array and adjacency array.
 * @tparam T the numeric type of the index arrays
 * @tparam TPB_X the number of threads to use per block for kernels
 * @param row_ind the input CSR row_ind array
 * @param n_rows number of total vertices in graph
 * @param adj an adjacency array
 * @param row_ind_ptr output CSR row_ind_ptr for adjacency graph
 * @param stream cuda stream to use
 */
template<typename T, int TPB_X = 32, typename Lambda = auto (T, T, T)->void>
void csr_adj_graph(T* const row_ind, T total_rows, T nnz,
        bool* const adj, T *row_ind_ptr, cudaStream_t stream, Lambda fused_op) {

    csr_adj_graph_batched<T, TPB_X>(row_ind, total_rows, nnz, total_rows, adj,
            row_ind_ptr, stream, fused_op);
}

template<typename T = int>
class WeakCCState {
    public:

        bool *xa;
        bool *fa;
        bool *m;
        bool owner;

        WeakCCState(T n): owner(true) {
            MLCommon::allocate(xa, n, true);
            MLCommon::allocate(fa, n, true);
            MLCommon::allocate(m, 1, true);
        }

        WeakCCState(bool *xa, bool *fa, bool *m):
            owner(false), xa(xa), fa(fa), m(m) {
        }

        ~WeakCCState() {
            if(owner) {
                try {
                    CUDA_CHECK(cudaFree(xa));
                    CUDA_CHECK(cudaFree(fa));
                    CUDA_CHECK(cudaFree(m));
                } catch(Exception &e) {
                    std::cout << "Exception freeing memory for WeakCCState: " <<
                            e.what() << std::endl;
                }
            }
        }
};

template <typename Type, int TPB_X = 32>
__global__ void weak_cc_label_device(
        Type *labels,
        Type *row_ind, Type *row_ind_ptr, Type nnz,
        bool *fa, bool *xa, bool *m,
        int startVertexId, int batchSize) {
    int tid = threadIdx.x + blockIdx.x*TPB_X;
    if(tid<batchSize) {
        if(fa[tid + startVertexId]) {
            fa[tid + startVertexId] = false;
            int start = int(row_ind[tid]);
            Type ci, cj;
            bool ci_mod = false;
            ci = labels[tid + startVertexId];

            Type degree = get_stop_idx(tid, batchSize,nnz, row_ind) - row_ind[tid];

            for(int j=0; j< int(degree); j++) { // TODO: Can't this be calculated from the ex_scan?
                cj = labels[row_ind_ptr[start + j]];
                if(ci<cj) {
                    atomicMin(labels + row_ind_ptr[start +j], ci);
                    xa[row_ind_ptr[start+j]] = true;
                    m[0] = true;
                }
                else if(ci>cj) {
                    ci = cj;
                    ci_mod = true;
                }
            }
            if(ci_mod) {
                atomicMin(labels + startVertexId + tid, ci);
                xa[startVertexId + tid] = true;
                m[0] = true;
            }
        }
    }
}


template <typename Type, int TPB_X = 32, typename Lambda>
__global__ void weak_cc_init_label_kernel(Type *labels, int startVertexId, int batchSize,
        Type MAX_LABEL, Lambda filter_op) {
    /** F1 and F2 in the paper correspond to fa and xa */
    /** Cd in paper corresponds to db_cluster */
    int tid = threadIdx.x + blockIdx.x*TPB_X;
    if(tid<batchSize) {
        if(filter_op(tid) && labels[tid + startVertexId]==MAX_LABEL)
            labels[startVertexId + tid] = Type(startVertexId + tid + 1);
    }
}

template <typename Type, int TPB_X = 32>
__global__ void weak_cc_init_all_kernel(Type *labels, bool *fa, bool *xa,
        Type N, Type MAX_LABEL) {
    int tid = threadIdx.x + blockIdx.x*TPB_X;
    if(tid<N) {
        labels[tid] = MAX_LABEL;
        fa[tid] = true;
        xa[tid] = false;
    }
}

template <typename Type, int TPB_X = 32, typename Lambda>
void weak_cc_label_batched(Type *labels,
        Type* const row_ind, Type* const row_ind_ptr, Type nnz, Type N,
        WeakCCState<Type> *state,
        Type startVertexId, Type batchSize,
        cudaStream_t stream, Lambda filter_op) {
    bool host_m;
    bool *host_fa = (bool*)malloc(sizeof(bool)*N);
    bool *host_xa = (bool*)malloc(sizeof(bool)*N);

    dim3 blocks(ceildiv(batchSize, TPB_X));
    dim3 threads(TPB_X);
    Type MAX_LABEL = std::numeric_limits<Type>::max();

    weak_cc_init_label_kernel<Type, TPB_X><<<blocks, threads, 0, stream>>>(labels,
            startVertexId, batchSize, MAX_LABEL, filter_op);
    CUDA_CHECK(cudaPeekAtLastError());
    do {
        CUDA_CHECK( cudaMemsetAsync(state->m, false, sizeof(bool), stream) );
        weak_cc_label_device<Type, TPB_X><<<blocks, threads, 0, stream>>>(
                labels,
                row_ind, row_ind_ptr, nnz,
                state->fa, state->xa, state->m,
                startVertexId, batchSize);
        CUDA_CHECK(cudaPeekAtLastError());

        //** swapping F1 and F2
        MLCommon::updateHost(host_fa, state->fa, N, stream);
        MLCommon::updateHost(host_xa, state->xa, N, stream);
        MLCommon::updateDevice(state->fa, host_xa, N, stream);
        MLCommon::updateDevice(state->xa, host_fa, N, stream);

        //** Updating m *
        MLCommon::updateHost(&host_m, state->m, 1, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } while(host_m);
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Array package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param startVertexId the starting vertex index for the current batch
 * @param batchSize number of vertices for current batch
 * @param state instance of inter-batch state management
 * @param stream the cuda stream to use
 * @param filter_op an optional filtering function to determine which points
 * should get considered for labeling.
 */
template<typename Type = int, int TPB_X = 32, typename Lambda = auto (Type)->bool>
void weak_cc_batched(Type *labels, Type* const row_ind,  Type* const row_ind_ptr,
        Type nnz, Type N, Type startVertexId, Type batchSize,
        WeakCCState<Type> *state, cudaStream_t stream, Lambda filter_op) {

    dim3 blocks(ceildiv(N, TPB_X));
    dim3 threads(TPB_X);

    Type MAX_LABEL = std::numeric_limits<Type>::max();
    if(startVertexId == 0) {
        weak_cc_init_all_kernel<Type, TPB_X><<<blocks, threads, 0, stream>>>
            (labels, state->fa, state->xa, N, MAX_LABEL);
        CUDA_CHECK(cudaPeekAtLastError());
    }
    weak_cc_label_batched<Type, TPB_X>(labels, row_ind, row_ind_ptr, nnz, N, state,
            startVertexId, batchSize, stream, filter_op);
}

template<typename Type = int, int TPB_X = 32>
void weak_cc_batched(Type *labels, Type* const row_ind,  Type* const row_ind_ptr,
        Type nnz, Type N, Type startVertexId, Type batchSize,
        WeakCCState<Type> *state, cudaStream_t stream) {

    weak_cc_batched(labels, row_ind, row_ind_ptr, nnz, N, startVertexId, batchSize,
            state, stream, [] __device__ (int tid) {return true;});
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Array package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param stream the cuda stream to use
 * @param filter_op an optional filtering function to determine which points
 * should get considered for labeling.
 */
template<typename Type = int, int TPB_X = 32, typename Lambda = auto (Type)->bool>
void weak_cc(Type *labels, Type* const row_ind, Type* const row_ind_ptr,
        Type nnz, Type N, cudaStream_t stream, Lambda filter_op) {

    WeakCCState<Type> state(N);
    weak_cc_batched<Type, TPB_X>(
            labels, row_ind, row_ind_ptr,
            nnz, N, 0, N, stream,
            filter_op);
}

/**
 * @brief Compute weakly connected components. Note that the resulting labels
 * may not be taken from a monotonically increasing set (eg. numbers may be
 * skipped). The MLCommon::Array package contains a primitive `make_monotonic`,
 * which will make a monotonically increasing set of labels.
 *
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * @tparam Type the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @tparam Lambda the type of an optional filter function (int)->bool
 * @param labels an array for the output labels
 * @param row_ind the compressed row index of the CSR array
 * @param row_ind_ptr the row index pointer of the CSR array
 * @param nnz the size of row_ind_ptr array
 * @param N number of vertices
 * @param stream the cuda stream to use
 * should get considered for labeling.
 */
template<typename Type = int, int TPB_X = 32>
void weak_cc(Type *labels, Type* const row_ind, Type* const row_ind_ptr,
        Type nnz, Type N, cudaStream_t stream) {

    WeakCCState<Type> state(N);
    weak_cc_batched<Type, TPB_X>(
            labels, row_ind, row_ind_ptr,
            nnz, N, 0, N, stream,
            [](Type t){return true;});
}



};
};
