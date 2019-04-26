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

#include "csr.h"

#include "cusparse_wrappers.h"

#include <cusparse_v2.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

#include "cuda_utils.h"
#include <cuda_runtime.h>

#include <iostream>

namespace MLCommon {

    namespace Sparse {

        template<typename T>
        class COO {
            public:
               int *rows;
               int *cols;
               T *vals;
               int nnz;
               int n_rows;
               int n_cols;

               COO(): rows(nullptr), cols(nullptr), vals(nullptr), nnz(-1), n_rows(-1), n_cols(-1){}

               COO(int *rows, int *cols, int *vals, int nnz, int n_rows = -1, int n_cols = -1) {
                   this->rows = rows;
                   this->cols = cols;
                   this->vals = vals;
                   this->nnz = nnz;
                   this->n_rows = n_rows;
                   this->n_cols = n_cols;
               }

               COO(int nnz, int n_rows = -1, int n_cols = -1, bool init = true):
                   rows(nullptr), cols(nullptr), vals(nullptr), nnz(nnz),
                   n_rows(n_rows), n_cols(n_cols) {
                   this->allocate(nnz, n_rows, n_cols, init);
               }

               ~COO() {
                   this->free();
               }

               bool validate_size() {
                   if(this->nnz < 0 || n_rows < 0 || n_cols < 0)
                       return false;
                   return true;
               }

               bool validate_mem() {
                   if(this->rows == nullptr ||
                           this->cols == nullptr ||
                           this->vals == nullptr) {
                       return false;
                   }

                   return true;
               }

               friend std::ostream & operator << (std::ostream &out, const COO &c) {
                   out << arr2Str(c->rows, c->nnz, "rows") << std::endl;
                   out << arr2Str(c->cols, c->nnz, "cols") << std::endl;
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
                   MLCommon::allocate(this->rows, this->nnz, init);
                   MLCommon::allocate(this->cols, this->nnz, init);
                   MLCommon::allocate(this->vals, this->nnz, init);
               }

               void free() {

                   try {
                       std::cout << "Cleaning up!" << std::endl;
                       if(rows != nullptr)
                           CUDA_CHECK(cudaFree(rows));

                       if(cols != nullptr)
                           CUDA_CHECK(cudaFree(cols));

                       if(vals != nullptr)
                           CUDA_CHECK(cudaFree(vals));

                       rows = nullptr;
                       cols = nullptr;
                       vals = nullptr;

                   } catch(Exception &e) {
                       std::cout << "An exception occurred freeing COO memory" << std::endl;
                   }
               }


               void remove_zeros(COO<T> *out_coo, cudaStream_t stream);
        };

        template<typename T>
        cusparseStatus_t cusparse_gthr(cusparseHandle_t handle,
                int nnz,
                float *vals,
                float *vals_sorted,
                int *d_P) {
            return cusparseSgthr(
                handle,
                nnz,
                vals,
                vals_sorted,
                d_P,
                CUSPARSE_INDEX_BASE_ZERO
            );
        }



        template<typename T>
        cusparseStatus_t cusparse_gthr(cusparseHandle_t handle,
                int nnz,
                double *vals,
                double *vals_sorted,
                int *d_P) {
            return cusparseDgthr(
                handle,
                nnz,
                vals,
                vals_sorted,
                d_P,
                CUSPARSE_INDEX_BASE_ZERO
            );
        }



        /**
         * Sorts the arrays that comprise the coo matrix
         * by row.
         *
         * @param m number of rows in coo matrix
         * @param n number of cols in coo matrix
         * @param rows rows array from coo matrix
         * @param cols cols array from coo matrix
         * @param vals vals array from coo matrix
         */
        template<typename T>
        void coo_sort(int m, int n, int nnz,
                      int *rows, int *cols, T *vals) {

            cusparseHandle_t handle = NULL;
            cudaStream_t stream = NULL;

            size_t pBufferSizeInBytes = 0;
            void *pBuffer = NULL;
            int *d_P = NULL;

            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

            CUSPARSE_CHECK(cusparseCreate(&handle));

            CUSPARSE_CHECK(cusparseSetStream(handle, stream));

            CUSPARSE_CHECK(cusparseXcoosort_bufferSizeExt(
                handle,
                m,
                n,
                nnz,
                rows,
                cols,
                &pBufferSizeInBytes
            ));

            allocate(d_P, nnz);
            cudaMalloc(&pBuffer, pBufferSizeInBytes*sizeof(char));

            CUSPARSE_CHECK(cusparseCreateIdentityPermutation(
                handle,
                nnz,
                d_P));

            CUSPARSE_CHECK(cusparseXcoosortByRow(
                handle,
                m,
                n,
                nnz,
                rows,
                cols,
                d_P,
                pBuffer
            ));

            T* vals_sorted;
            allocate(vals_sorted, nnz);

            CUSPARSE_CHECK(cusparse_gthr<T>(
                handle,
                nnz,
                vals,
                vals_sorted,
                d_P
            ));

            cudaDeviceSynchronize();


            copy(vals, vals_sorted, nnz);

            cudaFree(d_P);
            cudaFree(vals_sorted);
            cudaFree(pBuffer);
            CUSPARSE_CHECK(cusparseDestroy(handle));
            cudaStreamDestroy(stream);
        }


        template<typename T>
         void coo_sort(COO<T> *in) {
             coo_sort<T>(in->n_rows, in->n_cols, in->nnz,
                     in->rows, in->cols, in->vals);
         }

        /**
         * Remove any zero values from a COO sparse matrix. The input arrays
         * are processed in chunks, where each GPU thread is assigned a chunk.
         * A required argument is an exclusive scan array providing the indices
         * to the compressed output array. The exclusive scan array should be
         * created from an array of non-zero value counts per chunk.
         *
         * @param n: number of nonzero values in COO
         * @param rows: input array of rows (size n)
         * @param cols: input array of cols (size n)
         * @param vals: input array of vals (size n)
         * @param crows: compressed array of rows
         * @param ccols: compressed array of cols
         * @param cvals: compressed array of vals
         * @param ex_scan: an exclusive scan of nnz counts for each chunk
         * @param ex_scan_n: number of elements (chunks) in ex_scan
         */
        template<int TPB_X, typename T>
        __global__ void coo_remove_zeros_kernel(
                const int *rows, const int *cols, const T *vals, int nnz,
                int *crows, int *ccols, T *cvals,
                int *ex_scan, int *cur_ex_scan, int m) {

            int row = (blockIdx.x * TPB_X) + threadIdx.x;

            if (row < m) {
                int start = cur_ex_scan[row];
                int stop = MLCommon::Sparse::get_stop_idx(row, m, nnz, cur_ex_scan);
                int cur_out_idx = ex_scan[row];

//                printf("row=%d, start=%d, stop=%d, cur_out_idx=%d\n", row, start, stop, cur_out_idx);

                for (int idx = start; idx < stop; idx++) {
                    if (vals[idx] != 0.0) {
                        crows[cur_out_idx] = rows[idx];
                        ccols[cur_out_idx] = cols[idx];
                        cvals[cur_out_idx] = vals[idx];
                        ++cur_out_idx;
                    }
                }
            }
        }




        /**
         * Count all the rows in the coo row array and place them in the
         * results matrix, indexed by row.
         *
         * @param rows the rows array of the coo matrix
         * @param nnz the size of the rows array
         * @param results array to place results
         * @param n number of rows in coo matrix
         */
        template<int TPB_X>
        __global__ void coo_row_count_kernel(int *rows, int nnz,
                int *results, int n) {
            int row = (blockIdx.x * TPB_X) + threadIdx.x;
            if(row < nnz) {
                atomicAdd(results+rows[row], 1);
            }
        }

        template<int TPB_X>
        void coo_row_count(int *rows, int nnz, int *results, int n) {
            dim3 grid_rc(MLCommon::ceildiv(nnz, TPB_X), 1, 1);
            dim3 blk_rc(TPB_X, 1, 1);

            coo_row_count_kernel<TPB_X><<<grid_rc,blk_rc>>>(
                    rows, nnz, results, n);
        }

        template<int TPB_X, typename T>
        void coo_row_count(COO<T> *in, int *results) {
            dim3 grid_rc(MLCommon::ceildiv(in->nnz, TPB_X), 1, 1);
            dim3 blk_rc(TPB_X, 1, 1);

            coo_row_count_kernel<TPB_X><<<grid_rc,blk_rc>>>(
                    in->rows, in->nnz, results, in->n_rows);
        }

        /**
         * Count all the rows with non-zero values in the coo row and val
         * arrays. Place the counts in the results matrix, indexed by row.
         *
         * @param rows the rows array of the coo matrix
         * @param vals the vals array of the coo matrix
         * @param nnz the size of rows / vals
         * @param results array to place resulting counts
         * @param n number of rows in coo matrix
         */
        template<int TPB_X, typename T>
        __global__ void coo_row_count_nz_kernel(int *rows, T *vals, int nnz,
                int *results, int n) {
            int row = (blockIdx.x * TPB_X) + threadIdx.x;
            if(row < nnz && vals[row] > 0.0) {
                atomicAdd(results+rows[row], 1);
            }
        }

        template<int TPB_X, typename T>
        void coo_row_count_nz(int *rows, T *vals, int nnz, int *results, int n) {
            dim3 grid_rc(MLCommon::ceildiv(nnz, TPB_X), 1, 1);
            dim3 blk_rc(TPB_X, 1, 1);

            coo_row_count_nz_kernel<TPB_X, T><<<grid_rc,blk_rc>>>(
                    rows, vals, nnz, results, n);
        }

        template<int TPB_X, typename T>
        void coo_row_count_nz(COO<T> *in, int *results) {
            dim3 grid_rc(MLCommon::ceildiv(in->nnz, TPB_X), 1, 1);
            dim3 blk_rc(TPB_X, 1, 1);

            coo_row_count_nz_kernel<TPB_X, T><<<grid_rc,blk_rc>>>(
                    in->rows, in->vals, in->nnz, results, in->n_rows);
        }



        /**
         * Removes the zeros from a COO formatted sparse matrix.
         *
         * @param rows: input array of rows (size n)
         * @param cols: input array of cols (size n)
         * @param vals: input array of vals (size n)
         * @param nnz: size of current rows/cols/vals arrays
         * @param crows: compressed array of rows
         * @param ccols: compressed array of cols
         * @param cvals: compressed array of vals
         * @param cnnz: array of non-zero counts per row
         * @param cur_nnz array of counts per row
         * @param cnnz_n: size of cnnz array (eg. num chunks)
         */
        template<int TPB_X, typename T>
        void coo_remove_zeros(
                const int *rows, const int *cols, const T *vals, int nnz,
                int *crows, int *ccols, T *cvals,
                int *cnnz, int *cur_cnnz, int n) {

            int *ex_scan, *cur_ex_scan;
            MLCommon::allocate(ex_scan, n, true);
            MLCommon::allocate(cur_ex_scan, n, true);

//            std::cout << "Allocated" << std::endl;

            thrust::device_ptr<int> dev_cnnz = thrust::device_pointer_cast(
                    cnnz);
            thrust::device_ptr<int> dev_ex_scan =
                    thrust::device_pointer_cast(ex_scan);
            thrust::exclusive_scan(dev_cnnz, dev_cnnz + n, dev_ex_scan);
            CUDA_CHECK(cudaPeekAtLastError());

            thrust::device_ptr<int> dev_cur_cnnz = thrust::device_pointer_cast(
                    cur_cnnz);
            thrust::device_ptr<int> dev_cur_ex_scan =
                    thrust::device_pointer_cast(cur_ex_scan);
            thrust::exclusive_scan(dev_cur_cnnz, dev_cur_cnnz + n, dev_cur_ex_scan);
            CUDA_CHECK(cudaPeekAtLastError());

            dim3 grid(ceildiv(n, TPB_X), 1, 1);
            dim3 blk(TPB_X, 1, 1);

            coo_remove_zeros_kernel<TPB_X><<<grid, blk>>>(
                    rows, cols, vals, nnz,
                    crows, ccols, cvals,
                    dev_ex_scan.get(), dev_cur_ex_scan.get(), n
            );

            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaFree(ex_scan));
            CUDA_CHECK(cudaFree(cur_ex_scan));
        }

        template<int TPB_X, typename T>
        void coo_remove_zeros(COO<T> *in,
                            COO<T> *out,
                            cudaStream_t stream) {

            int *row_count_nz, *row_count;

            MLCommon::allocate(row_count, in->n_rows, true);
            MLCommon::allocate(row_count_nz, in->n_rows, true);

            MLCommon::Sparse::coo_row_count<TPB_X>(
                    in->rows, in->nnz, row_count, in->n_rows);
            CUDA_CHECK(cudaPeekAtLastError());

            MLCommon::Sparse::coo_row_count_nz<TPB_X>(
                    in->rows, in->vals, in->nnz, row_count_nz, in->n_rows);
            CUDA_CHECK(cudaPeekAtLastError());

            thrust::device_ptr<int> d_row_count_nz =
                    thrust::device_pointer_cast(row_count_nz);
            int out_nnz = thrust::reduce(thrust::cuda::par.on(stream),
                    d_row_count_nz, d_row_count_nz+in->n_rows);

            out->allocate(out_nnz);

            coo_remove_zeros<TPB_X, T>(
                in->rows, in->cols, in->vals, in->nnz,
                out->rows, out->cols, out->vals,
                row_count_nz, row_count, in->n_rows);
            CUDA_CHECK(cudaPeekAtLastError());

            out->n_rows = in->n_rows;
            out->n_cols = in->n_cols;

            CUDA_CHECK(cudaFree(row_count));
            CUDA_CHECK(cudaFree(row_count_nz));
        }


        template<int TPB_X, typename T>
        __global__ void from_knn_graph_kernel(long *knn_indices, T *knn_dists, int m, int k,
                int *rows, int *cols, T *vals) {

            int row = (blockIdx.x * TPB_X) + threadIdx.x;
            if(row < m) {

                for(int i = 0; i < k; i++) {
                    rows[row*k+i] = row;
                    cols[row*k+i] = knn_indices[row*k+i];
                    vals[row*k+i] = knn_dists[row*k+i];
                }
            }
        }

        /**
         * Converts a knn graph into a COO format.
         */
        template<typename T>
        void from_knn(long *knn_indices, T *knn_dists, int m, int k,
                int *rows, int *cols, T *vals) {

            dim3 grid(ceildiv(m, 32), 1, 1);
            dim3 blk(32, 1, 1);
            from_knn_graph_kernel<32, T><<<grid, blk>>>(
                    knn_indices, knn_dists, m, k, rows, cols, vals);
        }

        template<typename T>
        void sorted_coo_to_csr(
                T *rows, int nnz, T *row_ind, int m) {

            T *row_counts;
            MLCommon::allocate(row_counts, m, true);

            dim3 grid(ceildiv(m, 32), 1, 1);
            dim3 blk(32, 1, 1);

            coo_row_count<32>(rows, nnz, row_counts, m);

            std::cout << MLCommon::arr2Str(row_counts, m, "row_counts");

            // create csr compressed row index from row counts
            thrust::device_ptr<T> row_counts_d = thrust::device_pointer_cast(row_counts);
            thrust::device_ptr<T> c_ind_d = thrust::device_pointer_cast(row_ind);
            exclusive_scan(row_counts_d, row_counts_d + m, c_ind_d);

            CUDA_CHECK(cudaFree(row_counts));
        }

        template<typename T>
        void sorted_coo_to_csr(COO<T> *coo, int *row_ind) {
            sorted_coo_to_csr(coo->rows, coo->nnz, row_ind, coo->n_rows);
        }

    }

}
