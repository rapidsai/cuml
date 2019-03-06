/*
 * coo.h
 *
 *  Created on: Feb 13, 2019
 *      Author: cjnolet
 */

#pragma once

#include <cusparse_v2.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

#include "cuda_utils.h"
#include <cuda_runtime.h>

namespace MLCommon {

    template<typename T>
    void coo_sort(int m, int n, int nnz,
                  int *rows, int *cols, T *vals) {

        cusparseHandle_t handle = NULL;
        cudaStream_t stream = NULL;

        size_t pBufferSizeInBytes = 0;
        void *pBuffer = NULL;
        int *d_P = NULL;

        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

        cusparseCreate(&handle);

        cusparseSetStream(handle, stream);

        cusparseXcoosort_bufferSizeExt(
            handle,
            m,
            n,
            nnz,
            rows,
            cols,
            &pBufferSizeInBytes
        );

        cudaMalloc(&d_P, sizeof(int)*nnz);
        cudaMalloc(&pBuffer, sizeof(char)* pBufferSizeInBytes);

        cusparseCreateIdentityPermutation(
            handle,
            nnz,
            d_P);

        cusparseXcoosortByRow(
            handle,
            m,
            n,
            nnz,
            rows,
            cols,
            d_P,
            pBuffer
        );

        T* vals_sorted;
        allocate(vals_sorted, nnz);

        cusparseSgthr(
            handle,
            nnz,
            vals,
            vals_sorted,
            d_P,
            CUSPARSE_INDEX_BASE_ZERO
        );
        cudaDeviceSynchronize(); /* wait until the computation is done */

        copy(vals, vals_sorted, nnz);
        cudaDeviceSynchronize();

        cudaFree(d_P);
        cudaFree(vals_sorted);
        cudaFree(pBuffer);
        cusparseDestroy(handle);
        cudaStreamDestroy(stream);
    }

    /**
     * Remove any zero values from a COO sparse matrix. The input arrays
     * are processed in chunks, where each GPU thread is assigned a chunk.
     * A required argument is an exclusive scan array providing the indices
     * to the compressed output array. The exclusive scan array should be
     * created from an array of non-zero value counts per chunk.
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
    __global__ void coo_remove_zeros_kernel(int nnz,
            const int *rows, const int *cols, const T *vals,
            int *crows, int *ccols, T *cvals,
            int *ex_scan, int ex_scan_n) {

        int rows_per_chunk = int(nnz / ex_scan_n);

        int chunk = (blockIdx.x * TPB_X) + threadIdx.x;
        int i = chunk * rows_per_chunk; // 1 chunk per thread

        if (chunk < ex_scan_n) {
            int start = ex_scan[chunk];
            int cur_out_idx = start;

            for (int j = 0; j < rows_per_chunk; j++) {
                int idx = i + j;
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
     * Removes the zeros from a COO formatted sparse matrix.
     * @param nnz: size of rows/cols/vals arrays
     * @param rows: input array of rows (size n)
     * @param cols: input array of cols (size n)
     * @param vals: input array of vals (size n)
     * @param crows: compressed array of rows
     * @param ccols: compressed array of cols
     * @param cvals: compressed array of vals
     * @param cnnz: array of non-zero counts per chunk
     * @param cnnz_n: size of cnnz array (eg. num chunks)
     */
    template<int TPB_X, typename T>
    void coo_remove_zeros(int nnz,
            const int *rows, const int *cols, const T *vals,
            int *crows, int *ccols, T *cvals,
            int *cnnz, int cnnz_n) {

        int *ex_scan;
        MLCommon::allocate(ex_scan, cnnz_n);

        thrust::device_ptr<int> dev_cnnz = thrust::device_pointer_cast(
                cnnz);
        thrust::device_ptr<int> dev_ex_scan =
                thrust::device_pointer_cast(ex_scan);
        thrust::exclusive_scan(dev_cnnz, dev_cnnz + cnnz_n, dev_ex_scan);
        CUDA_CHECK(cudaPeekAtLastError());

        dim3 grid(ceildiv(cnnz_n, TPB_X), 1, 1);
        dim3 blk(TPB_X, 1, 1);

        coo_remove_zeros_kernel<TPB_X><<<grid, blk>>>(nnz, rows, cols, vals,
                crows, ccols, cvals, dev_ex_scan.get(), cnnz_n);

        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaFree(ex_scan));
    }
}
