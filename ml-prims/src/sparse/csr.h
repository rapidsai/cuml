#include <cuda_runtime.h>
#include <stdio.h>

namespace MLCommon {


    template<int TPB_X, typename T>
    __global__ void csr_row_normalize_l1(
            int *ia,    // csr row counts (sorted by row)
            T *vals, int nnz,  // array of values and number of non-zeros
            int m,          // num rows in csr
            int n,
            T *result) {    // output array

        // row-based matrix 1 thread per row
        int row = (blockIdx.x * TPB_X) + threadIdx.x;

        // sum all vals for row and divide each val by sum
        if(row < m) {
            int start_idx = ia[row];
            int stop_idx = 0;
            if(row < (m-1))
                stop_idx = ia[row+1];
            else
                stop_idx = nnz;
            T sum = 0.0;
            for(int j = start_idx; j < stop_idx; j++)
                sum += vals[j];

            for(int j = start_idx; j < stop_idx; j++)
                if(sum > 0.0)
                    vals[j] /= sum;
                else
                    vals[j] = 0.0;
        }
    }
}
