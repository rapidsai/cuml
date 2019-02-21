#include <cuda_runtime.h>

namespace MLCommon {


    template<int TPB_X, typename T>
    __global__ void csr_row_normalize_l1(
            const int *ia,    // csr row array (ex_scan of row counts)
            const T *vals, int nnz,  // array of values and number of nonzeros
            int m,          // num rows in csr
            int n,
            T *result) {    // output array

        // row-based matrix 1 thread per row
        int row = (blockIdx.x * TPB_X) + threadIdx.x;
        int i = row * n; // each thread processes one row of the dist matrix

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
                vals[j] /= sum;
        }
    }

}
