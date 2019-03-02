#include "hmm/gmm.h"

template <typename T>
void run(magma_int_t nCl, magma_int_t nDim, magma_int_t nObs)
{
// declaration:
        T *dX=NULL;
        T **dX_array=NULL;

        magma_int_t lddx = magma_roundup(nDim, RUP_SIZE);

        // allocation:
        allocate(dX, lddx * nObs);

        allocate(dX_array, nObs);

// filling:
        fill_matrix_gpu(nDim, nObs, dX, lddx);

// Batching
        split_to_batches(nObs, dX_array, dX, lddx);

// // computation:


// cleanup:
        CUDA_CHECK(cudaFree(dX));
        CUDA_CHECK(cudaFree(dX_array));
}


int main( int argc, char** argv )
{
        magma_init();


        magma_int_t nCl = 2;
        magma_int_t nDim = 3;
        magma_int_t nObs = 5;
        int isLog = false;

        run<double>(nCl, nDim, nObs);

        magma_finalize();
        return 0;
}
