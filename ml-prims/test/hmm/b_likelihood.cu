# pragma once

#include "hmm/magma/b_likelihood.h"

template <typename T>
void run(magma_int_t nCl, magma_int_t nDim, magma_int_t nObs)
{
// declaration:
        T *dX=NULL, *dmu=NULL, *dsigma=NULL, *dLlhd=NULL;
        T **dX_array=NULL, **dmu_array=NULL, **dsigma_array=NULL;

        magma_int_t lddx = magma_roundup(nDim, RUP_SIZE); // round up to multiple of 32 for best GPU performance
        magma_int_t lddmu = magma_roundup(nDim, RUP_SIZE); // round up to multiple of 32 for best GPU performance
        magma_int_t lddsigma = magma_roundup(nDim, RUP_SIZE); // round up to multiple of 32 for best GPU performance
        magma_int_t lddsigma_full = nDim * lddsigma;
        magma_int_t lddLlhd = magma_roundup(nCl, RUP_SIZE); // round up to multiple of 32 for best GPU performance

        // allocation:
        allocate(dX, lddx * nObs);
        allocate(dmu, lddmu * nCl);
        allocate(dsigma, lddsigma_full * nCl);
        allocate(dLlhd, lddLlhd * nObs);

        allocate(dX_array, lddx * nObs);
        allocate(dmu_array, lddmu * nCl);
        allocate(dsigma_array, lddsigma_full * nCl);

// filling:
        fill_matrix_gpu(nDim, nObs, dX, lddx);
        fill_matrix_gpu(nDim, nCl, dmu, lddmu);
        fill_matrix_gpu(lddsigma_full, nCl, dsigma, lddsigma_full);
        // TODO : zero out

        // print_matrix_device(lddsigma, nDim, dsigma, lddsigma, "dSigma matrix");

// Batching
        split_to_batches(nObs, dX_array, dX, lddx);
        split_to_batches(nCl, dmu_array, dmu, lddmu);
        split_to_batches(nCl, dsigma_array, dsigma, lddsigma_full);

        // print_matrix_batched(nDim, nDim, nCl, dsigma_array, lddsigma, "dSigma matrix2");


// computation:
        likelihood_batched(nCl, nDim, nObs,
                           dX_array, lddx,
                           dmu_array, lddmu,
                           dsigma_array, lddsigma_full, lddsigma,
                           dLlhd, lddLlhd);

// cleanup:
        CUDA_CHECK(cudaFree(dX));
        CUDA_CHECK(cudaFree(dmu));
        CUDA_CHECK(cudaFree(dsigma));
}


int main( int argc, char** argv )
{
        magma_init();


        magma_int_t nCl = 2;
        magma_int_t nDim = 3;
        magma_int_t nObs = 5;

        run<double>(nCl, nDim, nObs);

        magma_finalize();
        return 0;
}
