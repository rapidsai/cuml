// #include "magma/bilinear.h"

#include "magma/magma_test_utils.h"
#include "magma/magma_batched_wrappers.h"

using namespace MLCommon;
using namespace MLCommon::LinAlg;


template <typename T>
__global__
void batched_diffs_kernel()){

}

template <typename T>
void create_llhd_batches(){
        for (size_t clId = 0; clId < nCl; clId++) {
                for (size_t obsId = 0; obsId < nObs; obsId++) {
                        dDiff_array[IDX(clId, obsId, nCl)] = dX +IDX(0, obsId, lddX);
                        dsigma_array[IDX(clId, obsId, nCl)] = dsigma + IDX(0, clId, lddsigma);
                }
        }

}

template <typename T>
__host__ __device__
T lol_llhd_atomic(T det, T bil, int nDim){
        return -0.5 * (std::log(det) + nDim * std::log(2 * M_PI) + bil);
}

template <typename T>
__global__
void log_llhd_kernel()){
        int i_start = threadIdx.x + blockDim.x * blockIdx.x;
        int j_start = threadIdx.y + blockDim.y * blockIdx.y;

        int numThreads_x = blockDim.x * blockIdx.x;
        int numThreads_y = blockDim.y * blockIdx.y;

        for (size_t i = i_start; i < nCl; i+=numThreads_x) {
                for (size_t j = j_start; j < nObs; j+=numThreads_y) {
                        idx = IDX(i, j, lddLlhd);
                        dLlhd[idx] = lol_llhd_atomic(det[j], bil[idx], nDim);
                }
        }
}

template <typename T>
void likelihood_batched_run(){
// Compute sigma inverses
        inverse_batched_magma(dim, dSigma_array, lddsigma, dinvSigma_array, nCl, queue);

// Compute sigma inv dets
        det_batched_magma(dim, dinvSigma_array, lddsigma, ddet_array, nCl, queue);

// Create batches
        create_llhd_batches();

// Compute diffs
        diffs_batched();

// Compute bilinears
        magma_int_t batchCount = nObs * nCl;
        T* dB;
        allocate(dB, batchCount);
        bilinear_batched(nDim, nDim, dDiff_array, dinvSigma_array, lddsigma, dDiff_array, dB, batchCount, queue);

// Compute log likelihoods
        dim3 block(32, 32, 1);
        dim3 grid(ceildiv(nCl, (int)block.x), ceildiv(nObs, (int)block.y), 1);
        log_llhd_batched_kernel<T> <<< grid, block >>>();
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
void run(magma_int_t nCl, magma_int_t nDim, magma_int_t nObs, magma_int_t batchCount)
{
// declaration:
        T *dX=NULL, *dmu=NULL, *dsigma=NULL, *drho=NULL;
        T **dX_array=NULL, **dmu_array=NULL, **dsigma_array=NULL, **drho_array=NULL;

        magma_int_t lddX = magma_roundup(nDim, RUP_SIZE); // round up to multiple of 32 for best GPU performance
        magma_int_t lddmu = magma_roundup(nDim, RUP_SIZE); // round up to multiple of 32 for best GPU performance
        magma_int_t lddsigma = magma_roundup(nDim * nDim, RUP_SIZE); // round up to multiple of 32 for best GPU performance
        // magma_int_t lddrho = magma_roundup(nDim, RUP_SIZE); // round up to multiple of 32 for best GPU performance

        // allocation:
        allocate(dX, lddX * nObs);
        allocate(dmu, lddmu * nCl);
        allocate(dsigma, lddmu * nCl);
        // allocate(drho, lddrho);

        allocate(dX_array, nObs * nCl);
        allocate(dmu_array, nObs * nCl);
        allocate(dsigma_array, nObs * nCl);
        // allocate(drho_array, nObs * nCl);

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);

// filling:
        fill_matrix_gpu(nDim, nObs, dX, lddX);
        fill_matrix_gpu(nDim, nCl, dmu, lddmu);
        fill_matrix_gpu(nDim * nDim, nCl, dsigma, lddsigma);
        fill_matrix_gpu(nCl, 1, drho, lddrho);

// computation:
        likelihood_batched_run();

// cleanup:
        CUDA_CHECK(cudaFree(dX));
        CUDA_CHECK(cudaFree(dmu));
        CUDA_CHECK(cudaFree(dsigma));
        CUDA_CHECK(cudaFree(drho));

}


int main( int argc, char** argv )
{
        magma_init();

        magma_int_t m = 2;
        magma_int_t n = 2;
        magma_int_t batchCount = 3;

        run<double>(m, n, batchCount);

        magma_finalize();
        return 0;
}
