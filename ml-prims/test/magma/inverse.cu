// #include "magma/bilinear.h"

#include "magma/magma_test_utils.h"
#include "magma/magma_batched_wrappers.h"

using namespace MLCommon;
using namespace MLCommon::LinAlg;

// TODO : ADD batched cublas
// Using cublas for large batch sizes and magma otherwise
// https://github.com/pytorch/pytorch/issues/13546



template <typename T>
__global__ void ID_kernel (int n, T *A, int ldda) {
        int j = threadIdx.x + blockDim.x * blockIdx.x;
        int i = threadIdx.y + blockDim.y * blockIdx.y;
        if (i < n && j < n)
        {
                if (i == j)
                        A[IDX(i, j, ldda)] = 1.0;
                else
                        A[IDX(i, j, ldda)] = 0.0;
        }
}


template <typename T>
void make_ID_matrix(int n, T *A, int ldda) {
        dim3 block(32,32);
        dim3 grid(ceildiv(n, (int)block.x), ceildiv(n, (int)block.y));
        ID_kernel<T> <<< grid, block >>>(n, A, ldda);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
T test_inverse(magma_int_t n, T** dA_array, magma_int_t ldda, T** dinvA_array, magma_int_t batchCount, magma_queue_t queue){
        T alpha = 1, beta =0, mse=0;
        T **dO_array, **dIdM_array, *idMatrix;

// Allocate
        allocate_pointer_array(dO_array, ldda * n, batchCount);
        allocate(dIdM_array, batchCount);
        allocate(idMatrix, ldda * n);
        make_ID_matrix(n, idMatrix, ldda);
        fill_pointer_array(batchCount, dIdM_array, idMatrix);

// Compute error
        magmablas_gemm_batched(MagmaNoTrans, MagmaNoTrans, n, n, n, alpha, dA_array, ldda, dinvA_array, ldda, beta, dO_array, ldda, batchCount, queue);

        // print_matrix_batched(n, n, batchCount, dA_array, ldda, "dA_array");
        // print_matrix_batched(n, n, batchCount, dinvA_array, ldda, "dinvA_array");
        // print_matrix_batched(n, n, batchCount, dO_array, ldda, "O array");
        // print_matrix_batched(n, n, batchCount, dIdM_array, ldda, "idM array");
        // print_matrix_batched(n, n, batchCount, dO_array, ldda, "O array");
        mse = array_mse_batched(n, n, batchCount, dO_array, ldda, dIdM_array, ldda);

// free
        free_pointer_array(dO_array, batchCount);
        CUDA_CHECK(cudaFree(dIdM_array));
        CUDA_CHECK(cudaFree(idMatrix));

        return mse;
}

template <typename T>
void inverse_batched_magma(magma_int_t n, T** dA_array, magma_int_t ldda,
                           T**& dinvA_array, magma_int_t batchCount,
                           magma_queue_t queue){

        int **dipiv_array, *info_array;
        T **dA_array_cpy;
        allocate_pointer_array(dipiv_array, n, batchCount);
        allocate_pointer_array(dA_array_cpy, ldda * n, batchCount);
        allocate(info_array, batchCount);
        copy_batched(batchCount, dA_array_cpy, dA_array, ldda * n);

        magma_getrf_batched(n, n, dA_array_cpy, ldda, dipiv_array, info_array,
                            batchCount, queue);
        assert_batched(batchCount, info_array);

        magma_getri_outofplace_batched(n, dA_array_cpy, ldda, dipiv_array,
                                       dinvA_array, ldda, info_array,
                                       batchCount, queue);
        assert_batched(batchCount, info_array);

        free_pointer_array(dipiv_array, batchCount);
        free_pointer_array(dA_array_cpy, batchCount);
        CUDA_CHECK(cudaFree(info_array));
}

template <typename T>
void inverse_batched(magma_int_t n, T** dA_array, magma_int_t ldda,
                     T** dinvA_array, magma_int_t batchCount, magma_queue_t queue){
        inverse_batched_magma(n, dA_array, ldda, dinvA_array, batchCount, queue);

}

template <typename T>
void run(magma_int_t n, magma_int_t batchCount)
{
// declaration:
        T **dA_array=NULL, **dinvA_array=NULL;
        magma_int_t ldda = magma_roundup(n, RUP_SIZE); // round up to multiple of 32 for best GPU performance
        T error;

// allocation:
        allocate_pointer_array(dA_array, ldda * n, batchCount);
        allocate_pointer_array(dinvA_array, ldda * n, batchCount);

        int device = 0;  // CUDA device ID
        magma_queue_t queue;
        magma_queue_create(device, &queue);

// filling:
        fill_matrix_gpu_batched(n, n, batchCount, dA_array, ldda );

// computation:
        // print_matrix_batched(n, n, batchCount, dA_array, ldda, "A array");

        inverse_batched(n, dA_array, ldda, dinvA_array, batchCount, queue);

        // print_matrix_batched(n, n, batchCount, dA_array, ldda, "A array");
        // print_matrix_batched(n, n, batchCount, dinvA_array, ldda, "invA array");

// Error
        error = test_inverse(n, dA_array, ldda, dinvA_array, batchCount, queue);
        printf("Error : %f\n", (float) error);

// cleanup:
        free_pointer_array(dA_array, batchCount);
        free_pointer_array(dinvA_array, batchCount);
}


int main( int argc, char** argv )
{
        magma_init();

        magma_int_t n = 25;
        magma_int_t batchCount = 10;

        run<double>(n, batchCount);

        magma_finalize();
        return 0;
}
