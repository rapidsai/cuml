// #include "magma/bilinear.h"

#include "magma/magma_test_utils.h"
#include "magma/magma_batched_wrappers.h"

using namespace MLCommon;
using namespace MLCommon::LinAlg;

// Using cublas for large batch sizes and magma otherwise
// https://github.com/pytorch/pytorch/issues/13546
//
// template <typename T>
// void test_inverse(){
//         T alpha = 1, beta =0;
//         T** dO_array;
//         allocate_pointer_array(dO_array, ldda * n, batchCount);
//
//         magmablas_dgemm_batched(MagmaNoTrans, MagmaNoTrans, n, n, n, alpha, dA_array, ldda, dinvA_array, ldda, beta, **dO_array, ldda, batchCount, queue);
// }

template <typename T>
void inverse_batched_magma(magma_int_t n, T** dA_array, magma_int_t ldda,
                           T** dinvA_array, magma_int_t batchCount, magma_queue_t queue){

        int **dipiv_array, *info_array;
        allocate_pointer_array(dipiv_array, n, batchCount);
        allocate(info_array, batchCount);

        magma_getrf_batched(n, n, dA_array, ldda, dipiv_array, info_array,
                            batchCount, queue);
        assert_batched(batchCount, info_array);

        magma_dgetri_outofplace_batched (n, dA_array, ldda, dipiv_array,
                                         dinvA_array, ldda, info_array,
                                         batchCount, queue);
        assert_batched(batchCount, info_array);

        free_pointer_array(dipiv_array, batchCount);
        CUDA_CHECK(cudaFree(info_array));
}

template <typename T>
void inverse_batched(magma_int_t n, T** dA_array, magma_int_t ldda,
                     T** dinvA_array, magma_int_t batchCount, magma_queue_t queue){
        inverse_batched_magma(n, dA_array, ldda, dinvA_array, batchCount, queue);

}

template <typename T>
void run(magma_int_t m, magma_int_t n, magma_int_t batchCount)
{
// declaration:
        T **dA_array=NULL, **dinvA_array=NULL;
        magma_int_t ldda = magma_roundup(m, RUP_SIZE); // round up to multiple of 32 for best GPU performance

// allocation:
        allocate_pointer_array(dA_array, ldda * n, batchCount);
        allocate_pointer_array(dinvA_array, ldda * n, batchCount);

        int device = 0;  // CUDA device ID
        magma_queue_t queue;
        magma_queue_create(device, &queue);

// filling:
        fill_matrix_gpu_batched(m, n, batchCount, dA_array, ldda );

// computation:
        print_matrix_batched(m, n, batchCount, dA_array, ldda, "dA matrix");

        inverse_batched(n, dA_array, ldda, dinvA_array, batchCount, queue);

// cleanup:
        free_pointer_array(dA_array, batchCount);
        free_pointer_array(dinvA_array, batchCount);
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
