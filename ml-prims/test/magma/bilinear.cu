#include "magma/bilinear.h"

using namespace MLCommon;



// ------------------------------------------------------------
// Solve dA * dX = dB, where dA and dX are stored in GPU device memory.
// Internally, MAGMA uses a hybrid CPU + GPU algorithm.
template <typename T>
void run_bilinear( magma_int_t m, magma_int_t n, magma_int_t batchCount)
{
// declaration:
        T **dA_array=NULL, **dX_array=NULL, **dY_array=NULL, *dO=NULL;
        magma_int_t ldda = magma_roundup(m, RUP_SIZE); // round up to multiple of 32 for best GPU performance
        magma_int_t lddx = m;
        magma_int_t lddy = n;

// allocation:
        allocate_pointer_array(dA_array, ldda * n, batchCount);
        allocate_pointer_array(dX_array, m, batchCount);
        allocate_pointer_array(dY_array, n, batchCount);
        allocate(dO, batchCount);

        int device = 0;  // CUDA device ID
        magma_queue_t queue;
        magma_queue_create(device, &queue);

// filling:
        fill_matrix_gpu_batched(m, n, batchCount, dA_array, ldda );
        fill_matrix_gpu_batched(m, 1, batchCount, dX_array, lddx );
        fill_matrix_gpu_batched(n, 1, batchCount, dY_array, lddy );

// computation:
        print_matrix_batched(m, n, batchCount, dA_array, ldda, "dA matrix");
        print_matrix_batched(m, 1, batchCount, dX_array, lddx, "dX matrix");
        print_matrix_batched(n, 1, batchCount, dY_array, lddy, "dY matrix");

        naive_bilinear(m, n, dX_array, dA_array, ldda, dY_array, dO, batchCount);
        print_matrix_device(batchCount, 1, dO, batchCount, "dO matrix");

        bilinear(m, n, dX_array, dA_array, ldda, dY_array, dO, batchCount, queue);
        print_matrix_device(batchCount, 1, dO, batchCount, "dO matrix");

// cleanup:
        free_pointer_array(dA_array, batchCount);
        free_pointer_array(dX_array, batchCount);
        free_pointer_array(dY_array, batchCount);
}


int main( int argc, char** argv )
{
        magma_init();

        magma_int_t m = 2;
        magma_int_t n = 2;
        magma_int_t batchCount = 3;

        run_bilinear<double>(m, n, batchCount);

        magma_finalize();
        return 0;
}
