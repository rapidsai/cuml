#include <gtest/gtest.h>

#include "hmm/magma/b_inverse.h"

using namespace MLCommon;


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
T run(magma_int_t n, magma_int_t batchCount)
{
        // declaration:
        T **dA_array=NULL, **dinvA_array=NULL;
        magma_int_t ldda = magma_roundup(n, RUP_SIZE);   // round up to multiple of 32 for best GPU performance
        T error;

        // allocation:
        allocate_pointer_array(dA_array, ldda * n, batchCount);
        allocate_pointer_array(dinvA_array, ldda * n, batchCount);

        int device = 0;    // CUDA device ID
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

        // cleanup:
        free_pointer_array(dA_array, batchCount);
        free_pointer_array(dinvA_array, batchCount);
        // printf("%f\n", (float) error );
        return error;
}


template <typename T>
struct BatchedInverseInputs {
        T tolerance;
        magma_int_t n, batchCount;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const BatchedInverseInputs<T>& dims) {
        return os;
}

template <typename T>
class BatchedInverseTest : public ::testing::TestWithParam<BatchedInverseInputs<T> > {
protected:
void SetUp() override {
        params = ::testing::TestWithParam<BatchedInverseInputs<T> >::GetParam();
        tolerance = params.tolerance;

        magma_init();
        error = run<T>(params.n, params.batchCount);
        magma_finalize();
}

protected:
BatchedInverseInputs<T> params;
T error, tolerance;
};

const std::vector<BatchedInverseInputs<float> > BatchedInverseInputsf2 = {
        {0.000001f, 2, 4}
};

const std::vector<BatchedInverseInputs<double> > BatchedInverseInputsd2 = {
        {0.000001, 2, 4}
};


typedef BatchedInverseTest<float> BatchedInverseTestF;
TEST_P(BatchedInverseTestF, Result){
        EXPECT_LT(error, tolerance) << " error out of tol.";
}

typedef BatchedInverseTest<double> BatchedInverseTestD;
TEST_P(BatchedInverseTestD, Result){
        EXPECT_LT(error, tolerance) << " error out of tol.";
}

INSTANTIATE_TEST_CASE_P(BatchedInverseTests, BatchedInverseTestF,
                        ::testing::ValuesIn(BatchedInverseInputsf2));

INSTANTIATE_TEST_CASE_P(BatchedInverseTests, BatchedInverseTestD,
                        ::testing::ValuesIn(BatchedInverseInputsd2));
