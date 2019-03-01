#include <gtest/gtest.h>

#include "hmm/determinant.h"
#include "hmm/magma/determinant.h"


template <typename T>
void run_cuda_det(int batchCount, T** dA_array, T* dDet_cusolver, bool is_hermitian){
        T **A_array, *Det_cusolver;
        A_array = (T **)malloc(sizeof(T*) * batchCount);
        Det_cusolver = (T *)malloc(sizeof(T) * batchCount);

        CUSOLVER_CHECK(cusolverDnCreate(&cusolverHandle));
        Det = Determinant<T>(nDim, &cusolverHandle);

        updateHost(A_array, dA_array, batchCount);

        for(int bId = 0; bId < batchCount; bId++) {
                Det_cusolver[bId] = Det.compute(A_array[bId], is_hermitian);
        }

        updateDevice(dDet_cusolver, Det_cusolver, batchCount)

        CUDA_CHECK(cudaFree(A_array));
        CUDA_CHECK(cudaFree(Det_cusolver));
        CUSOLVER_CHECK(cusolverDnDestroy(cusolverHandle));
        Det->TearDown();
}

template <typename T>
T run(magma_int_t n, magma_int_t batchCount)
{
// declaration:
        T **dA_array=NULL, *dDet_cusolver=NULL, *dDet_magma=NULL;
        magma_int_t ldda = magma_roundup(n, RUP_SIZE); // round up to multiple of 32 for best GPU performance
        T *error_d, error = 0;

// allocation:
        allocate_pointer_array(dA_array, ldda * n, batchCount);
        allocate(dDet_magma, batchCount);
        allocate(error, 1);

        int device = 0;  // CUDA device ID
        magma_queue_t queue;
        magma_queue_create(device, &queue);

// filling:
        fill_matrix_gpu_batched(n, n, batchCount, dA_array, ldda);

// computation magma :
        print_matrix_batched(n, n, batchCount, dA_array, ldda, "A array");

        det_batched(n, dA_array, ldda, dDet_magma, batchCount, queue);

        print_matrix_device(n, 1, dDet_magma, n, "det array");

// computation cusolver :
        run_cuda_det(batchCount, dA_array, dDet_magma, is_hermitian);

// Error
        meanSquaredError(error_d, dDet_cusolver, dDet_magma, batchCount);
        updateHost(&error, error_d, 1);

// cleanup:
        free_pointer_array(dA_array, batchCount);
        CUDA_CHECK(cudaFree(dDet_magma));
        CUDA_CHECK(cudaFree(dDet_cusolver));
        CUDA_CHECK(cudaFree(error_d));

        return error;
}


template <typename T>
struct DeterminantInputs {
        T tolerance;
        bool is_hermitian;
        magma_int_t n, batchCount;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const DeterminantInputs<T>& dims) {
        return os;
}

template <typename T>
class DeterminantTest : public ::testing::TestWithParam<DeterminantInputs<T> > {
protected:
void SetUp() override {
        params = ::testing::TestWithParam<DeterminantInputs<T> >::GetParam();
        tolerance = params.tolerance;

        magma_init();
        error = run<T>(params.n, params.batchCount);
        magma_finalize();
}

protected:
DeterminantInputs<T> params;
T error, tolerance;
};

const std::vector<DeterminantInputs<float> > DeterminantInputsf2 = {
        {0.000001f, true, 2, 4}
};

const std::vector<DeterminantInputs<double> > DeterminantInputsd2 = {
        {0.000001, true, 2, 4}
};


typedef DeterminantTest<float> DeterminantTestF;
TEST_P(DeterminantTestF, Result){
        EXPECT_LT(error, tolerance) << " error out of tol.";
}

typedef DeterminantTest<double> DeterminantTestD;
TEST_P(DeterminantTestD, Result){
        EXPECT_LT(error, tolerance) << " error out of tol.";
}

INSTANTIATE_TEST_CASE_P(DeterminantTests, DeterminantTestF,
                        ::testing::ValuesIn(DeterminantInputsf2));

INSTANTIATE_TEST_CASE_P(DeterminantTests, DeterminantTestD,
                        ::testing::ValuesIn(DeterminantInputsd2));
