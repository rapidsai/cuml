
#pragma once
#include "utils.h"
#include <cusparse.h>

// From https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-generic-spmat-create-csr
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed with error (%d) at line %d\n",             \
               status, __LINE__);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}


typedef cusparseSpMatDescr_t CSR_t;
typedef cusparseDnMatDescr_t Dense_t;
typedef cusparseHandle_t Sparse_handle_t;


// Functions
namespace Sparse_ {

// Create cuSPARSE handle
#define createHandle(handle) CHECK_CUSPARSE( cusparseCreate(handle) );
#define destroyHandle(handle) CHECK_CUSPARSE( cusparseDestroy(handle) );


// Creates a CSR matrix
void createCSR(
    CSR_t *CSR_Matrix,
    const int n,
    const int p,
    const int NNZ,
    float * __restrict__ VAL,
    float * __restrict__ COL,
    float * __restrict__ ROW)
{
    CHECK_CUSPARSE(
        cusparseCreateCsr(
            CSR_Matrix, 
            (int64_t)n, (int64_t)p, (int64_t)NNZ,
            ROW, COL, VAL,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F)
        );
}
// Deletes CSR matrix
#define destroyCSR(CSR_Matrix) CHECK_CUSPARSE( cusparseDestroySpMat(CSR_Matrix) )


// Creates a Dense Matrix descriptor for use in sparse operations
void createDense(
    Dense_t *Dense_Matrix,
    const int n,
    const int p,
    float * __restrict__ data)
{
    CHECK_CUSPARSE(
        cusparseCreateDnMat(
            Dense_Matrix,
            (int64_t)n, (int64_t)p, (int64_t)n,
            data, CUDA_R_32F, 
            CUSPARSE_ORDER_COL)
        );
}
// Deletes Dense Matrix
#define destroyDense(Dense_Matrix) CHECK_CUSPARSE( cusparseDestroyDnMat(Dense_Matrix) )



// Create Buffer for spMM
void *createBuffer(
    Sparse_handle_t handle,
    const CSR_t CSR_Matrix,
    const Dense_t Dense_Matrix,
    Dense_t Output_Matrix)
{
    const float alpha = 1.0f;
    const float beta = 1.0f;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(
        cusparseSpMM_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            (const void*) &alpha,
            CSR_Matrix,
            Dense_Matrix,
            (const void*) &beta,
            Output_Matrix,
            CUDA_R_32F,
            CUSPARSE_MV_ALG_DEFAULT,
            &bufferSize)
        );

    void *buffer;
    cuda_malloc(buffer, bufferSize);
    return buffer;
}
// Destory buffer
#define destoryBuffer(buffer) cuda_free(buffer);



// Sparse Matrix @ Dense Matrix Multiply
void spMM(
    Sparse_handle_t handle,
    const CSR_t CSR_Matrix,
    const Dense_t Dense_Matrix,
    const float alpha,
    const float beta,
    Dense_t Output_Matrix,
    void *buffer)
{
    CHECK_CUSPARSE(
        cusparseSpMM(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            (const void*) &alpha,
            CSR_Matrix,
            Dense_Matrix,
            (const void*) &beta,
            Output_Matrix,
            CUDA_R_32F,
            CUSPARSE_MV_ALG_DEFAULT,
            buffer)
        );
}

// end namespace
}