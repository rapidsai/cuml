/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include "linalg/determinant.h"
#include "magma/b_determinant.h"
#include "magma/b_split.h"

using namespace MLCommon;

template <typename T>
void run_cuda_det(int n, int batchCount, T** dA_array, int ldda,
                  T* dDet_cusolver){

        T** dDet_cusolver_array, **Det_cusolver_array, **A_array;

        allocate(dDet_cusolver_array, batchCount);
        A_array = (T **)malloc(sizeof(T*) * batchCount);
        Det_cusolver_array = (T **)malloc(sizeof(T*) * batchCount);

        cusolverDnHandle_t cusolverHandle;
        CUSOLVER_CHECK(cusolverDnCreate(&cusolverHandle));

        size_t workspaceSize;
        determinantHandleCublas_t<T> detHandle;
        determinantCublas_bufferSize(detHandle,
                                     cusolverHandle,
                                     n, ldda,
                                     workspaceSize);

        T *workspace;
        CUDA_CHECK(cudaMalloc((void **)&workspace, workspaceSize));
        createDeterminantHandleCublas_t_new(detHandle, workspace);


        split_to_batches(batchCount, dDet_cusolver_array, dDet_cusolver, 1);

        print_matrix_batched(1, 1, batchCount, dDet_cusolver_array, 1, "dDet_cusolver_array");

        updateHost(Det_cusolver_array, dDet_cusolver_array, batchCount);
        updateHost(A_array, dA_array, batchCount);
        print_matrix_device(n, n,A_array[0], ldda, "A_array");

        for(int bId = 0; bId < batchCount; bId++) {
                det(Det_cusolver_array[bId],
                    n, A_array[bId], ldda,
                    cusolverHandle, detHandle);
        }

        // CUSOLVER_CHECK(cusolverDnDestroy(cusolverHandle));
}

template <typename T>
T run(magma_int_t n, magma_int_t batchCount, bool is_hermitian)
{
// declaration:
        T **dA_array=NULL, *dDet_cusolver=NULL, *dDet_magma=NULL;
        magma_int_t ldda = magma_roundup(n, RUP_SIZE); // round up to multiple of 32 for best GPU performance
        T *error_d, error = 0;

// allocation:
        allocate_pointer_array(dA_array, ldda * n, batchCount);
        allocate(dDet_magma, batchCount);
        allocate(dDet_cusolver, batchCount);
        allocate(error_d, 1);

        int device = 0;  // CUDA device ID
        magma_queue_t queue;
        magma_queue_create(device, &queue);

// filling:
        fill_matrix_gpu_batched(n, n, batchCount, dA_array, ldda);

// computation magma :
        size_t workspaceSize;

        determinantHandle_t<T> handle;
        determinant_bufferSize(handle,
                               n, ldda, batchCount,
                               workspaceSize);

        void *workspace;
        CUDA_CHECK(cudaMalloc((void **)&workspace, workspaceSize));
        createDeterminantHandle_t_new(handle, workspace);

        det_batched(n, dA_array, ldda, dDet_magma, batchCount, queue, handle);

// computation cusolver :
        run_cuda_det(n, batchCount, dA_array, ldda, dDet_cusolver);
        print_matrix_batched(n, n, batchCount, dA_array, ldda, "dA_array");
        print_matrix_device(batchCount, 1, dDet_cusolver, batchCount, "dets cusolver");
        print_matrix_device(batchCount, 1, dDet_magma, batchCount, "dets magma");

// Error
        meanSquaredError(error_d, dDet_cusolver, dDet_magma, batchCount);
        updateHost(&error, error_d, 1);



// cleanup:
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
        error = run<T>(params.n, params.batchCount, params.is_hermitian);
        magma_finalize();
}

protected:
DeterminantInputs<T> params;
T error, tolerance;
};

const std::vector<DeterminantInputs<float> > DeterminantInputsf2 = {
        {0.000001f, false, 2, 4}
};

const std::vector<DeterminantInputs<double> > DeterminantInputsd2 = {
        {0.000001, false, 2, 4}
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
