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

#include "magma/b_bilinear.h"

using namespace MLCommon;


template <typename T>
T run_bilinear( magma_int_t m, magma_int_t n, magma_int_t batchCount)
{
// declaration:
        T **dA_array=NULL, **dX_array=NULL, **dY_array=NULL, *dO_naive=NULL, *dO_magma=NULL;
        magma_int_t ldda = magma_roundup(m, RUP_SIZE);
        magma_int_t lddx = m;
        magma_int_t lddy = n;
        T *error_d, error = 0;

// allocation:
        allocate_pointer_array(dA_array, ldda * n, batchCount);
        allocate_pointer_array(dX_array, m, batchCount);
        allocate_pointer_array(dY_array, n, batchCount);
        allocate(dO_magma, batchCount);
        allocate(dO_naive, batchCount);
        allocate(error_d, 1);

        int device = 0;  // CUDA device ID
        magma_queue_t queue;
        magma_queue_create(device, &queue);

// filling:
        fill_matrix_gpu_batched(m, n, batchCount, dA_array, ldda );
        fill_matrix_gpu_batched(m, 1, batchCount, dX_array, lddx );
        fill_matrix_gpu_batched(n, 1, batchCount, dY_array, lddy );

// computation:
        naive_bilinear_batched(m, n, dX_array, dA_array, ldda, dY_array, dO_naive, batchCount);

        bilinear_batched(m, n, dX_array, dA_array, ldda, dY_array, dO_magma, batchCount, queue);

        // Error
        meanSquaredError(error_d, dO_naive, dO_magma, batchCount);
        updateHost(&error, error_d, 1);

// cleanup:
        free_pointer_array(dA_array, batchCount);
        free_pointer_array(dX_array, batchCount);
        free_pointer_array(dY_array, batchCount);
        CUDA_CHECK(cudaFree(dO_naive));
        CUDA_CHECK(cudaFree(dO_magma));

        return error;
}

template <typename T>
struct BilinearInputs {
        T tolerance;
        magma_int_t m, n, batchCount;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const BilinearInputs<T>& dims) {
        return os;
}

template <typename T>
class BilinearTest : public ::testing::TestWithParam<BilinearInputs<T> > {
protected:
void SetUp() override {
        params = ::testing::TestWithParam<BilinearInputs<T> >::GetParam();
        tolerance = params.tolerance;

        magma_init();
        error = run_bilinear<T>(params.m, params.n, params.batchCount);
        magma_finalize();
}

protected:
BilinearInputs<T> params;
T error, tolerance;
};

const std::vector<BilinearInputs<float> > BilinearInputsf2 = {
        {0.000001f, 5, 2, 4}
};

const std::vector<BilinearInputs<double> > BilinearInputsd2 = {
        {0.000001, 5, 2, 4}
};


typedef BilinearTest<float> BilinearTestF;
TEST_P(BilinearTestF, Result){
        EXPECT_LT(error, tolerance) << " error out of tol.";
}

typedef BilinearTest<double> BilinearTestD;
TEST_P(BilinearTestD, Result){
        EXPECT_LT(error, tolerance) << " error out of tol.";
}

INSTANTIATE_TEST_CASE_P(BilinearTests, BilinearTestF,
                        ::testing::ValuesIn(BilinearInputsf2));

INSTANTIATE_TEST_CASE_P(BilinearTests, BilinearTestD,
                        ::testing::ValuesIn(BilinearInputsd2));
