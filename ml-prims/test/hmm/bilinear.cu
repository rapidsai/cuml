/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>

#include "hmm/bilinear.h"
// #include "hmm/utils.h"
#include "cuda_utils.h"

#define IDX(i,j,lda) ((i)+(j)*(lda))

namespace MLCommon {
namespace HMM {


template <typename T>
T naiveBilinear(T* mat, int dim, T* x) {
        T result = 0;
        for (int i = 0; i < dim; ++i)
                for (int j = 0; j < dim; ++j)
                        result += mat[IDX(i, j, dim)] * x[i] * x[j];
        return result;
}


template <typename T>
T compute_error(T *matrix, T *x, T *matrix_d, T *x_d, int dim, cublasHandle_t cublas_handle){
        T true_val;
        true_val = naiveBilinear(matrix, dim, x);
        T computed, error;

        bilinear(matrix_d, dim, x_d, cublas_handle, &computed);
        error = std::abs(computed - true_val);
        return error;
}

template <typename T>
struct BilinearInputs {
        T tolerance;
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

        cublasHandle_t cublas_handle;
        CUBLAS_CHECK(cublasCreate(&cublas_handle));

        // allocate memory
        allocate(matrix_d, dim*dim);
        allocate(x_d, dim);

        // cpu mallocs
        matrix = (T *)malloc(dim * dim * sizeof(T));
        x = (T *)malloc(dim * sizeof(T));
        matrix[0] = 5.0;
        matrix[1] = 1.0;
        matrix[2] = 6.0;
        matrix[3] = 10.0;
        x[0] = 2.0;
        x[1] = 1.0;

        // copy data to gpu (available in ml-common/cuda_utils.h)
        updateDevice(matrix_d, matrix, dim * dim);
        updateDevice(x_d, x, dim);

        error = compute_error(matrix, x, matrix_d, x_d, dim, cublas_handle);
}

void TearDown() override {
        free(matrix);
        free(x);

        CUDA_CHECK(cudaFree(matrix_d));
        CUDA_CHECK(cudaFree(x_d));
}

protected:
BilinearInputs<T> params;
// random_matrix is generated with the primitive
// sums are the rowwize sums which should be equal to 1
T *matrix, *x, *matrix_d, *x_d;
int dim = 2;
unsigned long long seed;
T error, tolerance;
};

const std::vector<BilinearInputs<float> > inputsf2 = {
        {0.000001f}
};

const std::vector<BilinearInputs<double> > inputsd2 = {
        {0.000001}
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
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(BilinearTests, BilinearTestD,
                        ::testing::ValuesIn(inputsd2));

} // end namespace LinAlg
} // end namespace MLCommon
