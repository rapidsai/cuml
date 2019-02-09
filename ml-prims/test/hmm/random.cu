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
#include "hmm/random.h"

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>

#define IDX(i,j,lda) ((i)+(j)*(lda))

namespace MLCommon {
namespace HMM {


template <typename T>
__global__ void naiveMatrixSumKernel(T* sums, T* matrix, int n_rows, int n_cols) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if(idx < n_rows) {
                for( int j = 0; j < n_cols; j = j + 1 ) {
                        sums[idx] = sums[idx] + matrix[IDX(idx, j, n_rows)];
                }
        }
}


template <typename T>
void naiveMatrixSum(T* sums, T* matrix, int n_rows, int n_cols) {
        static const int TPB = 64;
        int nblks = ceildiv(n_rows, TPB);
        naiveMatrixSumKernel<T><<<nblks,TPB>>>(sums, matrix, n_rows, n_cols);
        CUDA_CHECK(cudaPeekAtLastError());
}


template <typename T>
struct error_functor
{
        __host__ __device__
        T operator()(const T& x) const {
                return std::abs(x - 1);
        }
};


template <typename T>
T compute_error(T* sums, int n_rows){
        // transfer to device
        thrust::device_ptr<T> sums_th(sums);

        // setup arguments
        error_functor<T> unary_op;
        thrust::plus<T> binary_op;
        T init = 0;

        // compute error
        T error = thrust::transform_reduce(sums_th, sums_th + n_rows,
                                           unary_op, init, binary_op);
        return error;
}

template <typename T>
struct RandomMatrixInputs {
        T tolerance;
        int n_rows, n_cols;
        unsigned long long int seed;
        int random_start, random_end;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const RandomMatrixInputs<T>& dims) {
        return os;
}

template <typename T>
class RandomMatrixTest : public ::testing::TestWithParam<RandomMatrixInputs<T> > {
protected:
void SetUp() override {
        params = ::testing::TestWithParam<RandomMatrixInputs<T> >::GetParam();
        Random::Rng<T> r(params.seed);
        tolerance = params.tolerance;
        n_rows = params.n_rows;
        n_cols = params.n_cols;
        // T random_start = params.random_start;
        // T random_end = params.random_end;
        array_size = n_rows * n_cols;
        seed = params.seed;
        paramsRandom<T> paramsRd(params.random_start, params.random_end, params.seed);
        // paramsRd.start = params.random_start;
        // paramsRd.end = params.random_end;
        // paramsRd.seed = params.seed;

        // allocate memory
        allocate(random_matrix, array_size);
        allocate(sums, n_rows);

        thrust::device_ptr<T> sums_th(sums);
        thrust::fill(sums_th, sums_th + n_rows, (T) 0);

        MLCommon::HMM::gen_array(random_matrix, array_size, &paramsRd);
        MLCommon::HMM::normalize_matrix(random_matrix, n_rows, n_cols);
        naiveMatrixSum(sums, random_matrix, n_rows, n_cols);
        error = compute_error(sums, n_cols);
}

void TearDown() override {
        CUDA_CHECK(cudaFree(sums));
        CUDA_CHECK(cudaFree(random_matrix));
}

protected:
RandomMatrixInputs<T> params;
// random_matrix is generated with the primitive
// sums are the rowwize sums which should be equal to 1
T *random_matrix, *sums;
int array_size;
int n_rows, n_cols;
unsigned long long seed;
T error, tolerance;
};

const std::vector<RandomMatrixInputs<float> > inputsf2 = {
        {0.000001f, 5, 3, 1234ULL, 0, 1}
};

const std::vector<RandomMatrixInputs<double> > inputsd2 = {
        {0.000001, 5, 3, 1234ULL, 0, 1}
};


typedef RandomMatrixTest<float> RandomMatrixTestF;
TEST_P(RandomMatrixTestF, Result){
        EXPECT_LT(error, tolerance) << " error out of tol.";
}

typedef RandomMatrixTest<double> RandomMatrixTestD;
TEST_P(RandomMatrixTestD, Result){
        EXPECT_LT(error, tolerance) << " error out of tol.";
}

INSTANTIATE_TEST_CASE_P(RandomMatrixTests, RandomMatrixTestF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(RandomMatrixTests, RandomMatrixTestD,
                        ::testing::ValuesIn(inputsd2));

} // end namespace LinAlg
} // end namespace MLCommon
