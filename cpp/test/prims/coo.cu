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

#include "coo.h"
#include <gtest/gtest.h>
#include "sparse/coo.h"
#include "random/rng.h"
#include "test_utils.h"

#include <iostream>

namespace MLCommon {
namespace Sparse {

template <typename T>
class COOTest : public ::testing::TestWithParam<COOInputs<T>> {
protected:
  void SetUp() override {}

  void TearDown() override {}

protected:
  COOInputs<T> params;
};

const std::vector<COOInputs<float>> inputsf = {
  {5, 10, 5, 1234ULL}};


typedef COOTest<float> SortedCOOToCSR;
TEST_P(SortedCOOToCSR, Result) {

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int nnz = 8;

    int *in, *out, *exp;

    int *in_h = new int[nnz]    { 0,   0,   1,   1,   2,   2,   3,   3 };
    int *exp_h = new int[4]   {0, 2, 4, 6 };

    allocate(in, nnz, true);
    allocate(exp, 4, true);
    allocate(out, 4, true);

    updateDevice(in, in_h, nnz, stream);
    updateDevice(exp, exp_h, 4, stream);

    sorted_coo_to_csr<int>(in, nnz, out, 4, stream);

    ASSERT_TRUE(devArrMatch<int>(out, exp, 4, Compare<int>()));

    cudaStreamDestroy(stream);

    delete in_h;
    delete exp_h;

    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(exp));
    CUDA_CHECK(cudaFree(out));

}


typedef COOTest<float> COOSymmetrize;
TEST_P(COOSymmetrize, Result) {

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int nnz = 8;

    int *in_rows_h = new int[nnz]    { 0,   0,   1,   1,   2,   2,   3,   3 };
    int *in_cols_h = new int[nnz]    { 1,   3,   2,   3,   0,   1,   0,   2 };
    float *in_vals_h = new float[nnz]{ 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0.5, 0.5 };

    int *exp_rows_h = new int[nnz*2]   {1,    0,    0,    0,    0,    1,    0,    0,    3,    1,    0,    0,    2,    0,    2,    3 };
    int *exp_cols_h = new int[nnz*2]   {0,    1,    0,    0,    3,    2,    0,    0,    1,    3,    0,    0,    1,    0,    3,    2 };
    float *exp_vals_h = new float[nnz*2] {0.5,  0.5,    0,    0,    1,  0.5,    0,    0,  0.5,  0.5,    0,    0,  0.5,    0,  0.5,  0.5 };

    COO<float> expected(exp_rows_h, exp_cols_h, exp_vals_h, nnz*2, 4, 4, false);

    COO<float> in(nnz, 4, 4);
    updateDevice(in.rows, *&in_rows_h, nnz, stream);
    updateDevice(in.cols, *&in_cols_h, nnz, stream);
    updateDevice(in.vals, *&in_vals_h, nnz, stream);

    COO<float> out;

    coo_symmetrize<32, float>(&in, &out,
            [] __device__ (int row, int col, float val, float trans) { return val+trans; },
            stream);

    ASSERT_TRUE(out.nnz == expected.nnz);
    ASSERT_TRUE(devArrMatch<int>(out.rows, expected.rows, out.nnz, Compare<int>()));
    ASSERT_TRUE(devArrMatch<int>(out.cols, expected.cols, out.nnz, Compare<int>()));
    ASSERT_TRUE(devArrMatch<float>(out.vals, expected.vals, out.nnz, Compare<float>()));

    cudaStreamDestroy(stream);

    delete in_rows_h;
    delete in_cols_h;
    delete in_vals_h;

    delete exp_rows_h;
    delete exp_cols_h;
    delete exp_vals_h;
}



typedef COOTest<float> COOSort;
TEST_P(COOSort, Result) {

    int *in_rows, *in_cols, *verify;
    float *in_vals;

    params = ::testing::TestWithParam<COOInputs<float>>::GetParam();
    Random::Rng r(params.seed);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(in_vals, params.nnz);
    r.uniform(in_vals, params.nnz, float(-1.0), float(1.0), stream);

    int *in_rows_h = (int*)malloc(params.nnz * sizeof(int));
    int *in_cols_h = (int*)malloc(params.nnz * sizeof(int));
    int *verify_h = (int*)malloc(params.nnz * sizeof(int));

    for(int i = 0; i < params.nnz; i++) {
        in_rows_h[i] = params.nnz-i-1;
        verify_h[i] = i;
        in_cols_h[i] = i;
    }

    allocate(in_rows, params.nnz);
    allocate(in_cols, params.nnz);
    allocate(verify, params.nnz);

    updateDevice(in_rows, in_rows_h, params.nnz, stream);

    updateDevice(in_cols, in_cols_h, params.nnz, stream);
    updateDevice(verify, verify_h, params.nnz, stream);

    coo_sort(params.m, params.n, params.nnz, in_rows, in_cols, in_vals);

    ASSERT_TRUE(devArrMatch<int>(verify, in_rows, params.nnz, Compare<int>()));

    free(in_rows_h);
    free(in_cols_h);
    free(verify_h);

    CUDA_CHECK(cudaFree(in_rows));
    CUDA_CHECK(cudaFree(in_cols));
    CUDA_CHECK(cudaFree(in_vals));
    CUDA_CHECK(cudaFree(verify));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

typedef COOTest<float> COORemoveZeros;
TEST_P(COORemoveZeros, Result) {

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    COO<float> in_h(params.nnz, 5, 5, false);
    COO<float> in(params.nnz, 5, 5);

    params = ::testing::TestWithParam<COOInputs<float>>::GetParam();

    Random::Rng r(params.seed);
    r.uniform(in.vals, params.nnz, float(-1.0), float(1.0), stream);

    updateHost(in_h.vals, in.vals, params.nnz, stream);

    in_h.vals[0] = 0;
    in_h.vals[2] = 0;
    in_h.vals[3] = 0;

    for(int i = 0; i < params.nnz; i++) {
        in_h.rows[i] = params.nnz-i-1;
        in_h.cols[i] = i;
    }

    updateDevice(in.rows, in_h.rows, params.nnz, stream);
    updateDevice(in.cols, in_h.cols, params.nnz, stream);
    updateDevice(in.vals, in_h.vals, params.nnz, stream);

    coo_sort<float>(&in);

    int out_rows_ref_h[2]  = { 0, 3 };
    int out_cols_ref_h[2] =  { 4, 1 };

    float *out_vals_ref_h = (float*)malloc(2*sizeof(float));
    out_vals_ref_h[0] = in_h.vals[4];
    out_vals_ref_h[1] = in_h.vals[1];

    COO<float> out_ref(2, 5, 5);
    COO<float> out;

    updateDevice(out_ref.rows, *&out_rows_ref_h, 2, stream);
    updateDevice(out_ref.cols, *&out_cols_ref_h, 2, stream);
    updateDevice(out_ref.vals, out_vals_ref_h, 2, stream);

    coo_remove_zeros<32, float>(&in, &out, stream);

    ASSERT_TRUE(devArrMatch<int>(out_ref.rows, out.rows, 2, Compare<int>()));
    ASSERT_TRUE(devArrMatch<int>(out_ref.cols, out.cols, 2, Compare<int>()));
    ASSERT_TRUE(devArrMatch<float>(out_ref.vals, out.vals, 2, Compare<float>()));

    CUDA_CHECK(cudaStreamDestroy(stream));
    free(out_vals_ref_h);
}


typedef COOTest<float> COORowCount;
TEST_P(COORowCount, Result) {

    int *in_rows,*verify, *results;

    int in_rows_h[5] = { 0, 0, 1, 2, 2 };
    int verify_h[5] = {2, 1, 2, 0, 0};

    allocate(in_rows, 5);
    allocate(verify, 5, true);
    allocate(results, 5, true);

    updateDevice(in_rows, *&in_rows_h, 5, 0);
    updateDevice(verify, *&verify_h, 5, 0);

    dim3 grid(ceildiv(5, 32), 1, 1);
    dim3 blk(32, 1, 1);
    coo_row_count<32>(in_rows, 5, results, 0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(devArrMatch<int>(verify, results, 5, Compare<int>()));

    CUDA_CHECK(cudaFree(in_rows));
    CUDA_CHECK(cudaFree(verify));
}

typedef COOTest<float> COORowCountNonzero;
TEST_P(COORowCountNonzero, Result) {

    int *in_rows,*verify, *results;
    float *in_vals;

    int in_rows_h[5] = { 0, 0, 1, 2, 2 };
    float in_vals_h[5] = { 0.0, 5.0, 0.0, 1.0, 1.0 };
    int verify_h[5] = {1, 0, 2, 0, 0};

    allocate(in_rows, 5);
    allocate(verify, 5, true);
    allocate(results, 5, true);
    allocate(in_vals, 5, true);

    updateDevice(in_rows, *&in_rows_h, 5, 0);
    updateDevice(verify, *&verify_h, 5, 0);
    updateDevice(in_vals, *&in_vals_h, 5, 0);

    dim3 grid(ceildiv(5, 32), 1, 1);
    dim3 blk(32, 1, 1);
    coo_row_count_nz<32, float>(in_rows, in_vals, 5, results);
    cudaDeviceSynchronize();

    ASSERT_TRUE(devArrMatch<int>(verify, results, 5, Compare<int>()));

    CUDA_CHECK(cudaFree(in_rows));
    CUDA_CHECK(cudaFree(verify));
}

INSTANTIATE_TEST_CASE_P(COOTests, SortedCOOToCSR,
                        ::testing::ValuesIn(inputsf));


INSTANTIATE_TEST_CASE_P(COOTests, COOSort,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(COOTests, COORemoveZeros,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(COOTests, COORowCount,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(COOTests, COORowCountNonzero,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(COOTests, COOSymmetrize,
                        ::testing::ValuesIn(inputsf));

}
}
