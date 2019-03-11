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

typedef COOTest<float> COOSort;
TEST_P(COOSort, Result) {

    int *in_rows, *in_cols, *verify;
    float *in_vals;

    params = ::testing::TestWithParam<COOInputs<float>>::GetParam();
    Random::Rng r(params.seed);
    allocate(in_vals, params.nnz);
    r.uniform(in_vals, params.nnz, float(-1.0), float(1.0));

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

    updateDevice(in_rows, in_rows_h, params.nnz);

    updateDevice(in_cols, in_cols_h, params.nnz);
    updateDevice(verify, verify_h, params.nnz);

    coo_sort(params.m, params.n, params.nnz, in_rows, in_cols, in_vals);

    ASSERT_TRUE(devArrMatch<int>(verify, in_rows, params.nnz, Compare<int>()));

    CUDA_CHECK(cudaFree(in_rows));
    CUDA_CHECK(cudaFree(in_cols));
    CUDA_CHECK(cudaFree(in_vals));
    CUDA_CHECK(cudaFree(verify));
}

typedef COOTest<float> COORemoveZeros;
TEST_P(COORemoveZeros, Result) {

    int *in_rows, *in_cols;
    float *in_vals;

    params = ::testing::TestWithParam<COOInputs<float>>::GetParam();
    Random::Rng r(params.seed);
    allocate(in_vals, params.nnz);
    r.uniform(in_vals, params.nnz, float(-1.0), float(1.0));

    float *vals_h = (float*)malloc(params.nnz * sizeof(float));
    updateHost(vals_h, in_vals, params.nnz);

    vals_h[0] = 0;
    vals_h[2] = 0;
    vals_h[3] = 0;

    updateDevice(in_vals, vals_h, params.nnz);

    int out_rows_ref_h[2]  = { 3, 0 };
    int out_cols_ref_h[2] =  { 1, 4 };

    float *out_vals_ref_h = (float*)malloc(2*sizeof(float));
    out_vals_ref_h[0] = vals_h[1];
    out_vals_ref_h[1] = vals_h[4];

    int *out_rows, *out_cols, *out_rows_ref, *out_cols_ref;
    float *out_vals, *out_vals_ref;
    allocate(out_rows, 2);
    allocate(out_cols, 2);
    allocate(out_vals, 2);
    allocate(out_rows_ref, 2);
    allocate(out_vals_ref, 2);
    allocate(out_cols_ref, 2);

    updateDevice(out_rows_ref, *&out_rows_ref_h, 2);
    updateDevice(out_cols_ref, *&out_cols_ref_h, 2);
    updateDevice(out_vals_ref, out_vals_ref_h, 2);

    int *cnnz, cnnz_h[5] = { 0, 1, 0, 0, 1 };
    allocate(cnnz, 5);

    updateDevice(cnnz, *&cnnz_h, 5);

    int *in_rows_h = (int*)malloc(params.nnz * sizeof(int));
    int *in_cols_h = (int*)malloc(params.nnz * sizeof(int));

    for(int i = 0; i < params.nnz; i++) {
        in_rows_h[i] = params.nnz-i-1;
        in_cols_h[i] = i;
    }

    allocate(in_rows, params.nnz);
    allocate(in_cols, params.nnz);

    updateDevice(in_rows, in_rows_h, params.nnz);
    updateDevice(in_cols, in_cols_h, params.nnz);

    coo_remove_zeros<16, float>(params.nnz, in_rows, in_cols, in_vals, out_rows, out_cols, out_vals, cnnz, 5);

    ASSERT_TRUE(devArrMatch<int>(out_rows_ref, out_rows, 2, Compare<int>()));
    ASSERT_TRUE(devArrMatch<int>(out_cols_ref, out_cols, 2, Compare<int>()));
    ASSERT_TRUE(devArrMatch<float>(out_vals_ref, out_vals, 2, Compare<float>()));

    CUDA_CHECK(cudaFree(in_rows));
    CUDA_CHECK(cudaFree(in_cols));
    CUDA_CHECK(cudaFree(in_vals));
    CUDA_CHECK(cudaFree(out_rows));
    CUDA_CHECK(cudaFree(out_cols));
    CUDA_CHECK(cudaFree(out_vals));
    CUDA_CHECK(cudaFree(out_rows_ref));
    CUDA_CHECK(cudaFree(out_cols_ref));
    CUDA_CHECK(cudaFree(out_vals_ref));
    free(out_vals_ref_h);
    free(in_rows_h);
    free(in_cols_h);
    free(vals_h);
}


typedef COOTest<float> COORowCount;
TEST_P(COORowCount, Result) {

    int *in_rows,*verify, *results;

    int in_rows_h[5] = { 0, 0, 1, 2, 2 };
    int verify_h[5] = {2, 1, 2, 0, 0};

    allocate(in_rows, 5);
    allocate(verify, 5, true);
    allocate(results, 5, true);

    updateDevice(in_rows, *&in_rows_h, 5);
    updateDevice(verify, *&verify_h, 5);

    dim3 grid(ceildiv(5, 32), 1, 1);
    dim3 blk(32, 1, 1);
    coo_row_count<32, float><<<grid, blk>>>(in_rows, 5, results, 5);

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

    updateDevice(in_rows, *&in_rows_h, 5);
    updateDevice(verify, *&verify_h, 5);
    updateDevice(in_vals, *&in_vals_h, 5);

    dim3 grid(ceildiv(5, 32), 1, 1);
    dim3 blk(32, 1, 1);
    coo_row_count_nz<32, float><<<grid, blk>>>(in_rows, in_vals, 5, results, 5);

    ASSERT_TRUE(devArrMatch<int>(verify, results, 5, Compare<int>()));

    CUDA_CHECK(cudaFree(in_rows));
    CUDA_CHECK(cudaFree(verify));
}


INSTANTIATE_TEST_CASE_P(COOTests, COOSort,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(COOTests, COORemoveZeros,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(COOTests, COORowCount,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(COOTests, COORowCountNonzero,
                        ::testing::ValuesIn(inputsf));
