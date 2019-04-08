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

#include "csr.h"
#include <gtest/gtest.h>
#include "sparse/csr.h"
#include "random/rng.h"
#include "test_utils.h"

#include <iostream>

namespace MLCommon {
namespace Sparse {

template <typename T>
class CSRTest : public ::testing::TestWithParam<CSRInputs<T>> {
protected:
  void SetUp() override {}

  void TearDown() override {}

protected:
  CSRInputs<T> params;
};

const std::vector<CSRInputs<float>> inputsf = {
  {5, 10, 5, 1234ULL}};

typedef CSRTest<float> CSRRowNormalizeL1;
TEST_P(CSRRowNormalizeL1, Result) {

    int *ex_scan;
    float *in_vals, *result, *verify;

    int ex_scan_h[4] = {0, 4, 8, 9 };
    float in_vals_h[10] = { 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0 };

    float verify_h[10] =  { 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1, 0.0 };

    allocate(in_vals, 10);
    allocate(verify, 10);
    allocate(ex_scan, 4);
    allocate(result, 10, true);

    updateDevice(ex_scan, *&ex_scan_h, 4);
    updateDevice(in_vals, *&in_vals_h, 10);
    updateDevice(verify, *&verify_h, 10);

    dim3 grid(ceildiv(10, 32), 1, 1);
    dim3 blk(32, 1, 1);

    csr_row_normalize_l1<32, float><<<grid, blk>>>(ex_scan, in_vals, 10, 4, result);
    cudaDeviceSynchronize();

    ASSERT_TRUE(devArrMatch<float>(verify, result, 10, Compare<float>()));

    CUDA_CHECK(cudaFree(ex_scan));
    CUDA_CHECK(cudaFree(in_vals));
    CUDA_CHECK(cudaFree(verify));
    CUDA_CHECK(cudaFree(result));
}

typedef CSRTest<float> CSRSum;
TEST_P(CSRSum, Result) {

    int *ex_scan, *ind_ptr_a, *ind_ptr_b;
    float *in_vals_a, *in_vals_b, *result, *verify;

    int ex_scan_h[4] = {0, 4, 8, 9 };

    int indptr_a_h[10] = { 1, 2, 3, 4, 1, 2, 3, 5, 0, 1 };
    int indptr_b_h[10] = { 1, 2, 5, 4, 0, 2, 3, 5, 1, 0 };

    float in_vals_h[10] = { 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 0.0 };

    float verify_h[10] =  { 2.0, 2.0, 0.5, 0.5, 0.2, 0.2, 0.0, 0.0, 0.2, 0.0 };

    allocate(in_vals_a, 10);
    allocate(in_vals_b, 10);
    allocate(verify, 10);
    allocate(ex_scan, 4);
    allocate(result, 10, true);

    allocate(ind_ptr_a, 10);
    allocate(ind_ptr_b, 10);

    updateDevice(ex_scan, *&ex_scan_h, 4);
    updateDevice(in_vals_a, *&in_vals_h, 10);
    updateDevice(in_vals_b, *&in_vals_h, 10);
    updateDevice(verify, *&verify_h, 10);
    updateDevice(ind_ptr_a, *&indptr_a_h, 10);
    updateDevice(ind_ptr_b, *&indptr_b_h, 10);

    int *result_ind;
    allocate(result_ind, 4);

    int nnz = 0;

    csr_add_calc_inds<float, 32>(
        ex_scan, ind_ptr_a, in_vals_a,
        ex_scan, ind_ptr_b, in_vals_b,
        10, 4,
        &nnz,
        result_ind
    );

    std::cout << MLCommon::arr2Str(result_ind, 4, "result_ind") << std::endl;
    std::cout << "final_nnz=" << nnz << std::endl;

    int *result_indptr;
    float *result_val;
    allocate(result_indptr, nnz);
    allocate(result_val, nnz);

    csr_add_finalize<float, 32>(
        ex_scan, ind_ptr_a, in_vals_a,
        ex_scan, ind_ptr_b, in_vals_b,
        10, 4,
        result_ind, result_indptr, result_val
    );
    std::cout << MLCommon::arr2Str(result_indptr, nnz, "result_intptr") << std::endl;

    std::cout << MLCommon::arr2Str(result_val, nnz, "result") << std::endl;

    ASSERT_TRUE(devArrMatch<float>(verify, result, 10, Compare<float>()));

    CUDA_CHECK(cudaFree(ex_scan));
    CUDA_CHECK(cudaFree(in_vals_a));
    CUDA_CHECK(cudaFree(in_vals_b));
    CUDA_CHECK(cudaFree(ind_ptr_a));
    CUDA_CHECK(cudaFree(ind_ptr_b));
    CUDA_CHECK(cudaFree(verify));
    CUDA_CHECK(cudaFree(result));
    CUDA_CHECK(cudaFree(result_indptr));
    CUDA_CHECK(cudaFree(result_val));
}


INSTANTIATE_TEST_CASE_P(CSRTests, CSRRowNormalizeL1,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(CSRTests, CSRSum,
                        ::testing::ValuesIn(inputsf));
}}

