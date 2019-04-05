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
#include "linalg/reduce_rows_by_key.h"
#include "random/rng.h"
#include "test_utils.h"
#include <iostream>

namespace MLCommon {
namespace LinAlg {


template <typename Type>
__global__ void naiveReduceRowsByKeyKernel(Type *d_A, int lda,
                          uint32_t *d_keys,char *d_char_keys, int nrows, 
                          int ncols, int nkeys, Type *d_sums) 
{
    int c=threadIdx.x + blockIdx.x*blockDim.x;
    if (c >= ncols) return;
    int this_key = threadIdx.y+blockIdx.y*blockDim.y;
    
    Type sum = 0.0;
    for (int r=0;r<nrows;r++) {
       if (this_key != d_keys[r]) continue;
       sum += d_A[lda*r+c];
    }
    d_sums[this_key*ncols + c] = sum;
}
template <typename Type>
void naiveReduceRowsByKey( Type* d_A, int lda, 
                          uint32_t *d_keys,char *d_char_keys, int nrows, 
                          int ncols, int nkeys, Type *d_sums, cudaStream_t stream)
{
    cudaMemset(d_sums, 0, sizeof(Type) * nkeys*ncols);

    naiveReduceRowsByKeyKernel
          <<< dim3((ncols+31)/32, nkeys ),
              dim3(32, 1), 0, stream>>>
                      (d_A,lda,d_keys,d_char_keys,nrows,ncols,nkeys,d_sums);
}


template <typename T>
struct ReduceRowsInputs {
    T tolerance;
    int nobs;
    uint32_t cols;
    uint32_t nkeys;
    unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const ReduceRowsInputs<T>& dims) {
    return os;
}

template <typename T>
class ReduceRowTest: public ::testing::TestWithParam<ReduceRowsInputs<T> > {
protected:
    void SetUp() override {
        params = ::testing::TestWithParam<ReduceRowsInputs<T>>::GetParam();
        Random::Rng r(params.seed);
        Random::Rng r_int(params.seed);
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        int nobs = params.nobs;
        uint32_t cols = params.cols;
        uint32_t nkeys = params.nkeys;
        allocate(in1, nobs*cols);
        allocate(in2, nobs);
        allocate(chars2, nobs);
        allocate(out_ref, nkeys*cols);
        allocate(out, nkeys*cols);
        r.uniform(in1, nobs*cols, T(0.0), T(2.0/nobs), stream);
        r_int.uniformInt(in2, nobs, (uint32_t)0, nkeys, stream);
        naiveReduceRowsByKey(in1, cols, in2, chars2,
                               nobs, cols, nkeys, out_ref, stream);
        reduce_rows_by_key(in1, cols, in2, chars2, 
                               nobs, cols, nkeys, out, stream);
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(in1));
        CUDA_CHECK(cudaFree(in2));
        CUDA_CHECK(cudaFree(chars2));
        CUDA_CHECK(cudaFree(out_ref));
        CUDA_CHECK(cudaFree(out));

    }

protected:
    ReduceRowsInputs<T> params;
    T *in1, *out_ref, *out, *out_2;
    uint32_t *in2;
    char *chars2;
    int device_count = 0;
};

// ReduceRowTestF
// 128 Obs, 32 cols, 6 clusters
const std::vector<ReduceRowsInputs<float> > inputsf2 = {
    {0.000001f, 128, 32, 6, 1234ULL}
};
typedef ReduceRowTest<float> ReduceRowTestF;
TEST_P(ReduceRowTestF, Result) {
    ASSERT_TRUE(devArrMatch(out_ref, out, params.cols*params.nkeys,
                            CompareApprox<float>(params.tolerance)));
    //if (device_count > 1) {
   // 	ASSERT_TRUE(devArrMatch(out_ref, out_2, params.len,
    //	                    CompareApprox<double>(params.tolerance)));
    //}
}
INSTANTIATE_TEST_CASE_P(ReduceRowTests, ReduceRowTestF,
                        ::testing::ValuesIn(inputsf2));

// ReduceRowTestD
// 128 Obs, 32 cols, 6 clusters, double precision 
const std::vector<ReduceRowsInputs<double> > inputsd2 = {
    {0.00000001, 128, 32, 6, 1234ULL}
};
typedef ReduceRowTest<double> ReduceRowTestD;
TEST_P(ReduceRowTestD, Result){
    ASSERT_TRUE(devArrMatch(out_ref, out, params.cols*params.nkeys,
                            CompareApprox<double>(params.tolerance)));

}
INSTANTIATE_TEST_CASE_P(ReduceRowTests, ReduceRowTestD,
                        ::testing::ValuesIn(inputsd2));

// ReduceRowTestSmallnKey
// 128 Obs, 32 cols, 3 clusters
const std::vector<ReduceRowsInputs<float> > inputsf_small_nkey = {
    {0.000001f, 128, 32, 3, 1234ULL}
};
typedef ReduceRowTest<float> ReduceRowTestSmallnKey;
TEST_P(ReduceRowTestSmallnKey, Result){
    ASSERT_TRUE(devArrMatch(out_ref, out, params.cols*params.nkeys,
                            CompareApprox<float>(params.tolerance)));

}
INSTANTIATE_TEST_CASE_P(ReduceRowTests, ReduceRowTestSmallnKey,
                        ::testing::ValuesIn(inputsf_small_nkey));

// ReduceRowTestBigSpace
// 512 Obs, 1024 cols, 32 clusters, double precision 
const std::vector<ReduceRowsInputs<double> > inputsd_big_space = {
    {0.00000001, 512, 1024, 40, 1234ULL}
};
typedef ReduceRowTest<double> ReduceRowTestBigSpace;
TEST_P(ReduceRowTestBigSpace, Result){
    ASSERT_TRUE(devArrMatch(out_ref, out, params.cols*params.nkeys,
                            CompareApprox<double>(params.tolerance)));

}
INSTANTIATE_TEST_CASE_P(ReduceRowTests, ReduceRowTestBigSpace,
                        ::testing::ValuesIn(inputsd_big_space));

// ReduceRowTestManyObs
// 100000 Obs, 37 cols, 32 clusters
const std::vector<ReduceRowsInputs<float> > inputsf_many_obs = {
    {0.00001f, 100000, 37, 32, 1234ULL}
};
typedef ReduceRowTest<float> ReduceRowTestManyObs;
TEST_P(ReduceRowTestManyObs, Result){
    ASSERT_TRUE(devArrMatch(out_ref, out, params.cols*params.nkeys,
                            CompareApprox<float>(params.tolerance)));

}
INSTANTIATE_TEST_CASE_P(ReduceRowTests, ReduceRowTestManyObs,
                        ::testing::ValuesIn(inputsf_many_obs));

} // end namespace LinAlg
} // end namespace MLCommon
