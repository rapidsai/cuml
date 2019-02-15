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
#include "kmeans/reduce_rows_by_key.h"
#include "random/rng.h"
#include "test_utils.h"
#include <iostream>

namespace MLCommon {
namespace KMeans {


template <typename Type>
__global__ void naiveReduceRowsByKeyKernel(Type *d_A, int lda,
                          int *d_keys,char *d_char_keys, int nrows, 
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
void naiveReduceRowsByKey( int stream, Type* d_A, int lda, 
                          int *d_keys,char *d_char_keys, int nrows, 
                          int ncols, int nkeys, Type *d_sums) 
{
    cudaMemset(d_sums, 0, sizeof(Type) * nkeys*ncols);

    naiveReduceRowsByKeyKernel
          <<< dim3((ncols+31)/32, nkeys ),
              dim3(32, 1) >>>
                      (d_A,lda,d_keys,d_char_keys,nrows,ncols,nkeys,d_sums);
}


template <typename T>
struct ReduceRowsInputs {
    T tolerance;
    int nobs;
    int cols;
    int nkeys;
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
        Random::Rng<T> r(params.seed);
        Random::Rng<int> r_int(params.seed);
        int nobs = params.nobs;
        int cols = params.cols;
        int nkeys = params.nkeys;
        allocate(in1, nobs*cols);
        allocate(in2, nobs);
        allocate(chars2, nobs);
        allocate(out_ref, nkeys*cols);
        allocate(out, nkeys*cols);
        r.uniform(in1, nobs*cols, T(0.0), T(2.0/nobs));
        r_int.randInt(in2, nobs, 0, nkeys);
        naiveReduceRowsByKey(0, in1, cols, in2, chars2,
                               nobs, cols, nkeys, out_ref );
        reduce_rows_by_key(0, in1, cols, in2, chars2, 
                               nobs, cols, nkeys, out );
        /*
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count > 1) {
        	T *h_in1 = (T *) malloc(len * sizeof(T));
        	T *h_in2 = (T *) malloc(len * sizeof(T));
        	updateHost(h_in1, in1, len);
        	updateHost(h_in2, in2, len);
        	addMGColSplitTest(h_in1, h_in2);
        	free(h_in1);
        	free(h_in2);
        }
        */
    }
/*
    void addMGColSplitTest(T *h_in1, T *h_in2) {
    	int n_gpus = 2;

    	TypeMG<T> d_in1[n_gpus];
    	TypeMG<T> d_in2[n_gpus];
    	TypeMG<T> d_out[n_gpus];

    	for (int i = 0; i < n_gpus; i++) {
    		d_in1[i].gpu_id = i;
    		d_in2[i].gpu_id = i;
    		d_out[i].gpu_id = i;
    		CUDA_CHECK(cudaSetDevice(d_in1[i].gpu_id));
    		CUDA_CHECK(cudaStreamCreate(&(d_in1[i].stream)));
    		d_in2[i].stream = d_in1[i].stream;
    		d_out[i].stream = d_in1[i].stream;
    	}

    	int len = params.len;
    	allocateMG(d_in1, n_gpus, 1, len, true, true, false);
    	allocateMG(d_in2, n_gpus, 1, len, true, true, false);
    	allocateMG(d_out, n_gpus, 1, len, true, true, false);

    	updateDeviceMG(d_in1, h_in1, n_gpus, false);
    	updateDeviceMG(d_in1, h_in1, n_gpus, false);

    	addMG(d_out, d_in1, d_in2, len, n_gpus, false);

    	T *h_out = (T *) malloc(len * sizeof(T));
    	updateHostMG(h_out, d_out, n_gpus, false);

    	streamSyncMG(d_in1, n_gpus);
    	streamDestroyGPUs(d_in1, n_gpus);

    	freeMG(d_in1, n_gpus);
    	freeMG(d_in2, n_gpus);
    	freeMG(d_out, n_gpus);

    	allocate(out_2, len);
    	updateDevice(out_2, h_out, len);

    	free(h_out);
    }
*/

    void TearDown() override {
        CUDA_CHECK(cudaFree(in1));
        CUDA_CHECK(cudaFree(in2));
        CUDA_CHECK(cudaFree(chars2));
        CUDA_CHECK(cudaFree(out_ref));
        CUDA_CHECK(cudaFree(out));

        //if (device_count > 1) {
        // CUDA_CHECK(cudaFree(out_2));
        //}
    }

protected:
    ReduceRowsInputs<T> params;
    T *in1, *out_ref, *out, *out_2;
    int *in2;
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
