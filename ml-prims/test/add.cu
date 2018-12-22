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
#include "linalg/add.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace LinAlg {


template <typename Type>
__global__ void naiveAddElemKernel(Type* out, const Type* in1, const Type* in2,
                               int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len) {
        out[idx] = in1[idx] + in2[idx];
    }
}

template <typename Type>
void naiveAddElem(Type* out, const Type* in1, const Type* in2, int len) {
    static const int TPB = 64;
    int nblks = ceildiv(len, TPB);
    naiveAddElemKernel<Type><<<nblks,TPB>>>(out, in1, in2, len);
    CUDA_CHECK(cudaPeekAtLastError());
}


template <typename T>
struct AddInputs {
    T tolerance;
    int len;
    unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const AddInputs<T>& dims) {
    return os;
}

template <typename T>
class AddTest: public ::testing::TestWithParam<AddInputs<T> > {
protected:
    void SetUp() override {
        params = ::testing::TestWithParam<AddInputs<T>>::GetParam();
        Random::Rng<T> r(params.seed);
        int len = params.len;
        allocate(in1, len);
        allocate(in2, len);
        allocate(out_ref, len);
        allocate(out, len);
        r.uniform(in1, len, T(-1.0), T(1.0));
        r.uniform(in2, len, T(-1.0), T(1.0));
        naiveAddElem(out_ref, in1, in2, len);
        add(out, in1, in2, len);
        add(in1, in1, in2, len);

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
    }

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

    void TearDown() override {
        CUDA_CHECK(cudaFree(in1));
        CUDA_CHECK(cudaFree(in2));
        CUDA_CHECK(cudaFree(out_ref));
        CUDA_CHECK(cudaFree(out));

        if (device_count > 1) {
        	CUDA_CHECK(cudaFree(out_2));
        }
    }

protected:
    AddInputs<T> params;
    T *in1, *in2, *out_ref, *out, *out_2;
    int device_count = 0;
};

const std::vector<AddInputs<float> > inputsf2 = {
    {0.000001f, 1024*1024, 1234ULL}
};

const std::vector<AddInputs<double> > inputsd2 = {
    {0.00000001, 1024*1024, 1234ULL}
};

typedef AddTest<float> AddTestF;
TEST_P(AddTestF, Result) {
    ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                            CompareApprox<float>(params.tolerance)));

    ASSERT_TRUE(devArrMatch(out_ref, in1, params.len,
                            CompareApprox<float>(params.tolerance)));

    if (device_count > 1) {
    	ASSERT_TRUE(devArrMatch(out_ref, out_2, params.len,
    	                    CompareApprox<float>(params.tolerance)));
    }
}

typedef AddTest<double> AddTestD;
TEST_P(AddTestD, Result){
    ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                            CompareApprox<double>(params.tolerance)));

    ASSERT_TRUE(devArrMatch(out_ref, in1, params.len,
                            CompareApprox<double>(params.tolerance)));

    if (device_count > 1) {
    	ASSERT_TRUE(devArrMatch(out_ref, out_2, params.len,
    	                    CompareApprox<double>(params.tolerance)));
    }
}

INSTANTIATE_TEST_CASE_P(AddTests, AddTestF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(AddTests, AddTestD,
                        ::testing::ValuesIn(inputsd2));

} // end namespace LinAlg
} // end namespace MLCommon
