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
#include "stats/sum.h"
#include "random/rng.h"
#include "test_utils.h"
#include "linalg/eltwise.h"

namespace MLCommon {
namespace Stats {

template<typename T>
struct SumInputs {
	T tolerance;
	int rows, cols;
	unsigned long long int seed;
};

template<typename T>
::std::ostream& operator<<(::std::ostream& os, const SumInputs<T>& dims) {
	return os;
}

template<typename T>
class SumTest: public ::testing::TestWithParam<SumInputs<T> > {
protected:
	void SetUp() override {
		params = ::testing::TestWithParam<SumInputs<T>>::GetParam();
		int rows = params.rows, cols = params.cols;
		int len = rows * cols;
		allocate(data, len);

		T data_h[len];
		for (int i = 0; i < len; i++) {
			data_h[i] = T(1);
		}

		updateDevice(data, data_h, len);

		allocate(sum_act, cols);
		sum(sum_act, data, cols, rows, false);

		CUDA_CHECK(cudaGetDeviceCount(&device_count));

		if (device_count > 1) {
			sumMGColSplitTest(data_h);
		}
	}

	void sumMGColSplitTest(T *h_data) {
		int n_gpus = 2;

		TypeMG<T> d_data[n_gpus];
		TypeMG<T> d_sum[n_gpus];

		for (int i = 0; i < n_gpus; i++) {
			d_data[i].gpu_id = i;
			d_sum[i].gpu_id = i;
			CUDA_CHECK(cudaSetDevice(d_data[i].gpu_id));
			CUDA_CHECK(cudaStreamCreate(&(d_data[i].stream)));
			d_sum[i].stream = d_data[i].stream;
		}

		allocateMG(d_data, n_gpus, params.rows, params.cols, true, true, false);
		allocateMG(d_sum, n_gpus, 1, params.cols, true, true, false);

		updateDeviceMG(d_data, h_data, n_gpus, false);

		sumMG(d_sum, d_data, params.cols, params.rows, n_gpus, false,
				false, false);


		int len = params.cols;
		T *h_sum = (T *) malloc(len * sizeof(T));
		updateHostMG(h_sum, d_sum, n_gpus, false);

		streamSyncMG(d_data, n_gpus);
		streamDestroyGPUs(d_data, n_gpus);

		freeMG(d_data, n_gpus);
		freeMG(d_sum, n_gpus);

		allocate(sum_act_2, len);
		updateDevice(sum_act_2, h_sum, len);

		free(h_sum);
	}

	void TearDown() override {
		CUDA_CHECK(cudaFree(data));
		CUDA_CHECK(cudaFree(sum_act));
		if (device_count > 1) {
			CUDA_CHECK(cudaFree(sum_act_2));
		}
	}

protected:
	SumInputs<T> params;
	T *data, *sum_act, *sum_act_2;
	int device_count = 0;
};

const std::vector<SumInputs<float> > inputsf = { { 0.05f, 1024, 32, 1234ULL }, {
		0.05f, 1024, 256, 1234ULL } };

const std::vector<SumInputs<double> > inputsd = { { 0.05, 1024, 32, 1234ULL }, {
		0.05, 1024, 256, 1234ULL } };

typedef SumTest<float> SumTestF;
TEST_P(SumTestF, Result) {
	ASSERT_TRUE(
			devArrMatch(float(params.rows), sum_act, params.cols,
					CompareApprox<float>(params.tolerance)));

	if (device_count > 1) {
		ASSERT_TRUE(
				devArrMatch(float(params.rows), sum_act_2, params.cols,
						CompareApprox<float>(params.tolerance)));
	}
}

typedef SumTest<double> SumTestD;
TEST_P(SumTestD, Result) {
	ASSERT_TRUE(
			devArrMatch(double(params.rows), sum_act, params.cols,
					CompareApprox<double>(params.tolerance)));

	if (device_count > 1) {
		ASSERT_TRUE(
				devArrMatch(double(params.rows), sum_act_2, params.cols,
						CompareApprox<double>(params.tolerance)));
	}
}

INSTANTIATE_TEST_CASE_P(SumTests, SumTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(SumTests, SumTestD, ::testing::ValuesIn(inputsd));

} // end namespace Stats
} // end namespace MLCommon
