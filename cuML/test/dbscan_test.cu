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
#include "random/rng.h"
#include "test_utils.h"
#include <cuda_utils.h>
#include "ml_utils.h"
#include "dbscan/dbscan.hpp"
#include <linalg/cublas_wrappers.h>
#include <vector>

namespace ML {

using namespace MLCommon;
using namespace std;

template<typename T>
struct DbscanInputs {
	T tolerance;
	int len;
	int n_row;
	int n_col;
	unsigned long long int seed;
};

template<typename T>
::std::ostream& operator<<(::std::ostream& os, const DbscanInputs<T>& dims) {
	return os;
}

class DbScanTestBase: public ::testing::Test {};

typedef DbScanTestBase DbScanTestBatchSize;
TEST(COOSort, Result) {

    cudaStream_t stream;
    CUDA_CHECK( cudaStreamCreate(&stream) );


    size_t max_elems = 2e4;

    double *data;

    long long seed = 10;

    Random::Rng r(1234ULL);
    int rows = 5000, cols = 500;

    allocate(data, rows*cols);

    r.uniform<double, double>(data, rows*cols, -1.0, 1.0, stream);

    int *labels;
    allocate(labels, rows);

    cumlHandle handle;
    handle.setStream(stream);

    // Should not fail
    dbscanFit(handle, data, rows, cols, 0.01, 1, labels, max_elems);


};

template<typename T>
class DbscanTest: public ::testing::TestWithParam<DbscanInputs<T> > {
protected:
	void basicTest() {
		cudaStream_t stream;
		CUDA_CHECK( cudaStreamCreate(&stream) );

		params = ::testing::TestWithParam<DbscanInputs<T>>::GetParam();
		Random::Rng r(params.seed);
		int len = params.len;

		allocate(data, len);

		std::vector<T> data_h = { 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 8.0, 7.0, 8.0, 8.0, 25.0, 80.0 };
		data_h.resize(len);
		updateDevice(data, data_h.data(), len, stream);

		allocate(labels, params.n_row);
		allocate(labels_ref, params.n_row);
		std::vector<int> labels_ref_h = { 0, 0, 0, 1, 1, -1 };
		labels_ref_h.resize(len);
		updateDevice(labels_ref, labels_ref_h.data(), params.n_row, stream);

		T eps = 3.0;
		int min_pts = 2;
		cumlHandle handle;
		handle.setStream(stream);
		dbscanFit(handle, data, params.n_row, params.n_col, eps, min_pts, labels, (size_t)2e6);
		CUDA_CHECK( cudaStreamSynchronize(stream) );
		CUDA_CHECK( cudaStreamDestroy(stream) );

	}

	void SetUp() override {
		basicTest();
	}

	void TearDown() override {
		CUDA_CHECK(cudaFree(data));
		CUDA_CHECK(cudaFree(labels));
		CUDA_CHECK(cudaFree(labels_ref));
	}

protected:
	DbscanInputs<T> params;
	T *data;
	int *labels, *labels_ref;

};

const std::vector<DbscanInputs<float> > inputsf2 = {
		{ 0.05f, 6 * 2, 6, 2, 1234ULL }};

const std::vector<DbscanInputs<double> > inputsd2 = {
		{ 0.05, 6 * 2, 6, 2, 1234ULL }};


typedef DbscanTest<float> DbscanTestF;
TEST_P(DbscanTestF, Result) {
	ASSERT_TRUE(
			devArrMatch(labels, labels_ref, params.n_row,
					CompareApproxAbs<float>(params.tolerance)));

}

typedef DbscanTest<double> DbscanTestD;
TEST_P(DbscanTestD, Result) {
	ASSERT_TRUE(
			devArrMatch(labels, labels_ref, params.n_row,
					CompareApproxAbs<double>(params.tolerance)));
}


INSTANTIATE_TEST_CASE_P(DbscanTests, DbscanTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(DbscanTests, DbscanTestD, ::testing::ValuesIn(inputsd2));

} // end namespace ML
