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
#include "umap/fuzzy_simpl_set/runner.h"
#include <linalg/cublas_wrappers.h>
#include <vector>

namespace ML {

using namespace MLCommon;
using namespace std;

template<typename T>
class UMAPFuzzySimplSetTest: public ::testing::Test {
protected:
	void basicTest() {

		Random::Rng<T> r(1);

		umap_params->

		allocate(data, len);

		std::vector<T> input_h = { 1.0, 2.0, 2.0,
								  2.0, 2.0, 3.0,
								  8.0, 7.0, 8.0,
								  8.0, 25.0, 80.0
		};

		input_h.resize(len);
		updateDevice(data, data_h.data(), len);

		allocate(labels, params.n_row);
		allocate(labels_ref, params.n_row);
		std::vector<int> labels_ref_h = { 0, 0, 0, 1, 1, -1 };
		labels_ref_h.resize(len);
		updateDevice(labels_ref, labels_ref_h.data(), params.n_row);

		T eps = 3.0;
		int min_pts = 2;

		dbscanFitImpl(data, params.n_row, params.n_col, eps, min_pts, labels);

	}

	void SetUp() override {
		basicTest();
	}

	void TearDown() override {
		CUDA_CHECK(cudaFree(dists));
		CUDA_CHECK(cudaFree(indices));
	}

protected:
	UMAPFuzzySimplSetInputs<T> params;

	UMAPParams *umap_params;



	float *dists_h;
	long *inds_h;

	float *dists_d;
	long *inds_d;
};


typedef UMAPFuzzySimplSetTest<float> UMAPFuzzySimplSetTestF;
TEST_P(UMAPFuzzySimplSetTestF, Result) {
	ASSERT_TRUE(
			devArrMatch(labels, labels_ref, params.n_row,
					CompareApproxAbs<float>(params.tolerance)));

}


} // end namespace ML
