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
#include "umap/umap.h"
#include <linalg/cublas_wrappers.h>
#include <vector>

using namespace ML;
using namespace MLCommon;
using namespace std;

class UMAPFuzzySimplSetTest: public ::testing::Test {
protected:
	void basicTest() {

		umap_params = new UMAPParams();
		umap_params->n_neighbors = 2;

		allocate(dists_d, n*k);
		allocate(inds_d, n*k);

		std::vector<float> dists_h = {
			0.0, 1.0,
			0.0, 1.0,
			0.0, 1.0,
			0.0, 1.0
		};

		std::vector<long> inds_h = {
			0, 1,
			1, 2,
			2, 3,
			3, 0,
		};

		dists_h.resize(n*k);
		inds_h.resize(n*k);

		updateDevice(dists_d, dists_h.data(), n*k);
		updateDevice(inds_d, inds_h.data(), n*k);

		int *rows, *cols;
		float *vals;

		allocate(rows, n*k);
		allocate(cols, n*k);
		allocate(vals, n*k);

		UMAP::FuzzySimplSet::run(n, inds_d, dists_d, rows, cols, vals, umap_params);



	}

	void SetUp() override {
		basicTest();
	}

	void TearDown() override {
		CUDA_CHECK(cudaFree(dists_d));
		CUDA_CHECK(cudaFree(inds_d));
	}

protected:
	UMAPParams *umap_params;

	int n = 4;
	int k = 2;

	float *dists_d;
	long *inds_d;
};


typedef UMAPFuzzySimplSetTest UMAPFuzzySimplSetTestF;
TEST_F(UMAPFuzzySimplSetTestF, Result) {
//	ASSERT_TRUE(
//			devArrMatch(labels, labels_ref, params.n_row,
//					CompareApproxAbs<float>(params.tolerance)));

}

