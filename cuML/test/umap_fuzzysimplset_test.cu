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

#include "umap/knn_graph/runner.h"
#include "umap/fuzzy_simpl_set/runner.h"
#include "umap/umap.h"

#include <linalg/cublas_wrappers.h>
#include <sparse/coo.h>

#include <vector>

using namespace ML;
using namespace UMAPAlgo;
using namespace MLCommon;
using namespace std;

class UMAPFuzzySimplSetTest: public ::testing::Test {
protected:
	void basicTest() {

		umap_params = new UMAPParams();
		umap_params->n_neighbors = k;


		std::vector<float> X = {
			1.0, 1.0, 34.0,
			76.0, 2.0, 29.0,
			34.0, 3.0, 13.0,
			23.0, 7.0, 80.0
		};

		float* X_d;
		MLCommon::allocate(X_d, n*d);
		MLCommon::updateDevice(X_d, X.data(), n*d);

		allocate(dists_d, n*k);
		allocate(inds_d, n*k);

		kNNGraph::run(X_d, n, d, inds_d, dists_d, umap_params);

		int *rows, *cols;
		float *vals;

		allocate(rows, n*k);
		allocate(cols, n*k);
		allocate(vals, n*k);

		FuzzySimplSet::run<float>(n, inds_d, dists_d, rows, cols, vals, umap_params);

//		int *rows_h, *cols_h;
//		float *vals_h;
//
//		MLCommon::updateHost(rows_h, rows, n*k);
//		MLCommon::updateHost(cols_h, rows, n*k);
//		MLCommon::updateHost(vals_h, rows, n*k);
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

	int d = 3;
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

