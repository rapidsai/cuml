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

#include "umap/umapparams.h"
#include "knn/knn.h"
#include "umap/runner.h"

#include "random/rng.h"
#include "test_utils.h"
#include <cuda_utils.h>
#include "ml_utils.h"
//#include "umap/runner.h"

#include <vector>

#include <iostream>

using namespace ML;
using namespace std;

class UMAPTest: public ::testing::Test {
protected:
	void basicTest() {

		umap_params = new UMAPParams();
		umap_params->n_neighbors = k;

		kNN *knn = new kNN(d);

		UMAPAlgo::find_ab(umap_params);

//		std::vector<float> X = {
//			1.0, 1.0, 34.0,
//			76.0, 2.0, 29.0,
//			34.0, 3.0, 13.0,
//			23.0, 7.0, 80.0
//		};

        std::vector<float> X = {
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0,
            1.0, 0.0, 0.0
        };



		float* X_d;
		MLCommon::allocate(X_d, n*d);
		MLCommon::updateDevice(X_d, X.data(), n*d);

		float *embeddings;
		MLCommon::allocate(embeddings, n*umap_params->n_components);

		std::cout << "Fitting UMAP..." << std::endl;

		UMAPAlgo::_fit<float>(X_d, n, d, knn, umap_params, embeddings);

		std::cout << "Done." << std::endl;

		float *xformed;
		MLCommon::allocate(xformed, n*umap_params->n_components);

        std::cout << "Transforming UMAP..." << std::endl;

		UMAPAlgo::_transform<float, 256>(X_d, n, d, embeddings, n, knn, umap_params, xformed);
//
        std::cout << "Done." << std::endl;

	}

	void SetUp() override {
		basicTest();
	}

	void TearDown() override {
//		CUDA_CHECK(cudaFree(dists_d));
//		CUDA_CHECK(cudaFree(inds_d));
	}

protected:

	UMAPParams *umap_params;

	int d = 3;
	int n = 4;
	int k = 2;

	float *dists_d;
	long *inds_d;
};


typedef UMAPTest UMAPFuzzySimplSetTestF;
TEST_F(UMAPFuzzySimplSetTestF, Result) {
//	ASSERT_TRUE(
//			devArrMatch(labels, labels_ref, params.n_row,
//					CompareApproxAbs<float>(params.tolerance)));

    std::cout << "HELLO!" << std::endl;
}

