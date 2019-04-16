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

#include <gtest/gtest.h>

#include "umap/umapparams.h"
#include "umap/runner.h"
#include "knn/knn.h"

#include <cuda_utils.h>

#include <vector>
#include <iostream>

using namespace ML;
using namespace std;

/**
 * For now, this is mostly to test the c++ algorithm is able to be built.
 * Comprehensive comparisons of resulting embeddings are being done in the
 * Python test suite. Next to come will be a CUDA implementation of t-SNE's
 * trustworthiness score, which will allow us to gtest embedding algorithms.
 */
class UMAPTest: public ::testing::Test {
protected:
	void basicTest() {

		umap_params = new UMAPParams();
		umap_params->n_neighbors = k;

		kNN *knn = new kNN(d);
		cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));
		UMAPAlgo::find_ab(umap_params, stream);

		std::vector<float> X = {
			1.0, 1.0, 34.0,
			76.0, 2.0, 29.0,
			34.0, 3.0, 13.0,
			23.0, 7.0, 80.0
		};

    float* X_d;

		MLCommon::allocate(X_d, n*d);
		MLCommon::updateDevice(X_d, X.data(), n*d);

		MLCommon::allocate(embeddings, n*umap_params->n_components);

		UMAPAlgo::_fit<float, 256>(X_d, n, d, knn, umap_params, embeddings, stream);

		float *xformed;
		MLCommon::allocate(xformed, n*umap_params->n_components);

		UMAPAlgo::_transform<float, 256>(X_d, n, d, embeddings, n, knn, umap_params, xformed, stream);
		CUDA_CHECK(cudaStreamDestroy(stream));
	}

	void SetUp() override {
		basicTest();
	}

	void TearDown() override {
	}

protected:

	UMAPParams *umap_params;

	int d = 3;
	int n = 4;
	int k = 2;

	float *embeddings;

};


typedef UMAPTest UMAPTestF;
TEST_F(UMAPTestF, Result) {}

