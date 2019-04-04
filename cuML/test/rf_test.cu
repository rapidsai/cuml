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
#include <cuda_utils.h>
#include <test_utils.h>
#include "ml_utils.h"
#include "randomforest/randomforest.h"

namespace ML {

using namespace MLCommon;

template<typename T> // template useless for now.
struct RfInputs {
	int n_rows;
	int n_cols;
	int n_trees;
	float max_features;
	float rows_sample;
	int n_inference_rows;
	int max_depth;
	int max_leaves;
	bool bootstrap;
	int n_bins;
};

template<typename T>
::std::ostream& operator<<(::std::ostream& os, const RfInputs<T>& dims) {
	return os;
}


template<typename T>
class RfTest: public ::testing::TestWithParam<RfInputs<T> > {
protected:
	void basicTest() {

		params = ::testing::TestWithParam<RfInputs<T>>::GetParam();

		//--------------------------------------------------------
		// Random Forest - Single tree 
		//--------------------------------------------------------

		int data_len = params.n_rows * params.n_cols;
		allocate(data, data_len);
		allocate(labels, params.n_rows);

		// Populate data (assume Col major)
		std::vector<T> data_h = {30.0f, 1.0f, 2.0f, 0.0f, 10.0f, 20.0f, 10.0f, 40.0f};
		data_h.resize(data_len);
	    updateDevice(data, data_h.data(), data_len);

		// Populate labels
		labels_h = {0, 1, 0, 4};
		labels_h.resize(params.n_rows);
		preprocess_labels(params.n_rows, labels_h, labels_map);
	    updateDevice(labels, labels_h.data(), params.n_rows);

		// Set selected rows: all for forest w/ single decision tree
		allocate(selected_rows, params.n_rows);
		std::vector<unsigned int> selected_rows_h = {0, 1, 2, 3};
		selected_rows_h.resize(params.n_rows);
		updateDevice(selected_rows, selected_rows_h.data(), params.n_rows);

		rf_classifier = new rfClassifier<float>::rfClassifier(params.n_trees, params.bootstrap, params.max_depth,
							params.max_leaves, 0, params.n_bins, params.rows_sample, params.max_features);
		rf_classifier->fit(data, params.n_rows, params.n_cols, labels, labels_map.size());

		// Inference data: same as train, but row major
		int inference_data_len = params.n_inference_rows * params.n_cols;
		inference_data_h = {30.0f, 10.0f, 1.0f, 20.0f, 2.0f, 10.0f, 0.0f, 40.0f};
		inference_data_h.resize(inference_data_len);

    }

 	void SetUp() override {
		basicTest();
	}

	void TearDown() override {
		postprocess_labels(params.n_rows, labels_h, labels_map);
		inference_data_h.clear();
		labels_h.clear();
		labels_map.clear();

		CUDA_CHECK(cudaFree(labels));
		CUDA_CHECK(cudaFree(data));
		CUDA_CHECK(cudaFree(selected_rows));
		delete rf_classifier;
	}

protected:

	RfInputs<T> params;
	T * data;
    int * labels;
	std::vector<T> inference_data_h;
	std::vector<int> labels_h;
	unsigned int * selected_rows;
	std::map<int, int> labels_map; //unique map of labels to int vals starting from 0

    rfClassifier<T> * rf_classifier;
};


const std::vector<RfInputs<float> > inputsf2 = {
		  { 4, 2, 1, 1.0f, 1.0f, 4, -1, -1, false, 8},	// single tree forest, bootstrap false, unlimited depth, 8 bins
		  { 4, 2, 1, 1.0f, 1.0f, 4, 8, -1, false, 8}	// single tree forest, bootstrap false, depth of 8, 8 bins
};


typedef RfTest<float> RfTestF;
TEST_P(RfTestF, Fit) {
	RF_metrics tmp = rf_classifier->cross_validate(inference_data_h.data(), labels_h.data(), params.n_inference_rows, params.n_cols, false);
	//rf_classifier->print_rf_detailed();
	ASSERT_TRUE((tmp.accuracy == 1.0f));
}

INSTANTIATE_TEST_CASE_P(RfTests, RfTestF, ::testing::ValuesIn(inputsf2));


} // end namespace ML
