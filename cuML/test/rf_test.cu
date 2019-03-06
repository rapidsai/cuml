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

#include "randomforest/randomforest.h"
#include <gtest/gtest.h>
#include <cuda_utils.h>
#include <test_utils.h>
#include "ml_utils.h"

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
		// Single tree predictor 
		//--------------------------------------------------------
		tree_cf = new DecisionTree::DecisionTreeClassifier();

		int data_len = params.n_rows * params.n_cols;
		allocate(data, data_len);
		allocate(labels, params.n_rows);

		// Populate data (assume Col major)
		std::vector<T> data_h = {30.0f, 1.0f, 2.0f, 0.0f, 10.0f, 20.0f, 10.0f, 40.0f};
		data_h.resize(data_len);
	    updateDevice(data, data_h.data(), data_len);

		// Populate labels
		std::vector<int> labels_h = {0, 1, 0, 4};
		labels_h.resize(params.n_rows);
	    updateDevice(labels, labels_h.data(), params.n_rows);

		// Set selected rows: all for single decision tree
		unsigned int * selected_rows;
		allocate(selected_rows, params.n_rows);
		std::vector<unsigned int> selected_rows_h = {0, 1, 2, 3};
		selected_rows_h.resize(params.n_rows);
		updateDevice(selected_rows, selected_rows_h.data(), params.n_rows);

		// Train single decision tree.
		std::cout << "Config: " << params.n_cols << " " << params.n_rows << " " << params.max_depth << " " << params.max_leaves  << " " << params.max_features << " " << params.bootstrap << std::endl;
		tree_cf->fit(data, params.n_cols, params.n_rows, labels, selected_rows, params.n_rows, params.max_depth, params.max_leaves, params.max_features);
		tree_cf->print();

		int single_tree_inference_data_len = params.n_cols;
		//std::vector<T> single_tree_inference_data_h = {1.0f, 20.0f};
		std::vector<T> single_tree_inference_data_h = {30.0f, 20.0f};
		single_tree_inference_data_h.resize(single_tree_inference_data_len);
		int predicted_val = tree_cf->predict(single_tree_inference_data_h.data());
	    std::cout << "Predicted " << predicted_val << std::endl;

		//--------------------------------------------------------
		// Random Forest
		//--------------------------------------------------------
		
 		rf_classifier = new ML::rfClassifier::rfClassifier(params.n_trees, params.bootstrap, 0, 0);
		rf_classifier->fit(data, params.n_rows, params.n_cols, labels, params.n_trees, params.max_features, params.rows_sample);

		int inference_data_len = params.n_inference_rows * params.n_cols;
		std::vector<T> inference_data_h = {30.0f, 10.0f, 1.0f, 20.0f, 2.0f, 10.0f, 0.0f, 40.0f};
		inference_data_h.resize(inference_data_len);

		predictions = rf_classifier->predict(inference_data_h.data(), params.n_inference_rows, params.n_cols, false);
		for (int i = 0; i < params.n_inference_rows; i++) {
			std::cout << "Random forest predicted " << predictions[i] << std::endl;
		}

		rf_classifier->cross_validate(inference_data_h.data(), labels_h.data(), params.n_inference_rows, params.n_cols, false);


		
    }

 	void SetUp() override {
		basicTest();
	}

	void TearDown() override {
		CUDA_CHECK(cudaFree(labels));
		CUDA_CHECK(cudaFree(data));
		delete tree_cf;
		delete predictions;
		delete rf_classifier;

	}

protected:

   	//placeholder for any params?
	RfInputs<T> params;
	T * data, * inference_data;
    int * labels, * predictions, * ref_predictions;

	
	DecisionTree::DecisionTreeClassifier * tree_cf;
    rfClassifier * rf_classifier;

};

/* NVM for now:
   Rf configs:
	- 10 rows, 4 features, 1 tree, 0.8 max features, 1.0 row sampling, 2 rows inf
	- 10 rows, 4 features, 2 trees, 0.8 max features, 0.8 row sampling, 2 rows inf
 */
const std::vector<RfInputs<float> > inputsf2 = {
		  //{ 4, 2, 1, 1.0f, 1.0f, 4}, //};
		  //{ 4, 2, 3, 1.0f, 1.0f, 4, -1, -1} };
		  { 4, 2, 3, 1.0f, 1.0f, 4, 8, -1, false},
		  { 4, 2, 3, 1.0f, 1.0f, 4, 8, -1, true}, 
		  { 4, 2, 1, 1.0f, 1.0f, 4, 8, -1, true} };


//FIXME Add tests for fit and predict. Identify what would make a comparison match (similar predictions?)
typedef RfTest<float> RfTestF;
TEST_P(RfTestF, Fit) {
	//ASSERT_TRUE(
	//		devArrMatch(ref_predictions, predictions, /*some row cnt? */ , Compare<float>()));
}

INSTANTIATE_TEST_CASE_P(RfTests, RfTestF, ::testing::ValuesIn(inputsf2));


} // end namespace ML
