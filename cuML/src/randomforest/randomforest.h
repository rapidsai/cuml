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

#pragma once
#include "decisiontree/decisiontree.h"
#include <iostream>
#include <utils.h>
#include "random/rng.h"
#include <map>
#include <common/cumlHandle.hpp>
#include <common/device_buffer.hpp>

namespace ML {

struct RF_metrics {
	float accuracy;

	RF_metrics(float cfg_accuracy);
	void print();
};

enum RF_type {
	CLASSIFICATION, REGRESSION,
};

struct RF_params {
	bool bootstrap = true;
	int n_trees;
	float rows_sample = 1.0f;
	DecisionTree::DecisionTreeParams tree_params;

	RF_params(int cfg_n_trees);
	RF_params(bool cfg_bootstrap, int cfg_n_trees, float cfg_rows_sample);
	RF_params(bool cfg_bootstrap, int cfg_n_trees, float cfg_rows_sample, DecisionTree::DecisionTreeParams cfg_tree_params);
	void validity_check() const;
	void print() const;
};

/* Update labels so they are unique from 0 to n_unique_vals.
   		Create an old_label to new_label map per random forest.
*/
void preprocess_labels(int n_rows, std::vector<int> & labels, std::map<int, int> & labels_map, bool verbose=false);

/* Revert preprocessing effect, if needed. */
void postprocess_labels(int n_rows, std::vector<int> & labels, std::map<int, int> & labels_map, bool verbose=false);

template<class T>
class rf {
	protected:
		RF_params rf_params;
		int rf_type;
		DecisionTree::DecisionTreeClassifier<T> * trees;

	public:
		rf(RF_params cfg_rf_params, int cfg_rf_type=RF_type::CLASSIFICATION);
		~rf();

		int get_ntrees();
		void print_rf_summary();
		void print_rf_detailed();
};

template <class T>
class rfClassifier : public rf<T> {
	public:

	rfClassifier(RF_params cfg_rf_params, int cfg_rf_type=RF_type::CLASSIFICATION);

	void fit(const cumlHandle& user_handle, T * input, int n_rows, int n_cols, int * labels, int n_unique_labels);
	void predict(const cumlHandle& user_handle, const T * input, int n_rows, int n_cols, int * predictions, bool verbose=false) const;
	RF_metrics cross_validate(const cumlHandle& user_handle, const T * input, const int * ref_labels, int n_rows, int n_cols, int * predictions, bool verbose=false) const;

};

// Stateless API functions: fit, predict and cross_validate.
void fit(const cumlHandle& user_handle, rfClassifier<float> * rf_classifier, float * input, int n_rows, int n_cols, int * labels, int n_unique_labels);
void fit(const cumlHandle& user_handle, rfClassifier<double> * rf_classifier, double * input, int n_rows, int n_cols, int * labels, int n_unique_labels);

void predict(const cumlHandle& user_handle, const rfClassifier<float> * rf_classifier, const float * input, int n_rows, int n_cols, int * predictions, bool verbose=false);
void predict(const cumlHandle& user_handle, const rfClassifier<double> * rf_classifier, const double * input, int n_rows, int n_cols, int * predictions, bool verbose=false);

RF_metrics cross_validate(const cumlHandle& user_handle, const rfClassifier<float> * rf_classifier, const float * input, const int * ref_labels,
							int n_rows, int n_cols, int * predictions, bool verbose=false);
RF_metrics cross_validate(const cumlHandle& user_handle, const rfClassifier<double> * rf_classifier, const double * input, const int * ref_labels,
							int n_rows, int n_cols, int * predictions, bool verbose=false);

};
