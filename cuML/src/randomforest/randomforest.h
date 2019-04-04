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

namespace ML {

	struct RF_metrics {
		float accuracy;

		RF_metrics(float cfg_accuracy);
		void print();
	};

	enum RF_type {
		CLASSIFICATION, REGRESSION,
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
			int n_trees, n_bins, rf_type;
			int max_depth, max_leaves, split_algo;
			bool bootstrap;
			float rows_sample;
			float max_features; // ratio of number of features (columns) to consider per node split.
         					    // TODO SKL's default is sqrt(n_cols)

			DecisionTree::DecisionTreeClassifier<T> * trees;

		public:
			rf(int cfg_n_trees, bool cfg_bootstrap=true, int cfg_max_depth=-1, int cfg_max_leaves=-1, int cfg_rf_type=RF_type::CLASSIFICATION, int cfg_n_bins=8,
			   float cfg_rows_sample=1.0f, float cfg_max_features=1.0f, int cfg_split_algo=SPLIT_ALGO::HIST);
			~rf();

			int get_ntrees();
			void print_rf_summary();
			void print_rf_detailed();
    };

	template <class T>
	class rfClassifier : public rf<T> {
		public:

		rfClassifier(int cfg_n_trees, bool cfg_bootstrap=true, int cfg_max_depth=-1, int cfg_max_leaves=-1, int cfg_rf_type=RF_type::CLASSIFICATION, int cfg_n_bins=8,
						float cfg_rows_sample=1.0f, float cfg_max_features=1.0f, int cfg_split_algo=SPLIT_ALGO::HIST);

		void fit(T * input, int n_rows, int n_cols, int * labels, int n_unique_labels);
		int * predict(const T * input, int n_rows, int n_cols, bool verbose=false);
		RF_metrics cross_validate(const T * input, const int * ref_labels, int n_rows, int n_cols, bool verbose=false);

	};
};
