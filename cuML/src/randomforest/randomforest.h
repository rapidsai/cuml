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
#include "decisiontree/tree.cuh"

namespace ML {
	
	enum RF_type {
		CLASSIFICATION, REGRESSION,
	};

	class rf {
		protected:
		int n_trees; 
		int max_depth; 
	    int max_leaves; 	
		int rf_type;

		DecisionTree::DecisionTreeClassifier * trees;
		
		public:
			rf(int n_trees, int max_depth=0, int max_leaves=0, int rf_type=RF_type::CLASSIFICATION) {
				n_trees = n_trees;
				max_depth = max_depth; //FIXME Set these during fit?
				max_leaves = max_leaves;
				trees = NULL; 
				rf_type = rf_type;
			}

			~rf() {
				delete trees;
			}

    };

//FIXME: input could be of different type.
//FIXME: labels for regression could be of different type too, potentially match input type. 
/*FIXME: there are many more hyperparameters to consider, as per SKL-RF. For example:
  - max_depth, max_leaves, criterion etc. */


	class rfClassifier : public rf {
		public:
		rfClassifier(int n_trees, int max_depth, int max_leaves, int rf_type) : rf(n_trees, max_depth, max_leaves, RF_type::CLASSIFICATION) {}
        /** 
         * Fit an RF classification model on input data with n_rows samples and n_cols features.
         * @param input			data array in FIXME row major format for now. 
         * @param n_rows		number of training? data rows
         * @param n_cols		number of features (columns)
         * @param labels		list of target features
         * @param n_trees		number of trees in the random forest
         * @param max_features	number of features to consider per split (default = sqrt(n_cols))
	     * @param rows_sample	ratio of n_rows used per tree
        */
		void fit(float * input, int n_rows, int n_cols, int * labels,
                         int n_trees, float rows_sample);
		void fit(float * input, int n_rows, int n_cols, int * labels,
                         int n_trees, float max_features, float rows_sample);

		void predict(const float * input, int n_rows, int n_cols, int * preds);
	};


	class rfRegressor : public rf {
	    public:
		rfRegressor(int n_trees, int max_depth, int max_leaves, int rf_type) : rf(n_trees, max_depth, max_leaves, RF_type::REGRESSION) {}
		void fit(float * input, int n_rows, int n_cols, int * labels,
                         int n_trees, float max_features, float rows_sample);

		void predict(const float * input, int n_rows, int n_cols, int * preds);
	}; 


}
