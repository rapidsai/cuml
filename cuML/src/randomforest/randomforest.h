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
#include <iostream>
#include <utils.h>
#include "random/rng.h"
#include "linalg/cublas_wrappers.h"
#include <map>

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
			bool bootstrap;

			DecisionTree::DecisionTreeClassifier * trees;
		
		public:
			rf(int cfg_n_trees, bool cfg_bootstrap=true, int cfg_max_depth=0, int cfg_max_leaves=0, int cfg_rf_type=RF_type::CLASSIFICATION) {
					n_trees = cfg_n_trees;
					max_depth = cfg_max_depth; //FIXME Set these during fit?
					max_leaves = cfg_max_leaves;
					trees = NULL; 
					rf_type = cfg_rf_type;
					bootstrap = cfg_bootstrap;
			}

			~rf() {
					delete trees;
			}

			int get_ntrees() {
				std::cout << std::dec << n_trees << " " << max_depth << " " << max_leaves << " " << rf_type << "\n";
				return n_trees;
			}


    };

//FIXME: input could be of different type.
//FIXME: labels for regression could be of different type too, potentially match input type. 
/*FIXME: there are many more hyperparameters to consider, as per SKL-RF. For example:
  - max_depth, max_leaves, criterion etc. */

	class rfClassifier : public rf {
		public:

		rfClassifier(int cfg_n_trees, bool cfg_bootstrap=true, int cfg_max_depth=0, int cfg_max_leaves=0, int cfg_rf_type=RF_type::CLASSIFICATION) 
					: rf::rf(cfg_n_trees, cfg_bootstrap, cfg_max_depth, cfg_max_leaves, cfg_rf_type) {};

        /** 
         * Fit an RF classification model on input data with n_rows samples and n_cols features.
         * @param input			data array in col major format for now (device ptr)
         * @param n_rows		number of training? data rows
         * @param n_cols		number of features (i.e, columns)
         * @param labels		list of target features (device ptr)
         * @param n_trees		number of trees in the random forest
         * @param max_features	number of features to consider per split (default = sqrt(n_cols))
	     * @param rows_sample	ratio of n_rows used per tree
        */
		void fit(float * input, int n_rows, int n_cols, int * labels,
                         int cfg_n_trees, float max_features, float rows_sample, int cfg_max_depth=-1, int cfg_max_leaves=-1) {
			ASSERT(!trees, "Cannot fit an existing forest.");
			ASSERT((rows_sample > 0) && (rows_sample <= 1.0), "rows_sample value %f outside permitted (0, 1] range", rows_sample);
			ASSERT((max_features > 0) && (max_features <= 1.0), "max_features value %f outside permitted (0, 1] range", max_features);
			ASSERT((n_rows > 0), "Invalid n_rows %d", n_rows);
			ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);
			ASSERT((cfg_n_trees > 0), "Invalid n_trees %d", cfg_n_trees);

			n_trees = cfg_n_trees;
			max_depth = cfg_max_depth;
			max_leaves = cfg_max_leaves;

			rfClassifier::trees = new DecisionTree::DecisionTreeClassifier[n_trees];
			int n_sampled_rows = rows_sample * n_rows;
			
			for (int i = 0; i < n_trees; i++) {
				// Select n_sampled_rows (with replacement) numbers from [0, n_rows) per tree.
				unsigned int * selected_rows; // randomly generated IDs for bootstrapped samples (w/ replacement); a device ptr.
				CUDA_CHECK(cudaMalloc((void **)& selected_rows, n_sampled_rows * sizeof(unsigned int)));
				
				if (bootstrap) {
					MLCommon::Random::Rng r(i * 1000); // Ensure the seed for each tree is different and meaningful.
					r.uniformInt(selected_rows, n_sampled_rows, (unsigned int) 0, (unsigned int) n_rows);
					/*
					//DBG
					std::cout << "Bootstrapping for tree " << i << std::endl;
					unsigned int h_selected_rows[n_sampled_rows];
					CUDA_CHECK(cudaMemcpy(h_selected_rows, selected_rows, n_sampled_rows * sizeof(unsigned int), cudaMemcpyDeviceToHost));
					for (int tmp = 0; tmp < n_sampled_rows; tmp++) {
						std::cout << h_selected_rows[tmp] << " ";
					}
					std::cout << std::endl;
					*/
				} else {
					std::vector<unsigned int> h_selected_rows(n_sampled_rows);
					std::iota(h_selected_rows.begin(), h_selected_rows.end(), 0);
					CUDA_CHECK(cudaMemcpy(selected_rows, h_selected_rows.data(), n_sampled_rows * sizeof(unsigned int), cudaMemcpyHostToDevice));
				}

				/* Build individual tree in the forest.
				   - input is a pointer to orig data that have n_cols features and n_rows rows.
				   - n_sampled_rows: # rows sampled for tree's bootstrap sample.
				   - selected_rows: points to a list of row #s (w/ n_sampled_rows elements) used to build the bootstrapped sample.  
					Expectation: Each tree node will contain (a) # n_sampled_rows and (b) a pointer to a list of row numbers w.r.t original data. 
				*/
				trees[i].fit(input, n_cols, n_rows, labels, selected_rows, n_sampled_rows, max_depth, max_leaves, max_features);

				//Cleanup
				CUDA_CHECK(cudaFree(selected_rows));

			}
		}	


		//Assuming input in row_major format. 
		int * predict(const float * input, int n_rows, int n_cols, bool verbose=false) {
			ASSERT(trees, "Cannot predict! No trees in the forest.");
			int * preds = new int[n_rows];

			int row_size = n_cols;

			for (int row_id = 0; row_id < n_rows; row_id++) {
				
				if (verbose) {
					std::cout << "\n\n";
					std::cout << "Predict for sample: ";
					for (int i = 0; i < n_cols; i++) std::cout << input[row_id*row_size + i] << ", ";
					std::cout << std::endl;
				}

				std::map<int, int> prediction_to_cnt;
				std::pair<std::map<int, int>::iterator, bool> ret;
				int max_cnt_so_far = 0;
				int majority_prediction = -1;

				for (int i = 0; i < n_trees; i++) {
					//Return prediction for one sample. 
					if (verbose) {
						std::cout << "Printing tree " << i << std::endl;
						trees[i].print();
					}
					int prediction = trees[i].predict(&input[row_id * row_size], verbose);

  					ret = prediction_to_cnt.insert ( std::pair<int, int>(prediction, 1));
  					if (!(ret.second)) {
						ret.first->second += 1;
					}
					if (max_cnt_so_far < ret.first->second) {
						max_cnt_so_far = ret.first->second;
						majority_prediction = ret.first->first; 
					}	
				}

				preds[row_id] = majority_prediction;
			}

			return preds;
		}

};


	class rfRegressor : public rf {
	    public:

		rfRegressor(int cfg_n_trees, bool cfg_bootstrap=true, int cfg_max_depth=0, int cfg_max_leaves=0, int cfg_rf_type=RF_type::REGRESSION) 
						: rf::rf(cfg_n_trees, cfg_bootstrap, cfg_max_depth, cfg_max_leaves, cfg_rf_type) {};

		void fit(float * input, int n_rows, int n_cols, int * labels,
                         int n_trees, float max_features, float rows_sample) {};

		void predict(const float * input, int n_rows, int n_cols, int * preds) {};
	}; 


};
