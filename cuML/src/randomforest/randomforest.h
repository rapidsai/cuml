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

	struct RF_metrics {
		float accuracy;

		RF_metrics(float cfg_accuracy) : accuracy(cfg_accuracy) {};

		void print() {
			std::cout << "Accuracy: " << accuracy << std::endl;
		}
	};	

	enum RF_type {
		CLASSIFICATION, REGRESSION,
	};

	/* Update labels so they are unique from 0 to n_unique_vals. 
   		Create an old_label to new_label map per random forest.
	*/
	void preprocess_labels(int n_rows, std::vector<int> & labels, std::map<int, int> & labels_map, bool verbose=false) {
	
		std::pair<std::map<int, int>::iterator, bool> ret;
		int n_unique_labels = 0;
	
		if (verbose) std::cout << "Preprocessing labels\n";
		for (int i = 0; i < n_rows; i++) {
	  				ret = labels_map.insert(std::pair<int, int>(labels[i], n_unique_labels));
	  				if (ret.second) {
				n_unique_labels += 1;
			}
			if (verbose) std::cout << "Mapping " << labels[i] << " to ";
			labels[i] = ret.first->second; //Update labels **IN-PLACE**
			if (verbose) std::cout << labels[i] << std::endl;
		}
		if (verbose) std::cout << "Finished preprocessing labels\n";
	
	}


	/* Revert preprocessing effect, if needed. */
	void postprocess_labels(int n_rows, std::vector<int> labels, std::map<int, int> & labels_map, bool verbose=false) {
	
		if (verbose) std::cout << "Postrocessing labels\n";
		std::map<int, int>::iterator it;
		int n_unique_cnt = labels_map.size();
		std::vector<int> reverse_map;
		reverse_map.resize(n_unique_cnt);
		for (auto it = labels_map.begin(); it != labels_map.end(); it++) {
			reverse_map[it->second] = it->first;
		}
	
		for (int i = 0; i < n_rows; i++) {
			if (verbose) std::cout << "Mapping " << labels[i] << " back to " << reverse_map[labels[i]] << std::endl;
			labels[i] = reverse_map[labels[i]];
		}
		if (verbose) std::cout << "Finished postrocessing labels\n";
	}


	class rf {
		protected:
			int n_trees, n_bins, rf_type;
			int max_depth, max_leaves; 
			bool bootstrap;
			float rows_sample, max_features; // ratio of n_rows used per tree
         	//max_features	number of features to consider per split (default = sqrt(n_cols))

			DecisionTree::DecisionTreeClassifier * trees;
		
		public:
			rf(int cfg_n_trees, bool cfg_bootstrap=true, int cfg_max_depth=-1, int cfg_max_leaves=-1, int cfg_rf_type=RF_type::CLASSIFICATION, int cfg_n_bins=8,
			   float cfg_rows_sample=1.0f, float cfg_max_features=1.0f) {

					n_trees = cfg_n_trees;
					max_depth = cfg_max_depth;
					max_leaves = cfg_max_leaves;
					trees = NULL; 
					rf_type = cfg_rf_type;
					bootstrap = cfg_bootstrap;
					n_bins = cfg_n_bins;
					rows_sample = cfg_rows_sample;
					max_features = cfg_max_features;

					ASSERT((n_trees > 0), "Invalid n_trees %d", n_trees);
					ASSERT((cfg_n_bins > 0), "Invalid n_bins %d", cfg_n_bins);
					ASSERT((rows_sample > 0) && (rows_sample <= 1.0), "rows_sample value %f outside permitted (0, 1] range", rows_sample);
					ASSERT((max_features > 0) && (max_features <= 1.0), "max_features value %f outside permitted (0, 1] range", max_features);
			}

			~rf() {
					delete [] trees;
			}

			int get_ntrees() {
				return n_trees;
			}

			void print_rf_summary() {

				if (!trees) std::cout << "Empty forest" << std::endl;
				else {
					std::cout << "Forest has " << n_trees << " trees, max_depth " << max_depth << ", and max_leaves " << max_leaves << std::endl;
					for (int i = 0; i < n_trees; i++) {
						std::cout << "Tree #" << i << std::endl;
						trees[i].print_tree_summary();
					}
				}
			}

			void print_rf_detailed() {

				if (!trees) std::cout << "Empty forest" << std::endl;
				else {
					std::cout << "Forest has " << n_trees << " trees, max_depth " << max_depth << ", and max_leaves " << max_leaves << std::endl;
					for (int i = 0; i < n_trees; i++) {
						std::cout << "Tree #" << i << std::endl;
						trees[i].print();
					}
				}
			}

    };

//FIXME: input could be of different type.
//FIXME: labels for regression could be of different type too, potentially match input type. 

	class rfClassifier : public rf {
		public:

		rfClassifier(int cfg_n_trees, bool cfg_bootstrap=true, int cfg_max_depth=-1, int cfg_max_leaves=-1, int cfg_rf_type=RF_type::CLASSIFICATION, int cfg_n_bins=8,
						float cfg_rows_sample=1.0f, float cfg_max_features=1.0f) 
					: rf::rf(cfg_n_trees, cfg_bootstrap, cfg_max_depth, cfg_max_leaves, cfg_rf_type, cfg_n_bins, cfg_rows_sample, cfg_max_features) {};


        /** 
         * Fit an RF classification model on input data with n_rows samples and n_cols features.
         * @param input			data array in col major format for now (device ptr)
         * @param n_rows		number of training data rows
         * @param n_cols		number of features (i.e, columns)
         * @param labels		list of target features (device ptr).
								Assumption: labels were preprocessed to map to ascending numbers from 0;
							    needed for current gini impl in decision tree
        */
		void fit(float * input, int n_rows, int n_cols, int * labels) {

			ASSERT(!trees, "Cannot fit an existing forest.");
			ASSERT((n_rows > 0), "Invalid n_rows %d", n_rows);
			ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);

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
				std::cout << "Fitting tree # " << i << std::endl;
				trees[i].fit(input, n_cols, n_rows, labels, selected_rows, n_sampled_rows, max_depth, max_leaves, max_features, n_bins);

				//Cleanup
				CUDA_CHECK(cudaFree(selected_rows));

			}

		}	


		//Assuming input in row_major format. input is a CPU ptr.
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

		
		/* Predict input data and validate against ref_labels. input and ref_labels are both CPU ptrs. */
		RF_metrics cross_validate(const float * input, const int * ref_labels, int n_rows, int n_cols, bool verbose=false) {

			int * predictions = predict(input, n_rows, n_cols, verbose);

			unsigned long long correctly_predicted = 0ULL;
			for (int i = 0; i < n_rows; i++) {
				correctly_predicted += (predictions[i] == ref_labels[i]);
			}

			float accuracy = correctly_predicted * 1.0f/n_rows;
			RF_metrics stats(accuracy);
			if (verbose) stats.print();

			/* TODO: Potentially augment RF_metrics w/ more metrics (e.g., precision, F1, etc.).
			   For non binary classification problems (i.e., one target and  > 2 labels), need avg for each of these metrics */

			return stats;
		}

};


	class rfRegressor : public rf {
	    public:

		rfRegressor(int cfg_n_trees, bool cfg_bootstrap=true, int cfg_max_depth=-1, int cfg_max_leaves=-1, int cfg_rf_type=RF_type::REGRESSION, int cfg_n_bins=8, 
						float cfg_rows_sample=1.0f, float cfg_max_features=1.0f) 
					: rf::rf(cfg_n_trees, cfg_bootstrap, cfg_max_depth, cfg_max_leaves, cfg_rf_type, cfg_n_bins, cfg_rows_sample, cfg_max_features) {};

		void fit(float * input, int n_rows, int n_cols, int * labels,
                         int n_trees, float max_features, float rows_sample) {};

		void predict(const float * input, int n_rows, int n_cols, int * preds) {};
	}; 


};
