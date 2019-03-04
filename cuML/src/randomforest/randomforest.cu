/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "randomforest.h"
#include <utils.h>
#include "random/rng.h"
#include "linalg/cublas_wrappers.h"
#include <map>


namespace ML {

using namespace MLCommon;

	void rfClassifier::fit(float * input, int n_rows, int n_cols, int * labels, int n_trees, float rows_sample) {
		rfClassifier::fit(input, n_rows, n_cols, labels, n_trees, sqrt(n_cols), rows_sample);
	}

	/* Allocate an array of n_trees DecisionTrees.
	   Call fit on each of these trees on a bootstrapped sample of input w/ rows_sample * n_rows rows sample per tree.
	*/
	void rfClassifier::fit(float * input, int n_rows, int n_cols, int * labels, int n_trees, float max_features, float rows_sample) {

		ASSERT(!trees, "Cannot fit an existing forest.");
		ASSERT((rows_sample > 0) && (rows_sample <= 1.0), "rows_sample value %f outside permitted (0, 1] range", rows_sample);
		ASSERT((max_features > 0) && (max_features <= 1.0), "max_features value %f outside permitted (0, 1] range", max_features);
		ASSERT((n_rows > 0), "Invalid n_rows %d", n_rows);
		ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);
		ASSERT((n_trees > 0), "Invalid n_trees %d", n_trees);

		rfClassifier::trees = new DecisionTree::DecisionTreeClassifier[n_trees];
		int n_sampled_rows = rows_sample * n_rows;

		for (int i = 0; i < n_trees; i++) {
			// Select n_sampled_rows (with replacement) numbers from [0, n_rows) per tree.

			unsigned int * selected_rows; // randomly generated IDs for bootstrapped samples (w/ replacement). 
			CUDA_CHECK(cudaMalloc((void **)& selected_rows, n_sampled_rows * sizeof(unsigned int)));

			Random::Rng r(i); //FIXME Ensure the seed for each tree is different and a meaningful one.
			r.uniformInt(selected_rows, n_sampled_rows, (unsigned int) 0, (unsigned int) n_rows-1);

			/* Build individual tree in the forest.
			   - input is a pointer to orig data that have n_cols features and n_rows rows.
			   - n_sampled_rows: # rows sampled for tree's bootstrap sample.
			   - selected_rows: points to a list of row #s (w/ n_sampled_rows elements) used to build the bootstrapped sample.  
			   Expectation: Each tree node will contain (a) # n_sampled_rows and (b) a pointer to a list of row numbers w.r.t original data. 
			*/
			trees[i].fit(input, n_cols, n_rows, max_features, labels, selected_rows, n_sampled_rows);

			//Cleanup
			CUDA_CHECK(cudaFree(selected_rows));

		}
	}


	//FIXME Assuming input in row_major format. 
	int * rfClassifier::predict(const float * input, int n_rows, int n_cols) {
		ASSERT(trees, "Cannot predict! No trees in the forest.");
		int * preds = new int[n_rows];

		int row_size = n_cols * sizeof(float); //FIXME a lot of assumptions about data format.
		for (int row_id = 0; row_id < n_rows; row_id++) {

			std::map<int, int> prediction_to_cnt;
			std::pair<std::map<int, int>::iterator, bool> ret;
			int max_cnt_so_far = 0;
			int majority_prediction = -1;

			for (int i = 0; i < n_trees; i++) {
				//Return prediction for one sample. 
				int prediction = trees[i].predict(&input[row_id * row_size]);

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

	void rfRegressor::fit(float * input, int n_rows, int n_cols, int * els, int n_trees, float max_features, float rows_sample) {
	}

	
	void rfRegressor::predict(const float * input, int n_rows, int n_cols, int * preds) {
	}


};
// end namespace ML
