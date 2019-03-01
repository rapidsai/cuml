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


#include "randomforest.h"
#include <utils.h>
#include "random/rng.h"
#include "linalg/cublas_wrappers.h"

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

			int * selected_rows; // randomly generated IDs for bootstrapped samples (w/ replacement). 
			CUDA_CHECK(cudaMalloc((void **)& selected_rows, n_sampled_rows * sizeof(int)));

			Random::Rng r(i); //FIXME Ensure the seed for each tree is different and a meaningful one.
			r.uniformInt(selected_rows, n_sampled_rows, 0, n_rows-1);

			float * bootstrapped_input, * col_major_bootstrapped_input; // Temporary memory allocation for bootstrapped sample
			CUDA_CHECK(cudaMalloc((void **)& bootstrapped_input, n_sampled_rows * n_cols * sizeof(float)));
			CUDA_CHECK(cudaMalloc((void **)& col_major_bootstrapped_input, n_sampled_rows * n_cols * sizeof(float)));

		
			//If data was originally in row major format (more convenient)
			int row_size = n_cols * sizeof(float);
			for (int cnt = 0; cnt < n_sampled_rows; cnt++) {
  				CUDA_CHECK(cudaMemcpy(&bootstrapped_input[cnt*n_cols], &input[selected_rows[i]*n_cols], row_size, cudaMemcpyDeviceToDevice));
			}
  			//CUDA_CHECK(cudaMemcpy(col_major_bootstrapped_input, bootstrapped_input, row_size * n_sampled_rows, cudaMemcpyDeviceToDevice));

			// Transpose sample per tree to row-major. FIXME this will need work.
			float const alpha = 1.0f, beta = 0.0f;
			cublasHandle_t handle;
			CUBLAS_CHECK(cublasCreate(&handle));
			// C = alpha * op(A) + beta * C.
			CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
									 n_cols, n_sampled_rows,  // rows, cols of resulting matrix
									 &alpha, bootstrapped_input, n_sampled_rows,  
 									 &beta, bootstrapped_input, n_sampled_rows,  
									 col_major_bootstrapped_input, n_cols));
			CUBLAS_CHECK(cublasDestroy(handle));



			//Build individual tree in the forest.
			trees[i].fit(col_major_bootstrapped_input, n_cols, n_rows, max_features, labels);

			delete bootstrapped_input;
			CUDA_CHECK(cudaFree(selected_rows));
			CUDA_CHECK(cudaFree(bootstrapped_input));
			CUDA_CHECK(cudaFree(col_major_bootstrapped_input));
		}
	}


	
	void rfClassifier::predict(const float * input, int n_rows, int n_cols, int * preds) {
		ASSERT(trees, "Cannot predict! No trees in the forest.");

	}

	void rfRegressor::fit(float * input, int n_rows, int n_cols, int * labels, int n_trees, float max_features, float rows_sample) {
	}

	
	void rfRegressor::predict(const float * input, int n_rows, int n_cols, int * preds) {
	}


};
// end namespace ML
