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


namespace ML {


	void rfClassifier::fit(float * input, int n_rows, int n_cols, int * labels, int n_trees, int max_features) {
	}

	
	void rfClassifier::predict(const float * input, int n_rows, int n_cols, int * preds) {
	}

	void rfRegressor::fit(float * input, int n_rows, int n_cols, int * labels, int n_trees, int max_features) {
	}

	
	void rfRegressor::predict(const float * input, int n_rows, int n_cols, int * preds) {
	}


};
// end namespace ML
