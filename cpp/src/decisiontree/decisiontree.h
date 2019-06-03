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
#include "decisiontree_rf_params.h"

namespace ML {

// Stateless API functions
void fit(const ML::cumlHandle& handle, DecisionTree::DecisionTreeClassifier<float> * dt_classifier, float *data, const int ncols, const int nrows, int *labels,
		unsigned int *rowids, const int n_sampled_rows, int unique_labels, DecisionTree::DecisionTreeParams tree_params);

void fit(const ML::cumlHandle& handle, DecisionTree::DecisionTreeClassifier<double> * dt_classifier, double *data, const int ncols, const int nrows, int *labels,
		unsigned int *rowids, const int n_sampled_rows, int unique_labels, DecisionTree::DecisionTreeParams tree_params);

void predict(const ML::cumlHandle& handle, const DecisionTree::DecisionTreeClassifier<float> * dt_classifier, const float * rows,
			const int n_rows, const int n_cols, int* predictions, bool verbose=false);
void predict(const ML::cumlHandle& handle, const DecisionTree::DecisionTreeClassifier<double> * dt_classifier, const double * rows,
			const int n_rows, const int n_cols, int* predictions, bool verbose=false);

} //End namespace ML
