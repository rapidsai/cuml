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

namespace ML {
	namespace DecisionTree {
		
		template<> void DecisionTreeClassifier<float>::fit(float *data, const int ncols, const int nrows, int *labels, unsigned int *rowids, const int n_sampled_rows, int unique_labels, int maxdepth, int max_leaf_nodes, const float colper, int n_bins, int split_algo);
		template<> int DecisionTreeClassifier<float>::predict(const float * row, bool verbose);
		template<> void DecisionTreeClassifier<float>::print_tree_summary();
		template<> void DecisionTreeClassifier<float>::print();
		
		template<> void DecisionTreeClassifier<double>::fit(double *data, const int ncols, const int nrows, int *labels, unsigned int *rowids, const int n_sampled_rows, int unique_labels, int maxdepth, int max_leaf_nodes, const float colper, int n_bins, int split_algo);
		template<> int DecisionTreeClassifier<double>::predict(const double * row, bool verbose);
		template<> void DecisionTreeClassifier<double>::print_tree_summary();
		template<> void DecisionTreeClassifier<double>::print();
	}
}
