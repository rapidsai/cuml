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
#include "algo_helper.h"
#include "Timer.h"
#include "kernels/gini_def.h"
#include "memory.cuh"
#include "Timer.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>
#include <climits>
#include <common/cumlHandle.hpp>

namespace ML {
namespace DecisionTree {

template<class T>
struct Question {
	int column;
	T value;
	void update(const GiniQuestion<T> & ques);

};

template<class T>
struct TreeNode {

	TreeNode *left = nullptr;
	TreeNode *right = nullptr;
	int class_predict;
	Question<T> question;
	T gini_val;

	void print(std::ostream& os) const;
};

struct DataInfo
{
	unsigned int NLocalrows;
	unsigned int NGlobalrows;
	unsigned int Ncols;
};

template<class T>
class DecisionTreeClassifier
{
private:
	int split_algo;
	TreeNode<T> *root = nullptr;
	int nbins;
	DataInfo dinfo;
	int treedepth;
	int depth_counter = 0;
	int maxleaves;
	int leaf_counter = 0;
	std::vector<TemporaryMemory<T>*> tempmem;
	size_t total_temp_mem;
	const int MAXSTREAMS = 1;
	size_t max_shared_mem;
	size_t shmem_used = 0;
	int n_unique_labels = -1; // number of unique labels in dataset
	double construct_time;

public:
	// Expects column major T dataset, integer labels
	// data, labels are both device ptr.
	// Assumption: labels are all mapped to contiguous numbers starting from 0 during preprocessing. Needed for gini hist impl.
	void fit(const ML::cumlHandle& handle, T *data, const int ncols, const int nrows, int *labels, unsigned int *rowids, const int n_sampled_rows, int unique_labels,
			int maxdepth = -1, int max_leaf_nodes = -1, const float colper = 1.0, int n_bins = 8, int split_algo=SPLIT_ALGO::HIST);

	/* Predict a label for single row for a given tree. */
	int predict(const T * row, bool verbose=false);

	// Printing utility for high level tree info.
	void print_tree_summary();

	// Printing utility for debug and looking at nodes and leaves.
	void print();

private:
	// Same as above fit, but planting is better for a tree then fitting.
	void plant(const cumlHandle_impl& handle, T *data, const int ncols, const int nrows, int *labels, unsigned int *rowids, const int n_sampled_rows, int unique_labels,
				int maxdepth = -1, int max_leaf_nodes = -1, const float colper = 1.0, int n_bins = 8, int split_algo_flag = SPLIT_ALGO::HIST);

	TreeNode<T> * grow_tree(T *data, const float colper, int *labels, int depth, unsigned int* rowids, const int n_sampled_rows, GiniInfo prev_split_info);

	/* depth is used to distinguish between root and other tree nodes for computations */
	void find_best_fruit_all(T *data, int *labels, const float colper, GiniQuestion<T> & ques, float& gain, unsigned int* rowids,
							const int n_sampled_rows, GiniInfo split_info[3], int depth);
	void split_branch(T *data, GiniQuestion<T> & ques, const int n_sampled_rows, int& nrowsleft, int& nrowsright, unsigned int* rowids);
	int classify(const T * row, const TreeNode<T> * const node, bool verbose=false) const;
	void print_node(const std::string& prefix, const TreeNode<T>* const node, bool isLeft) const;
}; // End DecisionTree Class

} //End namespace DecisionTree


// Stateless API functions
void fit(DecisionTree::DecisionTreeClassifier<float> * dt_classifier, const ML::cumlHandle& handle, float *data, const int ncols, const int nrows, int *labels,
		unsigned int *rowids, const int n_sampled_rows, int unique_labels, int maxdepth = -1, int max_leaf_nodes = -1, const float colper = 1.0,
		int n_bins = 8, int split_algo=SPLIT_ALGO::HIST);

void fit(DecisionTree::DecisionTreeClassifier<double> * dt_classifier, const ML::cumlHandle& handle, double *data, const int ncols, const int nrows, int *labels,
		unsigned int *rowids, const int n_sampled_rows, int unique_labels, int maxdepth = -1, int max_leaf_nodes = -1, const float colper = 1.0,
		int n_bins = 8, int split_algo=SPLIT_ALGO::HIST);

int predict(DecisionTree::DecisionTreeClassifier<float> * dt_classifier, const float * row, bool verbose=false);
int predict(DecisionTree::DecisionTreeClassifier<double> * dt_classifier, const double * row, bool verbose=false);

} //End namespace ML
