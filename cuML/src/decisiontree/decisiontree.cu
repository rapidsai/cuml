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

#include <utils.h>
#include "decisiontree.h"
#include "kernels/gini.cuh"
#include "kernels/split_labels.cuh"
#include "kernels/col_condenser.cuh"
#include "kernels/evaluate.cuh"
#include "kernels/quantile.cuh"

namespace ML {
namespace DecisionTree {

template<class T>
void Question<T>::update(const GiniQuestion<T> & ques) {
	column = ques.original_column;
	value = ques.value;
}

template<class T>
void TreeNode<T>::print(std::ostream& os) const {

	if (left == nullptr && right == nullptr)
		os << "(leaf, " << class_predict << ", " << gini_val << ")" ;
	else
		os << "(" << question.column << ", " << question.value << ", " << gini_val << ")" ;

	return;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const TreeNode<T> * const node) {
	node->print(os);
	return os;
}

DecisionTreeParams::DecisionTreeParams() {};
DecisionTreeParams::DecisionTreeParams(int cfg_max_depth, int cfg_max_leaves, float cfg_max_features, int cfg_n_bins, int cfg_split_algo,
										int cfg_min_rows_per_node):max_depth(cfg_max_depth), max_leaves(cfg_max_leaves),
										max_features(cfg_max_features), n_bins(cfg_n_bins), split_algo(cfg_split_algo),
										min_rows_per_node(cfg_min_rows_per_node) {};

void DecisionTreeParams::validity_check() const {
	ASSERT((max_depth == -1) || (max_depth > 0), "Invalid max depth %d", max_depth);
	ASSERT((max_leaves == -1) || (max_leaves > 0), "Invalid max leaves %d", max_leaves);
	ASSERT((max_features > 0) && (max_features <= 1.0), "max_features value %f outside permitted (0, 1] range", max_features);
	ASSERT((n_bins > 0), "Invalid n_bins %d", n_bins);
	ASSERT((split_algo >= 0) && (split_algo < SPLIT_ALGO::SPLIT_ALGO_END), "split_algo value %d outside permitted [0, %d) range",
			split_algo, SPLIT_ALGO::SPLIT_ALGO_END);
	ASSERT((min_rows_per_node > 0), "Invalid min # rows per node %d", min_rows_per_node);
}

void DecisionTreeParams::print() const {
	std::cout << "max_depth: " << max_depth << std::endl;
	std::cout << "max_leaves: " << max_leaves << std::endl;
	std::cout << "max_features: " << max_features << std::endl;
	std::cout << "n_bins: " << n_bins << std::endl;
	std::cout << "split_algo: " << split_algo << std::endl;
	std::cout << "min_rows_per_node: " << min_rows_per_node << std::endl;
}

/**
 * @brief Build (i.e., fit, train) Decision Tree classifier for input data.
 * @tparam T: data type for input data (float or double).
 * @param[in] handle: cumlHandle
 * @param[in] data: train data (nrows samples, ncols features) in column major format, excluding labels. Device pointer.
 * @param[in] nrows: number of training data samples of the whole unsampled dataset.
 * @param[in] ncols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (int only). One label per training sample. Device pointer.
				  Assumption: labels need to preprocessed to map to ascending numbers from 0;
				  needed for current gini impl in decision tree
 * @param[in] n_sampled_rows: number of training samples, after sampling. If using decsion tree directly over the whole dataset (n_sampled_rows = nrows)
 * @param[in,out] rowids: This array consists of integers from (0 - n_sampled_rows), the same array is then rearranged when splits are made. This allows, us to contruct trees without rearranging the actual dataset. Device pointer.
 * @param[in] n_unique_labels: #unique label values. Number of categories of classification.
 * @param[in] tree_params: Decision Tree training hyper parameter struct
 */
template<typename T>
void DecisionTreeClassifier<T>::fit(const ML::cumlHandle& handle, T *data, const int ncols, const int nrows, int *labels,
									unsigned int *rowids, const int n_sampled_rows, int unique_labels, DecisionTreeParams tree_params) {
	tree_params.validity_check();
	if (tree_params.n_bins > n_sampled_rows) {
		std::cout << "Warning! Calling with number of bins > number of rows! ";
		std::cout << "Resetting n_bins to " << n_sampled_rows << "." << std::endl;
		tree_params.n_bins = n_sampled_rows;
	}
	return plant(handle.getImpl(), data, ncols, nrows, labels, rowids, n_sampled_rows, unique_labels, tree_params.max_depth,
				tree_params.max_leaves, tree_params.max_features, tree_params.n_bins, tree_params.split_algo, tree_params.min_rows_per_node);
}

/**
 * @brief Predict target feature for input data; n-ary classification for single feature supported. Inference of tress is CPU only for now.
 * @tparam T: data type for input data (float or double).
 * @param[in] user_handle: cumlHandle (currently unused; API placeholder)
 * @param[in] rows: test data (n_rows samples, n_cols features) in row major format. CPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in,out] predictions: n_rows predicted labels. CPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
template<typename T>
void DecisionTreeClassifier<T>::predict(const ML::cumlHandle& handle, const T * rows, const int n_rows, const int n_cols, int* predictions, bool verbose) {
	ASSERT(root, "Cannot predict w/ empty tree!");
	ASSERT((n_rows > 0), "Invalid n_rows %d", n_rows);
	ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);
	classify_all(rows, n_rows, n_cols, predictions, verbose);
}

template<typename T>
void DecisionTreeClassifier<T>::print_tree_summary() {
	std::cout << " Decision Tree depth --> " << depth_counter << " and n_leaves --> " << leaf_counter << std::endl;
	std::cout << " Total temporary memory usage--> "<< ((double)total_temp_mem / (1024*1024)) << "  MB" << std::endl;
	std::cout << " Tree growing time --> " << construct_time << " seconds" << std::endl;
	std::cout << " Shared memory used --> " << shmem_used << "  bytes " << std::endl;
}

template<typename T>
void DecisionTreeClassifier<T>::print() {
	print_tree_summary();
	print_node("", root, false);
}

template<typename T>
void DecisionTreeClassifier<T>::plant(const cumlHandle_impl& handle, T *data, const int ncols, const int nrows, int *labels, unsigned int *rowids, const int n_sampled_rows,
									int unique_labels, int maxdepth, int max_leaf_nodes, const float colper, int n_bins, int split_algo_flag, int cfg_min_rows_per_node) {

	split_algo = split_algo_flag;
	dinfo.NLocalrows = nrows;
	dinfo.NGlobalrows = nrows;
	dinfo.Ncols = ncols;
	nbins = n_bins;
	treedepth = maxdepth;
	maxleaves = max_leaf_nodes;
	tempmem.resize(MAXSTREAMS);
	n_unique_labels = unique_labels;
	min_rows_per_node = cfg_min_rows_per_node;

	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	max_shared_mem = prop.sharedMemPerBlock;

	if (split_algo == SPLIT_ALGO::HIST) {
		shmem_used += 2 * sizeof(T) * ncols;
		shmem_used += nbins * n_unique_labels * sizeof(int) * ncols;
	} else {
		shmem_used += nbins * n_unique_labels * sizeof(int) * ncols;
	}
	ASSERT(shmem_used <= max_shared_mem, "Shared memory per block limit %zd , requested %zd \n", max_shared_mem, shmem_used);

	for (int i = 0; i < MAXSTREAMS; i++) {
		tempmem[i] = std::make_shared<TemporaryMemory<T>>(handle, n_sampled_rows, ncols, MAXSTREAMS, unique_labels, n_bins, split_algo);
		if (split_algo == SPLIT_ALGO::GLOBAL_QUANTILE) {
			preprocess_quantile(data, rowids, n_sampled_rows, ncols, dinfo.NLocalrows, n_bins, tempmem[i]);
		}
	}
	total_temp_mem = tempmem[0]->totalmem;
	total_temp_mem *= MAXSTREAMS;
	GiniInfo split_info;
	MLCommon::TimerCPU timer;
	root = grow_tree(data, colper, labels, 0, rowids, n_sampled_rows, split_info);
	construct_time = timer.getElapsedSeconds();

	for (int i = 0; i < MAXSTREAMS; i++) {
		tempmem[i].reset();
	}

	return;
}

template<typename T>
TreeNode<T>* DecisionTreeClassifier<T>::grow_tree(T *data, const float colper, int *labels, int depth, unsigned int* rowids,
												const int n_sampled_rows, GiniInfo prev_split_info) {

	TreeNode<T> *node = new TreeNode<T>();
	GiniQuestion<T> ques;
	Question<T> node_ques;
	float gain = 0.0;
	GiniInfo split_info[3]; // basis, left, right. Populate this
	split_info[0] = prev_split_info;

	bool condition = ((depth != 0) && (prev_split_info.best_gini == 0.0f));  // This node is a leaf, no need to search for best split
	condition = condition || (n_sampled_rows < min_rows_per_node); // Do not split a node with less than min_rows_per_node samples

	if (!condition)  {
		find_best_fruit_all(data, labels, colper, ques, gain, rowids, n_sampled_rows, &split_info[0], depth);  //ques and gain are output here
		condition = condition || (gain == 0.0f);
	}

	if (treedepth != -1)
		condition = (condition || (depth == treedepth));

	if (maxleaves != -1)
		condition = (condition || (leaf_counter >= maxleaves)); // FIXME not fully respecting maxleaves, but >= constraints it more than ==

	if (condition) {
		node->class_predict = get_class_hist(split_info[0].hist);
		node->gini_val = split_info[0].best_gini;

		leaf_counter++;
		if (depth > depth_counter)
			depth_counter = depth;
	} else {
		int nrowsleft, nrowsright;
		split_branch(data, ques, n_sampled_rows, nrowsleft, nrowsright, rowids); // populates ques.value
		node_ques.update(ques);
		node->question = node_ques;
		node->left = grow_tree(data, colper, labels, depth+1, &rowids[0], nrowsleft, split_info[1]);
		node->right = grow_tree(data, colper, labels, depth+1, &rowids[nrowsleft], nrowsright, split_info[2]);
		node->gini_val = split_info[0].best_gini;
	}
	return node;
}


template<typename T>
void DecisionTreeClassifier<T>::find_best_fruit_all(T *data, int *labels, const float colper, GiniQuestion<T> & ques, float& gain,
												unsigned int* rowids, const int n_sampled_rows, GiniInfo split_info[3], int depth) {

	// Bootstrap columns
	std::vector<int> colselector(dinfo.Ncols);
	std::iota(colselector.begin(), colselector.end(), 0);
	std::random_shuffle(colselector.begin(), colselector.end());
	colselector.resize((int)(colper * dinfo.Ncols ));

	CUDA_CHECK(cudaHostRegister(colselector.data(), sizeof(int) * colselector.size(), cudaHostRegisterDefault));
	// Copy sampled column IDs to device memory
	CUDA_CHECK(cudaMemcpyAsync(tempmem[0]->d_colids->data(), colselector.data(), sizeof(int) * colselector.size(), cudaMemcpyHostToDevice, tempmem[0]->stream));
	CUDA_CHECK(cudaStreamSynchronize(tempmem[0]->stream));
	
	// Optimize ginibefore; no need to compute except for root.
	if (depth == 0) {
		int *labelptr = tempmem[0]->sampledlabels->data();
		get_sampled_labels(labels, labelptr, rowids, n_sampled_rows, tempmem[0]->stream);
		gini(labelptr, n_sampled_rows, tempmem[0], split_info[0], n_unique_labels);
	}

	int current_nbins = (n_sampled_rows < nbins) ? n_sampled_rows : nbins;
	best_split_all_cols(data, rowids, labels, current_nbins, n_sampled_rows, n_unique_labels, dinfo.NLocalrows, colselector,
						tempmem[0], &split_info[0], ques, gain, split_algo);

	//Unregister
	CUDA_CHECK(cudaHostUnregister(colselector.data()));
}

template<typename T>
void DecisionTreeClassifier<T>::split_branch(T *data, GiniQuestion<T> & ques, const int n_sampled_rows, int& nrowsleft,
											int& nrowsright, unsigned int* rowids) {

	T *temp_data = tempmem[0]->temp_data->data();
	T *sampledcolumn = &temp_data[n_sampled_rows * ques.bootstrapped_column];
	make_split(sampledcolumn, ques, n_sampled_rows, nrowsleft, nrowsright, rowids, split_algo, tempmem[0]);
}

template<typename T>
void DecisionTreeClassifier<T>::classify_all(const T * rows, const int n_rows, const int n_cols, int* preds, bool verbose) const {
	for (int row_id = 0; row_id < n_rows; row_id++) {
		preds[row_id] = classify(&rows[row_id * n_cols], root, verbose);
	}
	return;
}

template<typename T>
int DecisionTreeClassifier<T>::classify(const T * row, const TreeNode<T>* const node, bool verbose) const {

	Question<T> q = node->question;
	if (node->left && (row[q.column] <= q.value)) {
		if (verbose)
			std::cout << "Classifying Left @ node w/ column " << q.column << " and value " << q.value << std::endl;
		return classify(row, node->left, verbose);
	} else if (node->right && (row[q.column] > q.value)) {
		if (verbose)
			std::cout << "Classifying Right @ node w/ column " << q.column << " and value " << q.value << std::endl;
		return classify(row, node->right, verbose);
	} else {
		if (verbose)
			std::cout << "Leaf node. Predicting " << node->class_predict << std::endl;
		return node->class_predict;
	}
}

template<typename T>
void DecisionTreeClassifier<T>::print_node(const std::string& prefix, const TreeNode<T>* const node, bool isLeft) const {

	if (node != nullptr) {
		std::cout << prefix;

		std::cout << (isLeft ? "├" : "└" );

		// print the value of the node
		std::cout << node << std::endl;

		// enter the next tree level - left and right branch
		print_node( prefix + (isLeft ? "│   " : "    "), node->left, true);
		print_node( prefix + (isLeft ? "│   " : "    "), node->right, false);
	}
}

//Class specializations
template class DecisionTreeClassifier<float>;
template class DecisionTreeClassifier<double>;

} //End namespace DecisionTree


/**
 * @brief Build (i.e., fit, train) Decision Tree classifier for input data.
 * @param[in] handle: cumlHandle
 * @param[in,out] dt_classifier: Pointer to Decision Tree Classifier object. The object holds the trained tree.
 * @param[in] data: train data in float (nrows samples, ncols features) in column major format, excluding labels. Device pointer.
 * @param[in] nrows: number of training data samples of the whole unsampled dataset.
 * @param[in] ncols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (int only). One label per training sample. Device pointer.
				  Assumption: labels need to preprocessed to map to ascending numbers from 0;
				  needed for current gini impl in decision tree
 * @param[in] n_sampled_rows: number of training samples, after sampling. If using decsion tree directly over the whole dataset (n_sampled_rows = nrows)
 * @param[in,out] rowids: This array consists of integers from (0 - n_sampled_rows), the same array is then rearranged when splits are made. This allows, us to contruct trees without rearranging the actual dataset. Device pointer.
 * @param[in] n_unique_labels: #unique label values. Number of categories of classification.
 * @param[in] tree_params: Decision Tree training hyper parameter struct
 */
void fit(const ML::cumlHandle& handle, DecisionTree::DecisionTreeClassifier<float> * dt_classifier, float *data, const int ncols, const int nrows, int *labels, 
		unsigned int *rowids, const int n_sampled_rows, int unique_labels, DecisionTree::DecisionTreeParams tree_params) {
	dt_classifier->fit(handle, data, ncols, nrows, labels, rowids, n_sampled_rows, unique_labels, tree_params);
}

/**
 * @brief Build (i.e., fit, train) Decision Tree classifier for input data.
 * @param[in] handle: cumlHandle
 * @param[in,out] dt_classifier: Pointer to Decision Tree Classifier object. The object holds the trained tree.
 * @param[in] data: train data in double (nrows samples, ncols features) in column major format, excluding labels. Device pointer.
 * @param[in] nrows: number of training data samples of the whole unsampled dataset.
 * @param[in] ncols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (int only). One label per training sample. Device pointer.
				  Assumption: labels need to preprocessed to map to ascending numbers from 0;
				  needed for current gini impl in decision tree
 * @param[in] n_sampled_rows: number of training samples, after sampling. If using decsion tree directly over the whole dataset (n_sampled_rows = nrows)
 * @param[in,out] rowids: This array consists of integers from (0 - n_sampled_rows), the same array is then rearranged when splits are made. This allows, us to contruct trees without rearranging the actual dataset. Device pointer.
 * @param[in] n_unique_labels: #unique label values. Number of categories of classification.
 * @param[in] tree_params: Decision Tree training hyper parameter struct
 */	
void fit(const ML::cumlHandle& handle, DecisionTree::DecisionTreeClassifier<double> * dt_classifier, double *data, const int ncols, const int nrows, int *labels, 
		unsigned int *rowids, const int n_sampled_rows, int unique_labels, DecisionTree::DecisionTreeParams tree_params) {
	dt_classifier->fit(handle, data, ncols, nrows, labels, rowids, n_sampled_rows, unique_labels, tree_params);
}

/**
 * @brief Predict target feature for input data; n-ary classification for single feature supported. Inference of tress is CPU only for now.
 * @param[in] handle: cumlHandle (currently unused; API placeholder)
 * @param[in] dt_classifier: Pointer to decision tree object, which holds the trained tree. 
 * @param[in] rows: test data type float (n_rows samples, n_cols features) in row major format. CPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in,out] predictions: n_rows predicted labels. CPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
void predict(const ML::cumlHandle& handle, const DecisionTree::DecisionTreeClassifier<float> * dt_classifier, const float * rows, const int n_rows, const int n_cols, int* predictions, bool verbose) {
	return dt_classifier->predict(handle, rows, n_rows, n_cols, predictions, verbose);
}
	
/**
 * @brief Predict target feature for input data; n-ary classification for single feature supported. Inference of tress is CPU only for now.
 * @param[in] handle: cumlHandle (currently unused; API placeholder)
 * @param[in] dt_classifier: Pointer to decision tree object, which holds the trained tree. 
 * @param[in] rows: test data type double (n_rows samples, n_cols features) in row major format. CPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in,out] predictions: n_rows predicted labels. CPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
void predict(const ML::cumlHandle& handle, const DecisionTree::DecisionTreeClassifier<double> * dt_classifier, const double * rows, const int n_rows, const int n_cols, int* predictions, bool verbose) {
	return dt_classifier->predict(handle, rows, n_rows, n_cols, predictions, verbose);
}


} //End namespace ML
