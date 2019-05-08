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
void Question<T>::update(const MetricQuestion<T> & ques) {
	column = ques.original_column;
	value = ques.value;
}

template<class T, class L>
void TreeNode<T, L>::print(std::ostream& os) const {

	if (left == nullptr && right == nullptr) {
		os << "(leaf, " << prediction << ", " << split_metric_val << ")" ;
	} else {
		os << "(" << question.column << ", " << question.value << ", " << split_metric_val << ")" ;
	}
	return;
}

template<typename T, typename L>
std::ostream& operator<<(std::ostream& os, const TreeNode<T, L> * const node) {
	node->print(os);
	return os;
}

/**
 * @brief Decision tree hyper-parameter object constructor. All DecisionTreeParams members have their default values.
 */
DecisionTreeParams::DecisionTreeParams() {}

/**
 * @brief Decision tree hyper-parameter object constructor to set all DecisionTreeParams members.
 */
DecisionTreeParams::DecisionTreeParams(int cfg_max_depth, int cfg_max_leaves, float cfg_max_features, int cfg_n_bins, int cfg_split_algo,
				       int cfg_min_rows_per_node, bool cfg_bootstrap_features):max_depth(cfg_max_depth), max_leaves(cfg_max_leaves),
											       max_features(cfg_max_features), n_bins(cfg_n_bins), split_algo(cfg_split_algo),
											       min_rows_per_node(cfg_min_rows_per_node), bootstrap_features(cfg_bootstrap_features) {}

/**
 * @brief Check validity of all decision tree hyper-parameters.
 */
void DecisionTreeParams::validity_check() const {
	ASSERT((max_depth == -1) || (max_depth > 0), "Invalid max depth %d", max_depth);
	ASSERT((max_leaves == -1) || (max_leaves > 0), "Invalid max leaves %d", max_leaves);
	ASSERT((max_features > 0) && (max_features <= 1.0), "max_features value %f outside permitted (0, 1] range", max_features);
	ASSERT((n_bins > 0), "Invalid n_bins %d", n_bins);
	ASSERT((split_algo >= 0) && (split_algo < SPLIT_ALGO::SPLIT_ALGO_END), "split_algo value %d outside permitted [0, %d) range",
			split_algo, SPLIT_ALGO::SPLIT_ALGO_END);
	ASSERT((min_rows_per_node > 0), "Invalid min # rows per node %d", min_rows_per_node);
}

/**
 * @brief Print all decision tree hyper-parameters.
 */
void DecisionTreeParams::print() const {
	std::cout << "max_depth: " << max_depth << std::endl;
	std::cout << "max_leaves: " << max_leaves << std::endl;
	std::cout << "max_features: " << max_features << std::endl;
	std::cout << "n_bins: " << n_bins << std::endl;
	std::cout << "split_algo: " << split_algo << std::endl;
	std::cout << "min_rows_per_node: " << min_rows_per_node << std::endl;
}


/**
 * @brief Print high-level tree information.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 */
template<typename T, typename L>
void dt<T, L>::print_tree_summary() const {
	std::cout << " Decision Tree depth --> " << depth_counter << " and n_leaves --> " << leaf_counter << std::endl;
	std::cout << " Total temporary memory usage--> "<< ((double)total_temp_mem / (1024*1024)) << "  MB" << std::endl;
	std::cout << " Tree growing time --> " << construct_time << " seconds" << std::endl;
	std::cout << " Shared memory used --> " << shmem_used << "  bytes " << std::endl;
}

/**
 * @brief Print detailed tree information.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 */
template<typename T, typename L>
void dt<T, L>::print() const {
	print_tree_summary();
	print_node("", this->root, false);
}


template<typename T, typename L>
void dt<T, L>::print_node(const std::string& prefix, const TreeNode<T, L>* const node, bool isLeft) const {

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


/**
 * @brief Predict target feature for input data; n-ary classification or regression for single feature supported. Inference of trees is CPU only for now.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] handle: cumlHandle (currently unused; API placeholder)
 * @param[in] rows: test data (n_rows samples, n_cols features) in row major format. CPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in,out] predictions: n_rows predicted labels. CPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
template<typename T, typename L>
void dt<T, L>::predict(const ML::cumlHandle& handle, const T * rows, const int n_rows, const int n_cols, L* predictions, bool verbose) const {
	ASSERT(root, "Cannot predict w/ empty tree!");
	ASSERT((n_rows > 0), "Invalid n_rows %d", n_rows);
	ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);
	predict_all(rows, n_rows, n_cols, predictions, verbose);
}

template<typename T, typename L>
void dt<T, L>::predict_all(const T * rows, const int n_rows, const int n_cols, L * preds, bool verbose) const {
	for (int row_id = 0; row_id < n_rows; row_id++) {
		preds[row_id] = predict_one(&rows[row_id * n_cols], this->root, verbose);
	}
}

template<typename T, typename L>
L dt<T, L>::predict_one(const T * row, const TreeNode<T, L>* const node, bool verbose) const {

	Question<T> q = node->question;
	if (node->left && (row[q.column] <= q.value)) {
		if (verbose) {
			std::cout << "Classifying Left @ node w/ column " << q.column << " and value " << q.value << std::endl;
		}
		return predict_one(row, node->left, verbose);
	} else if (node->right && (row[q.column] > q.value)) {
		if (verbose) {
			std::cout << "Classifying Right @ node w/ column " << q.column << " and value " << q.value << std::endl;
		}
		return predict_one(row, node->right, verbose);
	} else {
		if (verbose) {
			std::cout << "Leaf node. Predicting " << node->prediction << std::endl;
		}
		return node->prediction;
	}
}


/**
 * @brief Build (i.e., fit, train) Decision Tree classifier for input data.
 * @tparam T: data type for input data (float or double).
 * @param[in] handle: cumlHandle
 * @param[in] data: train data (nrows samples, ncols features) in column major format, excluding labels. Device pointer.
 * @param[in] ncols: number of features (i.e., columns) excluding target feature.
 * @param[in] nrows: number of training data samples of the whole unsampled dataset.
 * @param[in] labels: 1D array of target features (int only). One label per training sample. Device pointer.
				  Assumption: labels need to be preprocessed to map to ascending numbers from 0;
				  needed for current gini impl. in decision tree.
 * @param[in,out] rowids: array of n_sampled_rows integers in [0, nrows) range. Device pointer.
						  The same array is then rearranged when splits are made, allowing us to construct trees without rearranging the actual dataset.
 * @param[in] n_sampled_rows: number of training samples, after sampling. If using decision tree directly over the whole dataset: n_sampled_rows = nrows
 * @param[in] n_unique_labels: #unique label values. Number of categories of classification.
 * @param[in] tree_params: Decision Tree training hyper parameter struct.
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
		     tree_params.max_leaves, tree_params.max_features, tree_params.n_bins, tree_params.split_algo, tree_params.min_rows_per_node, tree_params.bootstrap_features);
}

template<typename T>
void DecisionTreeClassifier<T>::plant(const cumlHandle_impl& handle, T *data, const int ncols, const int nrows, int *labels, unsigned int *rowids, const int n_sampled_rows,
				      int unique_labels, int maxdepth, int max_leaf_nodes, const float colper, int n_bins, int split_algo_flag, int cfg_min_rows_per_node, bool cfg_bootstrap_features) {

	this->split_algo = split_algo_flag;
	this->dinfo.NLocalrows = nrows;
	this->dinfo.NGlobalrows = nrows;
	this->dinfo.Ncols = ncols;
	this->nbins = n_bins;
	this->treedepth = maxdepth;
	this->maxleaves = max_leaf_nodes;
	this->tempmem.resize(this->MAXSTREAMS);
	this->n_unique_labels = unique_labels;
	this->min_rows_per_node = cfg_min_rows_per_node;
	this->bootstrap_features = cfg_bootstrap_features;

	//Bootstrap features
	this->feature_selector.resize(this->dinfo.Ncols);
	if (this->bootstrap_features) {
		srand(n_bins);
		for(int i=0; i < this->dinfo.Ncols; i++) {
			this->feature_selector.push_back( rand() % this->dinfo.Ncols );
		}
	} else {
		std::iota(this->feature_selector.begin(), this->feature_selector.end(), 0);
	}

	std::random_shuffle(this->feature_selector.begin(), this->feature_selector.end());
	this->feature_selector.resize((int) (colper * this->dinfo.Ncols));

	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	this->max_shared_mem = prop.sharedMemPerBlock;

	if (this->split_algo == SPLIT_ALGO::HIST) {
		this->shmem_used += 2 * sizeof(T) * ncols;
		this->shmem_used += this->nbins * this->n_unique_labels * sizeof(int) * ncols;
	} else {
		this->shmem_used += this->nbins * this->n_unique_labels * sizeof(int) * ncols;
	}
	ASSERT(this->shmem_used <= this->max_shared_mem, "Shared memory per block limit %zd , requested %zd \n", this->max_shared_mem, this->shmem_used);

	for (int i = 0; i < this->MAXSTREAMS; i++) {
		this->tempmem[i] = std::make_shared<TemporaryMemory<T>>(handle, n_sampled_rows, ncols, this->MAXSTREAMS, unique_labels, n_bins, this->split_algo);
		if (this->split_algo == SPLIT_ALGO::GLOBAL_QUANTILE) {
			preprocess_quantile(data, rowids, n_sampled_rows, ncols, this->dinfo.NLocalrows, n_bins, this->tempmem[i]);
		}
	}
	this->total_temp_mem = this->tempmem[0]->totalmem;
	this->total_temp_mem *= this->MAXSTREAMS;
	MetricInfo split_info;
	MLCommon::TimerCPU timer;
	this->root = grow_tree(data, colper, labels, 0, rowids, n_sampled_rows, split_info);
	this->construct_time = timer.getElapsedSeconds();

	for (int i = 0; i < this->MAXSTREAMS; i++) {
		this->tempmem[i].reset();
	}

	return;
}

template<typename T>
TreeNode<T, int>* DecisionTreeClassifier<T>::grow_tree(T *data, const float colper, int *labels, int depth, unsigned int* rowids,
												const int n_sampled_rows, MetricInfo prev_split_info) {

	TreeNode<T, int> *node = new TreeNode<T, int>();
	MetricQuestion<T> ques;
	Question<T> node_ques;
	float gain = 0.0;
	MetricInfo split_info[3]; // basis, left, right. Populate this
	split_info[0] = prev_split_info;

	bool condition = ((depth != 0) && (prev_split_info.best_metric == 0.0f));  // This node is a leaf, no need to search for best split
	condition = condition || (n_sampled_rows < this->min_rows_per_node); // Do not split a node with less than min_rows_per_node samples

	if (!condition)  {
		find_best_fruit_all(data, labels, colper, ques, gain, rowids, n_sampled_rows, &split_info[0], depth);  //ques and gain are output here
		condition = condition || (gain == 0.0f);
	}

	if (this->treedepth != -1)
		condition = (condition || (depth == this->treedepth));

	if (this->maxleaves != -1)
		condition = (condition || (this->leaf_counter >= this->maxleaves)); // FIXME not fully respecting maxleaves, but >= constraints it more than ==

	if (condition) {
		node->prediction = get_class_hist(split_info[0].hist);
		node->split_metric_val = split_info[0].best_metric;

		this->leaf_counter++;
		if (depth > this->depth_counter)
			this->depth_counter = depth;
	} else {
		int nrowsleft, nrowsright;
		split_branch(data, ques, n_sampled_rows, nrowsleft, nrowsright, rowids); // populates ques.value
		node_ques.update(ques);
		node->question = node_ques;
		node->left = grow_tree(data, colper, labels, depth+1, &rowids[0], nrowsleft, split_info[1]);
		node->right = grow_tree(data, colper, labels, depth+1, &rowids[nrowsleft], nrowsright, split_info[2]);
		node->split_metric_val = split_info[0].best_metric;
	}
	return node;
}


template<typename T>
void DecisionTreeClassifier<T>::find_best_fruit_all(T *data, int *labels, const float colper, MetricQuestion<T> & ques, float& gain,
												unsigned int* rowids, const int n_sampled_rows, MetricInfo split_info[3], int depth) {
	std::vector<int>& colselector = this->feature_selector;

	// Optimize ginibefore; no need to compute except for root.
	if (depth == 0) {
		CUDA_CHECK(cudaHostRegister(colselector.data(), sizeof(int) * colselector.size(), cudaHostRegisterDefault));
		// Copy sampled column IDs to device memory
		CUDA_CHECK(cudaMemcpyAsync(this->tempmem[0]->d_colids->data(), colselector.data(), sizeof(int) * colselector.size(), cudaMemcpyHostToDevice, this->tempmem[0]->stream));
		CUDA_CHECK(cudaStreamSynchronize(this->tempmem[0]->stream));

		int *labelptr = this->tempmem[0]->sampledlabels->data();
		get_sampled_labels(labels, labelptr, rowids, n_sampled_rows, this->tempmem[0]->stream);
		gini(labelptr, n_sampled_rows, this->tempmem[0], split_info[0], this->n_unique_labels);
		//Unregister
		CUDA_CHECK(cudaHostUnregister(colselector.data()));
	}

	int current_nbins = (n_sampled_rows < this->nbins) ? n_sampled_rows : this->nbins;
	best_split_all_cols(data, rowids, labels, current_nbins, n_sampled_rows, this->n_unique_labels, this->dinfo.NLocalrows, colselector,
						this->tempmem[0], &split_info[0], ques, gain, this->split_algo);
}

template<typename T>
void DecisionTreeClassifier<T>::split_branch(T *data, MetricQuestion<T> & ques, const int n_sampled_rows, int& nrowsleft,
											int& nrowsright, unsigned int* rowids) {

	T *temp_data = this->tempmem[0]->temp_data->data();
	T *sampledcolumn = &temp_data[n_sampled_rows * ques.bootstrapped_column];
	make_split(sampledcolumn, ques, n_sampled_rows, nrowsleft, nrowsright, rowids, this->split_algo, this->tempmem[0]);
}

/**
 * @brief Build (i.e., fit, train) Decision Tree regressor for input data.
 * @tparam T: data type for input data (float or double).
 * @param[in] handle: cumlHandle
 * @param[in] data: train data (nrows samples, ncols features) in column major format, excluding labels. Device pointer.
 * @param[in] ncols: number of features (i.e., columns) excluding target feature.
 * @param[in] nrows: number of training data samples of the whole unsampled dataset.
 * @param[in] labels: 1D array of target features (float or double). One label per training sample. Device pointer.
 * @param[in,out] rowids: array of n_sampled_rows integers in [0, nrows) range. Device pointer.
						  The same array is then rearranged when splits are made, allowing us to construct trees without rearranging the actual dataset.
 * @param[in] n_sampled_rows: number of training samples, after sampling. If using decision tree directly over the whole dataset: n_sampled_rows = nrows
 * @param[in] tree_params: Decision Tree training hyper parameter struct.
 */
template<typename T>
void DecisionTreeRegressor<T>::fit(const ML::cumlHandle& handle, T *data, const int ncols, const int nrows, T *labels,
									unsigned int *rowids, const int n_sampled_rows, DecisionTreeParams tree_params) {
	tree_params.validity_check();
	if (tree_params.n_bins > n_sampled_rows) {
		std::cout << "Warning! Calling with number of bins > number of rows! ";
		std::cout << "Resetting n_bins to " << n_sampled_rows << "." << std::endl;
		tree_params.n_bins = n_sampled_rows;
	}
	// TODO FIXME placeholder
	//plant(handle.getImpl(), data, ncols, nrows, labels, rowids, n_sampled_rows, unique_labels, tree_params.max_depth,
    //		     tree_params.max_leaves, tree_params.max_features, tree_params.n_bins, tree_params.split_algo, tree_params.min_rows_per_node, tree_params.bootstrap_features);
}

// ---------------- Regression end


//Class specializations
template class dt<float, int>;
template class dt<float, float>;
template class dt<double, int>;
template class dt<double, double>;

template class DecisionTreeClassifier<float>;
template class DecisionTreeClassifier<double>;

template class DecisionTreeRegressor<float>;
template class DecisionTreeRegressor<double>;

} //End namespace DecisionTree

// Stateless API functions

// ----------------------------- Classification ----------------------------------- //

/**
 * @brief Build (i.e., fit, train) Decision Tree classifier for input data.
 * @param[in] handle: cumlHandle
 * @param[in,out] dt_classifier: Pointer to Decision Tree Classifier object. The object holds the trained tree.
 * @param[in] data: train data in float (nrows samples, ncols features) in column major format, excluding labels. Device pointer.
 * @param[in] ncols: number of features (i.e., columns) excluding target feature.
 * @param[in] nrows: number of training data samples of the whole unsampled dataset.
 * @param[in] labels: 1D array of target features (int only). One label per training sample. Device pointer.
				  Assumption: labels need to be preprocessed to map to ascending numbers from 0;
				  needed for current gini impl. in decision tree
 * @param[in,out] rowids: This array consists of integers from (0 - n_sampled_rows), the same array is then rearranged when splits are made. This allows, us to contruct trees without rearranging the actual dataset. Device pointer.
 * @param[in] n_sampled_rows: number of training samples, after sampling. If using decsion tree directly over the whole dataset (n_sampled_rows = nrows)
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
 * @param[in] ncols: number of features (i.e., columns) excluding target feature.
 * @param[in] nrows: number of training data samples of the whole unsampled dataset.
 * @param[in] labels: 1D array of target features (int only). One label per training sample. Device pointer.
				  Assumption: labels need to preprocessed to map to ascending numbers from 0;
				  needed for current gini impl in decision tree
 * @param[in,out] rowids: array of n_sampled_rows integers in [0, nrows) range. Device pointer.
						  The same array is then rearranged when splits are made, allowing us to construct trees without rearranging the actual dataset.
 * @param[in] n_sampled_rows: number of training samples, after sampling. If using decsion tree directly over the whole dataset (n_sampled_rows = nrows)
 * @param[in] n_unique_labels: #unique label values. Number of categories of classification.
 * @param[in] tree_params: Decision Tree training hyper parameter struct.
 */
void fit(const ML::cumlHandle& handle, DecisionTree::DecisionTreeClassifier<double> * dt_classifier, double *data, const int ncols, const int nrows, int *labels,
		unsigned int *rowids, const int n_sampled_rows, int unique_labels, DecisionTree::DecisionTreeParams tree_params) {
	dt_classifier->fit(handle, data, ncols, nrows, labels, rowids, n_sampled_rows, unique_labels, tree_params);
}

/**
 * @brief Predict target feature for input data; n-ary classification for single feature supported. Inference of trees is CPU only for now.
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
 * @brief Predict target feature for input data; n-ary classification for single feature supported. Inference of trees is CPU only for now.
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

// ----------------------------- Regression ----------------------------------- //

/**
 * @brief Build (i.e., fit, train) Decision Tree regressor for input data.
 * @param[in] handle: cumlHandle
 * @param[in,out] dt_regressor: Pointer to Decision Tree Regressor object. The object holds the trained tree.
 * @param[in] data: train data in float (nrows samples, ncols features) in column major format, excluding labels. Device pointer.
 * @param[in] ncols: number of features (i.e., columns) excluding target feature.
 * @param[in] nrows: number of training data samples of the whole unsampled dataset.
 * @param[in] labels: 1D array of target features (type float). One label per training sample. Device pointer.
 * @param[in,out] rowids: This array consists of integers from (0 - n_sampled_rows), the same array is then rearranged when splits are made. This allows, us to contruct trees without rearranging the actual dataset. Device pointer.
 * @param[in] n_sampled_rows: number of training samples, after sampling. If using decsion tree directly over the whole dataset (n_sampled_rows = nrows)
 * @param[in] tree_params: Decision Tree training hyper parameter struct
 */
void fit(const ML::cumlHandle& handle, DecisionTree::DecisionTreeRegressor<float> * dt_regressor, float *data, const int ncols, const int nrows, float *labels,
		unsigned int *rowids, const int n_sampled_rows, DecisionTree::DecisionTreeParams tree_params) {
	dt_regressor->fit(handle, data, ncols, nrows, labels, rowids, n_sampled_rows, tree_params);
}

/**
 * @brief Build (i.e., fit, train) Decision Tree regressor for input data.
 * @param[in] handle: cumlHandle
 * @param[in,out] dt_regressor: Pointer to Decision Tree Regressor object. The object holds the trained tree.
 * @param[in] data: train data in double (nrows samples, ncols features) in column major format, excluding labels. Device pointer.
 * @param[in] ncols: number of features (i.e., columns) excluding target feature.
 * @param[in] nrows: number of training data samples of the whole unsampled dataset.
 * @param[in] labels: 1D array of target features (type float). One label per training sample. Device pointer.
 * @param[in,out] rowids: array of n_sampled_rows integers in [0, nrows) range. Device pointer.
						  The same array is then rearranged when splits are made, allowing us to construct trees without rearranging the actual dataset.
 * @param[in] n_sampled_rows: number of training samples, after sampling. If using decsion tree directly over the whole dataset (n_sampled_rows = nrows)
 * @param[in] tree_params: Decision Tree training hyper parameter struct.
 */
void fit(const ML::cumlHandle& handle, DecisionTree::DecisionTreeRegressor<double> * dt_regressor, double *data, const int ncols, const int nrows, double *labels,
		unsigned int *rowids, const int n_sampled_rows, DecisionTree::DecisionTreeParams tree_params) {
	dt_regressor->fit(handle, data, ncols, nrows, labels, rowids, n_sampled_rows, tree_params);
}

/**
 * @brief Predict target feature for input data; regression for single feature supported. Inference of trees is CPU only for now.
 * @param[in] handle: cumlHandle (currently unused; API placeholder)
 * @param[in] dt_regressor: Pointer to decision tree object, which holds the trained tree.
 * @param[in] rows: test data type float (n_rows samples, n_cols features) in row major format. CPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in,out] predictions: n_rows predicted labels. CPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
void predict(const ML::cumlHandle& handle, const DecisionTree::DecisionTreeRegressor<float> * dt_regressor, const float * rows, const int n_rows, const int n_cols, float * predictions, bool verbose) {
	return dt_regressor->predict(handle, rows, n_rows, n_cols, predictions, verbose);
}

/**
 * @brief Predict target feature for input data; regression for single feature supported. Inference of trees is CPU only for now.
 * @param[in] handle: cumlHandle (currently unused; API placeholder)
 * @param[in] dt_regressor: Pointer to decision tree object, which holds the trained tree.
 * @param[in] rows: test data type double (n_rows samples, n_cols features) in row major format. CPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in,out] predictions: n_rows predicted labels. CPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
void predict(const ML::cumlHandle& handle, const DecisionTree::DecisionTreeRegressor<double> * dt_regressor, const double * rows, const int n_rows, const int n_cols, double * predictions, bool verbose) {
	return dt_regressor->predict(handle, rows, n_rows, n_cols, predictions, verbose);
}

} //End namespace ML
