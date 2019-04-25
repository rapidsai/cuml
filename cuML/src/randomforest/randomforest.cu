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

namespace ML {

/**
 * @brief Construct RF_metrics.
 * @param[in] cfg_accuracy: accuracy.
 */
RF_metrics::RF_metrics(float cfg_accuracy) : accuracy(cfg_accuracy) {};

/**
 * @brief Print accuracy metric.
 */
void RF_metrics::print() {
	std::cout << "Accuracy: " << accuracy << std::endl;
}

/**
 * @brief Update labels so they are unique from 0 to n_unique_labels values.
		  Create/update an old label to new label map per random forest.
 * @param[in] n_rows: number of rows (labels)
 * @param[in,out] labels: 1D labels array to be changed in-place.
 * @param[in,out] labels_map: map of old label values to new ones.
 * @param[in] verbose: debugging flag.
 */
void preprocess_labels(int n_rows, std::vector<int> & labels, std::map<int, int> & labels_map, bool verbose) {
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

/**
 * @brief Revert label preprocessing effect, if needed.
 * @param[in] n_rows: number of rows (labels)
 * @param[in,out] labels: 1D labels array to be changed in-place.
 * @param[in] labels_map: map of old to new label values used during preprocessing.
 * @param[in] verbose: debugging flag.
 */
void postprocess_labels(int n_rows, std::vector<int> & labels, std::map<int, int> & labels_map, bool verbose) {

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

/**
 * @brief Random forest hyper-parameter object constructor to set n_trees member.
 */
RF_params::RF_params(int cfg_n_trees):n_trees(cfg_n_trees) {};

/**
 * @brief Random forest hyper-parameter object constructor to set bootstrap, n_trees and rows_sample members.
 */
RF_params::RF_params(bool cfg_bootstrap, int cfg_n_trees, float cfg_rows_sample):bootstrap(cfg_bootstrap), n_trees(cfg_n_trees), rows_sample(cfg_rows_sample) {};

/**
 * @brief Random forest hyper-parameter object constructor to set all RF_params members.
 */
RF_params::RF_params(bool cfg_bootstrap, int cfg_n_trees, float cfg_rows_sample, DecisionTree::DecisionTreeParams cfg_tree_params):bootstrap(cfg_bootstrap),
					n_trees(cfg_n_trees), rows_sample(cfg_rows_sample), tree_params(cfg_tree_params) {};

/**
 * @brief Check validity of all random forest hyper-parameters.
 */
void RF_params::validity_check() const {
	ASSERT((n_trees > 0), "Invalid n_trees %d", n_trees);
	ASSERT((rows_sample > 0) && (rows_sample <= 1.0), "rows_sample value %f outside permitted (0, 1] range", rows_sample);
	tree_params.validity_check();
}

/**
 * @brief Print all random forest hyper-parameters.
 */
void RF_params::print() const {
	std::cout << "bootstrap: " << bootstrap << std::endl;
	std::cout << "n_trees: " << n_trees << std::endl;
	std::cout << "rows_sample: " << rows_sample << std::endl;
	tree_params.print();
}

/**
 * @brief Construct rf (random forest) object.
 * @tparam T: data type for input data (float or double).
 * @param[in] cfg_rf_params: Random forest hyper-parameter struct.
 * @param[in] cfg_rf_type: Random forest type. Only CLASSIFICATION is currently supported.
 */
template<typename T>
rf<T>::rf(RF_params cfg_rf_params, int cfg_rf_type):rf_params(cfg_rf_params), rf_type(cfg_rf_type), trees(nullptr) {
	rf_params.validity_check();
}

/**
 * @brief Destructor for random forest object.
 * @tparam T: data type for input data (float or double).
 */
template<typename T>
rf<T>::~rf() {
	delete [] trees;
}

/**
 * @brief Return number of trees in the forest.
 * @tparam T: data type for input data (float or double).
 */
template<typename T>
int rf<T>::get_ntrees() {
	return rf_params.n_trees;
}


/**
 * @brief Print summary for all trees in the random forest.
 * @tparam T: data type for input data (float or double).
 */
template<typename T>
void rf<T>::print_rf_summary() {

	if (!trees) {
		std::cout << "Empty forest" << std::endl;
	} else {
		std::cout << "Forest has " << rf_params.n_trees << " trees, max_depth " << rf_params.tree_params.max_depth;
		std::cout << ", and max_leaves " << rf_params.tree_params.max_leaves << std::endl;
		for (int i = 0; i < rf_params.n_trees; i++) {
			std::cout << "Tree #" << i << std::endl;
		trees[i].print_tree_summary();
		}
	}
}

/**
 * @brief Print detailed view of all trees in the random forest.
 * @tparam T: data type for input data (float or double).
 */
template<typename T>
void rf<T>::print_rf_detailed() {

	if (!trees) {
		std::cout << "Empty forest" << std::endl;
	} else {
		std::cout << "Forest has " << rf_params.n_trees << " trees, max_depth " << rf_params.tree_params.max_depth;
		std::cout << ", and max_leaves " << rf_params.tree_params.max_leaves << std::endl;
		for (int i = 0; i < rf_params.n_trees; i++) {
			std::cout << "Tree #" << i << std::endl;
			trees[i].print();
		}
	}
}


/**
 * @brief Construct rfClassifier object.
 * @tparam T: data type for input data (float or double).
 * @param[in] cfg_rf_params: Random forest hyper-parameter struct.
 */
template <typename T>
rfClassifier<T>::rfClassifier(RF_params cfg_rf_params): rf<T>::rf(cfg_rf_params, RF_type::CLASSIFICATION) {};

/**
 * @brief Build (i.e., fit, train) random forest classifier for input data.
 * @tparam T: data type for input data (float or double).
 * @param[in] user_handle: cumlHandle
 * @param[in] input: train data (n_rows samples, n_cols features) in column major format, excluding labels. Device pointer.
 * @param[in] n_rows: number of training data samples.
 * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (int only), with one label per training sample. Device pointer.
				  Assumption: labels were preprocessed to map to ascending numbers from 0;
				  needed for current gini impl in decision tree
 * @param[in] n_unique_labels: #unique label values (known during preprocessing)
 */
template <typename T>
void rfClassifier<T>::fit(const cumlHandle& user_handle, T * input, int n_rows, int n_cols, int * labels, int n_unique_labels) {

	ASSERT(!this->trees, "Cannot fit an existing forest.");
	ASSERT((n_rows > 0), "Invalid n_rows %d", n_rows);
	ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);

	rfClassifier::trees = new DecisionTree::DecisionTreeClassifier<T>[this->rf_params.n_trees];
	int n_sampled_rows = this->rf_params.rows_sample * n_rows;

	const cumlHandle_impl& handle = user_handle.getImpl();
	cudaStream_t stream = user_handle.getStream();

	for (int i = 0; i < this->rf_params.n_trees; i++) {
		// Select n_sampled_rows (with replacement) numbers from [0, n_rows) per tree.
		// selected_rows: randomly generated IDs for bootstrapped samples (w/ replacement); a device ptr.
		MLCommon::device_buffer<unsigned int> selected_rows(handle.getDeviceAllocator(), stream, n_sampled_rows);

		if (this->rf_params.bootstrap) {
			MLCommon::Random::Rng r(i * 1000); // Ensure the seed for each tree is different and meaningful.
			r.uniformInt(selected_rows.data(), n_sampled_rows, (unsigned int) 0, (unsigned int) n_rows, stream);
		} else {
			std::vector<unsigned int> h_selected_rows(n_sampled_rows);
			std::iota(h_selected_rows.begin(), h_selected_rows.end(), 0);
			MLCommon::updateDevice(selected_rows.data(), h_selected_rows.data(), n_sampled_rows);
		}

		/* Build individual tree in the forest.
		   - input is a pointer to orig data that have n_cols features and n_rows rows.
		   - n_sampled_rows: # rows sampled for tree's bootstrap sample.
		   - selected_rows: points to a list of row #s (w/ n_sampled_rows elements) used to build the bootstrapped sample.
			Expectation: Each tree node will contain (a) # n_sampled_rows and (b) a pointer to a list of row numbers w.r.t original data.
		*/
		this->trees[i].fit(user_handle, input, n_cols, n_rows, labels, selected_rows.data(), n_sampled_rows, n_unique_labels, this->rf_params.tree_params);

		//Cleanup
		selected_rows.release(stream);
	}
}


/**
 * @brief Predict target feature for input data; n-ary classification for single feature supported.
 * @tparam T: data type for input data (float or double).
 * @param[in] user_handle: cumlHandle (currently unused; API placeholder)
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. CPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. CPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
template<typename T>
void rfClassifier<T>::predict(const cumlHandle& user_handle, const T * input, int n_rows, int n_cols, int * predictions, bool verbose) const {

	ASSERT(this->trees, "Cannot predict! No trees in the forest.");
	ASSERT((n_rows > 0), "Invalid n_rows %d", n_rows);
	ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);
	ASSERT(predictions != nullptr, "Error! User has not allocated memory for predictions.");

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

		for (int i = 0; i < this->rf_params.n_trees; i++) {
			//Return prediction for one sample.
			if (verbose) {
				std::cout << "Printing tree " << i << std::endl;
				this->trees[i].print();
			}
			int prediction;
			this->trees[i].predict(user_handle, &input[row_id * row_size], 1, n_cols, &prediction, verbose);
			ret = prediction_to_cnt.insert(std::pair<int, int>(prediction, 1));
			if (!(ret.second)) {
				ret.first->second += 1;
			}
			if (max_cnt_so_far < ret.first->second) {
				max_cnt_so_far = ret.first->second;
				majority_prediction = ret.first->first;
			}
		}

		predictions[row_id] = majority_prediction;
	}
}


/**
 * @brief Predict target feature for input data and validate against ref_labels.
 * @tparam T: data type for input data (float or double).
 * @param[in] user_handle: cumlHandle (currently unused; API placeholder)
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. CPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); CPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. CPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
template<typename T>
RF_metrics rfClassifier<T>::cross_validate(const cumlHandle& user_handle, const T * input, const int * ref_labels, int n_rows, int n_cols, int * predictions, bool verbose) const {

	predict(user_handle, input, n_rows, n_cols, predictions, verbose);

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

template class rf<float>;
template class rf<double>;

template class rfClassifier<float>;
template class rfClassifier<double>;

// Stateless API functions: fit, predict and cross_validate

/**
 * @brief Build (i.e., fit, train) random forest classifier for input data of type float.
 * @param[in] user_handle: cumlHandle
 * @param[in,out] rf_classifier: pointer to the rfClassifier object, previously constructed by the user.
 * @param[in] input: train data (n_rows samples, n_cols features) in column major format, excluding labels. Device pointer.
 * @param[in] n_rows: number of training data samples.
 * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (int only), with one label per training sample. Device pointer.
				  Assumption: labels were preprocessed to map to ascending numbers from 0;
				  needed for current gini impl. in decision tree
 * @param[in] n_unique_labels: #unique label values (known during preprocessing)
 */
void fit(const cumlHandle& user_handle, rfClassifier<float> * rf_classifier, float * input, int n_rows, int n_cols, int * labels, int n_unique_labels) {
	rf_classifier->fit(user_handle, input, n_rows, n_cols, labels, n_unique_labels);
}

/**
 * @brief Build (i.e., fit, train) random forest classifier for input data of type double.
 * @param[in] user_handle: cumlHandle
 * @param[in,out] rf_classifier: pointer to the rfClassifier object, previously constructed by the user.
 * @param[in] input: train data (n_rows samples, n_cols features) in column major format, excluding labels. Device pointer.
 * @param[in] n_rows: number of training data samples.
 * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (int only), with one label per training sample. Device pointer.
				  Assumption: labels were preprocessed to map to ascending numbers from 0;
				  needed for current gini impl. in decision tree
 * @param[in] n_unique_labels: #unique label values (known during preprocessing)
 */
void fit(const cumlHandle& user_handle, rfClassifier<double> * rf_classifier, double * input, int n_rows, int n_cols, int * labels, int n_unique_labels) {
	rf_classifier->fit(user_handle, input, n_rows, n_cols, labels, n_unique_labels);
}

/**
 * @brief Predict target feature for input data of type float; n-ary classification for single feature supported.
 * @param[in] user_handle: cumlHandle (currently unused; API placeholder)
 * @param[in] rf_classifier: pointer to the rfClassifier object. The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. CPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. CPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
void predict(const cumlHandle& user_handle, const rfClassifier<float> * rf_classifier, const float * input, int n_rows, int n_cols, int * predictions, bool verbose) {
	rf_classifier->predict(user_handle, input, n_rows, n_cols, predictions, verbose);
}

/**
 * @brief Predict target feature for input data of type double; n-ary classification for single feature supported.
 * @param[in] user_handle: cumlHandle (currently unused; API placeholder)
 * @param[in] rf_classifier: pointer to the rfClassifier object. The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. CPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. CPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
void predict(const cumlHandle& user_handle, const rfClassifier<double> * rf_classifier, const double * input, int n_rows, int n_cols, int * predictions, bool verbose) {
	rf_classifier->predict(user_handle, input, n_rows, n_cols, predictions, verbose);
}

/**
 * @brief Predict target feature for input data of type float and validate against ref_labels.
 * @param[in] user_handle: cumlHandle (currently unused; API placeholder)
 * @param[in] rf_classifier: pointer to the rfClassifier object. The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. CPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); CPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. CPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
RF_metrics cross_validate(const cumlHandle& user_handle, const rfClassifier<float> * rf_classifier, const float * input, const int * ref_labels,
							int n_rows, int n_cols, int * predictions, bool verbose) {
	return rf_classifier->cross_validate(user_handle, input, ref_labels, n_rows, n_cols, predictions, verbose);
}

/**
 * @brief Predict target feature for input data of type double and validate against ref_labels.
 * @param[in] user_handle: cumlHandle (currently unused; API placeholder)
 * @param[in] rf_classifier: pointer to the rfClassifier object. The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. CPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); CPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. CPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
RF_metrics cross_validate(const cumlHandle& user_handle, const rfClassifier<double> * rf_classifier, const double * input, const int * ref_labels,
							int n_rows, int n_cols, int * predictions, bool verbose) {
	return rf_classifier->cross_validate(user_handle, input, ref_labels, n_rows, n_cols, predictions, verbose);
}

};
// end namespace ML
