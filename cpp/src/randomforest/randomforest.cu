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

#include "../decisiontree/kernels/quantile.h"
#include "../decisiontree/memory.h"
#include "random/permute.h"
#include "random/rng.h"
#include "randomforest.h"
#include "score/scores.h"

namespace ML {

/**
 * @brief Construct RF_metrics.
 * @param[in] cfg_accuracy: accuracy.
 */
RF_metrics::RF_metrics(float cfg_accuracy)
  : rf_type(RF_type::CLASSIFICATION), accuracy(cfg_accuracy){};

/**
 * @brief Construct RF_metrics.
 * @param[in] cfg_mean_abs_error: mean absolute error.
 * @param[in] cfg_mean_squared_error: mean squared error.
 * @param[in] cfg_median_abs_error: median absolute error.
 */
RF_metrics::RF_metrics(double cfg_mean_abs_error, double cfg_mean_squared_error,
                       double cfg_median_abs_error)
  : rf_type(RF_type::REGRESSION),
    mean_abs_error(cfg_mean_abs_error),
    mean_squared_error(cfg_mean_squared_error),
    median_abs_error(cfg_median_abs_error){};

/**
 * @brief Print either accuracy metric for classification, or mean absolute error, mean squared error, 
   and median absolute error metrics for regression.
 */
void RF_metrics::print() {
  if (rf_type == RF_type::CLASSIFICATION) {
    std::cout << "Accuracy: " << accuracy << std::endl;
  } else if (rf_type == RF_type::REGRESSION) {
    std::cout << "Mean Absolute Error: " << mean_abs_error << std::endl;
    std::cout << "Mean Squared Error: " << mean_squared_error << std::endl;
    std::cout << "Median Absolute Error: " << median_abs_error << std::endl;
  }
}

/**
 * @brief Update labels so they are unique from 0 to n_unique_labels values.
		  Create/update an old label to new label map per random forest.
 * @param[in] n_rows: number of rows (labels)
 * @param[in,out] labels: 1D labels array to be changed in-place.
 * @param[in,out] labels_map: map of old label values to new ones.
 * @param[in] verbose: debugging flag.
 */
void preprocess_labels(int n_rows, std::vector<int>& labels,
                       std::map<int, int>& labels_map, bool verbose) {
  std::pair<std::map<int, int>::iterator, bool> ret;
  int n_unique_labels = 0;

  if (verbose) std::cout << "Preprocessing labels\n";
  for (int i = 0; i < n_rows; i++) {
    ret = labels_map.insert(std::pair<int, int>(labels[i], n_unique_labels));
    if (ret.second) {
      n_unique_labels += 1;
    }
    if (verbose) std::cout << "Mapping " << labels[i] << " to ";
    labels[i] = ret.first->second;  //Update labels **IN-PLACE**
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
void postprocess_labels(int n_rows, std::vector<int>& labels,
                        std::map<int, int>& labels_map, bool verbose) {
  if (verbose) std::cout << "Postrocessing labels\n";
  std::map<int, int>::iterator it;
  int n_unique_cnt = labels_map.size();
  std::vector<int> reverse_map;
  reverse_map.resize(n_unique_cnt);
  for (auto it = labels_map.begin(); it != labels_map.end(); it++) {
    reverse_map[it->second] = it->first;
  }

  for (int i = 0; i < n_rows; i++) {
    if (verbose)
      std::cout << "Mapping " << labels[i] << " back to "
                << reverse_map[labels[i]] << std::endl;
    labels[i] = reverse_map[labels[i]];
  }
  if (verbose) std::cout << "Finished postrocessing labels\n";
}

/**
 * @brief Random forest hyper-parameter object default constructor (1 tree).
 */
RF_params::RF_params() : n_trees(1) {}

/**
 * @brief Random forest hyper-parameter object constructor to set n_trees member.
 */
RF_params::RF_params(int cfg_n_trees) : n_trees(cfg_n_trees) {}

/**
 * @brief Random forest hyper-parameter object constructor to set bootstrap, bootstrap_features, n_trees and rows_sample members.
 */
RF_params::RF_params(bool cfg_bootstrap, bool cfg_bootstrap_features,
                     int cfg_n_trees, float cfg_rows_sample)
  : bootstrap(cfg_bootstrap),
    bootstrap_features(cfg_bootstrap_features),
    n_trees(cfg_n_trees),
    rows_sample(cfg_rows_sample) {
  tree_params.bootstrap_features = cfg_bootstrap_features;
}

/**
 * @brief Random forest hyper-parameter object constructor to set all RF_params members.
 */
RF_params::RF_params(bool cfg_bootstrap, bool cfg_bootstrap_features,
                     int cfg_n_trees, float cfg_rows_sample,
                     DecisionTree::DecisionTreeParams cfg_tree_params)
  : bootstrap(cfg_bootstrap),
    bootstrap_features(cfg_bootstrap_features),
    n_trees(cfg_n_trees),
    rows_sample(cfg_rows_sample),
    tree_params(cfg_tree_params) {
  tree_params.bootstrap_features = cfg_bootstrap_features;
}

/**
 * @brief Check validity of all random forest hyper-parameters.
 */
void RF_params::validity_check() const {
  ASSERT((n_trees > 0), "Invalid n_trees %d", n_trees);
  ASSERT((rows_sample > 0) && (rows_sample <= 1.0),
         "rows_sample value %f outside permitted (0, 1] range", rows_sample);
  tree_params.validity_check();
}

/**
 * @brief Print all random forest hyper-parameters.
 */
void RF_params::print() const {
  std::cout << "bootstrap: " << bootstrap << std::endl;
  std::cout << "bootstrap features: " << bootstrap_features << std::endl;
  std::cout << "n_trees: " << n_trees << std::endl;
  std::cout << "rows_sample: " << rows_sample << std::endl;
  tree_params.print();
}

/**
 * @brief Construct rf (random forest) object.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] cfg_rf_params: Random forest hyper-parameter struct.
 * @param[in] cfg_rf_type: Random forest type. Only CLASSIFICATION is currently supported.
 */
template <typename T, typename L>
rf<T, L>::rf(RF_params cfg_rf_params, int cfg_rf_type)
  : rf_params(cfg_rf_params), rf_type(cfg_rf_type) {
  rf_params.validity_check();
}

/**
 * @brief Return number of trees in the forest.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 */
template <typename T, typename L>
int rf<T, L>::get_ntrees() {
  return rf_params.n_trees;
}

/**
 * @brief Print summary for all trees in the random forest.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 */
template <typename T, typename L>
void rf<T, L>::print_rf_summary() {
  const DecisionTree::DecisionTreeBase<T, L>* trees = get_trees_ptr();
  if (!trees) {
    std::cout << "Empty forest" << std::endl;
  } else {
    std::cout << "Forest has " << rf_params.n_trees << " trees, max_depth "
              << rf_params.tree_params.max_depth;
    std::cout << ", and max_leaves " << rf_params.tree_params.max_leaves
              << std::endl;
    for (int i = 0; i < rf_params.n_trees; i++) {
      std::cout << "Tree #" << i << std::endl;
      trees[i].print_tree_summary();
    }
  }
}

/**
 * @brief Print detailed view of all trees in the random forest.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 */
template <typename T, typename L>
void rf<T, L>::print_rf_detailed() {
  const DecisionTree::DecisionTreeBase<T, L>* trees = get_trees_ptr();
  if (!trees) {
    std::cout << "Empty forest" << std::endl;
  } else {
    std::cout << "Forest has " << rf_params.n_trees << " trees, max_depth "
              << rf_params.tree_params.max_depth;
    std::cout << ", and max_leaves " << rf_params.tree_params.max_leaves
              << std::endl;
    for (int i = 0; i < rf_params.n_trees; i++) {
      std::cout << "Tree #" << i << std::endl;
      trees[i].print();
    }
  }
}

/**
 * @brief Sample row IDs for tree fitting and bootstrap if requested.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] handle: cumlHandle
 * @param[in] tree_id: unique tree ID
 * @param[in] n_rows: total number of data samples.
 * @param[in] n_sampled_rows: number of rows used for training
 * @param[in, out] selected_rows: already allocated array w/ row IDs
 * @param[in, out] sorted_selected_rows: already allocated array. Will contain sorted row IDs.
 * @param[in, out] rows_temp_storage: temp. storage used for sorting (previously allocated).
 * @param[in] temp_storage_bytes: size in bytes of rows_temp_storage.
 */
template <typename T, typename L>
void rf<T, L>::prepare_fit_per_tree(const ML::cumlHandle_impl& handle,
                                    int tree_id, int n_rows, int n_sampled_rows,
                                    unsigned int* selected_rows,
                                    unsigned int* sorted_selected_rows,
                                    char* rows_temp_storage,
                                    size_t temp_storage_bytes) {
  cudaStream_t stream = handle.getStream();

  if (rf_params.bootstrap) {
    MLCommon::Random::Rng r(
      tree_id *
      1000);  // Ensure the seed for each tree is different and meaningful.
    r.uniformInt(selected_rows, n_sampled_rows, (unsigned int)0,
                 (unsigned int)n_rows, stream);
    //thrust::sequence(thrust::cuda::par.on(stream), sorted_selected_rows,
    //           sorted_selected_rows + n_sampled_rows);

    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
      (void*)rows_temp_storage, temp_storage_bytes, selected_rows,
      sorted_selected_rows, n_sampled_rows, 0, 8 * sizeof(unsigned int),
      stream));
  } else {  // Sampling w/o replacement
    MLCommon::device_buffer<unsigned int>* inkeys =
      new MLCommon::device_buffer<unsigned int>(handle.getDeviceAllocator(),
                                                stream, n_rows);
    MLCommon::device_buffer<unsigned int>* outkeys =
      new MLCommon::device_buffer<unsigned int>(handle.getDeviceAllocator(),
                                                stream, n_rows);
    thrust::sequence(thrust::cuda::par.on(stream), inkeys->data(),
                     inkeys->data() + n_rows);
    int* perms = nullptr;
    MLCommon::Random::permute(perms, outkeys->data(), inkeys->data(), 1, n_rows,
                              false, stream);
    // outkeys has more rows than selected_rows; doing the shuffling before the resize to differentiate the per-tree rows sample.
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
      (void*)rows_temp_storage, temp_storage_bytes, outkeys->data(),
      sorted_selected_rows, n_sampled_rows, 0, 8 * sizeof(unsigned int),
      stream));
    inkeys->release(stream);
    outkeys->release(stream);
    delete inkeys;
    delete outkeys;
  }
}

template <typename T, typename L>
void rf<T, L>::error_checking(const T* input, L* predictions, int n_rows,
                              int n_cols, bool predict) const {
  if (predict) {
    ASSERT(get_trees_ptr(), "Cannot predict! No trees in the forest.");
    ASSERT(predictions != nullptr,
           "Error! User has not allocated memory for predictions.");
  } else {
    ASSERT(!get_trees_ptr(), "Cannot fit an existing forest.");
  }
  ASSERT((n_rows > 0), "Invalid n_rows %d", n_rows);
  ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);

  bool input_is_dev_ptr = is_dev_ptr(input);
  bool preds_is_dev_ptr = is_dev_ptr(predictions);

  if (!input_is_dev_ptr || (input_is_dev_ptr != preds_is_dev_ptr)) {
    ASSERT(false,
           "RF Error: Expected both input and labels/predictions to be GPU "
           "pointers");
  }
}

/**
 * @brief Construct rfClassifier object.
 * @tparam T: data type for input data (float or double).
 * @param[in] cfg_rf_params: Random forest hyper-parameter struct.
 */
template <typename T>
rfClassifier<T>::rfClassifier(RF_params cfg_rf_params)
  : rf<T, int>::rf(cfg_rf_params, RF_type::CLASSIFICATION){};

/**
 * @brief Destructor for random forest classifier object.
 * @tparam T: data type for input data (float or double).
 */
template <typename T>
rfClassifier<T>::~rfClassifier() {
  delete[] trees;
}

/**
 * @brief Return a const pointer to decision trees.
 * @tparam T: data type for input data (float or double).
 */
template <typename T>
const DecisionTree::DecisionTreeClassifier<T>* rfClassifier<T>::get_trees_ptr()
  const {
  return trees;
}

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
void rfClassifier<T>::fit(const cumlHandle& user_handle, T* input, int n_rows,
                          int n_cols, int* labels, int n_unique_labels) {
  this->error_checking(input, labels, n_rows, n_cols, false);

  trees = new DecisionTree::DecisionTreeClassifier<T>[this->rf_params.n_trees];

  int n_sampled_rows = this->rf_params.rows_sample * n_rows;

  const cumlHandle_impl& handle = user_handle.getImpl();
  cudaStream_t stream = user_handle.getStream();

  // Select n_sampled_rows (with replacement) numbers from [0, n_rows) per tree.
  // selected_rows: randomly generated IDs for bootstrapped samples (w/ replacement); a device ptr.
  MLCommon::device_buffer<unsigned int> selected_rows(
    handle.getDeviceAllocator(), stream, n_sampled_rows);
  MLCommon::device_buffer<unsigned int> sorted_selected_rows(
    handle.getDeviceAllocator(), stream, n_sampled_rows);

  // Will sort selected_rows (row IDs), prior to fit, to improve access patterns
  MLCommon::device_buffer<char>* rows_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
    rows_temp_storage, temp_storage_bytes, selected_rows.data(),
    sorted_selected_rows.data(), n_sampled_rows, 0, 8 * sizeof(unsigned int),
    stream));
  // Allocate temporary storage
  rows_temp_storage = new MLCommon::device_buffer<char>(
    handle.getDeviceAllocator(), stream, temp_storage_bytes);
  std::shared_ptr<TemporaryMemory<T, int>> tempmem =
    std::make_shared<TemporaryMemory<T, int>>(
      user_handle.getImpl(), n_sampled_rows, n_cols, 1, n_unique_labels,
      this->rf_params.tree_params.n_bins,
      this->rf_params.tree_params.split_algo);
  if ((this->rf_params.tree_params.split_algo == SPLIT_ALGO::GLOBAL_QUANTILE) && !(this->rf_params.tree_params.quantile_per_tree)) {
    preprocess_quantile(input, nullptr, n_sampled_rows, n_cols, n_rows,
                        this->rf_params.tree_params.n_bins, tempmem);
  }
  for (int i = 0; i < this->rf_params.n_trees; i++) {
    this->prepare_fit_per_tree(handle, i, n_rows, n_sampled_rows,
                               selected_rows.data(),
                               sorted_selected_rows.data(),
                               rows_temp_storage->data(), temp_storage_bytes);

    /* Build individual tree in the forest.
		   - input is a pointer to orig data that have n_cols features and n_rows rows.
		   - n_sampled_rows: # rows sampled for tree's bootstrap sample.
		   - sorted_selected_rows: points to a list of row #s (w/ n_sampled_rows elements) used to build the bootstrapped sample.
		   Expectation: Each tree node will contain (a) # n_sampled_rows and (b) a pointer to a list of row numbers w.r.t original data.
		*/

    trees[i].fit(user_handle, input, n_cols, n_rows, labels,
                 sorted_selected_rows.data(), n_sampled_rows, n_unique_labels,
                 this->rf_params.tree_params, tempmem);
  }

  //Cleanup
  rows_temp_storage->release(stream);
  selected_rows.release(stream);
  sorted_selected_rows.release(stream);
  tempmem.reset();
  delete rows_temp_storage;
}

/**
 * @brief Predict target feature for input data; n-ary classification for single feature supported.
 * @tparam T: data type for input data (float or double).
 * @param[in] user_handle: cumlHandle.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
template <typename T>
void rfClassifier<T>::predict(const cumlHandle& user_handle, const T* input,
                              int n_rows, int n_cols, int* predictions,
                              bool verbose) const {
  this->error_checking(input, predictions, n_rows, n_cols, true);

  std::vector<int> h_predictions(n_rows);
  const cumlHandle_impl& handle = user_handle.getImpl();
  cudaStream_t stream = user_handle.getStream();

  std::vector<T> h_input(n_rows * n_cols);
  MLCommon::updateHost(h_input.data(), input, n_rows * n_cols, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int row_size = n_cols;

  for (int row_id = 0; row_id < n_rows; row_id++) {
    if (verbose) {
      std::cout << "\n\n";
      std::cout << "Predict for sample: ";
      for (int i = 0; i < n_cols; i++)
        std::cout << h_input[row_id * row_size + i] << ", ";
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
        trees[i].print();
      }
      int prediction;
      trees[i].predict(user_handle, &h_input[row_id * row_size], 1, n_cols,
                       &prediction, verbose);
      ret = prediction_to_cnt.insert(std::pair<int, int>(prediction, 1));
      if (!(ret.second)) {
        ret.first->second += 1;
      }
      if (max_cnt_so_far < ret.first->second) {
        max_cnt_so_far = ret.first->second;
        majority_prediction = ret.first->first;
      }
    }

    h_predictions[row_id] = majority_prediction;
  }

  MLCommon::updateDevice(predictions, h_predictions.data(), n_rows, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

/**
 * @brief Predict target feature for input data and validate against ref_labels.
 * @tparam T: data type for input data (float or double).
 * @param[in] user_handle: cumlHandle.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
template <typename T>
RF_metrics rfClassifier<T>::score(const cumlHandle& user_handle, const T* input,
                                  const int* ref_labels, int n_rows, int n_cols,
                                  int* predictions, bool verbose) const {
  predict(user_handle, input, n_rows, n_cols, predictions, verbose);

  cudaStream_t stream = user_handle.getImpl().getStream();
  auto d_alloc = user_handle.getDeviceAllocator();
  float accuracy = MLCommon::Score::accuracy_score(predictions, ref_labels,
                                                   n_rows, d_alloc, stream);
  RF_metrics stats(accuracy);
  if (verbose) stats.print();

  /* TODO: Potentially augment RF_metrics w/ more metrics (e.g., precision, F1, etc.).
	   For non binary classification problems (i.e., one target and  > 2 labels), need avg for each of these metrics */
  return stats;
}

/**
 * @brief Construct rfRegressor object.
 * @tparam T: data type for input data (float or double).
 * @param[in] cfg_rf_params: Random forest hyper-parameter struct.
 */
template <typename T>
rfRegressor<T>::rfRegressor(RF_params cfg_rf_params)
  : rf<T, T>::rf(cfg_rf_params, RF_type::REGRESSION){};

/**
 * @brief Destructor for random forest regressor object.
 * @tparam T: data type for input data (float or double).
 */
template <typename T>
rfRegressor<T>::~rfRegressor() {
  delete[] trees;
}

/**
 * @brief Return a const pointer to decision trees.
 * @tparam T: data type for input data (float or double).
 */
template <typename T>
const DecisionTree::DecisionTreeRegressor<T>* rfRegressor<T>::get_trees_ptr()
  const {
  return trees;
}

/**
 * @brief Build (i.e., fit, train) random forest regressor for input data.
 * @tparam T: data type for input data (float or double).
 * @param[in] user_handle: cumlHandle
 * @param[in] input: train data (n_rows samples, n_cols features) in column major format, excluding labels. Device pointer.
 * @param[in] n_rows: number of training data samples.
 * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (float or double), with one label per training sample. Device pointer.
 */
template <typename T>
void rfRegressor<T>::fit(const cumlHandle& user_handle, T* input, int n_rows,
                         int n_cols, T* labels) {
  this->error_checking(input, labels, n_rows, n_cols, false);

  trees = new DecisionTree::DecisionTreeRegressor<T>[this->rf_params.n_trees];

  int n_sampled_rows = this->rf_params.rows_sample * n_rows;

  const cumlHandle_impl& handle = user_handle.getImpl();
  cudaStream_t stream = user_handle.getStream();

  // Select n_sampled_rows (with replacement) numbers from [0, n_rows) per tree.
  // selected_rows: randomly generated IDs for bootstrapped samples (w/ replacement); a device ptr.
  MLCommon::device_buffer<unsigned int> selected_rows(
    handle.getDeviceAllocator(), stream, n_sampled_rows);
  MLCommon::device_buffer<unsigned int> sorted_selected_rows(
    handle.getDeviceAllocator(), stream, n_sampled_rows);

  // Will sort selected_rows (row IDs), prior to fit, to improve access patterns
  MLCommon::device_buffer<char>* rows_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
    rows_temp_storage, temp_storage_bytes, selected_rows.data(),
    sorted_selected_rows.data(), n_sampled_rows, 0, 8 * sizeof(unsigned int),
    stream));
  // Allocate temporary storage
  rows_temp_storage = new MLCommon::device_buffer<char>(
    handle.getDeviceAllocator(), stream, temp_storage_bytes);
  std::shared_ptr<TemporaryMemory<T, T>> tempmem =
    std::make_shared<TemporaryMemory<T, T>>(
      user_handle.getImpl(), n_sampled_rows, n_cols, 1, 1,
      this->rf_params.tree_params.n_bins,
      this->rf_params.tree_params.split_algo);

  if (this->rf_params.tree_params.split_algo == SPLIT_ALGO::GLOBAL_QUANTILE) {
    preprocess_quantile(input, nullptr, n_sampled_rows, n_cols, n_rows,
                        this->rf_params.tree_params.n_bins, tempmem);
  }
  for (int i = 0; i < this->rf_params.n_trees; i++) {
    this->prepare_fit_per_tree(handle, i, n_rows, n_sampled_rows,
                               selected_rows.data(),
                               sorted_selected_rows.data(),
                               rows_temp_storage->data(), temp_storage_bytes);

    /* Build individual tree in the forest.
		   - input is a pointer to orig data that have n_cols features and n_rows rows.
		   - n_sampled_rows: # rows sampled for tree's bootstrap sample.
		   - sorted_selected_rows: points to a list of row #s (w/ n_sampled_rows elements) used to build the bootstrapped sample.
		   Expectation: Each tree node will contain (a) # n_sampled_rows and (b) a pointer to a list of row numbers w.r.t original data.
		*/

    trees[i].fit(user_handle, input, n_cols, n_rows, labels,
                 sorted_selected_rows.data(), n_sampled_rows,
                 this->rf_params.tree_params, tempmem);
  }
  //Cleanup
  rows_temp_storage->release(stream);
  selected_rows.release(stream);
  sorted_selected_rows.release(stream);
  tempmem.reset();
  delete rows_temp_storage;
}

/**
 * @brief Predict target feature for input data; regression for single feature supported.
 * @tparam T: data type for input data (float or double).
 * @param[in] user_handle: cumlHandle.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
template <typename T>
void rfRegressor<T>::predict(const cumlHandle& user_handle, const T* input,
                             int n_rows, int n_cols, T* predictions,
                             bool verbose) const {
  this->error_checking(input, predictions, n_rows, n_cols, true);

  std::vector<T> h_predictions(n_rows);
  const cumlHandle_impl& handle = user_handle.getImpl();
  cudaStream_t stream = user_handle.getStream();

  std::vector<T> h_input(n_rows * n_cols);
  MLCommon::updateHost(h_input.data(), input, n_rows * n_cols, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int row_size = n_cols;

  for (int row_id = 0; row_id < n_rows; row_id++) {
    if (verbose) {
      std::cout << "\n\n";
      std::cout << "Predict for sample: ";
      for (int i = 0; i < n_cols; i++)
        std::cout << h_input[row_id * row_size + i] << ", ";
      std::cout << std::endl;
    }

    T sum_predictions = 0;

    for (int i = 0; i < this->rf_params.n_trees; i++) {
      //Return prediction for one sample.
      if (verbose) {
        std::cout << "Printing tree " << i << std::endl;
        trees[i].print();
      }
      T prediction;
      trees[i].predict(user_handle, &h_input[row_id * row_size], 1, n_cols,
                       &prediction, verbose);
      sum_predictions += prediction;
    }
    // Random forest's prediction is the arithmetic mean of all its decision tree predictions.
    h_predictions[row_id] = sum_predictions / this->rf_params.n_trees;
  }

  MLCommon::updateDevice(predictions, h_predictions.data(), n_rows, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

/**
 * @brief Predict target feature for input data and validate against ref_labels.
 * @tparam T: data type for input data (float or double).
 * @param[in] user_handle: cumlHandle.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
template <typename T>
RF_metrics rfRegressor<T>::score(const cumlHandle& user_handle,
                                 const T* input, const T* ref_labels,
                                 int n_rows, int n_cols,
                                 T* predictions, bool verbose) const {
  predict(user_handle, input, n_rows, n_cols, predictions, verbose);

  cudaStream_t stream = user_handle.getImpl().getStream();
  auto d_alloc = user_handle.getDeviceAllocator();

  double mean_abs_error, mean_squared_error, median_abs_error;
  MLCommon::Score::regression_metrics(predictions, ref_labels, n_rows, d_alloc,
                                      stream, mean_abs_error,
                                      mean_squared_error, median_abs_error);
  RF_metrics stats(mean_abs_error, mean_squared_error, median_abs_error);
  if (verbose) stats.print();

  return stats;
}

template class rf<float, int>;
template class rf<float, float>;
template class rf<double, int>;
template class rf<double, double>;

template class rfClassifier<float>;
template class rfClassifier<double>;

template class rfRegressor<float>;
template class rfRegressor<double>;

// Stateless API functions: fit, predict and score

// ----------------------------- Classification ----------------------------------- //

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
void fit(const cumlHandle& user_handle, rfClassifier<float>* rf_classifier,
         float* input, int n_rows, int n_cols, int* labels,
         int n_unique_labels) {
  rf_classifier->fit(user_handle, input, n_rows, n_cols, labels,
                     n_unique_labels);
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
void fit(const cumlHandle& user_handle, rfClassifier<double>* rf_classifier,
         double* input, int n_rows, int n_cols, int* labels,
         int n_unique_labels) {
  rf_classifier->fit(user_handle, input, n_rows, n_cols, labels,
                     n_unique_labels);
}

/**
 * @brief Predict target feature for input data of type float; n-ary classification for single feature supported.
 * @param[in] user_handle: cumlHandle.
 * @param[in] rf_classifier: pointer to the rfClassifier object. The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
void predict(const cumlHandle& user_handle,
             const rfClassifier<float>* rf_classifier, const float* input,
             int n_rows, int n_cols, int* predictions, bool verbose) {
  rf_classifier->predict(user_handle, input, n_rows, n_cols, predictions,
                         verbose);
}

/**
 * @brief Predict target feature for input data of type double; n-ary classification for single feature supported.
 * @param[in] user_handle: cumlHandle.
 * @param[in] rf_classifier: pointer to the rfClassifier object. The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
void predict(const cumlHandle& user_handle,
             const rfClassifier<double>* rf_classifier, const double* input,
             int n_rows, int n_cols, int* predictions, bool verbose) {
  rf_classifier->predict(user_handle, input, n_rows, n_cols, predictions,
                         verbose);
}

/**
 * @brief Predict target feature for input data of type float and validate against ref_labels.
 * @param[in] user_handle: cumlHandle.
 * @param[in] rf_classifier: pointer to the rfClassifier object. The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
RF_metrics score(const cumlHandle& user_handle,
                 const rfClassifier<float>* rf_classifier, const float* input,
                 const int* ref_labels, int n_rows, int n_cols,
                 int* predictions, bool verbose) {
  return rf_classifier->score(user_handle, input, ref_labels, n_rows, n_cols,
                              predictions, verbose);
}

/**
 * @brief Predict target feature for input data of type double and validate against ref_labels.
 * @param[in] user_handle: cumlHandle.
 * @param[in] rf_classifier: pointer to the rfClassifier object. The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
RF_metrics score(const cumlHandle& user_handle,
                 const rfClassifier<double>* rf_classifier, const double* input,
                 const int* ref_labels, int n_rows, int n_cols,
                 int* predictions, bool verbose) {
  return rf_classifier->score(user_handle, input, ref_labels, n_rows, n_cols,
                              predictions, verbose);
}

RF_params set_rf_class_obj(int max_depth, int max_leaves, float max_features,
                           int n_bins, int split_algo, int min_rows_per_node,
                           bool bootstrap_features, bool bootstrap, int n_trees,
                           float rows_sample, CRITERION split_criterion,
                           bool quantile_per_tree) {
  DecisionTree::DecisionTreeParams tree_params(
    max_depth, max_leaves, max_features, n_bins, split_algo, min_rows_per_node,
    bootstrap_features, split_criterion, quantile_per_tree);
  RF_params rf_params(bootstrap, bootstrap_features, n_trees, rows_sample,
                      tree_params);
  return rf_params;
}

// ----------------------------- Regression ----------------------------------- //

/**
 * @brief Build (i.e., fit, train) random forest regressor for input data of type float.
 * @param[in] user_handle: cumlHandle
 * @param[in,out] rf_regreesor: pointer to the rfRegressor object, previously constructed by the user.
 * @param[in] input: train data (n_rows samples, n_cols features) in column major format, excluding labels. Device pointer.
 * @param[in] n_rows: number of training data samples.
 * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (float), with one label per training sample. Device pointer.
 */
void fit(const cumlHandle& user_handle, rfRegressor<float>* rf_regressor,
         float* input, int n_rows, int n_cols, float* labels) {
  rf_regressor->fit(user_handle, input, n_rows, n_cols, labels);
}

/**
 * @brief Build (i.e., fit, train) random forest regressor for input data of type double.
 * @param[in] user_handle: cumlHandle
 * @param[in,out] rf_regressor: pointer to the rfRegressor object, previously constructed by the user.
 * @param[in] input: train data (n_rows samples, n_cols features) in column major format, excluding labels. Device pointer.
 * @param[in] n_rows: number of training data samples.
 * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (double), with one label per training sample. Device pointer.
 */
void fit(const cumlHandle& user_handle, rfRegressor<double>* rf_regressor,
         double* input, int n_rows, int n_cols, double* labels) {
  rf_regressor->fit(user_handle, input, n_rows, n_cols, labels);
}

/**
 * @brief Predict target feature for input data of type float; regression for single feature supported.
 * @param[in] user_handle: cumlHandle.
 * @param[in] rf_regressor: pointer to the rfRegressor object. The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
void predict(const cumlHandle& user_handle,
             const rfRegressor<float>* rf_regressor, const float* input,
             int n_rows, int n_cols, float* predictions, bool verbose) {
  rf_regressor->predict(user_handle, input, n_rows, n_cols, predictions,
                        verbose);
}

/**
 * @brief Predict target feature for input data of type double; regression for single feature supported.
 * @param[in] user_handle: cumlHandle.
 * @param[in] rf_regressor: pointer to the rfRegressor object. The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
void predict(const cumlHandle& user_handle,
             const rfRegressor<double>* rf_regressor, const double* input,
             int n_rows, int n_cols, double* predictions, bool verbose) {
  rf_regressor->predict(user_handle, input, n_rows, n_cols, predictions,
                        verbose);
}

/**
 * @brief Predict target feature for input data of type float and validate against ref_labels.
 * @param[in] user_handle: cumlHandle.
 * @param[in] rf_regressor: pointer to the rfRegressor object. The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
RF_metrics score(const cumlHandle& user_handle,
                 const rfRegressor<float>* rf_regressor,
                 const float* input, const float* ref_labels,
                 int n_rows, int n_cols, float* predictions,
                 bool verbose) {
  return rf_regressor->score(user_handle, input, ref_labels, n_rows,
                             n_cols, predictions, verbose);
}

/**
 * @brief Predict target feature for input data of type double and validate against ref_labels.
 * @param[in] user_handle: cumlHandle.
 * @param[in] rf_regressor: pointer to the rfRegressor object. The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 */
RF_metrics score(const cumlHandle& user_handle,
                 const rfRegressor<double>* rf_regressor,
                 const double* input, const double* ref_labels,
                 int n_rows, int n_cols, double* predictions,
                 bool verbose) {
  return rf_regressor->score(user_handle, input, ref_labels, n_rows,
                             n_cols, predictions, verbose);
}

};  // namespace ML
// end namespace ML
