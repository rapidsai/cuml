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
#ifndef _OPENMP
#define omp_get_thread_num() 0
#endif
#include "../decisiontree/kernels/quantile.h"
#include "../decisiontree/memory.h"
#include "random/permute.h"
#include "random/rng.h"
#include "randomforest_impl.h"
#include "score/scores.h"

namespace ML {

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
  validity_check(rf_params);
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
    // outkeys has more rows than selected_rows; doing the shuffling before the
    // resize to differentiate the per-tree rows sample.
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
    ASSERT(predictions != nullptr,
           "Error! User has not allocated memory for predictions.");
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
  : rf<T, int>::rf(cfg_rf_params, RF_type::CLASSIFICATION) {
  trees = new DecisionTree::DecisionTreeClassifier<T>[this->rf_params.n_trees];
};

/**
 * @brief Destructor for random forest classifier object.
 * @tparam T: data type for input data (float or double).
 */
template <typename T>
rfClassifier<T>::~rfClassifier() {
  delete[] trees;
}

/**
 * @brief Return a const pointer to decision tree classifiers.
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
 * @param[in] input: train data (n_rows samples, n_cols features) in column major format,
 *   excluding labels. Device pointer.
 * @param[in] n_rows: number of training data samples.
 * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (int only), with one label per training sample. Device pointer.
				  Assumption: labels were preprocessed to map to ascending numbers from 0;
				  needed for current gini impl in decision tree
 * @param[in] n_unique_labels: #unique label values (known during preprocessing)
 * @param[in] forest: CPU point to RandomForestMetaData struct.
 */
template <typename T>
void rfClassifier<T>::fit(const cumlHandle& user_handle, T* input, int n_rows,
                          int n_cols, int* labels, int n_unique_labels,
                          RandomForestMetaData<T, int>*& forest) {
  this->error_checking(input, labels, n_rows, n_cols, false);

  int n_sampled_rows = this->rf_params.rows_sample * n_rows;

  const cumlHandle_impl& handle = user_handle.getImpl();
  cudaStream_t stream = user_handle.getStream();

  // Select n_sampled_rows (with replacement) numbers from [0, n_rows) per tree.
  // selected_rows: randomly generated IDs for bootstrapped samples (w/ replacement); a device ptr.
  MLCommon::device_buffer<unsigned int>* selected_rows =
    new MLCommon::device_buffer<unsigned int>(handle.getDeviceAllocator(),
                                              stream, n_sampled_rows);
  MLCommon::device_buffer<unsigned int>* sorted_selected_rows =
    new MLCommon::device_buffer<unsigned int>(handle.getDeviceAllocator(),
                                              stream, n_sampled_rows);

  // Will sort selected_rows (row IDs), prior to fit, to improve access patterns
  MLCommon::device_buffer<char>* rows_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
    rows_temp_storage, temp_storage_bytes, selected_rows->data(),
    sorted_selected_rows->data(), n_sampled_rows, 0, 8 * sizeof(unsigned int),
    stream));
  // Allocate temporary storage
  rows_temp_storage = new MLCommon::device_buffer<char>(
    handle.getDeviceAllocator(), stream, temp_storage_bytes);
  std::shared_ptr<TemporaryMemory<T, int>> tempmem =
    std::make_shared<TemporaryMemory<T, int>>(
      user_handle.getImpl(), n_sampled_rows, n_cols, 1, n_unique_labels,
      this->rf_params.tree_params.n_bins,
      this->rf_params.tree_params.split_algo,
      this->rf_params.tree_params.max_depth);
  if ((this->rf_params.tree_params.split_algo == SPLIT_ALGO::GLOBAL_QUANTILE) &&
      !(this->rf_params.tree_params.quantile_per_tree)) {
    preprocess_quantile(input, nullptr, n_rows, n_cols, n_rows,
                        this->rf_params.tree_params.n_bins, tempmem);
  }
  for (int i = 0; i < this->rf_params.n_trees; i++) {
    this->prepare_fit_per_tree(handle, i, n_rows, n_sampled_rows,
                               selected_rows->data(),
                               sorted_selected_rows->data(),
                               rows_temp_storage->data(), temp_storage_bytes);

    /* Build individual tree in the forest.
       - input is a pointer to orig data that have n_cols features and n_rows rows.
       - n_sampled_rows: # rows sampled for tree's bootstrap sample.
       - sorted_selected_rows: points to a list of row #s (w/ n_sampled_rows elements)
         used to build the bootstrapped sample.
         Expectation: Each tree node will contain (a) # n_sampled_rows and
         (b) a pointer to a list of row numbers w.r.t original data.
    */
    DecisionTree::TreeMetaDataNode<T, int>* tree_ptr = &(forest->trees[i]);
    trees[i].fit(user_handle, input, n_cols, n_rows, labels,
                 sorted_selected_rows->data(), n_sampled_rows, n_unique_labels,
                 tree_ptr, this->rf_params.tree_params, tempmem);
  }

  //Cleanup
  rows_temp_storage->release(stream);
  selected_rows->release(stream);
  sorted_selected_rows->release(stream);
  tempmem.reset();
  delete rows_temp_storage;
  delete selected_rows;
  delete sorted_selected_rows;
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
                              const RandomForestMetaData<T, int>* forest,
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
      /*if (verbose) {
        std::cout << "Printing tree " << i << std::endl;
        trees[i].print(forest->trees[i].root);
      }*/
      int prediction;
      trees[i].predict(user_handle, &forest->trees[i],
                       &h_input[row_id * row_size], 1, n_cols, &prediction,
                       verbose);
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
                                  int* predictions,
                                  const RandomForestMetaData<T, int>* forest,
                                  bool verbose) const {
  predict(user_handle, input, n_rows, n_cols, predictions, forest, verbose);

  cudaStream_t stream = user_handle.getImpl().getStream();
  auto d_alloc = user_handle.getDeviceAllocator();
  float accuracy = MLCommon::Score::accuracy_score(predictions, ref_labels,
                                                   n_rows, d_alloc, stream);
  RF_metrics stats = set_rf_metrics_classification(accuracy);
  if (verbose) print(stats);

  /* TODO: Potentially augment RF_metrics w/ more metrics (e.g., precision, F1, etc.).
     For non binary classification problems (i.e., one target and  > 2 labels), need avg.
     for each of these metrics */
  return stats;
}

/**
 * @brief Construct rfRegressor object.
 * @tparam T: data type for input data (float or double).
 * @param[in] cfg_rf_params: Random forest hyper-parameter struct.
 */
template <typename T>
rfRegressor<T>::rfRegressor(RF_params cfg_rf_params)
  : rf<T, T>::rf(cfg_rf_params, RF_type::REGRESSION) {
  trees = new DecisionTree::DecisionTreeRegressor<T>[this->rf_params.n_trees];
}

/**
 * @brief Destructor for random forest regressor object.
 * @tparam T: data type for input data (float or double).
 */
template <typename T>
rfRegressor<T>::~rfRegressor() {
  delete[] trees;
}

/**
 * @brief Return a const pointer to decision tree regressors.
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
 * @param[in, out] forest: CPU pointer to RandomForestMetaData struct
 */
template <typename T>
void rfRegressor<T>::fit(const cumlHandle& user_handle, T* input, int n_rows,
                         int n_cols, T* labels,
                         RandomForestMetaData<T, T>*& forest) {
  this->error_checking(input, labels, n_rows, n_cols, false);

  int n_sampled_rows = this->rf_params.rows_sample * n_rows;
  int n_streams = this->rf_params.n_streams;

  const cumlHandle_impl& handle = user_handle.getImpl();
  cudaStream_t stream = user_handle.getStream();
  cumlHandle local_handle[n_streams];
  cudaStream_t local_stream[n_streams];
  for (int i = 0; i < n_streams; i++) {
    CUDA_CHECK(cudaStreamCreate(&local_stream[i]));
    local_handle[i].setStream(local_stream[i]);
  }
  // Select n_sampled_rows (with replacement) numbers from [0, n_rows) per tree.
  // selected_rows: randomly generated IDs for bootstrapped samples (w/ replacement); a device ptr.
  MLCommon::device_buffer<unsigned int>* selected_rows[n_streams];
  MLCommon::device_buffer<unsigned int>* sorted_selected_rows[n_streams];
  for (int i = 0; i < n_streams; i++) {
    selected_rows[i] = new MLCommon::device_buffer<unsigned int>(
      handle.getDeviceAllocator(), stream, n_sampled_rows);
    sorted_selected_rows[i] = new MLCommon::device_buffer<unsigned int>(
      handle.getDeviceAllocator(), stream, n_sampled_rows);
  }

  // Will sort selected_rows (row IDs), prior to fit, to improve access patterns
  MLCommon::device_buffer<char>* rows_temp_storage[n_streams];
  size_t temp_storage_bytes[n_streams];
  for (int i = 0; i < n_streams; i++) {
    rows_temp_storage[i] = nullptr;
    temp_storage_bytes[i] = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
      rows_temp_storage[i], temp_storage_bytes[i], selected_rows[i]->data(),
      sorted_selected_rows[i]->data(), n_sampled_rows, 0,
      8 * sizeof(unsigned int), stream));
    // Allocate temporary storage
    rows_temp_storage[i] = new MLCommon::device_buffer<char>(
      handle.getDeviceAllocator(), stream, temp_storage_bytes[i]);
  }
  std::shared_ptr<TemporaryMemory<T, T>> tempmem[n_streams];
  for (int i = 0; i < n_streams; i++) {
    tempmem[i] = std::make_shared<TemporaryMemory<T, T>>(
      local_handle[i].getImpl(), n_sampled_rows, n_cols, 1, 1,
      this->rf_params.tree_params.n_bins,
      this->rf_params.tree_params.split_algo,
      this->rf_params.tree_params.max_depth);
  }
  //Preprocess once only per forest
  if ((this->rf_params.tree_params.split_algo == SPLIT_ALGO::GLOBAL_QUANTILE) &&
      !(this->rf_params.tree_params.quantile_per_tree)) {
    preprocess_quantile(input, nullptr, n_rows, n_cols, n_rows,
                        this->rf_params.tree_params.n_bins, tempmem[0]);
    for (int i = 1; i < n_streams; i++) {
      CUDA_CHECK(cudaMemcpyAsync(
        tempmem[i]->d_quantile->data(), tempmem[0]->d_quantile->data(),
        this->rf_params.tree_params.n_bins * n_cols * sizeof(T),
        cudaMemcpyDeviceToDevice, tempmem[i]->stream));
      memcpy((void*)(tempmem[i]->h_quantile->data()),
             (void*)(tempmem[0]->h_quantile->data()),
             this->rf_params.tree_params.n_bins * n_cols * sizeof(T));
    }
  }

#pragma omp parallel for num_threads(n_streams)
  for (int i = 0; i < this->rf_params.n_trees; i++) {
    int stream_id = omp_get_thread_num();
    this->prepare_fit_per_tree(
      local_handle[stream_id].getImpl(), i, n_rows, n_sampled_rows,
      selected_rows[stream_id]->data(), sorted_selected_rows[stream_id]->data(),
      rows_temp_storage[stream_id]->data(), temp_storage_bytes[stream_id]);

    /* Build individual tree in the forest.
       - input is a pointer to orig data that have n_cols features and n_rows rows.
       - n_sampled_rows: # rows sampled for tree's bootstrap sample.
       - sorted_selected_rows: points to a list of row #s (w/ n_sampled_rows elements)
         used to build the bootstrapped sample. Expectation: Each tree node will contain
         (a) # n_sampled_rows and (b) a pointer to a list of row numbers w.r.t original data.
    */

    DecisionTree::TreeMetaDataNode<T, T>* tree_ptr = &(forest->trees[i]);
    trees[i].fit(local_handle[stream_id], input, n_cols, n_rows, labels,
                 sorted_selected_rows[stream_id]->data(), n_sampled_rows,
                 tree_ptr, this->rf_params.tree_params, tempmem[stream_id]);
  }
  //Cleanup
  for (int i = 0; i < n_streams; i++) {
    rows_temp_storage[i]->release(stream);
    selected_rows[i]->release(stream);
    sorted_selected_rows[i]->release(stream);
    tempmem[i].reset();
    delete rows_temp_storage[i];
    delete selected_rows[i];
    delete sorted_selected_rows[i];
  }
  for (int i = 0; i < n_streams; i++) {
    CUDA_CHECK(cudaStreamSynchronize(local_handle[i].getStream()));
  }
  for (int i = 0; i < n_streams; i++) {
    CUDA_CHECK(cudaStreamDestroy(local_stream[i]));
  }
  CUDA_CHECK(cudaStreamSynchronize(user_handle.getStream()));
}

/**
 * @brief Predict target feature for input data; regression for single feature supported.
 * @tparam T: data type for input data (float or double).
 * @param[in] user_handle: cumlHandle.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] forest: CPU pointer to RandomForestMetaData struct
 * @param[in] verbose: flag for debugging purposes.
 */
template <typename T>
void rfRegressor<T>::predict(const cumlHandle& user_handle, const T* input,
                             int n_rows, int n_cols, T* predictions,
                             const RandomForestMetaData<T, T>* forest,
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
      /*if (verbose) {
        std::cout << "Printing tree " << i << std::endl;
        trees[i].print(forest->trees[i].root);
      }*/
      T prediction;
      trees[i].predict(user_handle, &forest->trees[i],
                       &h_input[row_id * row_size], 1, n_cols, &prediction,
                       verbose);
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
 * @param[in] forest: CPU pointer to RandomForestMetaData struct
 * @param[in] verbose: flag for debugging purposes.
 */
template <typename T>
RF_metrics rfRegressor<T>::score(const cumlHandle& user_handle, const T* input,
                                 const T* ref_labels, int n_rows, int n_cols,
                                 T* predictions,
                                 const RandomForestMetaData<T, T>* forest,
                                 bool verbose) const {
  predict(user_handle, input, n_rows, n_cols, predictions, forest, verbose);

  cudaStream_t stream = user_handle.getImpl().getStream();
  auto d_alloc = user_handle.getDeviceAllocator();

  double mean_abs_error, mean_squared_error, median_abs_error;
  MLCommon::Score::regression_metrics(predictions, ref_labels, n_rows, d_alloc,
                                      stream, mean_abs_error,
                                      mean_squared_error, median_abs_error);
  RF_metrics stats = set_rf_metrics_regression(
    mean_abs_error, mean_squared_error, median_abs_error);
  if (verbose) print(stats);

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

}  //End namespace ML
