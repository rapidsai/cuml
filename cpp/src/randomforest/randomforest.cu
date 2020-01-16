/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#endif
#include <treelite/tree.h>
#include <cstdio>
#include <cuml/ensemble/randomforest.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "randomforest_impl.cuh"

namespace ML {

using namespace MLCommon;
using namespace std;
namespace tl = treelite;

/**
 * @brief Set RF_metrics.
 * @param[in] rf_type: Random Forest type: classification or regression
 * @param[in] cfg_accuracy: accuracy.
 * @param[in] mean_abs_error: mean absolute error.
 * @param[in] mean_squared_error: mean squared error.
 * @param[in] median_abs_error: median absolute error.
 * @return RF_metrics struct with classification or regression score.
 */
RF_metrics set_all_rf_metrics(RF_type rf_type, float accuracy,
                              double mean_abs_error, double mean_squared_error,
                              double median_abs_error) {
  RF_metrics rf_metrics;
  rf_metrics.rf_type = rf_type;
  rf_metrics.accuracy = accuracy;
  rf_metrics.mean_abs_error = mean_abs_error;
  rf_metrics.mean_squared_error = mean_squared_error;
  rf_metrics.median_abs_error = median_abs_error;
  return rf_metrics;
}

/**
 * @brief Set RF_metrics for classification.
 * @param[in] cfg_accuracy: accuracy.
 * @return RF_metrics struct with classification score.
 */
RF_metrics set_rf_metrics_classification(float accuracy) {
  return set_all_rf_metrics(RF_type::CLASSIFICATION, accuracy, -1.0, -1.0,
                            -1.0);
}

/**
 * @brief Set RF_metrics for regression.
 * @param[in] mean_abs_error: mean absolute error.
 * @param[in] mean_squared_error: mean squared error.
 * @param[in] median_abs_error: median absolute error.
 * @return RF_metrics struct with regression score.
 */
RF_metrics set_rf_metrics_regression(double mean_abs_error,
                                     double mean_squared_error,
                                     double median_abs_error) {
  return set_all_rf_metrics(RF_type::REGRESSION, -1.0, mean_abs_error,
                            mean_squared_error, median_abs_error);
}

/**
 * @brief Print either accuracy metric for classification, or mean absolute error,
 *   mean squared error, and median absolute error metrics for regression.
 * @param[in] rf_metrics: random forest metrics to print.
 */
void print(const RF_metrics rf_metrics) {
  if (rf_metrics.rf_type == RF_type::CLASSIFICATION) {
    std::cout << "Accuracy: " << rf_metrics.accuracy << std::endl;
  } else if (rf_metrics.rf_type == RF_type::REGRESSION) {
    std::cout << "Mean Absolute Error: " << rf_metrics.mean_abs_error
              << std::endl;
    std::cout << "Mean Squared Error: " << rf_metrics.mean_squared_error
              << std::endl;
    std::cout << "Median Absolute Error: " << rf_metrics.median_abs_error
              << std::endl;
  }
}

/**
 * @brief Update labels so they are unique from 0 to n_unique_labels values.
 *   Create/update an old label to new label map per random forest.
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
 * @brief Set RF_params parameters members; use default tree parameters.
 * @param[in,out] params: update with random forest parameters
 * @param[in] cfg_n_trees: number of trees; default 1
 * @param[in] cfg_bootstrap: bootstrapping; default true
 * @param[in] cfg_rows_sample: rows sample; default 1.0f
 * @param[in] cfg_n_streams: No of parallel CUDA for training forest
 */
void set_rf_params(RF_params& params, int cfg_n_trees, bool cfg_bootstrap,
                   float cfg_rows_sample, int cfg_seed, int cfg_n_streams) {
  params.n_trees = cfg_n_trees;
  params.bootstrap = cfg_bootstrap;
  params.rows_sample = cfg_rows_sample;
  params.seed = cfg_seed;
  params.n_streams = min(cfg_n_streams, omp_get_max_threads());
  if (params.n_streams == cfg_n_streams) {
    std::cout << "Warning! Max setting Max streams to max openmp threads "
              << omp_get_max_threads() << std::endl;
  }
  if (cfg_n_trees < params.n_streams) params.n_streams = cfg_n_trees;
  set_tree_params(params.tree_params);  // use default tree params
}

/**
 * @brief Set all RF_params parameters members, including tree parameters.
 * @param[in,out] params: update with random forest parameters
 * @param[in] cfg_n_trees: number of trees
 * @param[in] cfg_bootstrap: bootstrapping
 * @param[in] cfg_rows_sample: rows sample
 * @param[in] cfg_n_streams: No of parallel CUDA for training forest
 * @param[in] cfg_tree_params: tree parameters
 */
void set_all_rf_params(RF_params& params, int cfg_n_trees, bool cfg_bootstrap,
                       float cfg_rows_sample, int cfg_seed, int cfg_n_streams,
                       DecisionTree::DecisionTreeParams cfg_tree_params) {
  params.n_trees = cfg_n_trees;
  params.bootstrap = cfg_bootstrap;
  params.rows_sample = cfg_rows_sample;
  params.seed = cfg_seed;
  params.n_streams = min(cfg_n_streams, omp_get_max_threads());
  if (cfg_n_trees < params.n_streams) params.n_streams = cfg_n_trees;
  set_tree_params(params.tree_params);  // use input tree params
  params.tree_params = cfg_tree_params;
}

/**
 * @brief Check validity of all random forest hyper-parameters.
 * @param[in] rf_params: random forest hyper-parameters
 */
void validity_check(const RF_params rf_params) {
  ASSERT((rf_params.n_trees > 0), "Invalid n_trees %d", rf_params.n_trees);
  ASSERT((rf_params.rows_sample > 0) && (rf_params.rows_sample <= 1.0),
         "rows_sample value %f outside permitted (0, 1] range",
         rf_params.rows_sample);
  DecisionTree::validity_check(rf_params.tree_params);
}

/**
 * @brief Print all random forest hyper-parameters.
 * @param[in] rf_params: random forest hyper-parameters
 */
void print(const RF_params rf_params) {
  std::cout << "n_trees: " << rf_params.n_trees << std::endl;
  std::cout << "bootstrap: " << rf_params.bootstrap << std::endl;
  std::cout << "rows_sample: " << rf_params.rows_sample << std::endl;
  std::cout << "n_streams: " << rf_params.n_streams << std::endl;
  DecisionTree::print(rf_params.tree_params);
}

/**
 * @brief Set the trees pointer of RandomForestMetaData to nullptr.
 * @param[in, out] forest: CPU pointer to RandomForestMetaData.
 */
template <class T, class L>
void null_trees_ptr(RandomForestMetaData<T, L>*& forest) {
  forest->trees = nullptr;
}

/**
 * @brief Print summary for all trees in the random forest.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] forest: CPU pointer to RandomForestMetaData struct.
 */
template <class T, class L>
void print_rf_summary(const RandomForestMetaData<T, L>* forest) {
  if (!forest || !forest->trees) {
    std::cout << "Empty forest" << std::endl;
  } else {
    std::cout << "Forest has " << forest->rf_params.n_trees
              << " trees, max_depth "
              << forest->rf_params.tree_params.max_depth;
    std::cout << ", and max_leaves " << forest->rf_params.tree_params.max_leaves
              << std::endl;
    for (int i = 0; i < forest->rf_params.n_trees; i++) {
      std::cout << "Tree #" << i << std::endl;
      DecisionTree::print_tree_summary<T, L>(&(forest->trees[i]));
    }
  }
}

/**
 * @brief Print detailed view of all trees in the random forest.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] forest: CPU pointer to RandomForestMetaData struct.
 */
template <class T, class L>
void print_rf_detailed(const RandomForestMetaData<T, L>* forest) {
  if (!forest || !forest->trees) {
    std::cout << "Empty forest" << std::endl;
  } else {
    std::cout << "Forest has " << forest->rf_params.n_trees
              << " trees, max_depth "
              << forest->rf_params.tree_params.max_depth;
    std::cout << ", and max_leaves " << forest->rf_params.tree_params.max_leaves
              << std::endl;
    for (int i = 0; i < forest->rf_params.n_trees; i++) {
      std::cout << "Tree #" << i << std::endl;
      DecisionTree::print_tree<T, L>(&(forest->trees[i]));
    }
  }
}

template <class T, class L>
void build_treelite_forest(ModelHandle* model,
                           const RandomForestMetaData<T, L>* forest,
                           int num_features, int task_category,
                           std::vector<unsigned char>& data) {
  bool check_val = (data).empty();
  if (not check_val) {
    // create a temp file
    const char* filename = std::tmpnam(nullptr);
    // write the model bytes into the temp file
    std::ofstream file(filename, std::ios::binary);
    file.write((char*)&data[0], data.size());
    // read the file as a protobuf model
    TREELITE_CHECK(TreeliteLoadProtobufModel(filename, model));
  }

  else {
    // Non-zero value here for random forest models.
    // The value should be set to 0 if the model is gradient boosted trees.
    int random_forest_flag = 1;
    ModelBuilderHandle model_builder;
    // num_output_group is 1 for binary classification and regression
    // num_output_group is #class for multiclass classification which is the same as task_category
    int num_output_group = task_category > 2 ? task_category : 1;
    TREELITE_CHECK(TreeliteCreateModelBuilder(
      num_features, num_output_group, random_forest_flag, &model_builder));

    if (task_category > 2) {
      // Multi-class classification
      TREELITE_CHECK(TreeliteModelBuilderSetModelParam(
        model_builder, "pred_transform", "max_index"));
    }

    for (int i = 0; i < forest->rf_params.n_trees; i++) {
      DecisionTree::TreeMetaDataNode<T, L>* tree_ptr = &forest->trees[i];
      TreeBuilderHandle tree_builder;

      TREELITE_CHECK(TreeliteCreateTreeBuilder(&tree_builder));
      if (tree_ptr->sparsetree.size() != 0) {
        DecisionTree::build_treelite_tree<T, L>(tree_builder, tree_ptr,
                                                num_output_group);

        // The third argument -1 means append to the end of the tree list.
        TREELITE_CHECK(
          TreeliteModelBuilderInsertTree(model_builder, tree_builder, -1));
      }
    }

    TREELITE_CHECK(TreeliteModelBuilderCommitModel(model_builder, model));
    TREELITE_CHECK(TreeliteDeleteModelBuilder(model_builder));
  }
}

std::vector<unsigned char> save_model(ModelHandle model) {
  // create a temp file
  const char* filename = std::tmpnam(nullptr);
  // export the treelite model to protobuf nd save it in the temp file
  TreeliteExportProtobufModel(filename, model);
  // read from the temp file and obtain the model bytes
  std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
  in.seekg(0, std::ios::end);
  int size_of_file = in.tellg();
  vector<unsigned char> bytes_info(size_of_file, 0);
  ifstream infile(filename, ios::in | ios::binary);
  infile.read((char*)&bytes_info[0], bytes_info.size());
  return bytes_info;
}

/**
 * @defgroup Random Forest Classification - Fit function
 * @brief Build (i.e., fit, train) random forest classifier for input data.
 * @param[in] user_handle: cumlHandle
 * @param[in,out] forest: CPU pointer to RandomForestMetaData object. User allocated.
 * @param[in] input: train data (n_rows samples, n_cols features) in column major format,
 *   excluding labels. Device pointer.
 * @param[in] n_rows: number of training data samples.
 * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (int only), with one label per
 *   training sample. Device pointer.
 *   Assumption: labels were preprocessed to map to ascending numbers from 0;
 *   needed for current gini impl. in decision tree
 * @param[in] n_unique_labels: #unique label values (known during preprocessing)
 * @param[in] rf_params: Random Forest training hyper parameter struct.
 * @{
 */
void fit(const cumlHandle& user_handle, RandomForestClassifierF*& forest,
         float* input, int n_rows, int n_cols, int* labels, int n_unique_labels,
         RF_params rf_params) {
  ASSERT(!forest->trees, "Cannot fit an existing forest.");
  forest->trees =
    new DecisionTree::TreeMetaDataNode<float, int>[rf_params.n_trees];
  forest->rf_params = rf_params;

  std::shared_ptr<rfClassifier<float>> rf_classifier =
    std::make_shared<rfClassifier<float>>(rf_params);
  rf_classifier->fit(user_handle, input, n_rows, n_cols, labels,
                     n_unique_labels, forest);
}

void fit(const cumlHandle& user_handle, RandomForestClassifierD*& forest,
         double* input, int n_rows, int n_cols, int* labels,
         int n_unique_labels, RF_params rf_params) {
  ASSERT(!forest->trees, "Cannot fit an existing forest.");
  forest->trees =
    new DecisionTree::TreeMetaDataNode<double, int>[rf_params.n_trees];
  forest->rf_params = rf_params;

  std::shared_ptr<rfClassifier<double>> rf_classifier =
    std::make_shared<rfClassifier<double>>(rf_params);
  rf_classifier->fit(user_handle, input, n_rows, n_cols, labels,
                     n_unique_labels, forest);
}
/** @} */

/**
 * @defgroup Random Forest Classification - Predict function
 * @brief Predict target feature for input data; n-ary classification for
     single feature supported.
 * @param[in] user_handle: cumlHandle.
 * @param[in] forest: CPU pointer to RandomForestMetaData object.
 *   The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 * @{
 */
void predict(const cumlHandle& user_handle,
             const RandomForestClassifierF* forest, const float* input,
             int n_rows, int n_cols, int* predictions, bool verbose) {
  ASSERT(forest->trees, "Cannot predict! No trees in the forest.");
  std::shared_ptr<rfClassifier<float>> rf_classifier =
    std::make_shared<rfClassifier<float>>(forest->rf_params);
  rf_classifier->predict(user_handle, input, n_rows, n_cols, predictions,
                         forest, verbose);
}

void predict(const cumlHandle& user_handle,
             const RandomForestClassifierD* forest, const double* input,
             int n_rows, int n_cols, int* predictions, bool verbose) {
  ASSERT(forest->trees, "Cannot predict! No trees in the forest.");
  std::shared_ptr<rfClassifier<double>> rf_classifier =
    std::make_shared<rfClassifier<double>>(forest->rf_params);
  rf_classifier->predict(user_handle, input, n_rows, n_cols, predictions,
                         forest, verbose);
}
/** @} */
/**
 * @defgroup Random Forest Classification - Predict function
 * @brief Predict target feature for input data; n-ary classification for
     single feature supported.
 * @param[in] user_handle: cumlHandle.
 * @param[in] forest: CPU pointer to RandomForestMetaData object.
 *   The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 * @{
 */
void predictGetAll(const cumlHandle& user_handle,
                   const RandomForestClassifierF* forest, const float* input,
                   int n_rows, int n_cols, int* predictions, bool verbose) {
  ASSERT(forest->trees, "Cannot predict! No trees in the forest.");
  std::shared_ptr<rfClassifier<float>> rf_classifier =
    std::make_shared<rfClassifier<float>>(forest->rf_params);
  rf_classifier->predictGetAll(user_handle, input, n_rows, n_cols, predictions,
                               forest, verbose);
}

void predictGetAll(const cumlHandle& user_handle,
                   const RandomForestClassifierD* forest, const double* input,
                   int n_rows, int n_cols, int* predictions, bool verbose) {
  ASSERT(forest->trees, "Cannot predict! No trees in the forest.");
  std::shared_ptr<rfClassifier<double>> rf_classifier =
    std::make_shared<rfClassifier<double>>(forest->rf_params);
  rf_classifier->predictGetAll(user_handle, input, n_rows, n_cols, predictions,
                               forest, verbose);
}
/** @} */

/**
 * @defgroup Random Forest Classification - Score function
 * @brief Compare predicted features validate against ref_labels.
 * @param[in] user_handle: cumlHandle.
 * @param[in] forest: CPU pointer to RandomForestMetaData object.
 *   The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 * @return RF_metrics struct with classification score (i.e., accuracy)
 * @{
 */
RF_metrics score(const cumlHandle& user_handle,
                 const RandomForestClassifierF* forest, const int* ref_labels,
                 int n_rows, const int* predictions, bool verbose) {
  RF_metrics classification_score = rfClassifier<float>::score(
    user_handle, ref_labels, n_rows, predictions, verbose);
  return classification_score;
}

RF_metrics score(const cumlHandle& user_handle,
                 const RandomForestClassifierD* forest, const int* ref_labels,
                 int n_rows, const int* predictions, bool verbose) {
  RF_metrics classification_score = rfClassifier<double>::score(
    user_handle, ref_labels, n_rows, predictions, verbose);
  return classification_score;
}

/** @} */

RF_params set_rf_class_obj(int max_depth, int max_leaves, float max_features,
                           int n_bins, int split_algo, int min_rows_per_node,
                           float min_impurity_decrease, bool bootstrap_features,
                           bool bootstrap, int n_trees, float rows_sample,
                           int seed, CRITERION split_criterion,
                           bool quantile_per_tree, int cfg_n_streams) {
  DecisionTree::DecisionTreeParams tree_params;
  DecisionTree::set_tree_params(
    tree_params, max_depth, max_leaves, max_features, n_bins, split_algo,
    min_rows_per_node, min_impurity_decrease, bootstrap_features,
    split_criterion, quantile_per_tree);
  RF_params rf_params;
  set_all_rf_params(rf_params, n_trees, bootstrap, rows_sample, seed,
                    cfg_n_streams, tree_params);
  return rf_params;
}

/**
 * @defgroup Random Forest Regression - Fit function
 * @brief Build (i.e., fit, train) random forest regressor for input data.
 * @param[in] user_handle: cumlHandle
 * @param[in,out] forest: CPU pointer to RandomForestMetaData object. User allocated.
 * @param[in] input: train data (n_rows samples, n_cols features) in column major format,
 *   excluding labels. Device pointer.
 * @param[in] n_rows: number of training data samples.
 * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (float or double), with one label per
 *   training sample. Device pointer.
 * @param[in] rf_params: Random Forest training hyper parameter struct.
 * @{
 */
void fit(const cumlHandle& user_handle, RandomForestRegressorF*& forest,
         float* input, int n_rows, int n_cols, float* labels,
         RF_params rf_params) {
  ASSERT(!forest->trees, "Cannot fit an existing forest.");
  forest->trees =
    new DecisionTree::TreeMetaDataNode<float, float>[rf_params.n_trees];
  forest->rf_params = rf_params;

  std::shared_ptr<rfRegressor<float>> rf_regressor =
    std::make_shared<rfRegressor<float>>(rf_params);
  rf_regressor->fit(user_handle, input, n_rows, n_cols, labels, forest);
}

void fit(const cumlHandle& user_handle, RandomForestRegressorD*& forest,
         double* input, int n_rows, int n_cols, double* labels,
         RF_params rf_params) {
  ASSERT(!forest->trees, "Cannot fit an existing forest.");
  forest->trees =
    new DecisionTree::TreeMetaDataNode<double, double>[rf_params.n_trees];
  forest->rf_params = rf_params;

  std::shared_ptr<rfRegressor<double>> rf_regressor =
    std::make_shared<rfRegressor<double>>(rf_params);
  rf_regressor->fit(user_handle, input, n_rows, n_cols, labels, forest);
}
/** @} */

/**
 * @defgroup Random Forest Regression - Predict function
 * @brief Predict target feature for input data; regression for single feature supported.
 * @param[in] user_handle: cumlHandle.
 * @param[in] forest: CPU pointer to RandomForestMetaData object.
 *   The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 * @{
 */
void predict(const cumlHandle& user_handle,
             const RandomForestRegressorF* forest, const float* input,
             int n_rows, int n_cols, float* predictions, bool verbose) {
  ASSERT(forest->trees, "Cannot predict! No trees in the forest.");
  std::shared_ptr<rfRegressor<float>> rf_regressor =
    std::make_shared<rfRegressor<float>>(forest->rf_params);
  rf_regressor->predict(user_handle, input, n_rows, n_cols, predictions, forest,
                        verbose);
}

void predict(const cumlHandle& user_handle,
             const RandomForestRegressorD* forest, const double* input,
             int n_rows, int n_cols, double* predictions, bool verbose) {
  ASSERT(forest->trees, "Cannot predict! No trees in the forest.");
  std::shared_ptr<rfRegressor<double>> rf_regressor =
    std::make_shared<rfRegressor<double>>(forest->rf_params);
  rf_regressor->predict(user_handle, input, n_rows, n_cols, predictions, forest,
                        verbose);
}
/** @} */

/**
 * @defgroup Random Forest Regression - Score function
 * @brief Predict target feature for input data and validate against ref_labels.
 * @param[in] user_handle: cumlHandle.
 * @param[in] forest: CPU pointer to RandomForestMetaData object.
 *   The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbose: flag for debugging purposes.
 * @return RF_metrics struct with regression score (i.e., mean absolute error,
 *   mean squared error, median absolute error)
 * @{
 */
RF_metrics score(const cumlHandle& user_handle,
                 const RandomForestRegressorF* forest, const float* ref_labels,
                 int n_rows, const float* predictions, bool verbose) {
  RF_metrics regression_score = rfRegressor<float>::score(
    user_handle, ref_labels, n_rows, predictions, verbose);

  return regression_score;
}

RF_metrics score(const cumlHandle& user_handle,
                 const RandomForestRegressorD* forest, const double* ref_labels,
                 int n_rows, const double* predictions, bool verbose) {
  RF_metrics regression_score = rfRegressor<double>::score(
    user_handle, ref_labels, n_rows, predictions, verbose);
  return regression_score;
}
/** @} */

// Functions' specializations
template void print_rf_summary<float, int>(
  const RandomForestClassifierF* forest);
template void print_rf_summary<double, int>(
  const RandomForestClassifierD* forest);
template void print_rf_summary<float, float>(
  const RandomForestRegressorF* forest);
template void print_rf_summary<double, double>(
  const RandomForestRegressorD* forest);

template void print_rf_detailed<float, int>(
  const RandomForestClassifierF* forest);
template void print_rf_detailed<double, int>(
  const RandomForestClassifierD* forest);
template void print_rf_detailed<float, float>(
  const RandomForestRegressorF* forest);
template void print_rf_detailed<double, double>(
  const RandomForestRegressorD* forest);

template void null_trees_ptr<float, int>(RandomForestClassifierF*& forest);
template void null_trees_ptr<double, int>(RandomForestClassifierD*& forest);
template void null_trees_ptr<float, float>(RandomForestRegressorF*& forest);
template void null_trees_ptr<double, double>(RandomForestRegressorD*& forest);

template void build_treelite_forest<float, int>(
  ModelHandle* model, const RandomForestMetaData<float, int>* forest,
  int num_features, int task_category, std::vector<unsigned char>& data);
template void build_treelite_forest<double, int>(
  ModelHandle* model, const RandomForestMetaData<double, int>* forest,
  int num_features, int task_category, std::vector<unsigned char>& data);
template void build_treelite_forest<float, float>(
  ModelHandle* model, const RandomForestMetaData<float, float>* forest,
  int num_features, int task_category, std::vector<unsigned char>& data);
template void build_treelite_forest<double, double>(
  ModelHandle* model, const RandomForestMetaData<double, double>* forest,
  int num_features, int task_category, std::vector<unsigned char>& data);
}  // End namespace ML
