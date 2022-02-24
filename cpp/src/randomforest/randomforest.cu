/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cuml/common/logger.hpp>
#include <cuml/ensemble/randomforest.hpp>
#include <cuml/tree/flatnode.h>

#include <treelite/c_api.h>
#include <treelite/tree.h>

#include <raft/error.hpp>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "randomforest.cuh"

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
RF_metrics set_all_rf_metrics(RF_type rf_type,
                              float accuracy,
                              double mean_abs_error,
                              double mean_squared_error,
                              double median_abs_error)
{
  RF_metrics rf_metrics;
  rf_metrics.rf_type            = rf_type;
  rf_metrics.accuracy           = accuracy;
  rf_metrics.mean_abs_error     = mean_abs_error;
  rf_metrics.mean_squared_error = mean_squared_error;
  rf_metrics.median_abs_error   = median_abs_error;
  return rf_metrics;
}

/**
 * @brief Set RF_metrics for classification.
 * @param[in] cfg_accuracy: accuracy.
 * @return RF_metrics struct with classification score.
 */
RF_metrics set_rf_metrics_classification(float accuracy)
{
  return set_all_rf_metrics(RF_type::CLASSIFICATION, accuracy, -1.0, -1.0, -1.0);
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
                                     double median_abs_error)
{
  return set_all_rf_metrics(
    RF_type::REGRESSION, -1.0, mean_abs_error, mean_squared_error, median_abs_error);
}

/**
 * @brief Print either accuracy metric for classification, or mean absolute error,
 *   mean squared error, and median absolute error metrics for regression.
 * @param[in] rf_metrics: random forest metrics to print.
 */
void print(const RF_metrics rf_metrics)
{
  if (rf_metrics.rf_type == RF_type::CLASSIFICATION) {
    CUML_LOG_DEBUG("Accuracy: %f", rf_metrics.accuracy);
  } else if (rf_metrics.rf_type == RF_type::REGRESSION) {
    CUML_LOG_DEBUG("Mean Absolute Error: %f", rf_metrics.mean_abs_error);
    CUML_LOG_DEBUG("Mean Squared Error: %f", rf_metrics.mean_squared_error);
    CUML_LOG_DEBUG("Median Absolute Error: %f", rf_metrics.median_abs_error);
  }
}

/**
 * @brief Update labels so they are unique from 0 to n_unique_labels values.
 *   Create/update an old label to new label map per random forest.
 * @param[in] n_rows: number of rows (labels)
 * @param[in,out] labels: 1D labels array to be changed in-place.
 * @param[in,out] labels_map: map of old label values to new ones.
 * @param[in] verbosity: verbosity level for logging messages during execution
 */
void preprocess_labels(int n_rows,
                       std::vector<int>& labels,
                       std::map<int, int>& labels_map,
                       int verbosity)
{
  std::pair<std::map<int, int>::iterator, bool> ret;
  int n_unique_labels = 0;
  ML::Logger::get().setLevel(verbosity);

  CUML_LOG_DEBUG("Preprocessing labels");
  for (int i = 0; i < n_rows; i++) {
    ret = labels_map.insert(std::pair<int, int>(labels[i], n_unique_labels));
    if (ret.second) { n_unique_labels += 1; }
    auto prev = labels[i];
    labels[i] = ret.first->second;  // Update labels **IN-PLACE**
    CUML_LOG_DEBUG("Mapping %d to %d", prev, labels[i]);
  }
  CUML_LOG_DEBUG("Finished preprocessing labels");
}

/**
 * @brief Revert label preprocessing effect, if needed.
 * @param[in] n_rows: number of rows (labels)
 * @param[in,out] labels: 1D labels array to be changed in-place.
 * @param[in] labels_map: map of old to new label values used during preprocessing.
 * @param[in] verbosity: verbosity level for logging messages during execution
 */
void postprocess_labels(int n_rows,
                        std::vector<int>& labels,
                        std::map<int, int>& labels_map,
                        int verbosity)
{
  ML::Logger::get().setLevel(verbosity);
  CUML_LOG_DEBUG("Postrocessing labels");
  std::map<int, int>::iterator it;
  int n_unique_cnt = labels_map.size();
  std::vector<int> reverse_map;
  reverse_map.resize(n_unique_cnt);
  for (auto it = labels_map.begin(); it != labels_map.end(); it++) {
    reverse_map[it->second] = it->first;
  }

  for (int i = 0; i < n_rows; i++) {
    auto prev = labels[i];
    labels[i] = reverse_map[prev];
    CUML_LOG_DEBUG("Mapping %d back to %d", prev, labels[i]);
  }
  CUML_LOG_DEBUG("Finished postrocessing labels");
}

/**
 * @brief Deletes RandomForestMetaData object
 * @param[in] forest: CPU pointer to RandomForestMetaData.
 */
template <class T, class L>
void delete_rf_metadata(RandomForestMetaData<T, L>* forest)
{
  delete forest;
}

template <class T, class L>
std::string _get_rf_text(const RandomForestMetaData<T, L>* forest, bool summary)
{
  ML::PatternSetter _("%v");
  if (!forest) {
    return "Empty forest";
  } else {
    std::ostringstream oss;
    oss << "Forest has " << forest->rf_params.n_trees << " trees, "
        << "max_depth " << forest->rf_params.tree_params.max_depth << ", and max_leaves "
        << forest->rf_params.tree_params.max_leaves << "\n";
    for (int i = 0; i < forest->rf_params.n_trees; i++) {
      oss << "Tree #" << i << "\n";
      if (summary) {
        oss << DT::get_tree_summary_text<T, L>(forest->trees[i].get()) << "\n";
      } else {
        oss << DT::get_tree_text<T, L>(forest->trees[i].get()) << "\n";
      }
    }
    return oss.str();
  }
}

template <class T, class L>
std::string _get_rf_json(const RandomForestMetaData<T, L>* forest)
{
  if (!forest) { return "[]"; }
  std::ostringstream oss;
  oss << "[\n";
  for (int i = 0; i < forest->rf_params.n_trees; i++) {
    oss << DT::get_tree_json<T, L>(forest->trees[i].get());
    if (i < forest->rf_params.n_trees - 1) { oss << ",\n"; }
  }
  oss << "\n]";
  return oss.str();
}

/**
 * @brief Print summary for all trees in the random forest.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] forest: CPU pointer to RandomForestMetaData struct.
 */
template <class T, class L>
std::string get_rf_summary_text(const RandomForestMetaData<T, L>* forest)
{
  return _get_rf_text(forest, true);
}

/**
 * @brief Print detailed view of all trees in the random forest.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] forest: CPU pointer to RandomForestMetaData struct.
 */
template <class T, class L>
std::string get_rf_detailed_text(const RandomForestMetaData<T, L>* forest)
{
  return _get_rf_text(forest, false);
}

template <class T, class L>
std::string get_rf_json(const RandomForestMetaData<T, L>* forest)
{
  return _get_rf_json(forest);
}

template <class T, class L>
void build_treelite_forest(ModelHandle* model_handle,
                           const RandomForestMetaData<T, L>* forest,
                           int num_features)
{
  auto parent_model          = tl::Model::Create<T, T>();
  tl::ModelImpl<T, T>* model = dynamic_cast<tl::ModelImpl<T, T>*>(parent_model.get());
  ASSERT(model != nullptr, "Invalid downcast to tl::ModelImpl");

  // Determine number of outputs
  int num_outputs = forest->trees.front()->num_outputs;
  ASSERT(num_outputs > 0, "Invalid forest");
  for (const auto& tree : forest->trees) {
    ASSERT(num_outputs == tree->num_outputs, "Invalid forest");
  }

  if constexpr (std::is_integral_v<L>) {
    ASSERT(num_outputs > 1, "More than one variable expected for classification problem.");
    model->task_type = tl::TaskType::kMultiClfProbDistLeaf;
    std::strncpy(model->param.pred_transform, "max_index", sizeof(model->param.pred_transform));
  } else {
    model->task_type = tl::TaskType::kBinaryClfRegr;
  }

  model->task_param = tl::TaskParam{
    tl::TaskParam::OutputType::kFloat, false, (unsigned int)num_outputs, (unsigned int)num_outputs};
  model->num_feature         = num_features;
  model->average_tree_output = true;
  model->SetTreeLimit(forest->rf_params.n_trees);

#pragma omp parallel for
  for (int i = 0; i < forest->rf_params.n_trees; i++) {
    auto rf_tree = forest->trees[i];

    if (rf_tree->sparsetree.size() != 0) {
      model->trees[i] = DT::build_treelite_tree<T, L>(*rf_tree, num_outputs);
    }
  }

  *model_handle = static_cast<ModelHandle>(parent_model.release());
}

/**
 * @brief Compares the trees present in concatenated treelite forest with the trees
 *   of the forests present in the different workers. If there is a difference in the two
 *   then an error statement will be thrown.
 * @param[in] tree_from_concatenated_forest: Tree info from the concatenated forest.
 * @param[in] tree_from_individual_forest: Tree info from the forest present in each worker.
 */
template <class T, class L>
void compare_trees(tl::Tree<T, L>& tree_from_concatenated_forest,
                   tl::Tree<T, L>& tree_from_individual_forest)
{
  ASSERT(tree_from_concatenated_forest.num_nodes == tree_from_individual_forest.num_nodes,
         "Error! Mismatch the number of nodes present in a tree in the "
         "concatenated forest and"
         " the tree present in the individual forests");
  for (int each_node = 0; each_node < tree_from_concatenated_forest.num_nodes; each_node++) {
    ASSERT(tree_from_concatenated_forest.IsLeaf(each_node) ==
             tree_from_individual_forest.IsLeaf(each_node),
           "Error! mismatch in the position of a leaf between concatenated "
           "forest and the"
           " individual forests ");
    ASSERT(tree_from_concatenated_forest.LeafValue(each_node) ==
             tree_from_individual_forest.LeafValue(each_node),
           "Error! leaf value mismatch between concatenated forest and the"
           " individual forests ");
    ASSERT(tree_from_concatenated_forest.RightChild(each_node) ==
             tree_from_individual_forest.RightChild(each_node),
           "Error! mismatch in the position of the node between concatenated "
           "forest and the"
           " individual forests ");
    ASSERT(tree_from_concatenated_forest.LeftChild(each_node) ==
             tree_from_individual_forest.LeftChild(each_node),
           "Error! mismatch in the position of the node between concatenated "
           "forest and the"
           " individual forests ");
    ASSERT(tree_from_concatenated_forest.SplitIndex(each_node) ==
             tree_from_individual_forest.SplitIndex(each_node),
           "Error! split index value mismatch between concatenated forest and the"
           " individual forests ");
  }
}

/**
 * @brief Compares the concatenated treelite model with the information of the forest
 *   present in the different workers. If there is a difference in the two then an error
 *   statement will be thrown.
 * @param[in] concat_tree_handle: ModelHandle for the concatenated forest.
 * @param[in] treelite_handles: List containing ModelHandles for the forest present in
 *   each worker.
 */
void compare_concat_forest_to_subforests(ModelHandle concat_tree_handle,
                                         std::vector<ModelHandle> treelite_handles)
{
  size_t concat_forest;
  size_t total_num_trees = 0;
  for (std::size_t forest_idx = 0; forest_idx < treelite_handles.size(); forest_idx++) {
    size_t num_trees_each_forest;
    TREELITE_CHECK(TreeliteQueryNumTree(treelite_handles[forest_idx], &num_trees_each_forest));
    total_num_trees = total_num_trees + num_trees_each_forest;
  }

  TREELITE_CHECK(TreeliteQueryNumTree(concat_tree_handle, &concat_forest));

  ASSERT(concat_forest == total_num_trees,
         "Error! the number of trees in the concatenated forest and the sum "
         "of the trees present in the forests present in each worker are not equal");

  int concat_mod_tree_num = 0;
  tl::Model& concat_model = *(tl::Model*)(concat_tree_handle);
  for (std::size_t forest_idx = 0; forest_idx < treelite_handles.size(); forest_idx++) {
    tl::Model& model = *(tl::Model*)(treelite_handles[forest_idx]);

    ASSERT(concat_model.GetThresholdType() == model.GetThresholdType(),
           "Error! Concatenated forest does not have the same threshold type as "
           "the individual forests");
    ASSERT(concat_model.GetLeafOutputType() == model.GetLeafOutputType(),
           "Error! Concatenated forest does not have the same leaf output type as "
           "the individual forests");
    ASSERT(concat_model.num_feature == model.num_feature,
           "Error! number of features mismatch between concatenated forest and the"
           " individual forests");
    ASSERT(concat_model.task_param.num_class == model.task_param.num_class,
           "Error! number of classes mismatch between concatenated forest "
           "and the individual forests ");
    ASSERT(concat_model.average_tree_output == model.average_tree_output,
           "Error! average_tree_output flag value mismatch between "
           "concatenated forest and the individual forests");

    model.Dispatch([&concat_mod_tree_num, &concat_model](auto& model_inner) {
      // model_inner is of the concrete type tl::ModelImpl<T, L>
      using model_type         = std::remove_reference_t<decltype(model_inner)>;
      auto& concat_model_inner = dynamic_cast<model_type&>(concat_model);
      for (std::size_t indiv_trees = 0; indiv_trees < model_inner.trees.size(); indiv_trees++) {
        compare_trees(concat_model_inner.trees[concat_mod_tree_num + indiv_trees],
                      model_inner.trees[indiv_trees]);
      }
      concat_mod_tree_num = concat_mod_tree_num + model_inner.trees.size();
    });
  }
}

/**
 * @brief Concatenates the forest information present in different workers to
 *  create a single forest. This concatenated forest is stored in a new treelite model.
 *  The model created is owned by and must be freed by the user.
 * @param[in] concat_tree_handle: ModelHandle for the concatenated forest.
 * @param[in] treelite_handles: List containing ModelHandles for the forest present in
 *   each worker.
 */
ModelHandle concatenate_trees(std::vector<ModelHandle> treelite_handles)
{
  tl::Model& first_model  = *(tl::Model*)treelite_handles[0];
  tl::Model* concat_model = first_model.Dispatch([&treelite_handles](auto& first_model_inner) {
    // first_model_inner is of the concrete type tl::ModelImpl<T, L>
    using model_type   = std::remove_reference_t<decltype(first_model_inner)>;
    auto* concat_model = dynamic_cast<model_type*>(
      tl::Model::Create(first_model_inner.GetThresholdType(), first_model_inner.GetLeafOutputType())
        .release());
    for (std::size_t forest_idx = 0; forest_idx < treelite_handles.size(); forest_idx++) {
      tl::Model& model  = *(tl::Model*)treelite_handles[forest_idx];
      auto& model_inner = dynamic_cast<model_type&>(model);
      for (const auto& tree : model_inner.trees) {
        concat_model->trees.push_back(tree.Clone());
      }
    }
    concat_model->num_feature         = first_model_inner.num_feature;
    concat_model->task_param          = first_model_inner.task_param;
    concat_model->average_tree_output = first_model_inner.average_tree_output;
    concat_model->param               = first_model_inner.param;
    return static_cast<tl::Model*>(concat_model);
  });
  return concat_model;
}

/**
 * @defgroup RandomForestClassificationFit Random Forest Classification - Fit function
 * @brief Build (i.e., fit, train) random forest classifier for input data.
 * @param[in] user_handle: raft::handle_t
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
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @{
 */
void fit(const raft::handle_t& user_handle,
         RandomForestClassifierF*& forest,
         float* input,
         int n_rows,
         int n_cols,
         int* labels,
         int n_unique_labels,
         RF_params rf_params,
         int verbosity)
{
  raft::common::nvtx::range fun_scope("RF::fit @randomforest.cu");
  ML::Logger::get().setLevel(verbosity);
  ASSERT(forest->trees.empty(), "Cannot fit an existing forest.");
  forest->trees.resize(rf_params.n_trees);
  forest->rf_params = rf_params;

  std::shared_ptr<RandomForest<float, int>> rf_classifier =
    std::make_shared<RandomForest<float, int>>(rf_params, RF_type::CLASSIFICATION);
  rf_classifier->fit(user_handle, input, n_rows, n_cols, labels, n_unique_labels, forest);
}

void fit(const raft::handle_t& user_handle,
         RandomForestClassifierD*& forest,
         double* input,
         int n_rows,
         int n_cols,
         int* labels,
         int n_unique_labels,
         RF_params rf_params,
         int verbosity)
{
  raft::common::nvtx::range fun_scope("RF::fit @randomforest.cu");
  ML::Logger::get().setLevel(verbosity);
  ASSERT(forest->trees.empty(), "Cannot fit an existing forest.");
  forest->trees.resize(rf_params.n_trees);
  forest->rf_params = rf_params;

  std::shared_ptr<RandomForest<double, int>> rf_classifier =
    std::make_shared<RandomForest<double, int>>(rf_params, RF_type::CLASSIFICATION);
  rf_classifier->fit(user_handle, input, n_rows, n_cols, labels, n_unique_labels, forest);
}
/** @} */

/**
 * @defgroup RandomForestClassificationPredict Random Forest Classification - Predict function
 * @brief Predict target feature for input data; n-ary classification for
     single feature supported.
 * @param[in] user_handle: raft::handle_t.
 * @param[in] forest: CPU pointer to RandomForestMetaData object.
 *   The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @{
 */
void predict(const raft::handle_t& user_handle,
             const RandomForestClassifierF* forest,
             const float* input,
             int n_rows,
             int n_cols,
             int* predictions,
             int verbosity)
{
  ASSERT(!forest->trees.empty(), "Cannot predict! No trees in the forest.");
  std::shared_ptr<RandomForest<float, int>> rf_classifier =
    std::make_shared<RandomForest<float, int>>(forest->rf_params, RF_type::CLASSIFICATION);
  rf_classifier->predict(user_handle, input, n_rows, n_cols, predictions, forest, verbosity);
}

void predict(const raft::handle_t& user_handle,
             const RandomForestClassifierD* forest,
             const double* input,
             int n_rows,
             int n_cols,
             int* predictions,
             int verbosity)
{
  ASSERT(!forest->trees.empty(), "Cannot predict! No trees in the forest.");
  std::shared_ptr<RandomForest<double, int>> rf_classifier =
    std::make_shared<RandomForest<double, int>>(forest->rf_params, RF_type::CLASSIFICATION);
  rf_classifier->predict(user_handle, input, n_rows, n_cols, predictions, forest, verbosity);
}

/**
 * @defgroup RandomForestClassificationScore Random Forest Classification - Score function
 * @brief Compare predicted features validate against ref_labels.
 * @param[in] user_handle: raft::handle_t.
 * @param[in] forest: CPU pointer to RandomForestMetaData object.
 *   The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @return RF_metrics struct with classification score (i.e., accuracy)
 * @{
 */
RF_metrics score(const raft::handle_t& user_handle,
                 const RandomForestClassifierF* forest,
                 const int* ref_labels,
                 int n_rows,
                 const int* predictions,
                 int verbosity)
{
  RF_metrics classification_score = RandomForest<float, int>::score(
    user_handle, ref_labels, n_rows, predictions, verbosity, RF_type::CLASSIFICATION);
  return classification_score;
}

RF_metrics score(const raft::handle_t& user_handle,
                 const RandomForestClassifierD* forest,
                 const int* ref_labels,
                 int n_rows,
                 const int* predictions,
                 int verbosity)
{
  RF_metrics classification_score = RandomForest<double, int>::score(
    user_handle, ref_labels, n_rows, predictions, verbosity, RF_type::CLASSIFICATION);
  return classification_score;
}

/**
 * @brief Check validity of all random forest hyper-parameters.
 * @param[in] rf_params: random forest hyper-parameters
 */
void validity_check(const RF_params rf_params)
{
  ASSERT((rf_params.n_trees > 0), "Invalid n_trees %d", rf_params.n_trees);
  ASSERT((rf_params.max_samples > 0) && (rf_params.max_samples <= 1.0),
         "max_samples value %f outside permitted (0, 1] range",
         rf_params.max_samples);
}

RF_params set_rf_params(int max_depth,
                        int max_leaves,
                        float max_features,
                        int max_n_bins,
                        int min_samples_leaf,
                        int min_samples_split,
                        float min_impurity_decrease,
                        bool bootstrap,
                        int n_trees,
                        float max_samples,
                        uint64_t seed,
                        CRITERION split_criterion,
                        int cfg_n_streams,
                        int max_batch_size)
{
  DT::DecisionTreeParams tree_params;
  DT::set_tree_params(tree_params,
                      max_depth,
                      max_leaves,
                      max_features,
                      max_n_bins,
                      min_samples_leaf,
                      min_samples_split,
                      min_impurity_decrease,
                      split_criterion,
                      max_batch_size);
  RF_params rf_params;
  rf_params.n_trees     = n_trees;
  rf_params.bootstrap   = bootstrap;
  rf_params.max_samples = max_samples;
  rf_params.seed        = seed;
  rf_params.n_streams   = min(cfg_n_streams, omp_get_max_threads());
  if (n_trees < rf_params.n_streams) rf_params.n_streams = n_trees;
  rf_params.tree_params = tree_params;
  validity_check(rf_params);
  return rf_params;
}

/** @} */

/**
 * @defgroup RandomForestRegressorFit Random Forest Regression - Fit function
 * @brief Build (i.e., fit, train) random forest regressor for input data.
 * @param[in] user_handle: raft::handle_t
 * @param[in,out] forest: CPU pointer to RandomForestMetaData object. User allocated.
 * @param[in] input: train data (n_rows samples, n_cols features) in column major format,
 *   excluding labels. Device pointer.
 * @param[in] n_rows: number of training data samples.
 * @param[in] n_cols: number of features (i.e., columns) excluding target feature.
 * @param[in] labels: 1D array of target features (float or double), with one label per
 *   training sample. Device pointer.
 * @param[in] rf_params: Random Forest training hyper parameter struct.
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @{
 */
void fit(const raft::handle_t& user_handle,
         RandomForestRegressorF*& forest,
         float* input,
         int n_rows,
         int n_cols,
         float* labels,
         RF_params rf_params,
         int verbosity)
{
  raft::common::nvtx::range fun_scope("RF::fit @randomforest.cu");
  ML::Logger::get().setLevel(verbosity);
  ASSERT(forest->trees.empty(), "Cannot fit an existing forest.");
  forest->trees.resize(rf_params.n_trees);
  forest->rf_params = rf_params;

  std::shared_ptr<RandomForest<float, float>> rf_regressor =
    std::make_shared<RandomForest<float, float>>(rf_params, RF_type::REGRESSION);
  rf_regressor->fit(user_handle, input, n_rows, n_cols, labels, 1, forest);
}

void fit(const raft::handle_t& user_handle,
         RandomForestRegressorD*& forest,
         double* input,
         int n_rows,
         int n_cols,
         double* labels,
         RF_params rf_params,
         int verbosity)
{
  raft::common::nvtx::range fun_scope("RF::fit @randomforest.cu");
  ML::Logger::get().setLevel(verbosity);
  ASSERT(forest->trees.empty(), "Cannot fit an existing forest.");
  forest->trees.resize(rf_params.n_trees);
  forest->rf_params = rf_params;

  std::shared_ptr<RandomForest<double, double>> rf_regressor =
    std::make_shared<RandomForest<double, double>>(rf_params, RF_type::REGRESSION);
  rf_regressor->fit(user_handle, input, n_rows, n_cols, labels, 1, forest);
}
/** @} */

/**
 * @defgroup RandomForestRegressorPredict Random Forest Regression - Predict function
 * @brief Predict target feature for input data; regression for single feature supported.
 * @param[in] user_handle: raft::handle_t.
 * @param[in] forest: CPU pointer to RandomForestMetaData object.
 *   The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in, out] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @{
 */
void predict(const raft::handle_t& user_handle,
             const RandomForestRegressorF* forest,
             const float* input,
             int n_rows,
             int n_cols,
             float* predictions,
             int verbosity)
{
  std::shared_ptr<RandomForest<float, float>> rf_regressor =
    std::make_shared<RandomForest<float, float>>(forest->rf_params, RF_type::REGRESSION);
  rf_regressor->predict(user_handle, input, n_rows, n_cols, predictions, forest, verbosity);
}

void predict(const raft::handle_t& user_handle,
             const RandomForestRegressorD* forest,
             const double* input,
             int n_rows,
             int n_cols,
             double* predictions,
             int verbosity)
{
  std::shared_ptr<RandomForest<double, double>> rf_regressor =
    std::make_shared<RandomForest<double, double>>(forest->rf_params, RF_type::REGRESSION);
  rf_regressor->predict(user_handle, input, n_rows, n_cols, predictions, forest, verbosity);
}
/** @} */

/**
 * @defgroup RandomForestRegressorScore Random Forest Regression - Score function
 * @brief Predict target feature for input data and validate against ref_labels.
 * @param[in] user_handle: raft::handle_t.
 * @param[in] forest: CPU pointer to RandomForestMetaData object.
 *   The user should have previously called fit to build the random forest.
 * @param[in] input: test data (n_rows samples, n_cols features) in row major format. GPU pointer.
 * @param[in] ref_labels: label values for cross validation (n_rows elements); GPU pointer.
 * @param[in] n_rows: number of  data samples.
 * @param[in] n_cols: number of features (excluding target feature).
 * @param[in] predictions: n_rows predicted labels. GPU pointer, user allocated.
 * @param[in] verbosity: verbosity level for logging messages during execution
 * @return RF_metrics struct with regression score (i.e., mean absolute error,
 *   mean squared error, median absolute error)
 * @{
 */
RF_metrics score(const raft::handle_t& user_handle,
                 const RandomForestRegressorF* forest,
                 const float* ref_labels,
                 int n_rows,
                 const float* predictions,
                 int verbosity)
{
  RF_metrics regression_score = RandomForest<float, float>::score(
    user_handle, ref_labels, n_rows, predictions, verbosity, RF_type::REGRESSION);

  return regression_score;
}

RF_metrics score(const raft::handle_t& user_handle,
                 const RandomForestRegressorD* forest,
                 const double* ref_labels,
                 int n_rows,
                 const double* predictions,
                 int verbosity)
{
  RF_metrics regression_score = RandomForest<double, double>::score(
    user_handle, ref_labels, n_rows, predictions, verbosity, RF_type::REGRESSION);
  return regression_score;
}
/** @} */

// Functions' specializations
template std::string get_rf_summary_text<float, int>(const RandomForestClassifierF* forest);
template std::string get_rf_summary_text<double, int>(const RandomForestClassifierD* forest);
template std::string get_rf_summary_text<float, float>(const RandomForestRegressorF* forest);
template std::string get_rf_summary_text<double, double>(const RandomForestRegressorD* forest);

template std::string get_rf_detailed_text<float, int>(const RandomForestClassifierF* forest);
template std::string get_rf_detailed_text<double, int>(const RandomForestClassifierD* forest);
template std::string get_rf_detailed_text<float, float>(const RandomForestRegressorF* forest);
template std::string get_rf_detailed_text<double, double>(const RandomForestRegressorD* forest);

template std::string get_rf_json<float, int>(const RandomForestClassifierF* forest);
template std::string get_rf_json<double, int>(const RandomForestClassifierD* forest);
template std::string get_rf_json<float, float>(const RandomForestRegressorF* forest);
template std::string get_rf_json<double, double>(const RandomForestRegressorD* forest);

template void delete_rf_metadata<float, int>(RandomForestClassifierF* forest);
template void delete_rf_metadata<double, int>(RandomForestClassifierD* forest);
template void delete_rf_metadata<float, float>(RandomForestRegressorF* forest);
template void delete_rf_metadata<double, double>(RandomForestRegressorD* forest);

template void build_treelite_forest<float, int>(ModelHandle* model,
                                                const RandomForestMetaData<float, int>* forest,
                                                int num_features);
template void build_treelite_forest<double, int>(ModelHandle* model,
                                                 const RandomForestMetaData<double, int>* forest,
                                                 int num_features);
template void build_treelite_forest<float, float>(ModelHandle* model,
                                                  const RandomForestMetaData<float, float>* forest,
                                                  int num_features);
template void build_treelite_forest<double, double>(
  ModelHandle* model, const RandomForestMetaData<double, double>* forest, int num_features);
}  // End namespace ML
