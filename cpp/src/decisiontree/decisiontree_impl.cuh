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
#include <type_traits>
#include "decisiontree_impl.h"
#include "kernels/col_condenser.cuh"
#include "kernels/evaluate_classifier.cuh"
#include "kernels/evaluate_regressor.cuh"
#include "kernels/metric.cuh"
#include "kernels/quantile.cuh"
#include "kernels/split_labels.cuh"
#include "memory.cuh"

namespace ML {

bool is_dev_ptr(const void *p) {
  cudaPointerAttributes pointer_attr;
  cudaError_t err = cudaPointerGetAttributes(&pointer_attr, p);
  if (err == cudaSuccess) {
    return pointer_attr.devicePointer;
  } else {
    err = cudaGetLastError();
    return false;
  }
}

namespace DecisionTree {

template <class T, class L>
void null_tree_node_child_ptrs(TreeNode<T, L> &node) {
  node.left = nullptr;
  node.right = nullptr;
}

template <class T, class L>
void print(const TreeNode<T, L> &node, std::ostream &os) {
  if (node.left == nullptr && node.right == nullptr) {
    os << "(leaf, " << node.prediction << ", " << node.split_metric_val << ")";
  } else {
    os << "(" << node.question.column << ", " << node.question.value << ", "
       << node.split_metric_val << ")";
  }
  return;
}

template <class T, class L>
void print_node(const std::string &prefix, const TreeNode<T, L> *const node,
                bool isLeft) {
  if (node != nullptr) {
    std::cout << prefix;

    std::cout << (isLeft ? "├" : "└");

    // print the value of the node
    std::cout << node << std::endl;

    // enter the next tree level - left and right branch
    print_node(prefix + (isLeft ? "│   " : "    "), node->left, true);
    print_node(prefix + (isLeft ? "│   " : "    "), node->right, false);
  }
}

template <typename T, typename L>
std::ostream &operator<<(std::ostream &os, const TreeNode<T, L> *const node) {
  DecisionTree::print(*node, os);
  return os;
}

/**
 * @brief Print high-level tree information.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 */
template <typename T, typename L>
void DecisionTreeBase<T, L>::print_tree_summary() const {
  std::cout << " Decision Tree depth --> " << depth_counter
            << " and n_leaves --> " << leaf_counter << std::endl;
  std::cout << " Total temporary memory usage--> "
            << ((double)total_temp_mem / (1024 * 1024)) << "  MB" << std::endl;
  std::cout << " Shared memory used --> " << shmem_used << "  bytes "
            << std::endl;
  std::cout << " Tree Fitting - Overall time --> " << prepare_time + train_time
            << " seconds" << std::endl;
  std::cout << "   - preparing for fit time: " << prepare_time << " seconds"
            << std::endl;
  std::cout << "   - tree growing time: " << train_time << " seconds"
            << std::endl;
}

/**
 * @brief Print detailed tree information.
 * @tparam T: data type for input data (float or double).
 * @tparam L: data type for labels (int type for classification, T type for regression).
 * @param[in] root: pointer to tree's root node
 */
template <typename T, typename L>
void DecisionTreeBase<T, L>::print(const TreeNode<T, L> *root) const {
  DecisionTreeBase<T, L>::print_tree_summary();
  print_node<T, L>("", root, false);
}

template <typename T, typename L>
void DecisionTreeBase<T, L>::split_branch(T *data, MetricQuestion<T> &ques,
                                          const int n_sampled_rows,
                                          int &nrowsleft, int &nrowsright,
                                          unsigned int *rowids) {
  T *temp_data = tempmem[0]->temp_data->data();
  T *sampledcolumn = &temp_data[n_sampled_rows * ques.bootstrapped_column];
  make_split(sampledcolumn, ques, n_sampled_rows, nrowsleft, nrowsright, rowids,
             split_algo, tempmem[0]);
}

template <typename T, typename L>
void DecisionTreeBase<T, L>::plant(
  const cumlHandle_impl &handle, TreeNode<T, L> *&root, T *data,
  const int ncols, const int nrows, L *labels, unsigned int *rowids,
  const int n_sampled_rows, int unique_labels, int maxdepth, int max_leaf_nodes,
  const float colper, int n_bins, int split_algo_flag,
  int cfg_min_rows_per_node, bool cfg_bootstrap_features,
  CRITERION cfg_split_criterion, bool quantile_per_tree,
  std::shared_ptr<TemporaryMemory<T, L>> in_tempmem) {
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
  bootstrap_features = cfg_bootstrap_features;
  split_criterion = cfg_split_criterion;

  //Bootstrap features
  feature_selector.resize(dinfo.Ncols);
  if (bootstrap_features) {
    srand(n_bins);
    for (int i = 0; i < dinfo.Ncols; i++) {
      feature_selector.push_back(rand() % dinfo.Ncols);
    }
  } else {
    std::iota(feature_selector.begin(), feature_selector.end(), 0);
  }

  std::random_shuffle(feature_selector.begin(), feature_selector.end());
  feature_selector.resize((int)(colper * dinfo.Ncols));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, handle.getDevice()));
  max_shared_mem = prop.sharedMemPerBlock;

  if (split_algo == SPLIT_ALGO::HIST) {
    shmem_used += 2 * sizeof(T);
  }
  if (typeid(L) == typeid(int)) {  // Classification
    shmem_used += nbins * n_unique_labels * sizeof(int);
  } else {  // Regression
    shmem_used += nbins * sizeof(T) * 3;
    shmem_used += nbins * sizeof(int);
  }
  ASSERT(shmem_used <= max_shared_mem,
         "Shared memory per block limit %zd , requested %zd \n", max_shared_mem,
         shmem_used);

  for (int i = 0; i < MAXSTREAMS; i++) {
    if (in_tempmem != nullptr) {
      tempmem[i] = in_tempmem;
    } else {
      tempmem[i] = std::make_shared<TemporaryMemory<T, L>>(
        handle, n_sampled_rows, ncols, MAXSTREAMS, unique_labels, n_bins,
        split_algo);
      quantile_per_tree = true;
    }
    if (split_algo == SPLIT_ALGO::GLOBAL_QUANTILE && quantile_per_tree) {
      preprocess_quantile(data, rowids, n_sampled_rows, ncols, dinfo.NLocalrows,
                          n_bins, tempmem[i]);
    }
    CUDA_CHECK(cudaStreamSynchronize(
      tempmem[i]->stream));  // added to ensure accurate measurement
  }
  prepare_time = prepare_fit_timer.getElapsedSeconds();

  total_temp_mem = tempmem[0]->totalmem;
  total_temp_mem *= MAXSTREAMS;
  MetricInfo<T> split_info;
  MLCommon::TimerCPU timer;
  root = grow_tree(data, colper, labels, 0, rowids, n_sampled_rows, split_info);
  train_time = timer.getElapsedSeconds();
  if (in_tempmem == nullptr) {
    for (int i = 0; i < MAXSTREAMS; i++) {
      tempmem[i].reset();
    }
  }
}

template <typename T, typename L>
TreeNode<T, L> *DecisionTreeBase<T, L>::grow_tree(
  T *data, const float colper, L *labels, int depth, unsigned int *rowids,
  const int n_sampled_rows, MetricInfo<T> prev_split_info) {
  TreeNode<T, L> *node = new TreeNode<T, L>;
  null_tree_node_child_ptrs(*node);
  MetricQuestion<T> ques;
  Question<T> node_ques;
  float gain = 0.0;
  MetricInfo<T> split_info[3];  // basis, left, right. Populate this
  split_info[0] = prev_split_info;

  bool condition =
    ((depth != 0) &&
     (prev_split_info.best_metric ==
      0.0f));  // This node is a leaf, no need to search for best split
  condition =
    condition ||
    (n_sampled_rows <
     min_rows_per_node);  // Do not split a node with less than min_rows_per_node samples

  if (treedepth != -1) {
    condition = (condition || (depth == treedepth));
  }

  if (maxleaves != -1) {
    condition =
      (condition ||
       (leaf_counter >=
        maxleaves));  // FIXME not fully respecting maxleaves, but >= constraints it more than ==
  }

  if (!condition) {
    find_best_fruit_all(data, labels, colper, ques, gain, rowids,
                        n_sampled_rows, &split_info[0],
                        depth);  //ques and gain are output here
    condition = condition || (gain == 0.0f);
  }

  if (condition) {
    if (typeid(L) == typeid(int)) {  // classification
      node->prediction = get_class_hist(split_info[0].hist);
    } else {  // regression (typeid(L) == typeid(T))
      node->prediction = split_info[0].predict;
    }
    node->split_metric_val = split_info[0].best_metric;

    leaf_counter++;
    if (depth > depth_counter) {
      depth_counter = depth;
    }
  } else {
    int nrowsleft, nrowsright;
    split_branch(data, ques, n_sampled_rows, nrowsleft, nrowsright,
                 rowids);  // populates ques.value
    node_ques.column = ques.original_column;
    node_ques.value = ques.value;
    node->question = node_ques;
    node->left = grow_tree(data, colper, labels, depth + 1, &rowids[0],
                           nrowsleft, split_info[1]);
    node->right = grow_tree(data, colper, labels, depth + 1, &rowids[nrowsleft],
                            nrowsright, split_info[2]);
    node->split_metric_val = split_info[0].best_metric;
  }
  return node;
}

template <typename T, typename L>
void DecisionTreeBase<T, L>::init_depth_zero(
  const L *labels, std::vector<unsigned int> &colselector,
  const unsigned int *rowids, const int n_sampled_rows,
  const std::shared_ptr<TemporaryMemory<T, L>> tempmem) {
  CUDA_CHECK(cudaHostRegister(colselector.data(),
                              sizeof(unsigned int) * colselector.size(),
                              cudaHostRegisterDefault));
  // Copy sampled column IDs to device memory
  MLCommon::updateDevice(tempmem->d_colids->data(), colselector.data(),
                         colselector.size(), tempmem->stream);
  CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));

  L *labelptr = tempmem->sampledlabels->data();
  get_sampled_labels<L>(labels, labelptr, rowids, n_sampled_rows,
                        tempmem->stream);

  //Unregister
  CUDA_CHECK(cudaHostUnregister(colselector.data()));
}

template <typename T, typename L>
void DecisionTreeBase<T, L>::predict(const ML::cumlHandle &handle,
                                     const TreeMetaDataNode<T, L> *tree,
                                     const T *rows, const int n_rows,
                                     const int n_cols, L *predictions,
                                     bool verbose) const {
  ASSERT(!is_dev_ptr(rows) && !is_dev_ptr(predictions),
         "DT Error: Current impl. expects both input and predictions to be CPU "
         "pointers.\n");

  ASSERT(tree && tree->root, "Cannot predict w/ empty tree!");
  ASSERT((n_rows > 0), "Invalid n_rows %d", n_rows);
  ASSERT((n_cols > 0), "Invalid n_cols %d", n_cols);

  predict_all(tree, rows, n_rows, n_cols, predictions, verbose);
}

template <typename T, typename L>
void DecisionTreeBase<T, L>::predict_all(const TreeMetaDataNode<T, L> *tree,
                                         const T *rows, const int n_rows,
                                         const int n_cols, L *preds,
                                         bool verbose) const {
  for (int row_id = 0; row_id < n_rows; row_id++) {
    preds[row_id] = predict_one(&rows[row_id * n_cols], tree->root, verbose);
  }
}

template <typename T, typename L>
L DecisionTreeBase<T, L>::predict_one(const T *row,
                                      const TreeNode<T, L> *const node,
                                      bool verbose) const {
  Question<T> q = node->question;
  if (node->left && (row[q.column] <= q.value)) {
    if (verbose) {
      std::cout << "Classifying Left @ node w/ column " << q.column
                << " and value " << q.value << std::endl;
    }
    return predict_one(row, node->left, verbose);
  } else if (node->right && (row[q.column] > q.value)) {
    if (verbose) {
      std::cout << "Classifying Right @ node w/ column " << q.column
                << " and value " << q.value << std::endl;
    }
    return predict_one(row, node->right, verbose);
  } else {
    if (verbose) {
      std::cout << "Leaf node. Predicting " << node->prediction << std::endl;
    }
    return node->prediction;
  }
}

template <typename T, typename L>
void DecisionTreeBase<T, L>::set_metadata(TreeMetaDataNode<T, L> *&tree) {
  tree->depth_counter = depth_counter;
  tree->leaf_counter = leaf_counter;
  tree->train_time = train_time;
  tree->prepare_time = prepare_time;
}

template <typename T, typename L>
void DecisionTreeBase<T, L>::base_fit(
  const ML::cumlHandle &handle, T *data, const int ncols, const int nrows,
  L *labels, unsigned int *rowids, const int n_sampled_rows, int unique_labels,
  TreeNode<T, L> *&root, DecisionTreeParams &tree_params, bool is_classifier,
  std::shared_ptr<TemporaryMemory<T, L>> in_tempmem) {
  const char *CRITERION_NAME[] = {"GINI", "ENTROPY", "MSE", "MAE", "END"};
  CRITERION default_criterion =
    (is_classifier) ? CRITERION::GINI : CRITERION::MSE;
  CRITERION last_criterion =
    (is_classifier) ? CRITERION::ENTROPY : CRITERION::MAE;

  validity_check(tree_params);
  if (tree_params.n_bins > n_sampled_rows) {
    std::cout << "Warning! Calling with number of bins > number of rows! ";
    std::cout << "Resetting n_bins to " << n_sampled_rows << "." << std::endl;
    tree_params.n_bins = n_sampled_rows;
  }

  if (
    tree_params.split_criterion ==
    CRITERION::
      CRITERION_END) {  // Set default to GINI (classification) or MSE (regression)
    tree_params.split_criterion = default_criterion;
  }
  ASSERT((tree_params.split_criterion >= default_criterion) &&
           (tree_params.split_criterion <= last_criterion),
         "Unsupported criterion %s\n",
         CRITERION_NAME[tree_params.split_criterion]);

  plant(handle.getImpl(), root, data, ncols, nrows, labels, rowids,
        n_sampled_rows, unique_labels, tree_params.max_depth,
        tree_params.max_leaves, tree_params.max_features, tree_params.n_bins,
        tree_params.split_algo, tree_params.min_rows_per_node,
        tree_params.bootstrap_features, tree_params.split_criterion,

        tree_params.quantile_per_tree, in_tempmem);
}

template <typename T>
void DecisionTreeClassifier<T>::fit(
  const ML::cumlHandle &handle, T *data, const int ncols, const int nrows,
  int *labels, unsigned int *rowids, const int n_sampled_rows,
  const int unique_labels, TreeMetaDataNode<T, int> *&tree,
  DecisionTreeParams tree_params,
  std::shared_ptr<TemporaryMemory<T, int>> in_tempmem) {
  this->base_fit(handle, data, ncols, nrows, labels, rowids, n_sampled_rows,
                 unique_labels, tree->root, tree_params, true, in_tempmem);
  this->set_metadata(tree);
}

template <typename T>
void DecisionTreeClassifier<T>::find_best_fruit_all(
  T *data, int *labels, const float colper, MetricQuestion<T> &ques,
  float &gain, unsigned int *rowids, const int n_sampled_rows,
  MetricInfo<T> split_info[3], int depth) {
  std::vector<unsigned int> &colselector = this->feature_selector;

  // Optimize ginibefore; no need to compute except for root.
  if (depth == 0) {
    this->init_depth_zero(labels, colselector, rowids, n_sampled_rows,
                          this->tempmem[0]);
    int *labelptr = this->tempmem[0]->sampledlabels->data();
    if (this->split_criterion == CRITERION::GINI) {
      gini<T, GiniFunctor>(labelptr, n_sampled_rows, this->tempmem[0],
                           split_info[0], this->n_unique_labels);
    } else {
      gini<T, EntropyFunctor>(labelptr, n_sampled_rows, this->tempmem[0],
                              split_info[0], this->n_unique_labels);
    }
  }

  // Do not update bin count for the GLOBAL_QUANTILE split algorithm, as all potential split points were precomputed.
  int current_nbins = ((this->split_algo != SPLIT_ALGO::GLOBAL_QUANTILE) &&
                       (n_sampled_rows < this->nbins))
                        ? n_sampled_rows
                        : this->nbins;

  if (this->split_criterion == CRITERION::GINI) {
    best_split_all_cols_classifier<T, int, GiniFunctor>(
      data, rowids, labels, current_nbins, n_sampled_rows,
      this->n_unique_labels, this->dinfo.NLocalrows, colselector,
      this->tempmem[0], &split_info[0], ques, gain, this->split_algo,
      this->max_shared_mem);
  } else {
    best_split_all_cols_classifier<T, int, EntropyFunctor>(
      data, rowids, labels, current_nbins, n_sampled_rows,
      this->n_unique_labels, this->dinfo.NLocalrows, colselector,
      this->tempmem[0], &split_info[0], ques, gain, this->split_algo,
      this->max_shared_mem);
  }
}

template <typename T>
void DecisionTreeRegressor<T>::fit(
  const ML::cumlHandle &handle, T *data, const int ncols, const int nrows,
  T *labels, unsigned int *rowids, const int n_sampled_rows,
  TreeMetaDataNode<T, T> *&tree, DecisionTreeParams tree_params,
  std::shared_ptr<TemporaryMemory<T, T>> in_tempmem) {
  this->base_fit(handle, data, ncols, nrows, labels, rowids, n_sampled_rows, 1,
                 tree->root, tree_params, false, in_tempmem);
  this->set_metadata(tree);
}

template <typename T>
void DecisionTreeRegressor<T>::find_best_fruit_all(
  T *data, T *labels, const float colper, MetricQuestion<T> &ques, float &gain,
  unsigned int *rowids, const int n_sampled_rows, MetricInfo<T> split_info[3],
  int depth) {
  std::vector<unsigned int> &colselector = this->feature_selector;

  if (depth == 0) {
    this->init_depth_zero(labels, colselector, rowids, n_sampled_rows,
                          this->tempmem[0]);
    T *labelptr = this->tempmem[0]->sampledlabels->data();
    if (this->split_criterion == CRITERION::MSE) {
      mse<T, SquareFunctor>(labelptr, n_sampled_rows, this->tempmem[0],
                            split_info[0]);
    } else {
      mse<T, AbsFunctor>(labelptr, n_sampled_rows, this->tempmem[0],
                         split_info[0]);
    }
  }

  // Do not update bin count for the GLOBAL_QUANTILE split algorithm, as all potential split points were precomputed.
  int current_nbins = ((this->split_algo != SPLIT_ALGO::GLOBAL_QUANTILE) &&
                       (n_sampled_rows < this->nbins))
                        ? n_sampled_rows
                        : this->nbins;

  if (this->split_criterion == CRITERION::MSE) {
    best_split_all_cols_regressor<T, SquareFunctor>(
      data, rowids, labels, current_nbins, n_sampled_rows,
      this->dinfo.NLocalrows, colselector, this->tempmem[0], split_info, ques,
      gain, this->split_algo, this->max_shared_mem);
  } else {
    best_split_all_cols_regressor<T, AbsFunctor>(
      data, rowids, labels, current_nbins, n_sampled_rows,
      this->dinfo.NLocalrows, colselector, this->tempmem[0], split_info, ques,
      gain, this->split_algo, this->max_shared_mem);
  }
}

//Class specializations
template class DecisionTreeBase<float, int>;
template class DecisionTreeBase<float, float>;
template class DecisionTreeBase<double, int>;
template class DecisionTreeBase<double, double>;

template class DecisionTreeClassifier<float>;
template class DecisionTreeClassifier<double>;

template class DecisionTreeRegressor<float>;
template class DecisionTreeRegressor<double>;

}  //End namespace DecisionTree

}  //End namespace ML
