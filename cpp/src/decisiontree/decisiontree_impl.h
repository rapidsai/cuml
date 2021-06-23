/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <common/Timer.h>
#include <cuml/tree/algo_helper.h>
#include <cuml/tree/flatnode.h>
#include <treelite/c_api.h>
#include <treelite/tree.h>
#include <algorithm>
#include <climits>
#include <cuml/tree/decisiontree.hpp>
#include <map>
#include <numeric>
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/host/allocator.hpp>
#include <vector>

/** check for treelite runtime API errors and assert accordingly */
#define TREELITE_CHECK(call)                                            \
  do {                                                                  \
    int status = call;                                                  \
    ASSERT(status >= 0, "TREELITE FAIL: call='%s'. Reason:%s\n", #call, \
           TreeliteGetLastError());                                     \
  } while (0)

namespace ML {

namespace tl = treelite;

bool is_dev_ptr(const void *p);

namespace DT {

template <class T, class L>
void print(const SparseTreeNode<T, L> &node, std::ostream &os);

template <class T, class L>
std::string get_node_text(const std::string &prefix,
                          const std::vector<SparseTreeNode<T, L>> &sparsetree,
                          int idx, bool isLeft);

template <class T, class L>
std::string get_node_json(const std::string &prefix,
                          const std::vector<SparseTreeNode<T, L>> &sparsetree,
                          int idx);

template <class T, class L>
tl::Tree<T, T> build_treelite_tree(
  const DT::TreeMetaDataNode<T, L> &rf_tree, unsigned int num_class,
  std::vector<Node_ID_info<T, L>> &working_queue_1,
  std::vector<Node_ID_info<T, L>> &working_queue_2);

struct DataInfo {
  unsigned int NLocalrows;
  unsigned int NGlobalrows;
  unsigned int Ncols;
};

template <class T, class L>
class DecisionTree {
 protected:
  DataInfo dinfo;
  int depth_counter = 0;
  int leaf_counter = 0;
  int n_unique_labels = -1;  // number of unique labels in dataset
  double prepare_time = 0;
  double train_time = 0;
  MLCommon::TimerCPU prepare_fit_timer;
  DecisionTreeParams tree_params;

 public:
  // fit function
  void fit(const raft::handle_t &handle, const T *data, const int ncols,
           const int nrows, const L *labels, unsigned int *rowids,
           const int n_sampled_rows, int unique_labels, bool is_classifier,
           TreeMetaDataNode<T, L> *&tree, DecisionTreeParams tree_parameters,
           uint64_t seed, T *d_global_quantiles);

  // Printing utility for high level tree info.
  void print_tree_summary() const;

  // Printing utility for debug and looking at nodes and leaves.
  void print(const std::vector<SparseTreeNode<T, L>> &sparsetree) const;

  // Predict labels for n_rows rows, with n_cols features each, for a given tree. rows in row-major format.
  void predict(const raft::handle_t &handle, const TreeMetaDataNode<T, L> *tree,
               const T *rows, const int n_rows, const int n_cols,
               L *predictions, int verbosity = -1) const;
  void predict_all(const TreeMetaDataNode<T, L> *tree, const T *rows,
                   const int n_rows, const int n_cols, L *preds) const;
  L predict_one(const T *row,
                const std::vector<SparseTreeNode<T, L>> sparsetree,
                int idx) const;

  void set_metadata(TreeMetaDataNode<T, L> *&tree);

};  // End DecisionTree Class

}  //End namespace DT

}  //End namespace ML
