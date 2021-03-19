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
#include <vector>

/** check for treelite runtime API errors and assert accordingly */
#define TREELITE_CHECK(call)                                            \
  do {                                                                  \
    int status = call;                                                  \
    ASSERT(status >= 0, "TREELITE FAIL: call='%s'. Reason:%s\n", #call, \
           TreeliteGetLastError());                                     \
  } while (0)

template <class T, class L>
struct TemporaryMemory;

namespace ML {

namespace tl = treelite;

bool is_dev_ptr(const void *p);

namespace DecisionTree {

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
  const DecisionTree::TreeMetaDataNode<T, L> &rf_tree, unsigned int num_class,
  std::vector<Node_ID_info<T, L>> &working_queue_1,
  std::vector<Node_ID_info<T, L>> &working_queue_2);

struct DataInfo {
  unsigned int NLocalrows;
  unsigned int NGlobalrows;
  unsigned int Ncols;
};

template <class T, class L>
class DecisionTreeBase {
 protected:
  DataInfo dinfo;
  int depth_counter = 0;
  int leaf_counter = 0;
  std::shared_ptr<TemporaryMemory<T, L>> tempmem;
  size_t total_temp_mem;
  const int MAXSTREAMS = 1;
  size_t max_shared_mem;
  size_t shmem_used = 0;
  int n_unique_labels = -1;  // number of unique labels in dataset
  double prepare_time = 0;
  double train_time = 0;
  MLCommon::TimerCPU prepare_fit_timer;
  DecisionTreeParams tree_params;

  void plant(std::vector<SparseTreeNode<T, L>> &sparsetree, const T *data,
             const int ncols, const int nrows, const L *labels,
             unsigned int *rowids, const int n_sampled_rows, int unique_labels,
             const int treeid, uint64_t seed);

  virtual void grow_deep_tree(
    const T *data, const L *labels, unsigned int *rowids,
    const int n_sampled_rows, const int ncols, const float colper,
    const int nrows, std::vector<SparseTreeNode<T, L>> &sparsetree,
    const int treeid, std::shared_ptr<TemporaryMemory<T, L>> tempmem) = 0;

  void base_fit(
    const std::shared_ptr<MLCommon::deviceAllocator> device_allocator_in,
    const std::shared_ptr<MLCommon::hostAllocator> host_allocator_in,
    const cudaStream_t stream_in, const T *data, const int ncols,
    const int nrows, const L *labels, unsigned int *rowids,
    const int n_sampled_rows, int unique_labels,
    std::vector<SparseTreeNode<T, L>> &sparsetree, const int treeid,
    uint64_t seed, bool is_classifier, T *d_global_quantiles,
    std::shared_ptr<TemporaryMemory<T, L>> in_tempmem);

 public:
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

};  // End DecisionTreeBase Class

template <class T>
class DecisionTreeClassifier : public DecisionTreeBase<T, int> {
 public:
  // Expects column major T dataset, integer labels
  // data, labels are both device ptr.
  // Assumption: labels are all mapped to contiguous numbers starting from 0 during preprocessing. Needed for gini hist impl.
  void fit(const raft::handle_t &handle, const T *data, const int ncols,
           const int nrows, const int *labels, unsigned int *rowids,
           const int n_sampled_rows, const int unique_labels,
           TreeMetaDataNode<T, int> *&tree, DecisionTreeParams tree_parameters,
           uint64_t seed, T *d_quantiles,
           std::shared_ptr<TemporaryMemory<T, int>> in_tempmem = nullptr);

  //This fit fucntion does not take handle , used by RF
  void fit(const std::shared_ptr<MLCommon::deviceAllocator> device_allocator_in,
           const std::shared_ptr<MLCommon::hostAllocator> host_allocator_in,
           const cudaStream_t stream_in, const T *data, const int ncols,
           const int nrows, const int *labels, unsigned int *rowids,
           const int n_sampled_rows, const int unique_labels,
           TreeMetaDataNode<T, int> *&tree, DecisionTreeParams tree_parameters,
           uint64_t seed, T *d_quantiles,
           std::shared_ptr<TemporaryMemory<T, int>> in_tempmem);

 private:
  void grow_deep_tree(const T *data, const int *labels, unsigned int *rowids,
                      const int n_sampled_rows, const int ncols,
                      const float colper, const int nrows,
                      std::vector<SparseTreeNode<T, int>> &sparsetree,
                      const int treeid,
                      std::shared_ptr<TemporaryMemory<T, int>> tempmem);

};  // End DecisionTreeClassifier Class

template <class T>
class DecisionTreeRegressor : public DecisionTreeBase<T, T> {
 public:
  void fit(const raft::handle_t &handle, const T *data, const int ncols,
           const int nrows, const T *labels, unsigned int *rowids,
           const int n_sampled_rows, TreeMetaDataNode<T, T> *&tree,
           DecisionTreeParams tree_parameters, uint64_t seed, T *d_quantiles,
           std::shared_ptr<TemporaryMemory<T, T>> in_tempmem = nullptr);

  //This fit function does not take handle. Used by RF
  void fit(const std::shared_ptr<MLCommon::deviceAllocator> device_allocator_in,
           const std::shared_ptr<MLCommon::hostAllocator> host_allocator_in,
           const cudaStream_t stream_in, const T *data, const int ncols,
           const int nrows, const T *labels, unsigned int *rowids,
           const int n_sampled_rows, TreeMetaDataNode<T, T> *&tree,
           DecisionTreeParams tree_parameters, uint64_t seed, T *d_quantiles,
           std::shared_ptr<TemporaryMemory<T, T>> in_tempmem);

 private:
  void grow_deep_tree(const T *data, const T *labels, unsigned int *rowids,
                      const int n_sampled_rows, const int ncols,
                      const float colper, const int nrows,
                      std::vector<SparseTreeNode<T, T>> &sparsetree,
                      const int treeid,
                      std::shared_ptr<TemporaryMemory<T, T>> tempmem);

};  // End DecisionTreeRegressor Class

}  //End namespace DecisionTree

}  //End namespace ML
