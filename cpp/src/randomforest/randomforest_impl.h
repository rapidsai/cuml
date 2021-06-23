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
#include <decisiontree/decisiontree_impl.h>
#include <cuml/ensemble/randomforest.hpp>
#include <map>
#include <raft/mr/device/allocator.hpp>

namespace ML {

template <class T, class L>
class rf {
 protected:
  RF_params rf_params;
  int rf_type;
  virtual const DT::DecisionTree<T, L>* get_trees_ptr() const = 0;
  virtual ~rf() = default;
  void prepare_fit_per_tree(
    int tree_id, int n_rows, int n_sampled_rows, unsigned int* selected_rows,
    int num_sms, const cudaStream_t stream,
    const std::shared_ptr<raft::mr::device::allocator> device_allocator);

  void error_checking(const T* input, L* predictions, int n_rows, int n_cols,
                      bool is_predict) const;

 public:
  rf(RF_params cfg_rf_params, int cfg_rf_type = RF_type::CLASSIFICATION);

  int get_ntrees();
};

template <class T>
class rfClassifier : public rf<T, int> {
 private:
  DT::DecisionTree<T, int>* trees = nullptr;
  const DT::DecisionTree<T, int>* get_trees_ptr() const;

 public:
  rfClassifier(RF_params cfg_rf_params);
  ~rfClassifier();

  void fit(const raft::handle_t& user_handle, const T* input, int n_rows,
           int n_cols, int* labels, int n_unique_labels,
           RandomForestMetaData<T, int>*& forest);
  void predict(const raft::handle_t& user_handle, const T* input, int n_rows,
               int n_cols, int* predictions,
               const RandomForestMetaData<T, int>* forest, int verbosity) const;
  void predictGetAll(const raft::handle_t& user_handle, const T* input,
                     int n_rows, int n_cols, int* predictions,
                     const RandomForestMetaData<T, int>* forest, int verbosity);
  static RF_metrics score(const raft::handle_t& user_handle,
                          const int* ref_labels, int n_rows,
                          const int* predictions, int verbosity);
};

template <class T>
class rfRegressor : public rf<T, T> {
 private:
  DT::DecisionTree<T, T>* trees = nullptr;
  const DT::DecisionTree<T, T>* get_trees_ptr() const;

 public:
  rfRegressor(RF_params cfg_rf_params);
  ~rfRegressor();

  void fit(const raft::handle_t& user_handle, const T* input, int n_rows,
           int n_cols, T* labels, RandomForestMetaData<T, T>*& forest);
  void predict(const raft::handle_t& user_handle, const T* input, int n_rows,
               int n_cols, T* predictions,
               const RandomForestMetaData<T, T>* forest, int verbosity) const;
  static RF_metrics score(const raft::handle_t& user_handle,
                          const T* ref_labels, int n_rows, const T* predictions,
                          int verbosity);
};
}  //End namespace ML
