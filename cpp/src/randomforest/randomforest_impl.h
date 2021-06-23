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
class RandomForest {
 private:
  DT::DecisionTree<T, L>* trees = nullptr;
  const DT::DecisionTree<T, L>* get_trees_ptr() const;

 protected:
  RF_params rf_params;
  int rf_type;
  void prepare_fit_per_tree(
    int tree_id, int n_rows, int n_sampled_rows, unsigned int* selected_rows,
    int num_sms, const cudaStream_t stream,
    const std::shared_ptr<raft::mr::device::allocator> device_allocator);

  void error_checking(const T* input, L* predictions, int n_rows, int n_cols,
                      bool is_predict) const;

 public:
  RandomForest(RF_params cfg_rf_params,
               int cfg_rf_type = RF_type::CLASSIFICATION);
  ~RandomForest();

  int get_ntrees();

  void fit(const raft::handle_t& user_handle, const T* input, int n_rows,
           int n_cols, L* labels, int n_unique_labels,
           RandomForestMetaData<T, L>*& forest);
  void predict(const raft::handle_t& user_handle, const T* input, int n_rows,
               int n_cols, L* predictions,
               const RandomForestMetaData<T, L>* forest, int verbosity) const;
  void predictGetAll(const raft::handle_t& user_handle, const T* input,
                     int n_rows, int n_cols, L* predictions,
                     const RandomForestMetaData<T, L>* forest, int verbosity);
  static RF_metrics score(const raft::handle_t& user_handle,
                          const L* ref_labels, int n_rows, const L* predictions,
                          int verbosity, int rf_type = RF_type::CLASSIFICATION);
};

}  //End namespace ML
