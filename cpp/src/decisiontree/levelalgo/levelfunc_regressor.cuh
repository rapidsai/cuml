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
#pragma once
#include <iostream>
#include <numeric>
#include "flatnode.h"
#include "../decisiontree.hpp"
#include "../kernels/metric.cuh"
#include "../kernels/metric_def.h"
#include "levelhelper_regressor.cuh"
#include "common_helper.cuh"

template <typename T>
ML::DecisionTree::TreeNode<T, T>* grow_deep_tree_regression(
  const ML::cumlHandle_impl& handle, T* data, T* labels, unsigned int* rowids,
  const std::vector<unsigned int>& feature_selector, const int n_sampled_rows,
  const int nrows, const int nbins, int maxdepth, const int maxleaves,
  const int min_rows_per_node, const ML::CRITERION split_cr, int& depth_cnt,
  int& leaf_cnt, std::shared_ptr<TemporaryMemory<T, T>> tempmem) {
  const int ncols = feature_selector.size();
  MLCommon::updateDevice(tempmem->d_colids->data(), feature_selector.data(),
                         feature_selector.size(), tempmem->stream);
  
  unsigned int* flagsptr = tempmem->d_flags->data();
  unsigned int* sample_cnt = tempmem->d_sample_cnt->data();
  setup_sampling(flagsptr, sample_cnt, rowids, nrows, n_sampled_rows,
                 tempmem->stream);
  
  return (new ML::DecisionTree::TreeNode<T, T>());
}
