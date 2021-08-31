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

#include <raft/handle.hpp>

#include "builder_base.cuh"
#include "metrics.cuh"

#include <common/nvtx.hpp>

namespace ML {
namespace DT {

template <typename ObjectiveT,
          typename DataT  = typename ObjectiveT::DataT,
          typename LabelT = typename ObjectiveT::LabelT,
          typename IdxT   = typename ObjectiveT::IdxT>
void convertToSparse(const std::vector<Node<DataT, LabelT, IdxT>> h_nodes,
                     std::vector<SparseTreeNode<DataT, LabelT>>& sparsetree)
{
  auto len = sparsetree.size();
  sparsetree.resize(len + h_nodes.size());
  for (std::size_t i = 0; i < h_nodes.size(); ++i) {
    const auto& hnode   = h_nodes[i].info;
    sparsetree[i + len] = hnode;
    if (hnode.left_child_id != -1) sparsetree[i + len].left_child_id += len;
  }
}

///@todo: support col subsampling per node
template <typename ObjectiveT,
          typename DataT  = typename ObjectiveT::DataT,
          typename LabelT = typename ObjectiveT::LabelT,
          typename IdxT   = typename ObjectiveT::IdxT>
void grow_tree(const raft::handle_t& handle,
               const DataT* data,
               IdxT treeid,
               uint64_t seed,
               IdxT ncols,
               IdxT nrows,
               const LabelT* labels,
               const DataT* quantiles,
               IdxT* rowids,
               int n_sampled_rows,
               int unique_labels,
               const DecisionTreeParams& params,
               std::vector<SparseTreeNode<DataT, LabelT>>& sparsetree,
               IdxT& num_leaves,
               IdxT& depth)
{
  ML::PUSH_RANGE("DT::grow_tree in batched-levelalgo @builder.cuh");

  Builder<ObjectiveT> builder(handle,
                              treeid,
                              seed,
                              params,
                              data,
                              labels,
                              nrows,
                              ncols,
                              n_sampled_rows,
                              rowids,
                              unique_labels,
                              quantiles);
  auto h_nodes = builder.train(num_leaves, depth, handle.get_stream());
  CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
  convertToSparse<ObjectiveT>(h_nodes, sparsetree);
  ML::POP_RANGE();
}

/**
 * @defgroup GrowTree Main entry point function for batched-level algo to build
 *                    a decision tree
 *
 * @tparam DataT  data type
 * @tparam LabelT label type
 * @tparam IdxT   index type
 *
 * @param[in]  handle         raft handle
 * @param[in]  data           input dataset [on device] [col-major]
 *                            [dim = nrows x ncols]
 * @param[in]  ncols          number of features in the dataset
 * @param[in]  nrows          number of rows in the dataset
 * @param[in]  labels         labels for the input [on device] [len = nrows]
 * @param[in]  quantiles      histograms/quantiles of the input dataset
 *                            [on device] [col-major]
 *                            [dim = params.n_bins x ncols]
 * @param[in]  rowids         sampled rows [on device] [len = n_sampled_rows]
 * @param[in]  colids         sampled cols [on device]
 *                            [len = params.max_features * ncols]
 * @param[in]  n_sampled_rows number of sub-sampled rows
 * @param[in]  unique_labels  number of classes (meaningful only for
 *                            classification)
 * @param[in]  params         decisiontree learning params
 * @param[in]  stream         cuda stream
 * @param[out] sparsetree     output learned tree
 * @param[out] num_leaves     number of leaves created during tree build
 * @param[out] depth          max depth of the built tree
 * @{
 */
template <typename DataT, typename LabelT, typename IdxT>
void grow_tree(const raft::handle_t& handle,
               const DataT* data,
               IdxT treeid,
               uint64_t seed,
               IdxT ncols,
               IdxT nrows,
               const LabelT* labels,
               const DataT* quantiles,
               IdxT* rowids,
               int n_sampled_rows,
               int unique_labels,
               const DecisionTreeParams& params,
               std::vector<SparseTreeNode<DataT, LabelT>>& sparsetree,
               IdxT& num_leaves,
               IdxT& depth)
{
  // Dispatch objective
  if (params.split_criterion == CRITERION::GINI) {
    grow_tree<GiniObjectiveFunction<DataT, LabelT, IdxT>>(handle,
                                                          data,
                                                          treeid,
                                                          seed,
                                                          ncols,
                                                          nrows,
                                                          labels,
                                                          quantiles,
                                                          rowids,
                                                          n_sampled_rows,
                                                          unique_labels,
                                                          params,
                                                          sparsetree,
                                                          num_leaves,
                                                          depth);
  } else if (params.split_criterion == CRITERION::ENTROPY) {
    grow_tree<EntropyObjectiveFunction<DataT, LabelT, IdxT>>(handle,
                                                             data,
                                                             treeid,
                                                             seed,
                                                             ncols,
                                                             nrows,
                                                             labels,
                                                             quantiles,
                                                             rowids,
                                                             n_sampled_rows,
                                                             unique_labels,
                                                             params,
                                                             sparsetree,
                                                             num_leaves,
                                                             depth);
  } else if (params.split_criterion == CRITERION::POISSON) {
    grow_tree<PoissonObjectiveFunction<DataT, LabelT, IdxT>>(handle,
                                                             data,
                                                             treeid,
                                                             seed,
                                                             ncols,
                                                             nrows,
                                                             labels,
                                                             quantiles,
                                                             rowids,
                                                             n_sampled_rows,
                                                             unique_labels,
                                                             params,
                                                             sparsetree,
                                                             num_leaves,
                                                             depth);
  } else if (params.split_criterion == CRITERION::MSE) {
    grow_tree<MSEObjectiveFunction<DataT, LabelT, IdxT>>(handle,
                                                         data,
                                                         treeid,
                                                         seed,
                                                         ncols,
                                                         nrows,
                                                         labels,
                                                         quantiles,
                                                         rowids,
                                                         n_sampled_rows,
                                                         unique_labels,
                                                         params,
                                                         sparsetree,
                                                         num_leaves,
                                                         depth);
  } else {
    ASSERT(false, "Unknown split criterion.");
  }
}
}  // namespace DT
}  // namespace ML
