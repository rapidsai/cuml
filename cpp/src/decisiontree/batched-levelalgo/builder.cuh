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

#include <cuml/common/device_buffer.hpp>
#include <cuml/common/host_buffer.hpp>

#include "builder_base.cuh"

#include <common/nvtx.hpp>

namespace ML {
namespace DecisionTree {

template <typename Traits, typename DataT = typename Traits::DataT,
          typename LabelT = typename Traits::LabelT,
          typename IdxT = typename Traits::IdxT>
void convertToSparse(const Builder<Traits>& b,
                     const Node<DataT, LabelT, IdxT>* h_nodes,
                     std::vector<SparseTreeNode<DataT, LabelT>>& sparsetree) {
  auto len = sparsetree.size();
  sparsetree.resize(len + b.h_total_nodes);
  for (IdxT i = 0; i < b.h_total_nodes; ++i) {
    const auto& hnode = h_nodes[i].info;
    sparsetree[i + len] = hnode;
    sparsetree[i + len].instance_count = h_nodes[i].count;
    if (hnode.left_child_id != -1) sparsetree[i + len].left_child_id += len;
  }
}

///@todo: support col subsampling per node
template <typename Traits, typename DataT = typename Traits::DataT,
          typename LabelT = typename Traits::LabelT,
          typename IdxT = typename Traits::IdxT>
void grow_tree(std::shared_ptr<MLCommon::deviceAllocator> d_allocator,
               std::shared_ptr<MLCommon::hostAllocator> h_allocator,
               const DataT* data, IdxT treeid, uint64_t seed, IdxT ncols,
               IdxT nrows, const LabelT* labels, const DataT* quantiles,
               IdxT* rowids, int n_sampled_rows, int unique_labels,
               const DecisionTreeParams& params, cudaStream_t stream,
               std::vector<SparseTreeNode<DataT, LabelT>>& sparsetree,
               IdxT& num_leaves, IdxT& depth) {
  ML::PUSH_RANGE("DecisionTree::grow_tree in batched-levelalgo @builder.cuh");
  Builder<Traits> builder;
  size_t d_wsize, h_wsize;
  builder.workspaceSize(d_wsize, h_wsize, treeid, seed, params, data, labels,
                        nrows, ncols, n_sampled_rows,
                        IdxT(params.max_features * ncols), rowids,
                        unique_labels, quantiles);
  MLCommon::device_buffer<char> d_buff(d_allocator, stream, d_wsize);
  MLCommon::host_buffer<char> h_buff(h_allocator, stream, h_wsize);

  std::vector<Node<DataT, LabelT, IdxT>> h_nodes;
  h_nodes.reserve(builder.maxNodes);
  builder.assignWorkspace(d_buff.data(), h_buff.data());
  builder.train(h_nodes, num_leaves, depth, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  d_buff.release(stream);
  h_buff.release(stream);
  convertToSparse<Traits>(builder, h_nodes.data(), sparsetree);
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
 * @param[in]  d_allocator    device allocator
 * @param[in]  h_allocator    host allocator
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
void grow_tree(std::shared_ptr<MLCommon::deviceAllocator> d_allocator,
               std::shared_ptr<MLCommon::hostAllocator> h_allocator,
               const DataT* data, IdxT treeid, uint64_t seed, IdxT ncols,
               IdxT nrows, const LabelT* labels, const DataT* quantiles,
               IdxT* rowids, int n_sampled_rows, int unique_labels,
               const DecisionTreeParams& params, cudaStream_t stream,
               std::vector<SparseTreeNode<DataT, LabelT>>& sparsetree,
               IdxT& num_leaves, IdxT& depth) {
  typedef ClsTraits<DataT, LabelT, IdxT> Traits;
  grow_tree<Traits>(d_allocator, h_allocator, data, treeid, seed, ncols, nrows,
                    labels, quantiles, rowids, n_sampled_rows, unique_labels,
                    params, stream, sparsetree, num_leaves, depth);
}
template <typename DataT, typename IdxT>
void grow_tree(std::shared_ptr<MLCommon::deviceAllocator> d_allocator,
               std::shared_ptr<MLCommon::hostAllocator> h_allocator,
               const DataT* data, IdxT treeid, uint64_t seed, IdxT ncols,
               IdxT nrows, const DataT* labels, const DataT* quantiles,
               IdxT* rowids, int n_sampled_rows, int unique_labels,
               const DecisionTreeParams& params, cudaStream_t stream,
               std::vector<SparseTreeNode<DataT, DataT>>& sparsetree,
               IdxT& num_leaves, IdxT& depth) {
  typedef RegTraits<DataT, IdxT> Traits;
  grow_tree<Traits>(d_allocator, h_allocator, data, treeid, seed, ncols, nrows,
                    labels, quantiles, rowids, n_sampled_rows, unique_labels,
                    params, stream, sparsetree, num_leaves, depth);
}
/** @} */

}  // namespace DecisionTree
}  // namespace ML
