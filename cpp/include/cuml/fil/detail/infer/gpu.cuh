/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
#include <cuml/fil/constants.hpp>
#include <cuml/fil/detail/forest.hpp>
#include <cuml/fil/detail/gpu_introspection.hpp>
#include <cuml/fil/detail/index_type.hpp>
#include <cuml/fil/detail/infer_kernel/gpu.cuh>
#include <cuml/fil/detail/postprocessor.hpp>
#include <cuml/fil/detail/raft_proto/buffer.hpp>
#include <cuml/fil/detail/raft_proto/ceildiv.hpp>
#include <cuml/fil/detail/raft_proto/cuda_stream.hpp>
#include <cuml/fil/detail/raft_proto/device_id.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>
#include <cuml/fil/detail/raft_proto/gpu_support.hpp>
#include <cuml/fil/detail/raft_proto/padding.hpp>
#include <cuml/fil/detail/specializations/infer_macros.hpp>
#include <cuml/fil/exceptions.hpp>
#include <cuml/fil/infer_kind.hpp>

#include <cstddef>
#include <optional>
#include <type_traits>

namespace ML {
namespace fil {
namespace detail {
namespace inference {

inline auto compute_output_size(index_type row_output_size,
                                index_type threads_per_block,
                                index_type rows_per_block_iteration,
                                infer_kind infer_type = infer_kind::default_kind)
{
  auto result = row_output_size * rows_per_block_iteration;
  if (infer_type == infer_kind::default_kind) {
    result *= raft_proto::ceildiv(threads_per_block, rows_per_block_iteration);
  }
  return result;
}

/* A wrapper around the underlying inference kernels to support dispatching to
 * the right kernel
 *
 * This specialization is used for GPU inference. It performs any necessary
 * computation necessary prior to kernel launch and then launches the correct
 * inference kernel.
 *
 * @tparam D The type of device (CPU/GPU) on which to perform inference.
 * @tparam has_categorical_nodes Whether or not any node in the model has
 * categorical splits.
 * @tparam vector_output_t If non-nullptr_t, the type of vector leaf output
 * @tparam categorical_data_t If non-nullptr_t, the type of non-local
 * categorical data storage
 *
 * @param forest The forest to be used for inference.
 * @param postproc The postprocessor object to be used for postprocessing raw
 * output from the forest.
 * @param row_count The number of rows in the input
 * @param col_count The number of columns per row in the input
 * @param output_count The number of output elements per row
 * @param vector_output If non-nullptr, a pointer to storage for vector leaf
 * outputs
 * @param categorical_data If non-nullptr, a pointer to non-local storage for
 * data on categorical splits.
 * @param infer_type Type of inference to perform. Defaults to summing the outputs of all trees
 * and produce an output per row. If set to "per_tree", we will instead output all outputs of
 * individual trees. If set to "leaf_id", we will output the integer ID of the leaf node
 * for each tree.
 * @param specified_chunk_size If non-nullopt, the mini-batch size used for
 * processing rows in a batch. For GPU inference, this determines the number of
 * rows that are processed per iteration of inference in a single block. It
 * is difficult to predict the optimal value for this parameter, but tuning it
 * can result in a substantial improvement in performance. The optimal
 * value depends on hardware, model, and batch size. Valid values are any power
 * of 2 from 1 to 32.
 */
template <raft_proto::device_type D,
          bool has_categorical_nodes,
          typename forest_t,
          typename vector_output_t    = std::nullptr_t,
          typename categorical_data_t = std::nullptr_t>
std::enable_if_t<D == raft_proto::device_type::gpu, void> infer(
  forest_t const& forest,
  postprocessor<typename forest_t::io_type> const& postproc,
  typename forest_t::io_type* output,
  typename forest_t::io_type* input,
  index_type row_count,
  index_type col_count,
  index_type output_count,
  vector_output_t vector_output                  = nullptr,
  categorical_data_t categorical_data            = nullptr,
  infer_kind infer_type                          = infer_kind::default_kind,
  std::optional<index_type> specified_chunk_size = std::nullopt,
  raft_proto::device_id<D> device                = raft_proto::device_id<D>{},
  raft_proto::cuda_stream stream                 = raft_proto::cuda_stream{})
{
  using output_t = typename forest_t::template raw_output_type<vector_output_t>;

  auto sm_count                       = get_sm_count(device);
  auto const max_shared_mem_per_block = get_max_shared_mem_per_block(device);
  auto const max_shared_mem_per_sm    = get_max_shared_mem_per_sm(device);
  auto const max_overall_shared_mem   = std::min(max_shared_mem_per_block, max_shared_mem_per_sm);

  auto row_size_bytes  = index_type(index_type(sizeof(typename forest_t::io_type) * col_count));
  auto row_output_size = output_count;
  auto row_output_size_bytes = index_type(sizeof(typename forest_t::io_type) * row_output_size);

  // First determine the number of threads per block. This is the indicated
  // preferred value unless we cannot handle at least 1 row per block iteration
  // with available shared memory, in which case we must reduce the threads per
  // block.
  auto threads_per_block =
    min(MAX_THREADS_PER_BLOCK,
        raft_proto::downpadded_size(
          (max_shared_mem_per_block - row_size_bytes) / row_output_size_bytes, WARP_SIZE));

  // If we cannot do at least a warp per block when storing input rows in
  // shared mem, recalculate our threads per block without input storage
  if (threads_per_block < WARP_SIZE) {
    threads_per_block =
      min(MAX_THREADS_PER_BLOCK,
          raft_proto::downpadded_size(max_shared_mem_per_block / row_output_size_bytes, WARP_SIZE));
    if (threads_per_block >= WARP_SIZE) {
      row_size_bytes = index_type{};  // Do not store input rows in shared mem
    }
  }

  // If we cannot do at least a warp per block when storing output in
  // shared mem, recalculate our threads per block with ONLY input storage
  if (threads_per_block < WARP_SIZE) {
    threads_per_block =
      min(MAX_THREADS_PER_BLOCK,
          raft_proto::downpadded_size(max_shared_mem_per_block / row_size_bytes, WARP_SIZE));
  }

  // If we still cannot use at least a warp per block, give up on using
  // shared memory and just maximize occupancy
  if (threads_per_block < WARP_SIZE) { threads_per_block = MAX_THREADS_PER_BLOCK; }

  auto const max_resident_blocks = sm_count * (get_max_threads_per_sm(device) / threads_per_block);

  // Compute shared memory usage based on minimum or specified
  // rows_per_block_iteration
  auto rows_per_block_iteration          = specified_chunk_size.value_or(index_type{1});
  auto constexpr const output_item_bytes = index_type(sizeof(output_t));
  auto output_workspace_size =
    compute_output_size(row_output_size, threads_per_block, rows_per_block_iteration, infer_type);
  auto output_workspace_size_bytes = output_item_bytes * output_workspace_size;
  auto global_workspace            = raft_proto::buffer<output_t>{};

  if (output_workspace_size_bytes > max_shared_mem_per_block) {
    output_workspace_size_bytes = 0;
    row_output_size             = 0;
  }
  auto shared_mem_per_block =
    min(rows_per_block_iteration * row_size_bytes + output_workspace_size_bytes,
        max_overall_shared_mem);

  auto resident_blocks_per_sm =
    min(raft_proto::ceildiv(max_shared_mem_per_sm, shared_mem_per_block), max_resident_blocks);

  // If caller has not specified the number of rows per block iteration, apply
  // the following heuristic to identify an approximately optimal value
  if (!specified_chunk_size.has_value() && resident_blocks_per_sm >= MIN_BLOCKS_PER_SM) {
    rows_per_block_iteration = index_type{32};
  }

  if (row_output_size != 0 && rows_per_block_iteration > 1) {
    do {
      output_workspace_size = compute_output_size(
        row_output_size, threads_per_block, rows_per_block_iteration, infer_type);
      output_workspace_size_bytes = output_item_bytes * output_workspace_size;

      shared_mem_per_block =
        (rows_per_block_iteration * row_size_bytes + output_workspace_size_bytes);
      if (shared_mem_per_block > max_overall_shared_mem) {
        rows_per_block_iteration >>= index_type{1};
      }
    } while (shared_mem_per_block > max_overall_shared_mem && rows_per_block_iteration > 1);
  }

  shared_mem_per_block = std::min(shared_mem_per_block, max_overall_shared_mem);

  // Divide shared mem evenly
  shared_mem_per_block = std::min(
    max_overall_shared_mem, max_shared_mem_per_sm / (max_shared_mem_per_sm / shared_mem_per_block));

  auto num_blocks = std::min(raft_proto::ceildiv(row_count, rows_per_block_iteration), MAX_BLOCKS);
  if (row_output_size == 0) {
    global_workspace = raft_proto::buffer<output_t>{
      output_workspace_size * num_blocks, raft_proto::device_type::gpu, device.value(), stream};
  }
  if (rows_per_block_iteration <= 1) {
    infer_kernel<has_categorical_nodes, 1>
      <<<num_blocks, threads_per_block, shared_mem_per_block, stream>>>(forest,
                                                                        postproc,
                                                                        output,
                                                                        input,
                                                                        row_count,
                                                                        col_count,
                                                                        output_count,
                                                                        shared_mem_per_block,
                                                                        output_workspace_size,
                                                                        vector_output,
                                                                        categorical_data,
                                                                        infer_type,
                                                                        global_workspace.data());
  } else if (rows_per_block_iteration <= 2) {
    infer_kernel<has_categorical_nodes, 2>
      <<<num_blocks, threads_per_block, shared_mem_per_block, stream>>>(forest,
                                                                        postproc,
                                                                        output,
                                                                        input,
                                                                        row_count,
                                                                        col_count,
                                                                        output_count,
                                                                        shared_mem_per_block,
                                                                        output_workspace_size,
                                                                        vector_output,
                                                                        categorical_data,
                                                                        infer_type,
                                                                        global_workspace.data());
  } else if (rows_per_block_iteration <= 4) {
    infer_kernel<has_categorical_nodes, 4>
      <<<num_blocks, threads_per_block, shared_mem_per_block, stream>>>(forest,
                                                                        postproc,
                                                                        output,
                                                                        input,
                                                                        row_count,
                                                                        col_count,
                                                                        output_count,
                                                                        shared_mem_per_block,
                                                                        output_workspace_size,
                                                                        vector_output,
                                                                        categorical_data,
                                                                        infer_type,
                                                                        global_workspace.data());
  } else if (rows_per_block_iteration <= 8) {
    infer_kernel<has_categorical_nodes, 8>
      <<<num_blocks, threads_per_block, shared_mem_per_block, stream>>>(forest,
                                                                        postproc,
                                                                        output,
                                                                        input,
                                                                        row_count,
                                                                        col_count,
                                                                        output_count,
                                                                        shared_mem_per_block,
                                                                        output_workspace_size,
                                                                        vector_output,
                                                                        categorical_data,
                                                                        infer_type,
                                                                        global_workspace.data());
  } else if (rows_per_block_iteration <= 16) {
    infer_kernel<has_categorical_nodes, 16>
      <<<num_blocks, threads_per_block, shared_mem_per_block, stream>>>(forest,
                                                                        postproc,
                                                                        output,
                                                                        input,
                                                                        row_count,
                                                                        col_count,
                                                                        output_count,
                                                                        shared_mem_per_block,
                                                                        output_workspace_size,
                                                                        vector_output,
                                                                        categorical_data,
                                                                        infer_type,
                                                                        global_workspace.data());
  } else {
    infer_kernel<has_categorical_nodes, 32>
      <<<num_blocks, threads_per_block, shared_mem_per_block, stream>>>(forest,
                                                                        postproc,
                                                                        output,
                                                                        input,
                                                                        row_count,
                                                                        col_count,
                                                                        output_count,
                                                                        shared_mem_per_block,
                                                                        output_workspace_size,
                                                                        vector_output,
                                                                        categorical_data,
                                                                        infer_type,
                                                                        global_workspace.data());
  }
  raft_proto::cuda_check(cudaGetLastError());
}

/* This macro is invoked here to declare all standard specializations of this
 * template as extern. This ensures that this (relatively complex) code is
 * compiled as few times as possible. A macro is used because ever
 * specialization must be explicitly declared. The final argument to the macro
 * references the 8 specialization variants compiled in standard cuML FIL. */
CUML_FIL_INFER_ALL(extern template, raft_proto::device_type::gpu, 0)
CUML_FIL_INFER_ALL(extern template, raft_proto::device_type::gpu, 1)
CUML_FIL_INFER_ALL(extern template, raft_proto::device_type::gpu, 2)
CUML_FIL_INFER_ALL(extern template, raft_proto::device_type::gpu, 3)
CUML_FIL_INFER_ALL(extern template, raft_proto::device_type::gpu, 4)
CUML_FIL_INFER_ALL(extern template, raft_proto::device_type::gpu, 5)
CUML_FIL_INFER_ALL(extern template, raft_proto::device_type::gpu, 6)
CUML_FIL_INFER_ALL(extern template, raft_proto::device_type::gpu, 7)

}  // namespace inference
}  // namespace detail
}  // namespace fil
}  // namespace ML
