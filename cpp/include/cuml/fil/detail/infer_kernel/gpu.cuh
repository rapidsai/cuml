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
#include <cuml/common/utils.hpp>
#include <cuml/fil/detail/evaluate_tree.hpp>
#include <cuml/fil/detail/gpu_introspection.hpp>
#include <cuml/fil/detail/index_type.hpp>
#include <cuml/fil/detail/infer_kernel/shared_memory_buffer.cuh>
#include <cuml/fil/detail/postprocessor.hpp>
#include <cuml/fil/detail/raft_proto/ceildiv.hpp>
#include <cuml/fil/detail/raft_proto/padding.hpp>
#include <cuml/fil/infer_kind.hpp>

#include <stddef.h>

#include <cstddef>

namespace ML {
namespace fil {
namespace detail {

/**
 * The GPU kernel used to actually perform forest inference
 *
 * @tparam has_categorical_nodes Whether or not this kernel should be
 * compiled to operate on trees with categorical nodes.
 * @tparam chunk_size The number of rows to be simultaneously processed
 * in each iteration of inference within a single block. This is a
 * performance tuning parameter, and having it fixed at compile-time offers a
 * measurable performance benefit. In standard cuML FIL, we compile for all
 * powers of 2 from 1 to 32. A power of 2 is not guaranteed to optimize
 * performance for all batch sizes and models, but it is far more likely to
 * than other values.
 * @tparam forest_t The type of the forest object which will be used for
 * inference.
 * @tparam vector_output_t If non-nullptr_t, this indicates the type we expect
 * for outputs from vector leaves.
 * @tparam categorical_data_t If non-nullptr_t, this indicates the type we
 * expect for non-local categorical data storage.
 * @param forest The forest used to perform inference
 * @param postproc The postprocessor object used to store all necessary
 * data for postprocessing
 * @param output Pointer to the device-accessible buffer where output
 * should be written
 * @param input Pointer to the device-accessible buffer where input should be
 * read from
 * @param row_count The number of rows in the input
 * @param col_count The number of columns per row in the input
 * @param num_outputs The expected number of output elements per row
 * @param shared_mem_byte_size The number of bytes of shared memory allocated
 * to this kernel.
 * @param output_workspace_size The total number of temporary elements required
 * to be stored as an intermediate output during inference
 * @param vector_output_p If non-nullptr, a pointer to the stored leaf
 * vector outputs for all leaf nodes
 * @param categorical_data If non-nullptr, a pointer to where non-local
 * data on categorical splits are stored.
 * @param infer_type Type of inference to perform. Defaults to summing the outputs of all trees
 * and produce an output per row. If set to "per_tree", we will instead output all outputs of
 * individual trees. If set to "leaf_id", we will instead output the integer ID of the leaf node
 * for each tree.
 * @param global_mem_fallback_buffer Buffer to use as a fallback, when there isn't enough shared
 * memory. Set it to nullptr to disable
 */
template <bool has_categorical_nodes,
          index_type chunk_size,
          typename forest_t,
          typename vector_output_t    = std::nullptr_t,
          typename categorical_data_t = std::nullptr_t>
CUML_KERNEL void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM) infer_kernel(
  forest_t forest,
  postprocessor<typename forest_t::io_type> postproc,
  typename forest_t::io_type* output,
  typename forest_t::io_type const* input,
  index_type row_count,
  index_type col_count,
  index_type num_outputs,
  index_type shared_mem_byte_size,
  index_type output_workspace_size,
  vector_output_t vector_output_p     = nullptr,
  categorical_data_t categorical_data = nullptr,
  infer_kind infer_type               = infer_kind::default_kind,
  typename forest_t::template raw_output_type<vector_output_t>* workspace_fallback = nullptr)
{
  auto const default_num_outputs         = forest.num_outputs();
  auto constexpr has_vector_leaves       = !std::is_same_v<vector_output_t, std::nullptr_t>;
  auto constexpr has_nonlocal_categories = !std::is_same_v<categorical_data_t, std::nullptr_t>;
  using output_t = typename forest_t::template raw_output_type<vector_output_t>;
  extern __shared__ std::byte shared_mem_raw[];

  auto shared_mem = shared_memory_buffer(shared_mem_raw, shared_mem_byte_size);

  using node_t = typename forest_t::node_type;

  using io_t = typename forest_t::io_type;

  for (auto i = blockIdx.x * chunk_size; i < row_count; i += chunk_size * gridDim.x) {
    // i: the ID of the first row in the current chunk

    shared_mem.clear();
    auto* output_workspace = shared_mem.fill<output_t>(
      output_workspace_size, output_t{}, (workspace_fallback + blockIdx.x * output_workspace_size));

    // Handle as many rows as requested per loop or as many rows as are left to
    // process
    auto rows_in_this_iteration = min(chunk_size, row_count - i);

    auto* input_data = shared_mem.copy(input + i * col_count, rows_in_this_iteration, col_count);

    auto task_count = chunk_size * forest.tree_count();

    auto num_grove = raft_proto::ceildiv(min(index_type(blockDim.x), task_count), chunk_size) *
                       (infer_type == infer_kind::default_kind) +
                     (infer_type != infer_kind::default_kind);

    // Note that this sync is safe because every thread in the block will agree
    // on whether or not a sync is required
    shared_mem.sync();

    // Every thread must iterate the same number of times in order to avoid a
    // deadlock on __syncthreads, so we round the task_count up to the next
    // multiple of the number of threads in this block. We then only perform
    // work within the loop if the task_index is below the actual task_count.
    auto const task_count_rounded_up = blockDim.x * raft_proto::ceildiv(task_count, blockDim.x);

    // Infer on each tree and row
    for (auto task_index = threadIdx.x; task_index < task_count_rounded_up;
         task_index += blockDim.x) {
      auto row_index = task_index % chunk_size;
      auto real_task = task_index < task_count && row_index < rows_in_this_iteration;
      row_index *= real_task;
      auto tree_index  = task_index * real_task / chunk_size;
      auto grove_index = (threadIdx.x / chunk_size) * (infer_type == infer_kind::default_kind);

      auto tree_output  = std::conditional_t<has_vector_leaves,
                                             typename node_t::index_type,
                                             typename node_t::threshold_type>{};
      auto leaf_node_id = index_type{};
      if (infer_type == infer_kind::leaf_id) {
        leaf_node_id =
          evaluate_tree<has_vector_leaves, has_categorical_nodes, has_nonlocal_categories, true>(
            forest, tree_index, input_data + row_index * col_count, categorical_data);
      } else {
        tree_output =
          evaluate_tree<has_vector_leaves, has_categorical_nodes, has_nonlocal_categories, false>(
            forest, tree_index, input_data + row_index * col_count, categorical_data);
      }

      if (infer_type == infer_kind::leaf_id) {
        output_workspace[row_index * num_outputs * num_grove + tree_index * num_grove +
                         grove_index] = static_cast<typename forest_t::io_type>(leaf_node_id);
      } else {
        if constexpr (has_vector_leaves) {
          auto output_offset =
            (row_index * num_outputs * num_grove +
             tree_index * default_num_outputs * num_grove * (infer_type == infer_kind::per_tree) +
             grove_index);
          for (auto output_index = index_type{}; output_index < default_num_outputs;
               ++output_index) {
            if (real_task) {
              output_workspace[output_offset + output_index * num_grove] +=
                vector_output_p[tree_output * default_num_outputs + output_index];
            }
          }
        } else {
          auto output_offset =
            (row_index * num_outputs * num_grove +
             (tree_index % default_num_outputs) * num_grove *
               (infer_type == infer_kind::default_kind) +
             tree_index * num_grove * (infer_type == infer_kind::per_tree) + grove_index);
          if (real_task) { output_workspace[output_offset] += tree_output; }
        }
      }

      __syncthreads();
    }

    auto padded_num_groves = raft_proto::padded_size(num_grove, WARP_SIZE);
    for (auto row_index = threadIdx.x / WARP_SIZE; row_index < rows_in_this_iteration;
         row_index += blockDim.x / WARP_SIZE) {
      for (auto class_index = index_type{}; class_index < num_outputs; ++class_index) {
        auto grove_offset = (row_index * num_outputs * num_grove + class_index * num_grove);
        auto class_sum    = output_t{};
        for (auto grove_index = threadIdx.x % WARP_SIZE; grove_index < padded_num_groves;
             grove_index += WARP_SIZE) {
          auto real_thread = grove_index < num_grove;
          auto out_index   = grove_offset + grove_index * real_thread;
          class_sum *= (threadIdx.x % WARP_SIZE == 0);
          class_sum += output_workspace[out_index] * real_thread;
          for (auto thread_offset = (WARP_SIZE >> 1); thread_offset > 0; thread_offset >>= 1) {
            class_sum += __shfl_down_sync(0xFFFFFFFF, class_sum, thread_offset);
          }
        }
        if (threadIdx.x % WARP_SIZE == 0) { output_workspace[grove_offset] = class_sum; }
      }
      if (threadIdx.x % WARP_SIZE == 0) {
        postproc(output_workspace + row_index * num_outputs * num_grove,
                 num_outputs,
                 output + ((i + row_index) * num_outputs),
                 num_grove);
      }
    }
    __syncthreads();
  }
}

}  // namespace detail
}  // namespace fil
}  // namespace ML
