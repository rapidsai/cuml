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
#include <cuml/fil/detail/cpu_introspection.hpp>
#include <cuml/fil/detail/evaluate_tree.hpp>
#include <cuml/fil/detail/index_type.hpp>
#include <cuml/fil/detail/postprocessor.hpp>
#include <cuml/fil/detail/raft_proto/ceildiv.hpp>
#include <cuml/fil/infer_kind.hpp>

#ifdef _OPENMP
#include <omp.h>
#else
#ifdef omp_get_max_threads
#if omp_get_max_threads() != 1
#error "Inconsistent placeholders for omp_get_max_threads"
#endif
#else
#define omp_get_max_threads() 1
#endif
#endif

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <new>
#include <numeric>
#include <vector>

namespace ML {
namespace fil {
namespace detail {

/**
 * The CPU "kernel" used to actually perform forest inference
 *
 * @tparam has_categorical_nodes Whether or not this kernel should be
 * compiled to operate on trees with categorical nodes.
 * @tparam forest_t The type of the forest object which will be used for
 * inference.
 * @tparam vector_output_t If non-nullptr_t, this indicates the type we expect
 * for outputs from vector leaves.
 * @tparam categorical_data_t If non-nullptr_t, this indicates the type we
 * expect for non-local categorical data storage.
 * @param forest The forest used to perform inference
 * @param postproc The postprocessor object used to store all necessary
 * data for postprocessing
 * @param output Pointer to the host-accessible buffer where output
 * should be written
 * @param input Pointer to the host-accessible buffer where input should be
 * read from
 * @param row_count The number of rows in the input
 * @param col_count The number of columns per row in the input
 * @param num_outputs The expected number of output elements per row
 * @param chunk_size The number of rows for each thread to process with its
 * assigned trees before fetching a new set of trees/rows.
 * @param grove_size The number of trees to assign to a thread for each chunk
 * of rows it processes.
 * @param vector_output_p If non-nullptr, a pointer to the stored leaf
 * vector outputs for all leaf nodes
 * @param categorical_data If non-nullptr, a pointer to where non-local
 * data on categorical splits are stored.
 * @param infer_type Type of inference to perform. Defaults to summing the outputs of all trees
 * and produce an output per row. If set to "per_tree", we will instead output all outputs of
 * individual trees. If set to "leaf_id", we will output the integer ID of the leaf node
 * for each tree.
 */
template <bool has_categorical_nodes,
          bool predict_leaf,
          typename forest_t,
          typename vector_output_t    = std::nullptr_t,
          typename categorical_data_t = std::nullptr_t>
void infer_kernel_cpu(forest_t const& forest,
                      postprocessor<typename forest_t::io_type> const& postproc,
                      typename forest_t::io_type* output,
                      typename forest_t::io_type const* input,
                      index_type row_count,
                      index_type col_count,
                      index_type num_outputs,
                      index_type chunk_size               = hardware_constructive_interference_size,
                      index_type grove_size               = hardware_constructive_interference_size,
                      vector_output_t vector_output_p     = nullptr,
                      categorical_data_t categorical_data = nullptr,
                      infer_kind infer_type               = infer_kind::default_kind)
{
  auto constexpr has_vector_leaves       = !std::is_same_v<vector_output_t, std::nullptr_t>;
  auto constexpr has_nonlocal_categories = !std::is_same_v<categorical_data_t, std::nullptr_t>;

  using node_t = typename forest_t::node_type;

  using output_t = typename forest_t::template raw_output_type<vector_output_t>;

  auto const num_tree  = forest.tree_count();
  auto const num_grove = raft_proto::ceildiv(num_tree, grove_size);
  auto const num_chunk = raft_proto::ceildiv(row_count, chunk_size);

  auto output_workspace = std::vector<output_t>(row_count * num_outputs * num_grove, output_t{});
  auto const task_count = num_grove * num_chunk;

#pragma omp parallel num_threads(std::min(index_type(omp_get_max_threads()), task_count))
  {
    // Infer on each grove and chunk
#pragma omp for
    for (auto task_index = index_type{}; task_index < task_count; ++task_index) {
      auto const grove_index = task_index / num_chunk;
      auto const chunk_index = task_index % num_chunk;
      auto const start_row   = chunk_index * chunk_size;
      auto const end_row     = std::min(start_row + chunk_size, row_count);
      auto const start_tree  = grove_index * grove_size;
      auto const end_tree    = std::min(start_tree + grove_size, num_tree);

      for (auto row_index = start_row; row_index < end_row; ++row_index) {
        for (auto tree_index = start_tree; tree_index < end_tree; ++tree_index) {
          auto tree_output =
            std::conditional_t<predict_leaf,
                               index_type,
                               std::conditional_t<has_vector_leaves,
                                                  typename node_t::index_type,
                                                  typename node_t::threshold_type>>{};
          tree_output = evaluate_tree<has_vector_leaves,
                                      has_categorical_nodes,
                                      has_nonlocal_categories,
                                      predict_leaf>(
            forest, tree_index, input + row_index * col_count, categorical_data);
          if constexpr (predict_leaf) {
            output_workspace[row_index * num_outputs * num_grove + tree_index * num_grove +
                             grove_index] = static_cast<typename forest_t::io_type>(tree_output);
          } else {
            auto const default_num_outputs = forest.num_outputs();
            if constexpr (has_vector_leaves) {
              auto output_offset = (row_index * num_outputs * num_grove +
                                    tree_index * default_num_outputs * num_grove *
                                      (infer_type == infer_kind::per_tree) +
                                    grove_index);
              for (auto output_index = index_type{}; output_index < default_num_outputs;
                   ++output_index) {
                output_workspace[output_offset + output_index * num_grove] +=
                  vector_output_p[tree_output * default_num_outputs + output_index];
              }
            } else {
              auto output_offset =
                (row_index * num_outputs * num_grove +
                 (tree_index % default_num_outputs) * num_grove *
                   (infer_type == infer_kind::default_kind) +
                 tree_index * num_grove * (infer_type == infer_kind::per_tree) + grove_index);
              output_workspace[output_offset] += tree_output;
            }
          }
        }  // Trees
      }  // Rows
    }  // Tasks

    // Sum over grove and postprocess
#pragma omp for
    for (auto row_index = index_type{}; row_index < row_count; ++row_index) {
      for (auto output_index = index_type{}; output_index < num_outputs; ++output_index) {
        auto grove_offset = (row_index * num_outputs * num_grove + output_index * num_grove);

        output_workspace[grove_offset] =
          std::accumulate(std::begin(output_workspace) + grove_offset,
                          std::begin(output_workspace) + grove_offset + num_grove,
                          output_t{});
      }
      postproc(output_workspace.data() + row_index * num_outputs * num_grove,
               num_outputs,
               output + row_index * num_outputs,
               num_grove);
    }
  }  // End omp parallel
}

}  // namespace detail
}  // namespace fil
}  // namespace ML
