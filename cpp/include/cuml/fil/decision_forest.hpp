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
#include <cuml/fil/detail/device_initialization.hpp>
#include <cuml/fil/detail/forest.hpp>
#include <cuml/fil/detail/index_type.hpp>
#include <cuml/fil/detail/infer.hpp>
#include <cuml/fil/detail/postprocessor.hpp>
#include <cuml/fil/detail/raft_proto/buffer.hpp>
#include <cuml/fil/detail/raft_proto/cuda_stream.hpp>
#include <cuml/fil/detail/raft_proto/exceptions.hpp>
#include <cuml/fil/detail/specialization_types.hpp>
#include <cuml/fil/exceptions.hpp>
#include <cuml/fil/infer_kind.hpp>
#include <cuml/fil/postproc_ops.hpp>
#include <cuml/fil/tree_layout.hpp>

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <variant>

namespace ML {
namespace fil {

/**
 * A general-purpose decision forest implementation
 *
 * This template provides an optimized but generic implementation of a decision
 * forest. Template parameters are used to specialize the
 * implementation based on the size and characteristics of the forest.
 * For instance, the smallest integer that can express the offset between a
 * parent and child node within a tree is used in order to minimize the size
 * of a node, increasing the number that can fit within the L2 or L1 cache.
 *
 * @tparam layout_v The in-memory layout of nodes in this forest
 * @tparam threshold_t The floating-point type used for quantitative splits
 * @tparam index_t The integer type used for storing many things within a
 * forest, including the category value of categorical nodes and the index at
 * which vector output for a leaf node is stored.
 * @tparam metadata_storage_t The type used for storing node metadata.
 * The first several bits will be used to store flags indicating various
 * characteristics of the node, and the remaining bits provide the integer
 * index of the feature for this node's split
 * @tparam offset_t An integer used to indicate the offset between a node and
 * its most distant child. This type must be large enough to store the
 * largest such offset in the entire forest.
 */
template <tree_layout layout_v,
          typename threshold_t,
          typename index_t,
          typename metadata_storage_t,
          typename offset_t>
struct decision_forest {
  /**
   * The in-memory layout of nodes in this forest
   */
  auto constexpr static const layout = layout_v;
  /**
   * The type of the forest object which is actually passed to the CPU/GPU
   * for inference
   */
  using forest_type = forest<layout, threshold_t, index_t, metadata_storage_t, offset_t>;
  /**
   * The type of nodes within the forest
   */
  using node_type = typename forest_type::node_type;
  /**
   * The type used for input and output to the model
   */
  using io_type = typename forest_type::io_type;
  /**
   * The type used for quantitative splits within the model
   */
  using threshold_type = threshold_t;
  /**
   * The type used to indicate how leaf output should be post-processed
   */
  using postprocessor_type = postprocessor<io_type>;
  /**
   * The type used for storing data on categorical nodes
   */
  using categorical_storage_type = typename node_type::index_type;

  /**
   * Construct an empty decision forest
   */
  decision_forest()
    : nodes_{},
      root_node_indexes_{},
      node_id_mapping_{},
      vector_output_{},
      categorical_storage_{},
      num_features_{},
      num_outputs_{},
      leaf_size_{},
      has_categorical_nodes_{false},
      row_postproc_{},
      elem_postproc_{},
      average_factor_{},
      bias_{},
      postproc_constant_{}
  {
  }

  /**
   * Construct a decision forest with the indicated data
   *
   * @param nodes A buffer containing all nodes within the forest
   * @param root_node_indexes A buffer containing the index of the root node
   * of every tree in the forest
   * @param node_id_mapping Mapping to use to convert FIL's internal node ID into Treelite's node
   * ID. Only relevant when predict_type == infer_kind::leaf_id
   * @param num_features The number of features per input sample for this model
   * @param num_outputs The number of outputs per row from this model
   * @param has_categorical_nodes Whether this forest contains any
   * categorical nodes
   * @param vector_output A buffer containing the output from all vector
   * leaves for this model. Each leaf node will specify the offset within
   * this buffer at which its vector output begins, and leaf_size will be
   * used to determine how many subsequent entries from the buffer should be
   * used to construct the vector output. A value of std::nullopt indicates
   * that this is not a vector leaf model.
   * @param categorical_storage For models with inputs on too many categories
   * to be stored in the bits of an `index_t`, it may be necessary to store
   * categorical information external to the node itself. This buffer
   * contains the necessary storage for this information.
   * @param leaf_size The number of output values per leaf (1 for non-vector
   * leaves; >1 for vector leaves)
   * @param row_postproc The post-processing operation to be applied to an
   * entire row of the model output
   * @param elem_postproc The per-element post-processing operation to be
   * applied to the model output
   * @param average_factor A factor which is used for output
   * normalization
   * @param bias The bias term that is applied to the output after
   * normalization
   * @param postproc_constant A constant used by some post-processing
   * operations, including sigmoid, exponential, and
   * logarithm_one_plus_exp
   */
  decision_forest(raft_proto::buffer<node_type>&& nodes,
                  raft_proto::buffer<index_type>&& root_node_indexes,
                  raft_proto::buffer<index_type>&& node_id_mapping,
                  index_type num_features,
                  index_type num_outputs                                     = index_type{2},
                  bool has_categorical_nodes                                 = false,
                  std::optional<raft_proto::buffer<io_type>>&& vector_output = std::nullopt,
                  std::optional<raft_proto::buffer<typename node_type::index_type>>&&
                    categorical_storage     = std::nullopt,
                  index_type leaf_size      = index_type{1},
                  row_op row_postproc       = row_op::disable,
                  element_op elem_postproc  = element_op::disable,
                  io_type average_factor    = io_type{1},
                  io_type bias              = io_type{0},
                  io_type postproc_constant = io_type{1})
    : nodes_{nodes},
      root_node_indexes_{root_node_indexes},
      node_id_mapping_{node_id_mapping},
      vector_output_{vector_output},
      categorical_storage_{categorical_storage},
      num_features_{num_features},
      num_outputs_{num_outputs},
      leaf_size_{leaf_size},
      has_categorical_nodes_{has_categorical_nodes},
      row_postproc_{row_postproc},
      elem_postproc_{elem_postproc},
      average_factor_{average_factor},
      bias_{bias},
      postproc_constant_{postproc_constant}
  {
    if (nodes.memory_type() != root_node_indexes.memory_type()) {
      throw raft_proto::mem_type_mismatch(
        "Nodes and indexes of forest must both be stored on either host or device");
    }
    if (nodes.device_index() != root_node_indexes.device_index()) {
      throw raft_proto::mem_type_mismatch(
        "Nodes and indexes of forest must both be stored on same device");
    }
    detail::initialize_device<forest_type>(nodes.device());
  }

  /** The number of features per row expected by the model */
  auto num_features() const { return num_features_; }
  /** The number of trees in the model */
  auto num_trees() const { return root_node_indexes_.size(); }
  /** Whether or not leaf nodes have vector outputs */
  auto has_vector_leaves() const { return vector_output_.has_value(); }

  /** The number of outputs per row generated by the model for the given
   * type of inference */
  auto num_outputs(infer_kind inference_kind = infer_kind::default_kind) const
  {
    auto result = num_outputs_;
    if (inference_kind == infer_kind::per_tree) {
      result = num_trees();
      if (has_vector_leaves()) { result *= num_outputs_; }
    } else if (inference_kind == infer_kind::leaf_id) {
      result = num_trees();
    }
    return result;
  }

  /** The operation used for postprocessing all outputs for a single row */
  auto row_postprocessing() const { return row_postproc_; }
  // Setter for row_postprocessing
  void set_row_postprocessing(row_op val) { row_postproc_ = val; }
  /** The operation used for postprocessing each element of the output for a
   * single row */
  auto elem_postprocessing() const { return elem_postproc_; }

  /** The type of memory (device/host) where the model is stored */
  auto memory_type() { return nodes_.memory_type(); }
  /** The ID of the device on which this model is loaded */
  auto device_index() { return nodes_.device_index(); }

  /**
   * Perform inference with this model
   *
   * @param[out] output The buffer where the model output should be stored.
   * This must be of size ROWS x num_outputs().
   * @param[in] input The buffer containing the input data
   * @param[in] stream For GPU execution, the CUDA stream. For CPU execution,
   * this optional parameter can be safely omitted.
   * @param[in] predict_type Type of inference to perform. Defaults to summing
   * the outputs of all trees and produce an output per row. If set to
   * "per_tree", we will instead output all outputs of individual trees.
   * If set to "leaf_id", we will output the integer ID of the leaf node
   * for each tree.
   * @param[in] specified_rows_per_block_iter If non-nullopt, this value is
   * used to determine how many rows are evaluated for each inference
   * iteration within a CUDA block. Runtime performance is quite sensitive
   * to this value, but it is difficult to predict a priori, so it is
   * recommended to perform a search over possible values with realistic
   * batch sizes in order to determine the optimal value. Any power of 2 from
   * 1 to 32 is a valid value, and in general larger batches benefit from
   * larger values.
   */
  void predict(raft_proto::buffer<typename forest_type::io_type>& output,
               raft_proto::buffer<typename forest_type::io_type> const& input,
               raft_proto::cuda_stream stream                          = raft_proto::cuda_stream{},
               infer_kind predict_type                                 = infer_kind::default_kind,
               std::optional<index_type> specified_rows_per_block_iter = std::nullopt)
  {
    if (output.memory_type() != memory_type() || input.memory_type() != memory_type()) {
      throw raft_proto::wrong_device_type{
        "Tried to use host I/O data with model on device or vice versa"};
    }
    if (output.device_index() != device_index() || input.device_index() != device_index()) {
      throw raft_proto::wrong_device{"I/O data on different device than model"};
    }
    auto* vector_output_data =
      (vector_output_.has_value() ? vector_output_->data() : static_cast<io_type*>(nullptr));
    auto* categorical_storage_data =
      (categorical_storage_.has_value() ? categorical_storage_->data()
                                        : static_cast<categorical_storage_type*>(nullptr));
    switch (nodes_.device().index()) {
      case 0:
        fil::detail::infer(obj(),
                           get_postprocessor(predict_type),
                           output.data(),
                           input.data(),
                           index_type(input.size() / num_features_),
                           num_features_,
                           num_outputs(predict_type),
                           has_categorical_nodes_,
                           vector_output_data,
                           categorical_storage_data,
                           predict_type,
                           specified_rows_per_block_iter,
                           std::get<0>(nodes_.device()),
                           stream);
        break;
      case 1:
        fil::detail::infer(obj(),
                           get_postprocessor(predict_type),
                           output.data(),
                           input.data(),
                           index_type(input.size() / num_features_),
                           num_features_,
                           num_outputs(predict_type),
                           has_categorical_nodes_,
                           vector_output_data,
                           categorical_storage_data,
                           predict_type,
                           specified_rows_per_block_iter,
                           std::get<1>(nodes_.device()),
                           stream);
        break;
    }
  }

 private:
  /** The nodes for all trees in the forest */
  raft_proto::buffer<node_type> nodes_;
  /** The index of the root node for each tree in the forest */
  raft_proto::buffer<index_type> root_node_indexes_;
  /** Mapping to apply to node IDs. Only relevant when predict_type == infer_kind::leaf_id */
  raft_proto::buffer<index_type> node_id_mapping_;
  /** Buffer of outputs for all leaves in vector-leaf models */
  std::optional<raft_proto::buffer<io_type>> vector_output_;
  /** Buffer of elements used as backing data for bitsets which specify
   * categories for all categorical nodes in the model. */
  std::optional<raft_proto::buffer<categorical_storage_type>> categorical_storage_;

  // Metadata
  index_type num_features_;
  index_type num_outputs_;
  index_type leaf_size_;
  bool has_categorical_nodes_ = false;
  // Postprocessing constants
  row_op row_postproc_;
  element_op elem_postproc_;
  io_type average_factor_;
  io_type bias_;
  io_type postproc_constant_;

  auto obj() const
  {
    return forest_type{nodes_.data(),
                       root_node_indexes_.data(),
                       node_id_mapping_.data(),
                       static_cast<index_type>(root_node_indexes_.size()),
                       num_outputs_};
  }

  auto get_postprocessor(infer_kind inference_kind = infer_kind::default_kind) const
  {
    auto result = postprocessor_type{};
    if (inference_kind == infer_kind::default_kind) {
      result = postprocessor_type{
        row_postproc_, elem_postproc_, average_factor_, bias_, postproc_constant_};
    }
    return result;
  }

  auto leaf_size() const { return leaf_size_; }
};

namespace detail {
/**
 * A convenience wrapper to simplify template instantiation of
 * decision_forest
 *
 * This template takes the large range of available template parameters
 * and reduces them to just three standard choices.
 *
 * @tparam layout The in-memory layout of nodes in this forest
 * @tparam double_precision Whether this model should use double-precision
 * for floating-point evaluation and 64-bit integers for indexes
 * @tparam large_trees Whether this forest expects more than 2**(16 -3) - 1 =
 * 8191 features or contains nodes whose child is offset more than 2**16 - 1 = 65535 nodes away.
 */
template <tree_layout layout, bool double_precision, bool large_trees>
using preset_decision_forest = decision_forest<
  layout,
  typename specialization_types<layout, double_precision, large_trees>::threshold_type,
  typename specialization_types<layout, double_precision, large_trees>::index_type,
  typename specialization_types<layout, double_precision, large_trees>::metadata_type,
  typename specialization_types<layout, double_precision, large_trees>::offset_type>;

}  // namespace detail

/** A variant containing all standard decision_forest instantiations */
using decision_forest_variant = std::variant<
  detail::preset_decision_forest<
    std::variant_alternative_t<0, detail::specialization_variant>::layout,
    std::variant_alternative_t<0, detail::specialization_variant>::is_double_precision,
    std::variant_alternative_t<0, detail::specialization_variant>::has_large_trees>,
  detail::preset_decision_forest<
    std::variant_alternative_t<1, detail::specialization_variant>::layout,
    std::variant_alternative_t<1, detail::specialization_variant>::is_double_precision,
    std::variant_alternative_t<1, detail::specialization_variant>::has_large_trees>,
  detail::preset_decision_forest<
    std::variant_alternative_t<2, detail::specialization_variant>::layout,
    std::variant_alternative_t<2, detail::specialization_variant>::is_double_precision,
    std::variant_alternative_t<2, detail::specialization_variant>::has_large_trees>,
  detail::preset_decision_forest<
    std::variant_alternative_t<3, detail::specialization_variant>::layout,
    std::variant_alternative_t<3, detail::specialization_variant>::is_double_precision,
    std::variant_alternative_t<3, detail::specialization_variant>::has_large_trees>,
  detail::preset_decision_forest<
    std::variant_alternative_t<4, detail::specialization_variant>::layout,
    std::variant_alternative_t<4, detail::specialization_variant>::is_double_precision,
    std::variant_alternative_t<4, detail::specialization_variant>::has_large_trees>,
  detail::preset_decision_forest<
    std::variant_alternative_t<5, detail::specialization_variant>::layout,
    std::variant_alternative_t<5, detail::specialization_variant>::is_double_precision,
    std::variant_alternative_t<5, detail::specialization_variant>::has_large_trees>,
  detail::preset_decision_forest<
    std::variant_alternative_t<6, detail::specialization_variant>::layout,
    std::variant_alternative_t<6, detail::specialization_variant>::is_double_precision,
    std::variant_alternative_t<6, detail::specialization_variant>::has_large_trees>,
  detail::preset_decision_forest<
    std::variant_alternative_t<7, detail::specialization_variant>::layout,
    std::variant_alternative_t<7, detail::specialization_variant>::is_double_precision,
    std::variant_alternative_t<7, detail::specialization_variant>::has_large_trees>,
  detail::preset_decision_forest<
    std::variant_alternative_t<8, detail::specialization_variant>::layout,
    std::variant_alternative_t<8, detail::specialization_variant>::is_double_precision,
    std::variant_alternative_t<8, detail::specialization_variant>::has_large_trees>,
  detail::preset_decision_forest<
    std::variant_alternative_t<9, detail::specialization_variant>::layout,
    std::variant_alternative_t<9, detail::specialization_variant>::is_double_precision,
    std::variant_alternative_t<9, detail::specialization_variant>::has_large_trees>,
  detail::preset_decision_forest<
    std::variant_alternative_t<10, detail::specialization_variant>::layout,
    std::variant_alternative_t<10, detail::specialization_variant>::is_double_precision,
    std::variant_alternative_t<10, detail::specialization_variant>::has_large_trees>,
  detail::preset_decision_forest<
    std::variant_alternative_t<11, detail::specialization_variant>::layout,
    std::variant_alternative_t<11, detail::specialization_variant>::is_double_precision,
    std::variant_alternative_t<11, detail::specialization_variant>::has_large_trees>>;

/**
 * Determine the variant index of the decision_forest type to used based on
 * model characteristics
 *
 * @param use_double_thresholds Whether single or double-precision floating
 * point values should be used for quantitative splits
 * @param max_node_offset The largest offset between a parent node and either
 * of its children
 * @param num_features The number of input features per row
 * @param num_categorical_nodes The total number of categorical nodes in the
 * forest
 * @param max_num_categories The maximum number of categories in any
 * categorical feature used by the model
 * @param num_vector_leaves The total number of leaf nodes which produce vector
 * outputs. For non-vector-leaf models, this should be 0. For vector-leaf
 * models, this should be the total number of leaf nodes.
 * @param layout The in-memory layout to be used for nodes in the forest
 */
inline auto get_forest_variant_index(bool use_double_thresholds,
                                     index_type max_node_offset,
                                     index_type num_features,
                                     index_type num_categorical_nodes = index_type{},
                                     index_type max_num_categories    = index_type{},
                                     index_type num_vector_leaves     = index_type{},
                                     tree_layout layout               = preferred_tree_layout)
{
  using small_index_t =
    typename detail::specialization_types<preferred_tree_layout, false, false>::index_type;
  auto max_local_categories = index_type(sizeof(small_index_t) * 8);
  // If the index required for pointing to categorical storage bins or vector
  // leaf output exceeds what we can store in a uint32_t, uint64_t will be used
  //
  // TODO(wphicks): We are overestimating categorical storage required here
  auto double_indexes_required =
    (max_num_categories > max_local_categories &&
     ((raft_proto::ceildiv(max_num_categories, max_local_categories) + 1 * num_categorical_nodes) >
      std::numeric_limits<small_index_t>::max())) ||
    num_vector_leaves > std::numeric_limits<small_index_t>::max();

  auto double_precision = use_double_thresholds || double_indexes_required;

  using small_metadata_t =
    typename detail::specialization_types<preferred_tree_layout, false, false>::metadata_type;
  using small_offset_t =
    typename detail::specialization_types<preferred_tree_layout, false, false>::offset_type;

  auto large_trees =
    (num_features > (std::numeric_limits<small_metadata_t>::max() >> reserved_node_metadata_bits) ||
     max_node_offset > std::numeric_limits<small_offset_t>::max());

  auto layout_value = static_cast<std::underlying_type_t<tree_layout>>(layout);

  return ((index_type{layout_value} << index_type{2}) +
          (index_type{double_precision} << index_type{1}) + index_type{large_trees});
}
}  // namespace fil
}  // namespace ML
