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
#include <cuml/fil/detail/bitset.hpp>
#include <cuml/fil/detail/forest.hpp>
#include <cuml/fil/detail/index_type.hpp>
#include <cuml/fil/detail/raft_proto/buffer.hpp>
#include <cuml/fil/detail/raft_proto/ceildiv.hpp>
#include <cuml/fil/detail/raft_proto/cuda_stream.hpp>
#include <cuml/fil/detail/raft_proto/device_type.hpp>
#include <cuml/fil/exceptions.hpp>
#include <cuml/fil/postproc_ops.hpp>

#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <vector>

namespace ML {
namespace fil {
namespace detail {

/*
 * Exception indicating that FIL model could not be built from given input
 */
struct model_builder_error : std::exception {
  model_builder_error() : model_builder_error("Error while building model") {}
  model_builder_error(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

/*
 * Struct used to build FIL forests
 */
template <typename decision_forest_t>
struct decision_forest_builder {
  /* The type for nodes in the given decision_forest type */
  using node_type = typename decision_forest_t::node_type;

  /* Add a node with a categorical split */
  template <typename iter_t>
  void add_categorical_node(
    iter_t vec_begin,
    iter_t vec_end,
    std::optional<int> tl_node_id                     = std::nullopt,
    std::size_t depth                                 = std::size_t{1},
    bool default_to_distant_child                     = false,
    typename node_type::metadata_storage_type feature = typename node_type::metadata_storage_type{},
    typename node_type::offset_type offset            = typename node_type::offset_type{})
  {
    auto constexpr const bin_width = index_type(sizeof(typename node_type::index_type) * 8);
    auto node_value                = typename node_type::index_type{};
    auto set_storage               = &node_value;
    auto max_node_categories =
      (vec_begin != vec_end) ? *std::max_element(vec_begin, vec_end) + 1 : 1;
    if (max_num_categories_ > bin_width) {
      // TODO(wphicks): Check for overflow here
      node_value         = categorical_storage_.size();
      auto bins_required = raft_proto::ceildiv(max_node_categories, bin_width);
      categorical_storage_.push_back(max_node_categories);
      categorical_storage_.resize(categorical_storage_.size() + bins_required);
      set_storage = &(categorical_storage_[node_value + 1]);
    }
    auto set = bitset{set_storage, max_node_categories};
    std::for_each(vec_begin, vec_end, [&set](auto&& cat_index) { set.set(cat_index); });

    add_node(
      node_value, tl_node_id, depth, false, default_to_distant_child, true, feature, offset, false);
  }

  /* Add a leaf node with vector output */
  template <typename iter_t>
  void add_leaf_vector_node(iter_t vec_begin,
                            iter_t vec_end,
                            std::optional<int> tl_node_id = std::nullopt,
                            std::size_t depth             = std::size_t{1})
  {
    auto leaf_index = typename node_type::index_type(vector_output_.size() / output_size_);
    std::copy(vec_begin, vec_end, std::back_inserter(vector_output_));

    add_node(leaf_index,
             tl_node_id,
             depth,
             true,
             false,
             false,
             typename node_type::metadata_storage_type{},
             typename node_type::offset_type{},
             false);
  }

  /* Add a node to the model */
  template <typename value_t>
  void add_node(
    value_t val,
    std::optional<int> tl_node_id                     = std::nullopt,
    std::size_t depth                                 = std::size_t{1},
    bool is_leaf_node                                 = true,
    bool default_to_distant_child                     = false,
    bool is_categorical_node                          = false,
    typename node_type::metadata_storage_type feature = typename node_type::metadata_storage_type{},
    typename node_type::offset_type offset            = typename node_type::offset_type{},
    bool is_inclusive                                 = false)
  {
    if (depth == std::size_t{}) {
      if (alignment_ != index_type{}) {
        if (cur_node_index_ % alignment_ != index_type{}) {
          auto padding = (alignment_ - cur_node_index_ % alignment_);
          for (auto i = index_type{}; i < padding; ++i) {
            add_node(typename node_type::threshold_type{}, std::nullopt);
          }
        }
      }
      root_node_indexes_.push_back(cur_node_index_);
    }

    if (is_inclusive) { val = std::nextafter(val, std::numeric_limits<value_t>::infinity()); }
    nodes_.emplace_back(
      val, is_leaf_node, default_to_distant_child, is_categorical_node, feature, offset);
    // 0 indicates the lack of ID mapping for a particular node
    node_id_mapping_.push_back(static_cast<index_type>(tl_node_id.value_or(0)));
    ++cur_node_index_;
  }

  /* Set the element-wise postprocessing operation for this model */
  void set_element_postproc(element_op val) { element_postproc_ = val; }
  /* Set the row-wise postprocessing operation for this model */
  void set_row_postproc(row_op val) { row_postproc_ = val; }
  /* Set the value to divide by during postprocessing */
  void set_average_factor(double val) { average_factor_ = val; }
  /* Set the the bias term to remove during postprocessing */
  void set_bias(double val) { bias_ = val; }
  /* Set the the value of the constant used in the postprocessing operation
   * (if any) */
  void set_postproc_constant(double val) { postproc_constant_ = val; }
  /* Set the number of outputs per row for this model */
  void set_output_size(index_type val)
  {
    if (output_size_ != index_type{1} && output_size_ != val) {
      throw model_import_error("Inconsistent leaf vector size");
    }
    output_size_ = val;
  }

  decision_forest_builder(index_type max_num_categories = index_type{},
                          index_type align_bytes        = index_type{})
    : cur_node_index_{},
      max_num_categories_{max_num_categories},
      alignment_{std::lcm(align_bytes, index_type(sizeof(node_type)))},
      output_size_{1},
      row_postproc_{},
      element_postproc_{},
      average_factor_{},
      bias_{},
      postproc_constant_{},
      nodes_{},
      root_node_indexes_{},
      vector_output_{}
  {
  }

  /* Return the FIL decision forest built by this builder */
  auto get_decision_forest(index_type num_feature,
                           index_type num_class,
                           raft_proto::device_type mem_type = raft_proto::device_type::cpu,
                           int device                       = 0,
                           raft_proto::cuda_stream stream   = raft_proto::cuda_stream{})
  {
    // Allow narrowing for preprocessing constants. They are stored as doubles
    // for consistency in the builder but must be converted to the proper types
    // for the concrete forest model.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
    return decision_forest_t{
      raft_proto::buffer{
        raft_proto::buffer{nodes_.data(), nodes_.size()}, mem_type, device, stream},
      raft_proto::buffer{raft_proto::buffer{root_node_indexes_.data(), root_node_indexes_.size()},
                         mem_type,
                         device,
                         stream},
      raft_proto::buffer{raft_proto::buffer{node_id_mapping_.data(), node_id_mapping_.size()},
                         mem_type,
                         device,
                         stream},
      num_feature,
      num_class,
      max_num_categories_ != 0,
      vector_output_.empty()
        ? std::nullopt
        : std::make_optional<raft_proto::buffer<typename node_type::threshold_type>>(
            raft_proto::buffer{vector_output_.data(), vector_output_.size()},
            mem_type,
            device,
            stream),
      categorical_storage_.empty()
        ? std::nullopt
        : std::make_optional<raft_proto::buffer<typename node_type::index_type>>(
            raft_proto::buffer{categorical_storage_.data(), categorical_storage_.size()},
            mem_type,
            device,
            stream),
      output_size_,
      row_postproc_,
      element_postproc_,
      static_cast<typename node_type::threshold_type>(average_factor_),
      static_cast<typename node_type::threshold_type>(bias_),
      static_cast<typename node_type::threshold_type>(postproc_constant_)};
#pragma GCC diagnostic pop
  }

 private:
  index_type cur_node_index_;
  index_type max_num_categories_;
  index_type alignment_;
  index_type output_size_;
  row_op row_postproc_;
  element_op element_postproc_;
  double average_factor_;
  double bias_;
  double postproc_constant_;

  std::vector<node_type> nodes_;
  std::vector<index_type> root_node_indexes_;
  std::vector<typename node_type::threshold_type> vector_output_;
  std::vector<typename node_type::index_type> categorical_storage_;
  std::vector<index_type> node_id_mapping_;
};

}  // namespace detail
}  // namespace fil
}  // namespace ML
