/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
#include <iterator>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ML {
namespace fil {
namespace detail {

struct floating_point_truncation_error : std::exception {
  floating_point_truncation_error() {}
  floating_point_truncation_error(std::string msg) : msg_{msg} {}
  floating_point_truncation_error(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_.c_str(); }

 private:
  std::string msg_;
};

template <typename To, typename From>
To safe_cast_floating_point(From x)
{
  static_assert(std::is_floating_point_v<From> && std::is_floating_point_v<To>,
                "Source and destination types must be both floating-point types.");
  if constexpr (sizeof(To) >= sizeof(From)) {
    // Widening cast
    return static_cast<To>(x);
  } else {
    // Narrowing cast: should be checked
    if (!std::isfinite(x)) {
      throw floating_point_truncation_error{"Cannot cast an INF or NaN value"};
    }
    auto constexpr lower_limit = static_cast<From>(std::numeric_limits<To>::lowest());
    auto constexpr upper_limit = static_cast<From>(std::numeric_limits<To>::max());
    if (x < lower_limit) {
      std::ostringstream ss;
      ss << "Input must be at least " << lower_limit << ".";
      throw floating_point_truncation_error{ss.str()};
    }
    if (x > upper_limit) {
      std::ostringstream ss;
      ss << "Input must be at most " << upper_limit << ".";
      throw floating_point_truncation_error{ss.str()};
    }
    return static_cast<To>(x);
  }
}

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
    auto constexpr const bin_width =
      typename node_type::index_type{sizeof(typename node_type::index_type) * 8};
    auto node_value  = typename node_type::index_type{};
    auto set_storage = &node_value;

    // Check invariants for data types
    using cat_t   = typename std::iterator_traits<iter_t>::value_type;
    using index_t = typename node_type::index_type;
    static_assert(std::is_same_v<cat_t, std::uint32_t>, "Category value must be uint32_t");
    static_assert(std::is_same_v<index_t, std::uint32_t> || std::is_same_v<index_t, std::uint64_t>,
                  "Index type in tree node must be either uint32_t or uint64_t");

    // Ensure that (max_cat + 1) can be represented as index_t to prevent integer overflow.
    auto max_cat = (vec_begin != vec_end) ? *std::max_element(vec_begin, vec_end) : cat_t{0};
    if constexpr (std::is_same_v<cat_t, index_t>) {
      if (max_cat == std::numeric_limits<index_t>::max()) {
        throw model_import_error{std::string{"Category index must be at most "} +
                                 std::to_string(std::numeric_limits<index_t>::max() - 1)};
      }
    }
    auto max_cat_plus_one = static_cast<index_t>(max_cat) + index_t{1};

    if (max_num_categories_ > bin_width) {
      node_value         = categorical_storage_.size();
      auto bins_required = raft_proto::ceildiv(max_cat_plus_one, bin_width);
      categorical_storage_.push_back(max_cat_plus_one);
      categorical_storage_.resize(categorical_storage_.size() + bins_required);
      set_storage = &(categorical_storage_[node_value + 1]);
    }
    auto set = bitset{set_storage, max_cat_plus_one};
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
  /* Set the bias term, which is added to the output. The bias term
   * should have the same length as output_size. */
  void set_bias(std::vector<double> val)
  {
    bias_.resize(val.size());
    std::transform(val.begin(), val.end(), bias_.begin(), [](double e) {
      return static_cast<typename node_type::threshold_type>(e);
    });
  }
  /* Set the value of the constant used in the postprocessing operation
   * (if any) */
  void set_postproc_constant(double val) { postproc_constant_ = val; }
  /* Set the number of outputs per row for this model */
  void set_output_size(index_type val)
  {
    if (output_size_ != index_type{1} && output_size_ != val) {
      throw unusable_model_exception("Inconsistent leaf vector size");
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
      postproc_constant_{},
      nodes_{},
      root_node_indexes_{},
      vector_output_{},
      bias_{}
  {
  }

  /* Return the FIL decision forest built by this builder */
  auto get_decision_forest(index_type num_feature,
                           index_type num_class,
                           raft_proto::device_type mem_type = raft_proto::device_type::cpu,
                           int device                       = 0,
                           raft_proto::cuda_stream stream   = raft_proto::cuda_stream{})
  {
    // Validate forest invariants the inference kernel relies on. After this
    // function returns, the forest is treated as trusted by the kernel.

    // tree_index arithmetic in the kernel uses index_type, so the tree count
    // must fit without narrowing.
    if (root_node_indexes_.size() > std::numeric_limits<index_type>::max()) {
      throw model_import_error{std::string{"Forest has "} +
                               std::to_string(root_node_indexes_.size()) +
                               " trees, which exceeds the maximum representable in index_type (" +
                               std::to_string(std::numeric_limits<index_type>::max()) + ")"};
    }

    // forest::get_tree_root(tree_index) dereferences nodes_ + root_index.
    // Ensure each root index points into the nodes buffer.
    for (auto i = std::size_t{0}; i < root_node_indexes_.size(); ++i) {
      if (root_node_indexes_[i] >= nodes_.size()) {
        throw model_import_error{
          std::string{"Tree "} + std::to_string(i) + ": root node index out of bounds (" +
          std::to_string(root_node_indexes_[i]) + " >= " + std::to_string(nodes_.size()) + ")"};
      }
    }

    auto constexpr const cat_bin_width =
      typename node_type::index_type{sizeof(typename node_type::index_type) * 8};
    if (max_num_categories_ > cat_bin_width) {
      auto const storage_size = categorical_storage_.size();
      for (auto i = std::size_t{0}; i < nodes_.size(); ++i) {
        auto const& n = nodes_[i];
        if (n.is_leaf() || !n.is_categorical()) { continue; }
        auto const offset = n.index();

        // evaluate_tree_impl() reads categorical_storage[offset] as the number
        // of categories for this node; offset must be in-range.
        if (offset >= storage_size) {
          throw model_import_error{std::string{"Categorical node "} + std::to_string(i) +
                                   ": storage offset out of bounds (" + std::to_string(offset) +
                                   " >= " + std::to_string(storage_size) + ")"};
        }
        auto const stored_num_cats = categorical_storage_[offset];
        auto const bins_required   = raft_proto::ceildiv(stored_num_cats, cat_bin_width);

        // evaluate_tree_impl() reconstructs a bitset from
        // [offset + 1, offset + 1 + bins_required). Compute this range using
        // size_t to keep the arithmetic explicit and overflow-safe.
        auto const bits_begin = static_cast<std::size_t>(offset) + std::size_t{1};
        auto const bits_end   = bits_begin + static_cast<std::size_t>(bins_required);
        if (bits_end > storage_size) {
          throw model_import_error{std::string{"Categorical node "} + std::to_string(i) +
                                   ": bitset extends past categorical_storage end"};
        }
      }
    }

    // Safely cast average_factor_ and postproc_constant_ to node_type::threshold_type
    auto average_factor_casted    = typename node_type::threshold_type{};
    auto postproc_constant_casted = typename node_type::threshold_type{};
    try {
      average_factor_casted =
        safe_cast_floating_point<typename node_type::threshold_type>(average_factor_);
      // We can't use cuda::narrow here, because it throws for imprecise conversion, i.e. casting
      // double{3.1} to float.
    } catch (const floating_point_truncation_error& e) {
      throw unusable_model_exception{std::string{"Found an invalid value for averaging factor: "} +
                                     e.what()};
    }
    try {
      postproc_constant_casted =
        safe_cast_floating_point<typename node_type::threshold_type>(postproc_constant_);
    } catch (const floating_point_truncation_error& e) {
      throw unusable_model_exception{
        std::string{"Found an invalid value for postprocessing constant: "} + e.what()};
    }
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
      raft_proto::buffer{raft_proto::buffer{bias_.data(), bias_.size()}, mem_type, device, stream},
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
      average_factor_casted,
      postproc_constant_casted};
  }

 private:
  index_type cur_node_index_;
  index_type max_num_categories_;
  index_type alignment_;
  index_type output_size_;
  row_op row_postproc_;
  element_op element_postproc_;
  double average_factor_;
  double postproc_constant_;

  std::vector<node_type> nodes_;
  std::vector<index_type> root_node_indexes_;
  std::vector<typename node_type::threshold_type> vector_output_;
  std::vector<typename node_type::threshold_type> bias_;
  std::vector<typename node_type::index_type> categorical_storage_;
  std::vector<index_type> node_id_mapping_;
};

}  // namespace detail
}  // namespace fil
}  // namespace ML
