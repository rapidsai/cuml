#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdint.h>
#include <numeric>
#include <optional>
#include <vector>
#include <cuml/experimental/fil/postproc_ops.hpp>
#include <cuml/experimental/fil/detail/forest.hpp>
#include <cuml/experimental/fil/detail/index_type.hpp>
#include <cuml/experimental/fil/exceptions.hpp>
#include <cuml/experimental/kayak/buffer.hpp>
#include <cuml/experimental/kayak/bitset.hpp>
#include <cuml/experimental/kayak/ceildiv.hpp>
#include <cuml/experimental/kayak/cuda_stream.hpp>
#include <cuml/experimental/kayak/device_type.hpp>

namespace ML {
namespace experimental {
namespace fil {
namespace detail {

struct model_builder_error : std::exception {
  model_builder_error() : model_builder_error("Error while building model") {}
  model_builder_error(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

template<typename decision_forest_t>
struct decision_forest_builder {

  using node_type = typename decision_forest_t::node_type;

  void start_new_tree() {
    if (root_node_indexes_.size() == index_type{}) {
      root_node_indexes_.emplace_back();
    } else {
      max_tree_size_ = std::max(cur_tree_size_, max_tree_size_);
      if (alignment_ != index_type{}) {
        if (cur_tree_size_ % alignment_ != index_type{}) {
          auto padding = (alignment_ - cur_tree_size_ % alignment_);
          for (auto i = index_type{}; i < padding; ++i) {
            add_node(typename node_type::threshold_type{});
          }
        }
      }
      root_node_indexes_.push_back(
        root_node_indexes_.back() + cur_tree_size_
      );
      cur_tree_size_ = index_type{};
    }
  }

  template<typename iter_t>
  void add_categorical_node(
    iter_t vec_begin,
    iter_t vec_end,
    bool default_to_distant_child=false,
    typename node_type::metadata_storage_type feature = typename node_type::metadata_storage_type{},
    typename node_type::offset_type offset = typename node_type::offset_type{}
  ) {
    auto constexpr const bin_width = index_type(sizeof(typename node_type::index_type) * 8);
    auto node_value = typename node_type::index_type{};
    auto set_storage = &node_value;
    auto max_node_categories = *std::max_element(vec_begin, vec_end) + 1;
    if (max_num_categories_ > bin_width) {
      // TODO(wphicks): Check for overflow here
      node_value = categorical_storage_.size();
      auto bins_required = kayak::ceildiv(max_node_categories, bin_width);
      categorical_storage_.push_back(max_node_categories);
      categorical_storage_.resize(categorical_storage_.size() + bins_required);
      set_storage = &(categorical_storage_[node_value + 1]);
    }
    auto set = kayak::bitset{set_storage, max_node_categories};
    std::for_each(
      vec_begin,
      vec_end, 
      [&set](auto&& cat_index) {
        set.set(cat_index);
      }
    );

    add_node(
      node_value,
      false,
      default_to_distant_child,
      true,
      feature,
      offset,
      false
    );
  }

  template<typename iter_t>
  void add_leaf_vector_node(
    iter_t vec_begin,
    iter_t vec_end
  ) {
    auto leaf_index = typename node_type::index_type(vector_output_.size() / output_size_);
    std::copy(vec_begin, vec_end, std::back_inserter(vector_output_));
    nodes_.emplace_back(
      leaf_index,
      true,
      false,
      false,
      typename node_type::metadata_storage_type{},
      typename node_type::offset_type{}
    );
    ++cur_tree_size_;
  }

  template<typename value_t>
  void add_node(
    value_t val,
    bool is_leaf_node=true,
    bool default_to_distant_child=false,
    bool is_categorical_node=false,
    typename node_type::metadata_storage_type feature = typename node_type::metadata_storage_type{},
    typename node_type::offset_type offset = typename node_type::offset_type{},
    bool is_inclusive=false
  ) {
    nodes_.emplace_back(
      val, is_leaf_node, default_to_distant_child, is_categorical_node, feature, offset
    );
    ++cur_tree_size_;
  }

  void set_element_postproc(element_op val) { element_postproc_ = val; }
  void set_row_postproc(row_op val) { row_postproc_ = val; }
  void set_average_factor(double val) { average_factor_ = val; }
  void set_bias(double val) { bias_ = val; }
  void set_postproc_constant(double val) { postproc_constant_ = val; }
  void set_output_size(index_type val) {
    if (output_size_ != index_type{1} && output_size_ != val) {
      throw model_import_error("Inconsistent leaf vector size");
    }
    output_size_ = val;
  }

  decision_forest_builder(
      index_type max_num_categories=index_type{},
      index_type align_bytes=index_type{}
    ) :
    cur_tree_size_{},
    max_num_categories_{max_num_categories},
    alignment_{std::lcm(align_bytes, index_type(sizeof(node_type)))},
    output_size_{1},
    element_postproc_{},
    average_factor_{},
    row_postproc_{},
    bias_{},
    postproc_constant_{},
    max_tree_size_{},
    nodes_{},
    root_node_indexes_{},
    vector_output_{} {
  }

  auto get_decision_forest(
      index_type num_feature,
      index_type num_class,
      kayak::device_type mem_type=kayak::device_type::cpu,
      int device=0,
      kayak::cuda_stream stream=kayak::cuda_stream{}
  ) {

    // Allow narrowing for preprocessing constants. They are stored as doubles
    // for consistency in the builder but must be converted to the proper types
    // for the concrete forest model.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
    return decision_forest_t{
      kayak::buffer{
        kayak::buffer{nodes_.data(), nodes_.size()},
        mem_type,
        device,
        stream
      },
      kayak::buffer{
        kayak::buffer{root_node_indexes_.data(), root_node_indexes_.size()},
        mem_type,
        device,
        stream
      },
      num_feature,
      num_class,
      max_num_categories_ != 0,
      vector_output_.size() == 0 ?
        std::nullopt :
        std::make_optional<kayak::buffer<typename node_type::threshold_type>>(
          kayak::buffer{vector_output_.data(), vector_output_.size()},
          mem_type,
          device,
          stream
        ),
      categorical_storage_.size() == 0 ?
        std::nullopt :
        std::make_optional<kayak::buffer<typename node_type::index_type>>(
          kayak::buffer{categorical_storage_.data(), categorical_storage_.size()},
          mem_type,
          device,
          stream
        ),
      output_size_,
      row_postproc_,
      element_postproc_,
      average_factor_,
      bias_,
      postproc_constant_
    };
#pragma GCC diagnostic pop
  }


 private:
  index_type cur_tree_size_;
  index_type max_num_categories_;
  index_type alignment_;
  index_type output_size_;
  row_op row_postproc_;
  element_op element_postproc_;
  double average_factor_;
  double bias_;
  double postproc_constant_;
  index_type max_tree_size_;

  std::vector<node_type> nodes_;
  std::vector<index_type> root_node_indexes_;
  std::vector<typename node_type::threshold_type> vector_output_;
  std::vector<typename node_type::index_type> categorical_storage_;
};

}
}
}
}
