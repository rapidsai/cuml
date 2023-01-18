#pragma once
#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include <cstddef>
#include <cuml/experimental/fil/constants.hpp>
#include <cuml/experimental/fil/postproc_ops.hpp>
#include <cuml/experimental/fil/detail/device_initialization.hpp>
#include <cuml/experimental/fil/detail/index_type.hpp>
#include <cuml/experimental/fil/detail/infer.hpp>
#include <cuml/experimental/fil/detail/postprocessor.hpp>
#include <cuml/experimental/fil/detail/specialization_types.hpp>
#include <cuml/experimental/fil/exceptions.hpp>
#include <cuml/experimental/fil/detail/forest.hpp>
#include <cuml/experimental/kayak/buffer.hpp>
#include <cuml/experimental/kayak/cuda_stream.hpp>
#include <cuml/experimental/kayak/exceptions.hpp>
#include <cuml/experimental/kayak/tree_layout.hpp>
#include <limits>
#include <optional>
#include <raft/core/nvtx.hpp>
#include <variant>

namespace ML {
namespace experimental {
namespace fil {

template <kayak::tree_layout layout_v, typename threshold_t, typename index_t, typename metadata_storage_t, typename offset_t>
struct decision_forest {

  auto constexpr static const layout = layout_v;
  using forest_type = forest<
    layout,
    threshold_t,
    index_t,
    metadata_storage_t,
    offset_t
  >;
  using node_type = typename forest_type::node_type;
  using io_type = typename forest_type::io_type;
  using threshold_type = threshold_t;
  using postprocessor_type = postprocessor<io_type>;
  using categorical_storage_type = typename node_type::index_type;

  decision_forest() :
    nodes_{},
    root_node_indexes_{},
    vector_output_{},
    categorical_storage_{},
    num_feature_{},
    num_class_{},
    leaf_size_{},
    has_categorical_nodes_{false},
    row_postproc_{},
    elem_postproc_{},
    average_factor_{},
    bias_{},
    postproc_constant_{} {}

  decision_forest(
    kayak::buffer<node_type>&& nodes,
    kayak::buffer<index_type>&& root_node_indexes,
    index_type num_feature,
    index_type num_class=index_type{2},
    bool has_categorical_nodes = false,
    std::optional<kayak::buffer<io_type>>&& vector_output=std::nullopt,
    std::optional<kayak::buffer<typename node_type::index_type>>&& categorical_storage=std::nullopt,
    index_type leaf_size=index_type{1},
    row_op row_postproc=row_op::disable,
    element_op elem_postproc=element_op::disable,
    io_type average_factor=io_type{1},
    io_type bias=io_type{0},
    io_type postproc_constant=io_type{1}
  ) :
    nodes_{nodes},
    root_node_indexes_{root_node_indexes},
    vector_output_{vector_output},
    categorical_storage_{categorical_storage},
    num_feature_{num_feature},
    num_class_{num_class},
    leaf_size_{leaf_size},
    has_categorical_nodes_{has_categorical_nodes},
    row_postproc_{row_postproc},
    elem_postproc_{elem_postproc},
    average_factor_{average_factor},
    bias_{bias},
    postproc_constant_{postproc_constant}
  {
    if (nodes.memory_type() != root_node_indexes.memory_type()) {
      throw kayak::mem_type_mismatch(
        "Nodes and indexes of forest must both be stored on either host or device"
      );
    }
    if (nodes.device_index() != root_node_indexes.device_index()) {
      throw kayak::mem_type_mismatch(
        "Nodes and indexes of forest must both be stored on same device"
      );
    }
    detail::initialize_device<forest_type>(nodes.device());
  }

  auto num_feature() const { return num_feature_; }
  auto num_outputs() const { return num_class_; }

  auto memory_type() {
    return nodes_.memory_type();
  }
  auto device_index() {
    return nodes_.device_index();
  }

  void predict(
    kayak::buffer<typename forest_type::io_type>& output,
    kayak::buffer<typename forest_type::io_type> const& input,
    kayak::cuda_stream stream = kayak::cuda_stream{},
    std::optional<index_type> specified_rows_per_block_iter=std::nullopt
  ) {
    auto nvtx_range = raft::common::nvtx::range{"decision_forest.predict"};
    if (output.memory_type() != memory_type() || input.memory_type() != memory_type()) {
      throw kayak::wrong_device_type{
        "Tried to use host I/O data with model on device or vice versa"
      };
    }
    if (output.device_index() != device_index() || input.device_index() != device_index()) {
      throw kayak::wrong_device{
        "I/O data on different device than model"
      };
    }
    auto* vector_output_data = (
      vector_output_.has_value() ? vector_output_->data() : static_cast<io_type*>(nullptr)
    );
    auto* categorical_storage_data = (
      categorical_storage_.has_value() ? categorical_storage_->data() : static_cast<categorical_storage_type*>(nullptr)
    );
    switch(nodes_.device().index()) {
      case 0:
        fil::detail::infer(
          obj(),
          get_postprocessor(),
          output.data(),
          input.data(),
          index_type(input.size() / num_feature_),
          num_feature_,
          num_class_,
          has_categorical_nodes_,
          vector_output_data,
          categorical_storage_data,
          specified_rows_per_block_iter,
          std::get<0>(nodes_.device()),
          stream
        );
        break;
      case 1:
        fil::detail::infer(
          obj(),
          get_postprocessor(),
          output.data(),
          input.data(),
          index_type(input.size() / num_feature_),
          num_feature_,
          num_class_,
          has_categorical_nodes_,
          vector_output_data,
          categorical_storage_data,
          specified_rows_per_block_iter,
          std::get<1>(nodes_.device()),
          stream
        );
        break;
    }
  }

 private:
  /** The nodes for all trees in the forest */
  kayak::buffer<node_type> nodes_;
  /** The index of the root node for each tree in the forest */
  kayak::buffer<index_type> root_node_indexes_;
  /** Buffer of outputs for all leaves in vector-leaf models */
  std::optional<kayak::buffer<io_type>> vector_output_;
  /** Buffer of outputs for all leaves in vector-leaf models */
  std::optional<kayak::buffer<categorical_storage_type>> categorical_storage_;

  // Metadata
  index_type num_feature_;
  index_type num_class_;
  index_type leaf_size_;
  bool has_categorical_nodes_ = false;
  // Postprocessing constants
  row_op row_postproc_;
  element_op elem_postproc_;
  io_type average_factor_;
  io_type bias_;
  io_type postproc_constant_;

  auto obj() const {
    return forest_type{
      nodes_.data(),
      root_node_indexes_.data(),
      static_cast<index_type>(root_node_indexes_.size())
    };
  }

  auto get_postprocessor() const {
    return postprocessor_type {
      row_postproc_,
      elem_postproc_,
      average_factor_,
      bias_,
      postproc_constant_
    };
  }

  auto leaf_size() const { return leaf_size_; }
};

namespace detail {
template<
  kayak::tree_layout layout,
  bool double_precision,
  bool large_trees
>
using preset_decision_forest = decision_forest<
  layout,
  typename specialization_types<double_precision, large_trees>::threshold_type,
  typename specialization_types<double_precision, large_trees>::index_type,
  typename specialization_types<double_precision, large_trees>::metadata_type,
  typename specialization_types<double_precision, large_trees>::offset_type
>;

}

using decision_forest_variant = std::variant<
  detail::preset_decision_forest<preferred_tree_layout, false, false>,
  detail::preset_decision_forest<preferred_tree_layout, false, true>,
  detail::preset_decision_forest<preferred_tree_layout, true, false>,
  detail::preset_decision_forest<preferred_tree_layout, true, true>
>;

inline auto get_forest_variant_index(
  bool use_double_thresholds,
  index_type max_node_offset,
  index_type num_features,
  index_type num_categorical_nodes = index_type{},
  index_type max_num_categories = index_type{},
  index_type num_vector_leaves = index_type{}
) {
  using small_index_t = typename detail::specialization_types<false, false>::index_type;
  auto max_local_categories = index_type(sizeof(small_index_t) * 8);
  // If the index required for pointing to categorical storage bins or vector
  // leaf output exceeds what we can store in a uint32_t, uint64_t will be used
  //
  // TODO(wphicks): We are overestimating categorical storage required here
  auto double_indexes_required = (
    max_num_categories > max_local_categories
    && (
      (
        kayak::ceildiv(max_num_categories, max_local_categories) + 1
        * num_categorical_nodes
      ) > std::numeric_limits<small_index_t>::max()
    )
  ) || num_vector_leaves > std::numeric_limits<small_index_t>::max();

  auto double_precision = use_double_thresholds || double_indexes_required;

  using small_metadata_t = typename detail::specialization_types<false, false>::metadata_type;
  using small_offset_t = typename detail::specialization_types<false, false>::offset_type;

  auto large_trees = (
    num_features > (
      std::numeric_limits<small_metadata_t>::max() >> reserved_node_metadata_bits
    ) || max_node_offset > std::numeric_limits<small_offset_t>::max()
  );

  return (
    (index_type{double_precision} << index_type{1})
    + index_type{large_trees}
  );
}
}
}
}
