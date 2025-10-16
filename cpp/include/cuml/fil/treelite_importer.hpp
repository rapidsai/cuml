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
#include <cuml/fil/decision_forest.hpp>
#include <cuml/fil/detail/decision_forest_builder.hpp>
#include <cuml/fil/detail/degenerate_trees.hpp>
#include <cuml/fil/detail/index_type.hpp>
#include <cuml/fil/exceptions.hpp>
#include <cuml/fil/forest_model.hpp>
#include <cuml/fil/postproc_ops.hpp>
#include <cuml/fil/tree_layout.hpp>
#include <cuml/forest/integrations/treelite.hpp>

#include <raft/core/error.hpp>

#include <treelite/c_api.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/tree_node_type.h>
#include <treelite/enum/typeinfo.h>
#include <treelite/tree.h>

#include <cmath>
#include <variant>

namespace ML {
namespace fil {

namespace detail {

struct postproc_params_t {
  element_op element = element_op::disable;
  row_op row         = row_op::disable;
  double constant    = 1.0;
};
}  // namespace detail

/**
 * Struct used to import a model from Treelite to FIL
 *
 * @tparam layout The in-memory layout for nodes to be loaded into FIL
 */
template <tree_layout layout>
struct treelite_importer {
  auto static constexpr const traversal_order = []() constexpr {
    if constexpr (layout == tree_layout::depth_first) {
      return ML::forest::forest_order::depth_first;
    } else if constexpr (layout == tree_layout::breadth_first) {
      return ML::forest::forest_order::breadth_first;
    } else if constexpr (layout == tree_layout::layered_children_together) {
      return ML::forest::forest_order::layered_children_together;
    } else {
      static_assert(layout == tree_layout::depth_first,
                    "Layout not yet implemented in treelite importer for FIL");
    }
  }();

  auto get_node_count(treelite::Model const& tl_model)
  {
    return ML::forest::tree_accumulate(
      tl_model, index_type{}, [](auto&& count, auto&& tree) { return count + tree.num_nodes; });
  }

  /* Return vector of offsets between each node and its most distant child */
  auto get_offsets(treelite::Model const& tl_model)
  {
    auto node_count     = get_node_count(tl_model);
    auto result         = std::vector<index_type>(node_count);
    auto parent_indexes = std::vector<index_type>{};
    parent_indexes.reserve(node_count);
    ML::forest::node_transform<traversal_order>(
      tl_model,
      std::back_inserter(parent_indexes),
      [](auto&& tree_id, auto&& node, auto&& depth, auto&& parent_index) { return parent_index; });
    for (auto i = std::size_t{}; i < node_count; ++i) {
      result[parent_indexes[i]] = i - parent_indexes[i];
    }
    return result;
  }

  auto num_trees(treelite::Model const& tl_model)
  {
    auto result = index_type{};
    std::visit([&result](auto&& concrete_tl_model) { result = concrete_tl_model.trees.size(); },
               tl_model.variant_);
    return result;
  }

  auto get_tree_sizes(treelite::Model const& tl_model)
  {
    auto result = std::vector<index_type>{};
    tree_transform(
      tl_model, std::back_inserter(result), [](auto&& tree) { return tree.num_nodes; });
    return result;
  }

  auto get_num_class(treelite::Model const& tl_model)
  {
    return static_cast<index_type>(tl_model.num_class[0]);
  }

  auto get_num_feature(treelite::Model const& tl_model)
  {
    return static_cast<index_type>(tl_model.num_feature);
  }

  auto get_max_num_categories(treelite::Model const& tl_model)
  {
    return ML::forest::node_accumulate<traversal_order>(
      tl_model,
      index_type{},
      [](auto&& cur_accum, auto&& tree_id, auto&& node, auto&& depth, auto&& parent_index) {
        return std::max(cur_accum, static_cast<index_type>(node.max_num_categories()));
      });
  }

  auto get_num_categorical_nodes(treelite::Model const& tl_model)
  {
    return ML::forest::node_accumulate<traversal_order>(
      tl_model,
      index_type{},
      [](auto&& cur_accum, auto&& tree_id, auto&& node, auto&& depth, auto&& parent_index) {
        return cur_accum + static_cast<index_type>(node.is_categorical());
      });
  }

  auto get_num_leaf_vector_nodes(treelite::Model const& tl_model)
  {
    return ML::forest::node_accumulate<traversal_order>(
      tl_model,
      index_type{},
      [](auto&& cur_accum, auto&& tree_id, auto&& node, auto&& depth, auto&& parent_index) {
        auto accum = cur_accum;
        if (node.is_leaf() && node.get_output().size() > 1) { ++accum; }
        return accum;
      });
  }

  auto get_average_factor(treelite::Model const& tl_model)
  {
    auto result = double{};
    if (tl_model.average_tree_output) {
      if (tl_model.task_type == treelite::TaskType::kMultiClf &&
          tl_model.leaf_vector_shape[1] == 1) {  // grove-per-class
        result = num_trees(tl_model) / tl_model.num_class[0];
      } else {
        result = num_trees(tl_model);
      }
    } else {
      result = 1.0;
    }
    return result;
  }

  auto get_bias(treelite::Model const& tl_model)
  {
    return static_cast<double>(tl_model.base_scores[0]);
  }

  auto get_postproc_params(treelite::Model const& tl_model)
  {
    auto result            = detail::postproc_params_t{};
    auto tl_pred_transform = tl_model.postprocessor;
    if (tl_pred_transform == std::string{"identity"} ||
        tl_pred_transform == std::string{"identity_multiclass"}) {
      result.element = element_op::disable;
      result.row     = row_op::disable;
    } else if (tl_pred_transform == std::string{"signed_square"}) {
      result.element = element_op::signed_square;
    } else if (tl_pred_transform == std::string{"hinge"}) {
      result.element = element_op::hinge;
    } else if (tl_pred_transform == std::string{"sigmoid"}) {
      result.constant = tl_model.sigmoid_alpha;
      result.element  = element_op::sigmoid;
    } else if (tl_pred_transform == std::string{"exponential"}) {
      result.element = element_op::exponential;
    } else if (tl_pred_transform == std::string{"exponential_standard_ratio"}) {
      result.constant = -tl_model.ratio_c / std::log(2);
      result.element  = element_op::exponential;
    } else if (tl_pred_transform == std::string{"logarithm_one_plus_exp"}) {
      result.element = element_op::logarithm_one_plus_exp;
    } else if (tl_pred_transform == std::string{"max_index"}) {
      result.row = row_op::max_index;
    } else if (tl_pred_transform == std::string{"softmax"}) {
      result.row = row_op::softmax;
    } else if (tl_pred_transform == std::string{"multiclass_ova"}) {
      result.constant = tl_model.sigmoid_alpha;
      result.element  = element_op::sigmoid;
    } else {
      throw model_import_error{"Unrecognized Treelite pred_transform string"};
    }
    return result;
  }

  auto uses_double_thresholds(treelite::Model const& tl_model)
  {
    auto result = false;
    switch (tl_model.GetThresholdType()) {
      case treelite::TypeInfo::kFloat64: result = true; break;
      case treelite::TypeInfo::kFloat32: result = false; break;
      default: throw model_import_error("Unrecognized Treelite threshold type");
    }
    return result;
  }

  auto uses_double_outputs(treelite::Model const& tl_model)
  {
    auto result = false;
    switch (tl_model.GetThresholdType()) {
      case treelite::TypeInfo::kFloat64: result = true; break;
      case treelite::TypeInfo::kFloat32: result = false; break;
      case treelite::TypeInfo::kUInt32: result = false; break;
      default: throw model_import_error("Unrecognized Treelite threshold type");
    }
    return result;
  }

  auto uses_integer_outputs(treelite::Model const& tl_model)
  {
    auto result = false;
    switch (tl_model.GetThresholdType()) {
      case treelite::TypeInfo::kFloat64: result = false; break;
      case treelite::TypeInfo::kFloat32: result = false; break;
      case treelite::TypeInfo::kUInt32: result = true; break;
      default: throw model_import_error("Unrecognized Treelite threshold type");
    }
    return result;
  }

  /**
   * Assuming that the correct decision_forest variant has been
   * identified, import to that variant
   */
  template <index_type variant_index>
  auto import_to_specific_variant(index_type target_variant_index,
                                  treelite::Model const& tl_model,
                                  index_type num_class,
                                  index_type num_feature,
                                  index_type max_num_categories,
                                  std::vector<index_type> const& offsets,
                                  index_type align_bytes           = index_type{},
                                  raft_proto::device_type mem_type = raft_proto::device_type::cpu,
                                  int device                       = 0,
                                  raft_proto::cuda_stream stream   = raft_proto::cuda_stream{})
  {
    auto result = decision_forest_variant{};
    if constexpr (variant_index != std::variant_size_v<decision_forest_variant>) {
      if (variant_index == target_variant_index) {
        using forest_model_t = std::variant_alternative_t<variant_index, decision_forest_variant>;
        if constexpr (traversal_order == ML::forest::forest_order::layered_children_together) {
          // Cannot align whole trees with layered traversal order, since trees
          // are mingled together
          align_bytes = index_type{};
        }
        auto builder =
          detail::decision_forest_builder<forest_model_t>(max_num_categories, align_bytes);
        auto node_index = index_type{};
        ML::forest::node_for_each<traversal_order>(
          tl_model,
          [&builder, &offsets, &node_index](
            auto&& tree_id, auto&& node, auto&& depth, auto&& parent_index) {
            if (node.is_leaf()) {
              auto output = node.get_output();
              builder.set_output_size(output.size());
              if (output.size() > index_type{1}) {
                builder.add_leaf_vector_node(
                  std::begin(output), std::end(output), node.get_treelite_id(), depth);
              } else {
                builder.add_node(
                  typename forest_model_t::io_type(output[0]), node.get_treelite_id(), depth, true);
              }
            } else {
              if (node.is_categorical()) {
                auto categories = node.get_categories();
                builder.add_categorical_node(std::begin(categories),
                                             std::end(categories),
                                             node.get_treelite_id(),
                                             depth,
                                             node.default_distant(),
                                             node.get_feature(),
                                             offsets[node_index]);
              } else {
                builder.add_node(typename forest_model_t::threshold_type(node.threshold()),
                                 node.get_treelite_id(),
                                 depth,
                                 false,
                                 node.default_distant(),
                                 false,
                                 node.get_feature(),
                                 offsets[node_index],
                                 node.is_inclusive());
              }
            }
            ++node_index;
          });

        builder.set_average_factor(get_average_factor(tl_model));
        builder.set_bias(get_bias(tl_model));
        auto postproc_params = get_postproc_params(tl_model);
        builder.set_element_postproc(postproc_params.element);
        builder.set_row_postproc(postproc_params.row);
        builder.set_postproc_constant(postproc_params.constant);

        result.template emplace<variant_index>(
          builder.get_decision_forest(num_feature, num_class, mem_type, device, stream));
      } else {
        result = import_to_specific_variant<variant_index + 1>(target_variant_index,
                                                               tl_model,
                                                               num_class,
                                                               num_feature,
                                                               max_num_categories,
                                                               offsets,
                                                               align_bytes,
                                                               mem_type,
                                                               device,
                                                               stream);
      }
    }
    return result;
  }

  /**
   * Import a treelite model to FIL
   *
   * Load a model from Treelite to a FIL forest_model. The model will be
   * inspected to determine the correct underlying decision_forest variant to
   * use within the forest_model object.
   *
   * @param tl_model The Treelite Model to load
   * @param align_bytes If non-zero, ensure that each tree is stored in a
   * multiple of this value of bytes by padding with empty nodes. This can
   * be useful for increasing the likelihood that successive reads will take
   * place within a single cache line. On GPU, a value of 128 can be used for
   * this purpose. On CPU, a value of either 0 or 64 typically produces
   * optimal performance.
   * @param use_double_precision Whether or not to use 64 bit floats for model
   * evaluation and 64 bit ints for applicable indexing
   * @param dev_type Which device type to use for inference (CPU or GPU)
   * @param device For GPU execution, the device id for the device on which this
   * model is to be loaded
   * @param stream The CUDA stream to use for loading this model (can be
   * omitted for CPU).
   */
  forest_model import(treelite::Model const& tl_model,
                      index_type align_bytes                   = index_type{},
                      std::optional<bool> use_double_precision = std::nullopt,
                      raft_proto::device_type dev_type         = raft_proto::device_type::cpu,
                      int device                               = 0,
                      raft_proto::cuda_stream stream           = raft_proto::cuda_stream{})
  {
    // Handle degenerate trees (a single root node with no child)
    if (auto processed_tl_model = detail::convert_degenerate_trees(tl_model); processed_tl_model) {
      return import(
        *processed_tl_model.get(), align_bytes, use_double_precision, dev_type, device, stream);
    }

    ASSERT(tl_model.num_target == 1, "FIL does not support multi-target model");
    // Check tree annotation (assignment)
    if (tl_model.task_type == treelite::TaskType::kMultiClf) {
      // Must be either vector leaf or grove-per-class
      if (tl_model.leaf_vector_shape[1] > 1) {  // vector-leaf
        ASSERT(tl_model.leaf_vector_shape[1] == int(tl_model.num_class[0]),
               "Vector leaf must be equal to num_class = %d",
               tl_model.num_class[0]);
        auto tree_count = num_trees(tl_model);
        for (decltype(tree_count) tree_id = 0; tree_id < tree_count; ++tree_id) {
          ASSERT(tl_model.class_id[tree_id] == -1, "Tree %d has invalid class assignment", tree_id);
        }
      } else {  // grove-per-class
        auto tree_count = num_trees(tl_model);
        for (decltype(tree_count) tree_id = 0; tree_id < tree_count; ++tree_id) {
          ASSERT(tl_model.class_id[tree_id] == int(tree_id % tl_model.num_class[0]),
                 "Tree %d has invalid class assignment",
                 tree_id);
        }
      }
    }
    // Check base_scores
    for (std::int32_t class_id = 1; class_id < tl_model.num_class[0]; ++class_id) {
      ASSERT(tl_model.base_scores[0] == tl_model.base_scores[class_id],
             "base_scores must be identical for all classes");
    }

    auto result                = decision_forest_variant{};
    auto num_feature           = get_num_feature(tl_model);
    auto max_num_categories    = get_max_num_categories(tl_model);
    auto num_categorical_nodes = get_num_categorical_nodes(tl_model);
    auto num_leaf_vector_nodes = get_num_leaf_vector_nodes(tl_model);
    auto use_double_thresholds = use_double_precision.value_or(uses_double_thresholds(tl_model));

    auto offsets    = get_offsets(tl_model);
    auto max_offset = *std::max_element(std::begin(offsets), std::end(offsets));

    auto variant_index = get_forest_variant_index(use_double_thresholds,
                                                  max_offset,
                                                  num_feature,
                                                  num_categorical_nodes,
                                                  max_num_categories,
                                                  num_leaf_vector_nodes,
                                                  layout);
    auto num_class     = get_num_class(tl_model);
    return forest_model{import_to_specific_variant<index_type{}>(variant_index,
                                                                 tl_model,
                                                                 num_class,
                                                                 num_feature,
                                                                 max_num_categories,
                                                                 offsets,
                                                                 align_bytes,
                                                                 dev_type,
                                                                 device,
                                                                 stream)};
  }
};

/**
 * Import a treelite model to FIL
 *
 * Load a model from Treelite to a FIL forest_model. The model will be
 * inspected to determine the correct underlying decision_forest variant to
 * use within the forest_model object.
 *
 * @param tl_model The Treelite Model to load
 * @param layout The in-memory layout of nodes in the loaded forest
 * @param align_bytes If non-zero, ensure that each tree is stored in a
 * multiple of this value of bytes by padding with empty nodes. This can
 * be useful for increasing the likelihood that successive reads will take
 * place within a single cache line. On GPU, a value of 128 can be used for
 * this purpose. On CPU, a value of either 0 or 64 typically produces
 * optimal performance.
 * @param use_double_precision Whether or not to use 64 bit floats for model
 * evaluation and 64 bit ints for applicable indexing
 * @param dev_type Which device type to use for inference (CPU or GPU)
 * @param device For GPU execution, the device id for the device on which this
 * model is to be loaded
 * @param stream The CUDA stream to use for loading this model (can be
 * omitted for CPU).
 */
auto import_from_treelite_model(treelite::Model const& tl_model,
                                tree_layout layout                       = preferred_tree_layout,
                                index_type align_bytes                   = index_type{},
                                std::optional<bool> use_double_precision = std::nullopt,
                                raft_proto::device_type dev_type = raft_proto::device_type::cpu,
                                int device                       = 0,
                                raft_proto::cuda_stream stream   = raft_proto::cuda_stream{})
{
  auto result = forest_model{};
  switch (layout) {
    case tree_layout::depth_first:
      result = treelite_importer<tree_layout::depth_first>{}.import(
        tl_model, align_bytes, use_double_precision, dev_type, device, stream);
      break;
    case tree_layout::breadth_first:
      result = treelite_importer<tree_layout::breadth_first>{}.import(
        tl_model, align_bytes, use_double_precision, dev_type, device, stream);
      break;
    case tree_layout::layered_children_together:
      result = treelite_importer<tree_layout::layered_children_together>{}.import(
        tl_model, align_bytes, use_double_precision, dev_type, device, stream);
      break;
  }
  return result;
}

/**
 * Import a treelite model handle to FIL
 *
 * Load a model from a Treelite model handle (type-erased treelite::Model
 * object) to a FIL forest_model. The model will be inspected to determine the
 * correct underlying decision_forest variant to use within the forest_model
 * object.
 *
 * @param tl_handle The Treelite ModelHandle to load
 * @param layout The in-memory layout of nodes in the loaded forest
 * @param align_bytes If non-zero, ensure that each tree is stored in a
 * multiple of this value of bytes by padding with empty nodes. This can
 * be useful for increasing the likelihood that successive reads will take
 * place within a single cache line. On GPU, a value of 128 can be used for
 * this purpose. On CPU, a value of either 0 or 64 typically produces
 * optimal performance.
 * @param use_double_precision Whether or not to use 64 bit floats for model
 * evaluation and 64 bit ints for applicable indexing
 * @param dev_type Which device type to use for inference (CPU or GPU)
 * @param device For GPU execution, the device id for the device on which this
 * model is to be loaded
 * @param stream The CUDA stream to use for loading this model (can be
 * omitted for CPU).
 */
auto import_from_treelite_handle(TreeliteModelHandle tl_handle,
                                 tree_layout layout                       = preferred_tree_layout,
                                 index_type align_bytes                   = index_type{},
                                 std::optional<bool> use_double_precision = std::nullopt,
                                 raft_proto::device_type dev_type = raft_proto::device_type::cpu,
                                 int device                       = 0,
                                 raft_proto::cuda_stream stream   = raft_proto::cuda_stream{})
{
  return import_from_treelite_model(*static_cast<treelite::Model*>(tl_handle),
                                    layout,
                                    align_bytes,
                                    use_double_precision,
                                    dev_type,
                                    device,
                                    stream);
}

}  // namespace fil
}  // namespace ML
