/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <cmath>
#include <cstddef>
#include <queue>
#include <stack>
#include <treelite/c_api.h>
#include <treelite/tree.h>
#include <treelite/typeinfo.h>
#include <cuml/experimental/fil/constants.hpp>
#include <cuml/experimental/fil/decision_forest.hpp>
#include <cuml/experimental/fil/detail/decision_forest_builder.hpp>
#include <cuml/experimental/fil/detail/index_type.hpp>
#include <cuml/experimental/fil/exceptions.hpp>
#include <cuml/experimental/fil/forest_model.hpp>
#include <cuml/experimental/fil/postproc_ops.hpp>
#include <cuml/experimental/fil/tree_layout.hpp>

namespace ML {
namespace experimental {
namespace fil {

namespace detail {
/** A template for storing nodes in either a depth or breadth-first traversal
 */
template <tree_layout layout, typename T>
struct traversal_container {
  using backing_container_t = std::conditional_t<
    layout == tree_layout::depth_first,
    std::stack<T>,
    std::queue<T>
  >;
  void add(T const& val) {
    data_.push(val);
  }
  void add(T const& hot, T const& distant) {
    if constexpr (layout == tree_layout::depth_first) {
      data_.push(distant);
      data_.push(hot);
    } else {
      data_.push(hot);
      data_.push(distant);
    }
  }
  auto next() {
    if constexpr (std::is_same_v<backing_container_t, std::stack<T>>) {
      auto result = data_.top();
      data_.pop();
      return result;
    } else {
      auto result = data_.front();
      data_.pop();
      return result;
    }
  }
  auto peek() {
    if constexpr (std::is_same_v<backing_container_t, std::stack<T>>) {
      return data_.top();
    } else {
      return data_.front();
    }
  }
  [[nodiscard]] auto empty() {
    return data_.empty();
  }
  auto size() {
    return data_.size();
  }
 private:
  backing_container_t data_;
};

  struct postproc_params_t {
    element_op element = element_op::disable;
    row_op row = row_op::disable;
    double constant = 1.0;
  };
} // namespace detail

/**
 * Struct used to import a model from Treelite to FIL
 *
 * @tparam layout The in-memory layout for nodes to be loaded into FIL
 */
template<tree_layout layout>
struct treelite_importer {
  template<typename tl_threshold_t, typename tl_output_t>
  struct treelite_node {
    treelite::Tree<tl_threshold_t, tl_output_t> const& tree;
    int node_id;
    index_type parent_index;
    index_type own_index;

    auto is_leaf() {
      return tree.IsLeaf(node_id);
    }

    auto get_output() {
      auto result = std::vector<tl_output_t>{};
      if (tree.HasLeafVector(node_id)) {
        result = tree.LeafVector(node_id);
      } else {
        result.push_back(tree.LeafValue(node_id));
      }
      return result;
    }

    auto get_categories() {
      return tree.MatchingCategories(node_id);
    }

    auto get_feature() {
      return tree.SplitIndex(node_id);
    }

    auto is_categorical() {
      return tree.SplitType(node_id) == treelite::SplitFeatureType::kCategorical;
    }

    auto default_distant() {
      auto result = false;
      auto default_child = tree.DefaultChild(node_id);
      if (is_categorical()) {
        if (tree.CategoriesListRightChild(node_id)) {
          result = (default_child == tree.RightChild(node_id));
        } else {
          result = (default_child == tree.LeftChild(node_id));
        }
      } else {
        auto tl_operator = tree.ComparisonOp(node_id);
        if (tl_operator == treelite::Operator::kLT || tl_operator == treelite::Operator::kLE) {
          result = (default_child == tree.LeftChild(node_id));
        } else {
          result = (default_child == tree.RightChild(node_id));
        }
      }
      return result;
    }

    auto threshold() {
      return tree.Threshold(node_id);
    }

    auto categories() {
      auto result = decltype(tree.MatchingCategories(node_id)){};
      if (is_categorical()) {
        result = tree.MatchingCategories(node_id);
      }
      return result;
    }

    auto is_inclusive() {
      auto tl_operator = tree.ComparisonOp(node_id);
      return tl_operator == treelite::Operator::kGT || tl_operator == treelite::Operator::kLE;
    }
  };

  template<typename tl_threshold_t, typename tl_output_t, typename lambda_t>
  void node_for_each(treelite::Tree<tl_threshold_t, tl_output_t> const& tl_tree, lambda_t&& lambda) {
    using node_index_t = decltype(tl_tree.LeftChild(0));
    auto to_be_visited = detail::traversal_container<layout, node_index_t>{};
    to_be_visited.add(node_index_t{});

    auto parent_indices = detail::traversal_container<layout, index_type>{};
    auto cur_index = index_type{};
    parent_indices.add(cur_index);

    while (!to_be_visited.empty()) {
      auto node_id = to_be_visited.next();
      auto remaining_size = to_be_visited.size();

      auto tl_node = treelite_node<tl_threshold_t, tl_output_t>{
        tl_tree, node_id, parent_indices.next(), cur_index
      };
      lambda(tl_node);

      if (!tl_tree.IsLeaf(node_id)) {
        auto tl_left_id = tl_tree.LeftChild(node_id);
        auto tl_right_id = tl_tree.RightChild(node_id);
        auto tl_operator = tl_tree.ComparisonOp(node_id);
        if (!tl_node.is_categorical()) {
          if (tl_operator == treelite::Operator::kLT || tl_operator == treelite::Operator::kLE) {
            to_be_visited.add(tl_right_id, tl_left_id);
          } else if (tl_operator == treelite::Operator::kGT || tl_operator == treelite::Operator::kGE) {
            to_be_visited.add(tl_left_id, tl_right_id);
          } else {
            throw model_import_error("Unrecognized Treelite operator");
          }
        } else {
          if (tl_tree.CategoriesListRightChild(node_id)) {
            to_be_visited.add(tl_left_id, tl_right_id);
          } else {
            to_be_visited.add(tl_right_id, tl_left_id);
          }
        }
        parent_indices.add(cur_index, cur_index);
      }
      ++cur_index;
    }
  }

  template<typename tl_threshold_t, typename tl_output_t, typename iter_t, typename lambda_t>
  void node_transform(treelite::Tree<tl_threshold_t, tl_output_t> const& tl_tree, iter_t output_iter, lambda_t&& lambda) {
    node_for_each(
      tl_tree,
      [&output_iter, &lambda](auto&& tl_node) {
        *output_iter = lambda(tl_node);
        ++output_iter;
      }
    );
  }

  template<typename tl_threshold_t, typename tl_output_t, typename T, typename lambda_t>
  auto node_accumulate(treelite::Tree<tl_threshold_t, tl_output_t> const& tl_tree, T init, lambda_t&& lambda) {
    auto result = init;
    node_for_each(
      tl_tree,
      [&result, &lambda](auto&& tl_node) {
        result = lambda(result, tl_node);
      }
    );
    return result;
  }

  template<typename tl_threshold_t, typename tl_output_t>
  auto get_nodes(treelite::Tree<tl_threshold_t, tl_output_t> const& tl_tree) {
    auto result = std::vector<treelite_node<tl_threshold_t, tl_output_t>>{};
    result.reserve(tl_tree.num_nodes);
    node_transform(tl_tree, std::back_inserter(result), [](auto&& node) { return node; });
    return result;
  }

  template<typename tl_threshold_t, typename tl_output_t>
  auto get_offsets(treelite::Tree<tl_threshold_t, tl_output_t> const& tl_tree) {
    auto result = std::vector<index_type>(tl_tree.num_nodes);
    auto nodes = get_nodes(tl_tree);
    for (auto i = index_type{}; i < nodes.size(); ++i) {
      // Current index should always be greater than or equal to parent index.
      // Later children will overwrite values set by earlier children, ensuring
      // that most distant offset is used.
      result[nodes[i].parent_index] = index_type{i - nodes[i].parent_index};
    }

    return result;
  }

  template<typename lambda_t>
  void tree_for_each(treelite::Model const& tl_model, lambda_t&& lambda) {
    tl_model.Dispatch([&lambda](auto&& concrete_tl_model) {
      std::for_each(
        std::begin(concrete_tl_model.trees),
        std::end(concrete_tl_model.trees),
        lambda
      );
    });
  }

  template<typename iter_t, typename lambda_t>
  void tree_transform(treelite::Model const& tl_model, iter_t output_iter, lambda_t&& lambda) {
    tl_model.Dispatch([&output_iter, &lambda](auto&& concrete_tl_model) {
      std::transform(
        std::begin(concrete_tl_model.trees),
        std::end(concrete_tl_model.trees),
        output_iter,
        lambda
      );
    });
  }

  template<typename T, typename lambda_t>
  auto tree_accumulate(treelite::Model const& tl_model, T init, lambda_t&& lambda) {
    auto result = init;
    tree_for_each(
      tl_model,
      [&result, &lambda](auto&& tree) {
        result = lambda(result, tree);
      }
    );
    return result;
  }

  auto num_trees(treelite::Model const& tl_model) {
    auto result = index_type{};
    tl_model.Dispatch([&result](auto&& concrete_tl_model) {
      result = concrete_tl_model.trees.size();
    });
    return result;
  }

  auto get_offsets(treelite::Model const& tl_model) {
    auto result = std::vector<std::vector<index_type>>{};
    result.reserve(num_trees(tl_model));
    tree_transform(tl_model, std::back_inserter(result), [this](auto&&tree) {
      return get_offsets(tree);
    });
    return result;
  }

  auto get_tree_sizes(treelite::Model const& tl_model) {
    auto result = std::vector<index_type>{};
    tree_transform(
      tl_model,
      std::back_inserter(result),
      [](auto&& tree) { return tree.num_nodes; }
    );
    return result;
  }

  auto get_num_class(treelite::Model const& tl_model) {
    auto result = index_type{};
    tl_model.Dispatch([&result](auto&& concrete_tl_model) {
      result = concrete_tl_model.task_param.num_class;
    });
    return result;
  }

  auto get_num_feature(treelite::Model const& tl_model) {
    auto result = index_type{};
    tl_model.Dispatch([&result](auto&& concrete_tl_model) {
      result = concrete_tl_model.num_feature;
    });
    return result;
  }

  auto get_max_num_categories(treelite::Model const& tl_model) {
    return tree_accumulate(tl_model, index_type{}, [this](auto&& accum, auto&& tree) {
      return node_accumulate(tree, accum, [](auto&& cur_accum, auto&& tl_node) {
        auto result = cur_accum;
        for (auto&& cat : tl_node.categories()) {
          result = (cat + 1 > result) ? cat + 1 : result;
        }
        return result;
      });
    });
  }

  auto get_num_categorical_nodes(treelite::Model const& tl_model) {
    return tree_accumulate(tl_model, index_type{}, [this](auto&& accum, auto&& tree) {
      return node_accumulate(tree, accum, [](auto&& cur_accum, auto&& tl_node) {
        return cur_accum + tl_node.is_categorical();
      });
    });
  }

  auto get_num_leaf_vector_nodes(treelite::Model const& tl_model) {
    return tree_accumulate(tl_model, index_type{}, [this](auto&& accum, auto&& tree) {
      return node_accumulate(tree, accum, [](auto&& cur_accum, auto&& tl_node) {
        return cur_accum + (tl_node.is_leaf() && tl_node.get_output().size() > 1);
      });
    });
  }

  auto get_average_factor(treelite::Model const& tl_model) {
    auto result = double{};
    tl_model.Dispatch([&result](auto&& concrete_tl_model) {
      if (concrete_tl_model.average_tree_output) {
        if (concrete_tl_model.task_type == treelite::TaskType::kMultiClfGrovePerClass) {
          result = concrete_tl_model.trees.size() / concrete_tl_model.task_param.num_class;
        } else {
          result = concrete_tl_model.trees.size();
        }
      } else {
        result = 1.0;
      }
    });
    return result;
  }

  auto get_bias(treelite::Model const& tl_model) {
    auto result = double{};
    tl_model.Dispatch([&result](auto&& concrete_tl_model) {
      result = concrete_tl_model.param.global_bias;
    });
    return result;
  }

  auto get_postproc_params(treelite::Model const& tl_model) {
    auto result = detail::postproc_params_t{};
    tl_model.Dispatch([&result](auto&& concrete_tl_model) {
      auto tl_pred_transform = std::string{concrete_tl_model.param.pred_transform};
      if (
          tl_pred_transform == std::string{"identity"} ||
          tl_pred_transform == std::string{"identity_multiclass"}) {
        result.element = element_op::disable;
        result.row = row_op::disable;
      } else if (tl_pred_transform == std::string{"signed_square"}) {
        result.element = element_op::signed_square;
      } else if (tl_pred_transform == std::string{"hinge"}) {
        result.element = element_op::hinge;
      } else if (tl_pred_transform == std::string{"sigmoid"}) {
        result.constant = concrete_tl_model.param.sigmoid_alpha;
        result.element = element_op::sigmoid;
      } else if (tl_pred_transform == std::string{"exponential"}) {
        result.element = element_op::exponential;
      } else if (tl_pred_transform == std::string{"exponential_standard_ratio"}) {
        result.constant = -concrete_tl_model.param.ratio_c / std::log(2);
        result.element = element_op::exponential;
      } else if (tl_pred_transform == std::string{"logarithm_one_plus_exp"}) {
        result.element = element_op::logarithm_one_plus_exp;
      } else if (tl_pred_transform == std::string{"max_index"}) {
        result.row = row_op::max_index;
      } else if (tl_pred_transform == std::string{"softmax"}) {
        result.row = row_op::softmax;
      } else if (tl_pred_transform == std::string{"multiclass_ova"}) {
        result.constant = concrete_tl_model.param.sigmoid_alpha;
        result.element = element_op::sigmoid;
      } else {
        throw model_import_error{"Unrecognized Treelite pred_transform string"};
      }
    });
    return result;
  }

  auto uses_double_thresholds(treelite::Model const& tl_model) {
    auto result = false;
    switch (tl_model.GetThresholdType()) {
      case treelite::TypeInfo::kFloat64:
        result = true;
        break;
      case treelite::TypeInfo::kFloat32:
        result = false;
        break;
      default:
        throw model_import_error("Unrecognized Treelite threshold type");
    }
    return result;
  }

  auto uses_double_outputs(treelite::Model const& tl_model) {
    auto result = false;
    switch (tl_model.GetThresholdType()) {
      case treelite::TypeInfo::kFloat64:
        result = true;
        break;
      case treelite::TypeInfo::kFloat32:
        result = false;
        break;
      case treelite::TypeInfo::kUInt32:
        result = false;
        break;
      default:
        throw model_import_error("Unrecognized Treelite threshold type");
    }
    return result;
  }

  auto uses_integer_outputs(treelite::Model const& tl_model) {
    auto result = false;
    switch (tl_model.GetThresholdType()) {
      case treelite::TypeInfo::kFloat64:
        result = false;
        break;
      case treelite::TypeInfo::kFloat32:
        result = false;
        break;
      case treelite::TypeInfo::kUInt32:
        result = true;
        break;
      default:
        throw model_import_error("Unrecognized Treelite threshold type");
    }
    return result;
  }

  /**
   * Assuming that the correct decision_forest variant has been
   * identified, import to that variant
   */
  template<index_type variant_index>
  auto import_to_specific_variant(
    index_type target_variant_index,
    treelite::Model const& tl_model,
    index_type num_class,
    index_type num_feature,
    index_type max_num_categories,
    std::vector<std::vector<index_type>> const& offsets,
    index_type align_bytes = index_type{},
    raft_proto::device_type mem_type=raft_proto::device_type::cpu,
    int device=0,
    raft_proto::cuda_stream stream=raft_proto::cuda_stream{}
  ) {
    auto result = decision_forest_variant{};
    if constexpr (variant_index != std::variant_size_v<decision_forest_variant>) {
      if (variant_index == target_variant_index) {
        using forest_model_t = std::variant_alternative_t<variant_index, decision_forest_variant>;
        auto builder = detail::decision_forest_builder<forest_model_t>(max_num_categories, align_bytes);
        auto tree_count = num_trees(tl_model);
        auto tree_index = index_type{};
        tree_for_each(tl_model, [this, &builder, &tree_index, &offsets](auto&& tree) {
          builder.start_new_tree();
          auto node_index = index_type{};
          node_for_each(tree, [&builder, &tree_index, &node_index, &offsets](auto&& node) {
            if (node.is_leaf()) {
              auto output = node.get_output();
              builder.set_output_size(output.size());
              if (output.size() > index_type{1}) {
                builder.add_leaf_vector_node(
                  std::begin(output),
                  std::end(output)
                );
              } else {
                builder.add_node(
                  typename forest_model_t::io_type(output[0]),
                  true
                );
              }
            } else {
              if (node.is_categorical()) {
                auto categories = node.get_categories();
                builder.add_categorical_node(
                  std::begin(categories),
                  std::end(categories),
                  node.default_distant(),
                  node.get_feature(),
                  offsets[tree_index][node_index]
                );
              } else {
                builder.add_node(
                  typename forest_model_t::threshold_type(node.threshold()),
                  false,
                  node.default_distant(),
                  false,
                  node.get_feature(),
                  offsets[tree_index][node_index],
                  node.is_inclusive()
                );
              }
            }
            ++node_index;
          });
          ++tree_index;
        });

        builder.set_average_factor(get_average_factor(tl_model));
        builder.set_bias(get_bias(tl_model));
        auto postproc_params = get_postproc_params(tl_model);
        builder.set_element_postproc(postproc_params.element);
        builder.set_row_postproc(postproc_params.row);
        builder.set_postproc_constant(postproc_params.constant);

        result.template emplace<variant_index>(builder.get_decision_forest(num_feature, num_class, mem_type, device, stream));
      } else {
        result = import_to_specific_variant<variant_index + 1>(
          target_variant_index,
          tl_model,
          num_class,
          num_feature,
          max_num_categories,
          offsets,
          align_bytes,
          mem_type,
          device,
          stream
        );
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
  auto import(
    treelite::Model const& tl_model,
    index_type align_bytes = index_type{},
    std::optional<bool> use_double_precision = std::nullopt,
    raft_proto::device_type dev_type=raft_proto::device_type::cpu,
    int device=0,
    raft_proto::cuda_stream stream=raft_proto::cuda_stream{}
  ) {
    auto result = decision_forest_variant{};
    auto num_feature = get_num_feature(tl_model);
    auto max_num_categories = get_max_num_categories(tl_model);
    auto num_categorical_nodes = get_num_categorical_nodes(tl_model);
    auto num_leaf_vector_nodes = get_num_leaf_vector_nodes(tl_model);
    auto use_double_thresholds = use_double_precision.value_or(uses_double_thresholds(tl_model));

    auto offsets = get_offsets(tl_model);
    auto max_offset = std::accumulate(
      std::begin(offsets),
      std::end(offsets),
      index_type{},
      [&offsets](auto&& cur_max, auto&& tree_offsets) {
        return std::max(cur_max, *std::max_element(std::begin(tree_offsets), std::end(tree_offsets)));
      }
    );
    auto tree_sizes = std::vector<index_type>{};
    std::transform(
      std::begin(offsets),
      std::end(offsets),
      std::back_inserter(tree_sizes),
      [](auto&& tree_offsets) {
        return tree_offsets.size();
      }
    );

    auto variant_index = get_forest_variant_index(
      use_double_thresholds,
      max_offset,
      num_feature,
      num_categorical_nodes,
      max_num_categories,
      num_leaf_vector_nodes,
      layout
    );
    auto num_class = get_num_class(tl_model);
    return forest_model{import_to_specific_variant<index_type{}>(
      variant_index,
      tl_model,
      num_class,
      num_feature,
      max_num_categories,
      offsets,
      align_bytes,
      dev_type,
      device,
      stream
    )};
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
auto import_from_treelite_model(
  treelite::Model const& tl_model,
  tree_layout layout=preferred_tree_layout,
  index_type align_bytes = index_type{},
  std::optional<bool> use_double_precision = std::nullopt,
  raft_proto::device_type dev_type=raft_proto::device_type::cpu,
  int device=0,
  raft_proto::cuda_stream stream=raft_proto::cuda_stream{}
) {
  auto result = forest_model{};
  switch(layout) {
    case tree_layout::depth_first:
      result = treelite_importer<tree_layout::depth_first>{}.import(
        tl_model,
        align_bytes,
        use_double_precision,
        dev_type,
        device,
        stream
      );
      break;
    case tree_layout::breadth_first:
      result = treelite_importer<tree_layout::breadth_first>{}.import(
        tl_model,
        align_bytes,
        use_double_precision,
        dev_type,
        device,
        stream
      );
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
auto import_from_treelite_handle(
  ModelHandle tl_handle,
  tree_layout layout=preferred_tree_layout,
  index_type align_bytes = index_type{},
  std::optional<bool> use_double_precision = std::nullopt,
  raft_proto::device_type dev_type=raft_proto::device_type::cpu,
  int device=0,
  raft_proto::cuda_stream stream=raft_proto::cuda_stream{}
) {
  return import_from_treelite_model(
    *static_cast<treelite::Model*>(tl_handle),
    layout,
    align_bytes,
    use_double_precision,
    dev_type,
    device,
    stream
  );
}

}
}
}
