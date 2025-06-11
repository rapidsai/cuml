/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cuml/forest/integrations/treelite.hpp>

#include <treelite/tree.h>

#include <cstdint>
#include <memory>
#include <type_traits>

namespace ML::fil::detail {

// This function returns a modified copy of a given Treelite model if it contains
// at least one degenerate tree (a single root node with no child).
// If the model contains no degenerate tree, then the function returns nullptr.
std::unique_ptr<treelite::Model> convert_degenerate_trees(treelite::Model const& tl_model)
{
  bool contains_degenerate =
    ML::forest::tree_accumulate(tl_model, false, [](auto&& contains, auto&& tree) {
      return contains || tree.IsLeaf(ML::forest::TREELITE_NODE_ID_T{});
    });

  if (contains_degenerate) {
    // Make a copy of the Treelite model, and then update the trees in-place
    auto modified_model = treelite::ConcatenateModelObjects({&tl_model});
    std::visit(
      [](auto&& concrete_tl_model) {
        using model_t = std::remove_const_t<std::remove_reference_t<decltype(concrete_tl_model)>>;
        using tree_t =
          treelite::Tree<typename model_t::threshold_type, typename model_t::leaf_output_type>;
        auto modified_trees = std::vector<tree_t>{};
        const auto root_id  = ML::forest::TREELITE_NODE_ID_T{};
        for (tree_t& tree : concrete_tl_model.trees) {
          if (tree.IsLeaf(root_id)) {
            const auto inst_cnt =
              tree.HasDataCount(root_id) ? tree.DataCount(root_id) : std::uint64_t{};
            auto new_tree = tree_t{};
            new_tree.Init();
            const auto root_id   = new_tree.AllocNode();
            const auto cleft_id  = new_tree.AllocNode();
            const auto cright_id = new_tree.AllocNode();
            new_tree.SetChildren(root_id, cleft_id, cright_id);
            new_tree.SetNumericalTest(
              root_id, int{}, typename model_t::threshold_type{}, true, treelite::Operator::kLE);
            if (tree.HasLeafVector(root_id)) {
              const auto leaf_vector = tree.LeafVector(root_id);
              new_tree.SetLeafVector(cleft_id, leaf_vector);
              new_tree.SetLeafVector(cright_id, leaf_vector);
            } else {
              const auto leaf_value = tree.LeafValue(root_id);
              new_tree.SetLeaf(cleft_id, leaf_value);
              new_tree.SetLeaf(cright_id, leaf_value);
            }
            new_tree.SetDataCount(root_id, inst_cnt);
            new_tree.SetDataCount(cleft_id, inst_cnt);
            new_tree.SetDataCount(cright_id, std::uint64_t{});
            modified_trees.push_back(std::move(new_tree));
          } else {
            modified_trees.push_back(std::move(tree));
          }
        }
        concrete_tl_model.trees = std::move(modified_trees);
      },
      modified_model->variant_);
    return modified_model;
  } else {
    return std::unique_ptr<treelite::Model>();
  }
}

}  // namespace ML::fil::detail
