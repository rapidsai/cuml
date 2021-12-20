/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <GPUTreeShap/gpu_treeshap.h>
#include <thrust/device_ptr.h>
#include <treelite/tree.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuml/explainer/tree_shap.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <raft/error.hpp>
#include <type_traits>
#include <vector>

namespace tl = treelite;

/* All functions and classes defined in this anonymous namespace are strictly
 * for internal use by GPUTreeSHAP. */
namespace {

template <typename ThresholdType>
struct SplitCondition {
  SplitCondition() = default;
  SplitCondition(ThresholdType feature_lower_bound,
                 ThresholdType feature_upper_bound,
                 tl::Operator comparison_op)
    : feature_lower_bound(feature_lower_bound),
      feature_upper_bound(feature_upper_bound),
      comparison_op(comparison_op)
  {
    if (feature_lower_bound > feature_upper_bound) {
      RAFT_FAIL("Lower bound cannot exceed upper bound");
    }
    if (comparison_op != tl::Operator::kLT && comparison_op != tl::Operator::kLE &&
        comparison_op != tl::Operator::kNone) {
      RAFT_FAIL("Unsupported comparison operator");
    }
  }

  // Lower and upper bounds on feature values flowing down this path
  ThresholdType feature_lower_bound;
  ThresholdType feature_upper_bound;
  // Comparison operator used in the test. For now only < (kLT) and <= (kLE)
  // are supported.
  tl::Operator comparison_op;

  // Does this instance flow down this path?
  __host__ __device__ bool EvaluateSplit(ThresholdType x) const
  {
    if (comparison_op == tl::Operator::kLE) {
      return x > feature_lower_bound && x <= feature_upper_bound;
    }
    return x >= feature_lower_bound && x < feature_upper_bound;
  }

  // Combine two split conditions on the same feature
  __host__ __device__ void Merge(const SplitCondition& other)
  {  // Combine duplicate features
    feature_lower_bound = max(feature_lower_bound, other.feature_lower_bound);
    feature_upper_bound = min(feature_upper_bound, other.feature_upper_bound);
  }
  static_assert(std::is_same<ThresholdType, float>::value ||
                  std::is_same<ThresholdType, double>::value,
                "ThresholdType must be a float or double");
};

template <typename ThresholdType>
class TreePathInfoImpl : public ML::Explainer::TreePathInfo {
 public:
  ThresholdTypeEnum threshold_type;
  int num_tree;
  float global_bias;
  tl::TaskType task_type;
  tl::TaskParam task_param;
  bool average_tree_output;
  std::vector<gpu_treeshap::PathElement<SplitCondition<ThresholdType>>> paths;

  static_assert(std::is_same<ThresholdType, float>::value ||
                  std::is_same<ThresholdType, double>::value,
                "ThresholdType must be a float or double");

  TreePathInfoImpl()
  {
    if constexpr (std::is_same<ThresholdType, double>::value) {
      threshold_type = ThresholdTypeEnum::kDouble;
    } else {
      threshold_type = ThresholdTypeEnum::kFloat;
    }
  }
  virtual ~TreePathInfoImpl() = default;

  ThresholdTypeEnum GetThresholdType() const override { return threshold_type; }
};

class DenseDatasetWrapper {
  const float* data;
  std::size_t num_rows;
  std::size_t num_cols;

 public:
  DenseDatasetWrapper() = default;
  DenseDatasetWrapper(const float* data, int num_rows, int num_cols)
    : data(data), num_rows(num_rows), num_cols(num_cols)
  {
  }
  __device__ float GetElement(std::size_t row_idx, std::size_t col_idx) const
  {
    return data[row_idx * num_cols + col_idx];
  }
  __host__ __device__ std::size_t NumRows() const { return num_rows; }
  __host__ __device__ std::size_t NumCols() const { return num_cols; }
};

template <typename ThresholdType>
void gpu_treeshap_impl(const TreePathInfoImpl<ThresholdType>* path_info,
                       const float* data,
                       std::size_t n_rows,
                       std::size_t n_cols,
                       float* out_preds)
{
  DenseDatasetWrapper X(data, n_rows, n_cols);

  std::size_t num_groups = 1;
  if (path_info->task_param.num_class > 1) {
    num_groups = static_cast<std::size_t>(path_info->task_param.num_class);
  }
  std::size_t pred_size = n_rows * num_groups * (n_cols + 1);

  thrust::device_ptr<float> out_preds_ptr = thrust::device_pointer_cast(out_preds);
  gpu_treeshap::GPUTreeShap(X,
                            path_info->paths.begin(),
                            path_info->paths.end(),
                            num_groups,
                            out_preds_ptr,
                            out_preds_ptr + pred_size);

  // Post-processing
  auto count_iter  = thrust::make_counting_iterator(0);
  auto num_tree    = path_info->num_tree;
  auto global_bias = path_info->global_bias;
  if (path_info->average_tree_output) {
    thrust::for_each(
      thrust::device, count_iter, count_iter + pred_size, [=] __device__(std::size_t idx) {
        out_preds[idx] /= num_tree;
      });
  }
  thrust::for_each(
    thrust::device,
    count_iter,
    count_iter + (n_rows * num_groups),
    [=] __device__(std::size_t idx) { out_preds[(idx + 1) * (n_cols + 1) - 1] += global_bias; });
}

}  // anonymous namespace

namespace ML {
namespace Explainer {

template <bool use_vector_leaf, typename ThresholdType, typename LeafType>
void extract_path_info_from_tree(const tl::Tree<ThresholdType, LeafType>& tree,
                                 int num_groups,
                                 int& tree_idx,
                                 std::size_t& path_idx,
                                 TreePathInfoImpl<ThresholdType>& path_info)
{
  if (num_groups < 1) { RAFT_FAIL("num_groups must be at least 1"); }

  std::vector<int> parent_id(tree.num_nodes, -1);
  // Compute parent ID of each node
  for (int i = 0; i < tree.num_nodes; i++) {
    if (!tree.IsLeaf(i)) {
      parent_id[tree.LeftChild(i)]  = i;
      parent_id[tree.RightChild(i)] = i;
    }
  }

  // Find leaf nodes
  // Work backwards from leaf to root, order does not matter
  // It's also possible to work from root to leaf
  for (int nid = 0; nid < tree.num_nodes; nid++) {
    if (tree.IsLeaf(nid)) {
      std::vector<gpu_treeshap::PathElement<SplitCondition<ThresholdType>>> tmp_paths;
      int child_idx              = nid;
      int parent_idx             = parent_id[child_idx];
      constexpr auto inf         = std::numeric_limits<ThresholdType>::infinity();
      tl::Operator comparison_op = tl::Operator::kNone;
      while (parent_idx != -1) {
        double zero_fraction = 1.0;
        bool has_count_info  = false;
        if (tree.HasSumHess(parent_idx) && tree.HasSumHess(child_idx)) {
          zero_fraction  = static_cast<double>(tree.SumHess(child_idx) / tree.SumHess(parent_idx));
          has_count_info = true;
        }
        if (!has_count_info && tree.HasDataCount(parent_idx) && tree.HasDataCount(child_idx)) {
          zero_fraction =
            static_cast<double>(tree.DataCount(child_idx)) / tree.DataCount(parent_idx);
          has_count_info = true;
        }
        if (!has_count_info) { RAFT_FAIL("Tree model doesn't have data count information"); }
        // Encode the range of feature values that flow down this path
        bool is_left_path = tree.LeftChild(parent_idx) == child_idx;
        if (tree.SplitType(parent_idx) == tl::SplitFeatureType::kCategorical) {
          RAFT_FAIL(
            "Only trees with numerical splits are supported. "
            "Trees with categorical splits are not supported yet.");
        }
        ThresholdType lower_bound = is_left_path ? -inf : tree.Threshold(parent_idx);
        ThresholdType upper_bound = is_left_path ? tree.Threshold(parent_idx) : inf;
        comparison_op             = tree.ComparisonOp(parent_idx);
        // Build temporary path segments with unknown path_idx, group_id and leaf value
        tmp_paths.push_back(gpu_treeshap::PathElement<SplitCondition<ThresholdType>>{
          ~std::size_t(0),
          tree.SplitIndex(parent_idx),
          -1,
          SplitCondition{lower_bound, upper_bound, comparison_op},
          zero_fraction,
          std::numeric_limits<float>::quiet_NaN()});
        child_idx  = parent_idx;
        parent_idx = parent_id[child_idx];
      }
      // Root node has feature -1
      comparison_op = tree.ComparisonOp(child_idx);
      // Build temporary path segments with unknown path_idx, group_id and leaf value
      tmp_paths.push_back(gpu_treeshap::PathElement<SplitCondition<ThresholdType>>{
        ~std::size_t(0),
        -1,
        -1,
        SplitCondition{-inf, inf, comparison_op},
        1.0,
        std::numeric_limits<float>::quiet_NaN()});

      // If use_vector_leaf=True:
      // * Duplicate tmp_paths N times, where N = num_groups
      // * Insert into path_info.paths
      // If use_vector_leaf=False:
      // * Insert tmp_paths into path_info.paths
      auto path_insertor = [&tmp_paths, &path_info](auto leaf_value, auto path_idx, int group_id) {
        for (auto& e : tmp_paths) {
          e.path_idx = path_idx;
          e.v        = static_cast<float>(leaf_value);
          e.group    = group_id;
        }
        path_info.paths.insert(path_info.paths.end(), tmp_paths.cbegin(), tmp_paths.cend());
      };
      if constexpr (use_vector_leaf) {
        auto leaf_vector = tree.LeafVector(nid);
        if (leaf_vector.size() != static_cast<std::size_t>(num_groups)) {
          RAFT_FAIL("Expected leaf vector of length %d but got %d instead",
                    num_groups,
                    static_cast<int>(leaf_vector.size()));
        }
        for (int group_id = 0; group_id < num_groups; ++group_id) {
          path_insertor(leaf_vector[group_id], path_idx, group_id);
          path_idx++;
        }
      } else {
        auto leaf_value = tree.LeafValue(nid);
        int group_id    = tree_idx % num_groups;
        path_insertor(leaf_value, path_idx, group_id);
        path_idx++;
      }
    }
  }
  tree_idx++;
}

template <typename ThresholdType, typename LeafType>
std::unique_ptr<TreePathInfo> extract_path_info_impl(
  const tl::ModelImpl<ThresholdType, LeafType>& model)
{
  if (!std::is_same<ThresholdType, LeafType>::value) {
    RAFT_FAIL("ThresholdType and LeafType must be identical");
  }
  if (!std::is_same<ThresholdType, float>::value && !std::is_same<ThresholdType, double>::value) {
    RAFT_FAIL("ThresholdType must be either float32 or float64");
  }

  std::unique_ptr<TreePathInfo> path_info_ptr = std::make_unique<TreePathInfoImpl<ThresholdType>>();
  auto* path_info = dynamic_cast<TreePathInfoImpl<ThresholdType>*>(path_info_ptr.get());

  std::size_t path_idx = 0;
  int tree_idx         = 0;
  int num_groups       = 1;
  if (model.task_param.num_class > 1) { num_groups = model.task_param.num_class; }
  if (model.task_type == tl::TaskType::kBinaryClfRegr ||
      model.task_type == tl::TaskType::kMultiClfGrovePerClass) {
    for (const tl::Tree<ThresholdType, LeafType>& tree : model.trees) {
      extract_path_info_from_tree<false>(tree, num_groups, tree_idx, path_idx, *path_info);
    }
  } else if (model.task_type == tl::TaskType::kMultiClfProbDistLeaf) {
    for (const tl::Tree<ThresholdType, LeafType>& tree : model.trees) {
      extract_path_info_from_tree<true>(tree, num_groups, tree_idx, path_idx, *path_info);
    }
  }
  path_info->global_bias         = model.param.global_bias;
  path_info->task_type           = model.task_type;
  path_info->task_param          = model.task_param;
  path_info->average_tree_output = model.average_tree_output;
  path_info->num_tree            = static_cast<int>(model.trees.size());

  return path_info_ptr;
}

std::unique_ptr<TreePathInfo> extract_path_info(ModelHandle model)
{
  const tl::Model& model_ref = *static_cast<tl::Model*>(model);

  return model_ref.Dispatch([&](const auto& model_inner) {
    // model_inner is of the concrete type tl::ModelImpl<threshold_t, leaf_t>
    return extract_path_info_impl(model_inner);
  });
}

void gpu_treeshap(const TreePathInfo* path_info,
                  const float* data,
                  std::size_t n_rows,
                  std::size_t n_cols,
                  float* out_preds)
{
  switch (path_info->GetThresholdType()) {
    case TreePathInfo::ThresholdTypeEnum::kDouble: {
      const auto* path_info_casted = dynamic_cast<const TreePathInfoImpl<double>*>(path_info);
      gpu_treeshap_impl(path_info_casted, data, n_rows, n_cols, out_preds);
    } break;
    case TreePathInfo::ThresholdTypeEnum::kFloat:
    default: {
      const auto* path_info_casted = dynamic_cast<const TreePathInfoImpl<float>*>(path_info);
      gpu_treeshap_impl(path_info_casted, data, n_rows, n_cols, out_preds);
    } break;
  }
}

}  // namespace Explainer
}  // namespace ML
