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

#include <cuml/explainer/tree_shap.hpp>
#include <raft/error.hpp>
#include <thrust/device_ptr.h>
#include <GPUTreeShap/gpu_treeshap.h>
#include <treelite/tree.h>
#include <memory>
#include <type_traits>
#include <iostream>
#include <vector>

namespace {

namespace tl = treelite;

// Define a custom split condition implementing EvaluateSplit and Merge
template <typename T>
struct MySplitCondition {
  MySplitCondition() = default;
  MySplitCondition(T feature_lower_bound, T feature_upper_bound)
      : feature_lower_bound(feature_lower_bound),
        feature_upper_bound(feature_upper_bound) {
    assert(feature_lower_bound <= feature_upper_bound);
  }

  /*! Feature values >= lower and < upper flow down this path. */
  T feature_lower_bound;
  T feature_upper_bound;

  // Does this instance flow down this path?
  __host__ __device__ bool EvaluateSplit(T x) const {
    return x >= feature_lower_bound && x < feature_upper_bound;
  }

  // Combine two split conditions on the same feature
  __host__ __device__ void Merge(
      const MySplitCondition& other) {  // Combine duplicate features
    feature_lower_bound = max(feature_lower_bound, other.feature_lower_bound);
    feature_upper_bound = min(feature_upper_bound, other.feature_upper_bound);
  }
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "T must be a float or double");
};

enum class CondType : uint8_t {
  kFloat, kDouble
};

class ExtractedPath {
 public:
  CondType cond_type;
  float global_bias;
  tl::TaskType task_type;
  tl::TaskParam task_param;
  bool average_tree_output;

  virtual CondType GetCondType() = 0;
  virtual ~ExtractedPath() = default;

  template <typename T>
  static std::unique_ptr<ExtractedPath> Create();

  template <typename Func, typename ...Args>
  auto Dispatch(Func func, Args&& ...args);
};

template <typename T>
class ExtractedPathImpl : public ExtractedPath {
 public:
  std::vector<gpu_treeshap::PathElement<MySplitCondition<T>>> paths;
  ExtractedPathImpl() {
    if (std::is_same<T, double>::value) {
      cond_type = CondType::kDouble;
    } else {
      cond_type = CondType::kFloat;
    }
  }
  virtual ~ExtractedPathImpl() = default;

  CondType GetCondType() override {
    return cond_type;
  }
};

template <typename T>
std::unique_ptr<ExtractedPath> ExtractedPath::Create() {
  std::unique_ptr<ExtractedPath> model
    = std::make_unique<ExtractedPathImpl<T>>();
  return model;
}

template <typename Func, typename ...Args>
auto
ExtractedPath::Dispatch(Func func, Args&& ...args) {
  switch (this->cond_type) {
   case CondType::kDouble:
    func(*dynamic_cast<ExtractedPathImpl<double>*>(this),
         std::forward<Args>(args)...);
    break;
   case CondType::kFloat:
   default:
    func(*dynamic_cast<ExtractedPathImpl<float>*>(this),
         std::forward<Args>(args)...);
    break;
  }
}

template <typename T>
std::ostream& operator<<(
    std::ostream& os,
    const std::vector<gpu_treeshap::PathElement<MySplitCondition<T>>>& paths) {
  std::vector<gpu_treeshap::PathElement<MySplitCondition<T>>> tmp(paths);
  std::sort(tmp.begin(), tmp.end(),
            [&](const gpu_treeshap::PathElement<MySplitCondition<T>>& a,
                const gpu_treeshap::PathElement<MySplitCondition<T>>& b) {
              if (a.path_idx < b.path_idx) return true;
              if (b.path_idx < a.path_idx) return false;

              if (a.feature_idx < b.feature_idx) return true;
              if (b.feature_idx < a.feature_idx) return false;
              return false;
            });

  for (auto i = 0ull; i < tmp.size(); i++) {
    auto e = tmp[i];
    if (i == 0 || e.path_idx != tmp[i - 1].path_idx) {
      os << "path_idx:" << e.path_idx << ", leaf value:" << e.v;
      os << "\n";
    }
    os << " (feature:" << e.feature_idx << ", pz:" << e.zero_fraction << ", ["
       << e.split_condition.feature_lower_bound << "<=x<"
       << e.split_condition.feature_upper_bound << "])";
    os << "\n";
  }
  return os;
}

class DenseDatasetWrapper {
  const float* data;
  std::size_t num_rows;
  std::size_t num_cols;

 public:
  DenseDatasetWrapper() = default;
  DenseDatasetWrapper(const float* data, int num_rows, int num_cols)
      : data(data), num_rows(num_rows), num_cols(num_cols) {}
  __device__ float GetElement(std::size_t row_idx, std::size_t col_idx) const {
    return data[row_idx * num_cols + col_idx];
  }
  __host__ __device__ std::size_t NumRows() const { return num_rows; }
  __host__ __device__ std::size_t NumCols() const { return num_cols; }
};


}  // anonymous namespace

namespace ML {
namespace Explainer {

template <typename ThresholdType, typename LeafType>
ExtractedPathHandle
extract_paths_impl(const tl::ModelImpl<ThresholdType, LeafType>& model) {
  if (!std::is_same<ThresholdType, LeafType>::value) {
    RAFT_FAIL("ThresholdType and LeafType must be identical");
  }
  if (model.task_type != tl::TaskType::kBinaryClfRegr
      && model.task_type != tl::TaskType::kMultiClfGrovePerClass) {
    RAFT_FAIL("Only tree models with task type kBinaryClfRegr, "
              "kMultiClfGrovePerClass are supported. So for example, "
              "scikit-learn trees are not supported.");
  }
  std::unique_ptr<ExtractedPath> path_container
    = ExtractedPath::Create<ThresholdType>();
  ExtractedPathImpl<ThresholdType>* paths
    = dynamic_cast<ExtractedPathImpl<ThresholdType>*>(path_container.get());

  std::size_t path_idx = 0;
  int tree_idx = 0;
  int num_groups = 1;
  if (model.task_type == tl::TaskType::kMultiClfGrovePerClass
      && model.task_param.num_class > 1) {
    num_groups = model.task_param.num_class;
  }
  for (const tl::Tree<ThresholdType, LeafType>& tree : model.trees) {
    std::vector<int> parent_id(tree.num_nodes, -1);
    // Compute parent ID of each node
    for (int i = 0; i < tree.num_nodes; i++) {
      if (!tree.IsLeaf(i)) {
        parent_id[tree.LeftChild(i)] = i;
        parent_id[tree.RightChild(i)] = i;
      }
    }

    // Find leaf nodes
    // Work backwards from leaf to root, order does not matter
    // It's also possible to work from root to leaf
    for (int i = 0; i < tree.num_nodes; i++) {
      if (tree.IsLeaf(i)) {
        float v = static_cast<float>(tree.LeafValue(i));
        int child_idx = i;
        int parent_idx = parent_id[child_idx];
        const auto inf = std::numeric_limits<ThresholdType>::infinity();
        while (parent_idx != -1) {
          double zero_fraction = 1.0;
          bool has_count_info = false;
          if (tree.HasSumHess(parent_idx) && tree.HasSumHess(child_idx)) {
            zero_fraction = static_cast<double>(
                tree.SumHess(child_idx) / tree.SumHess(parent_idx));
            has_count_info = true;
          }
          if (tree.HasDataCount(parent_idx) && tree.HasDataCount(child_idx)) {
            zero_fraction = static_cast<double>(
                tree.DataCount(child_idx)) / tree.DataCount(parent_idx);
            has_count_info = true;
          }
          if (!has_count_info) {
            RAFT_FAIL("Lacking sufficient info");
          }
          // Encode the range of feature values that flow down this path
          bool is_left_path = tree.LeftChild(parent_idx) == child_idx;
          if (tree.SplitType(parent_idx) == tl::SplitFeatureType::kCategorical) {
            RAFT_FAIL("For now only trees with numerical splits are supported. "
                      "Trees with categorical splits are not supported yet.");
          }
          ThresholdType lower_bound =
            is_left_path ? -inf : tree.Threshold(parent_idx);
          ThresholdType upper_bound =
            is_left_path ? tree.Threshold(parent_idx) : inf;
          int group_id = tree_idx % num_groups;
          paths->paths.push_back(
              gpu_treeshap::PathElement<MySplitCondition<ThresholdType>>{
                path_idx,
                tree.SplitIndex(parent_idx),
                group_id,
                {lower_bound, upper_bound},
                zero_fraction,
                v});
          child_idx = parent_idx;
          parent_idx = parent_id[child_idx];
        }
        // Root node has feature -1
        {
          int group_id = tree_idx % num_groups;
          paths->paths.push_back(
              gpu_treeshap::PathElement<MySplitCondition<ThresholdType>>{
                path_idx, -1, group_id, {-inf, inf}, 1.0, v});
          path_idx++;
        }
      }
    }
    tree_idx++;
  }
  paths->global_bias = model.param.global_bias;
  paths->task_type = model.task_type;
  paths->task_param = model.task_param;
  paths->average_tree_output = model.average_tree_output;

  return static_cast<ExtractedPathHandle>(path_container.release());
}

void extract_paths(ModelHandle model, ExtractedPathHandle* extracted_paths) {
  const tl::Model& model_ref = *static_cast<tl::Model*>(model);

  *extracted_paths = model_ref.Dispatch([&](const auto& model_inner) {
    // model_inner is of the concrete type tl::ModelImpl<threshold_t, leaf_t>
    return extract_paths_impl(model_inner);
  });
}

void gpu_treeshap(ExtractedPathHandle extracted_paths, const float* data,
                  std::size_t n_rows, std::size_t n_cols, float* out_preds) {
  ExtractedPath& path_ref = *static_cast<ExtractedPath*>(extracted_paths);
  DenseDatasetWrapper X(data, n_rows, n_cols);
  std::size_t num_groups = 1;
  if (path_ref.task_type == tl::TaskType::kMultiClfGrovePerClass
      && path_ref.task_param.num_class > 1) {
    num_groups = static_cast<std::size_t>(path_ref.task_param.num_class);
  }
  thrust::device_ptr<float> out_preds_ptr
    = thrust::device_pointer_cast(out_preds);
  path_ref.Dispatch([&](auto& paths) {
    gpu_treeshap::GPUTreeShap(X, paths.paths.begin(), paths.paths.end(),
                              num_groups, out_preds_ptr,
                              out_preds_ptr
                              + (n_rows * num_groups * (n_cols + 1)));
  });
  float global_bias = path_ref.global_bias;
  auto count_iter = thrust::make_counting_iterator(0);
  thrust::for_each(thrust::device, count_iter,
      count_iter + (n_rows * num_groups),
    [=] __device__(std::size_t idx) {
      out_preds[(idx + 1) * (n_cols + 1) - 1] += global_bias;
    });
}

void free_extracted_paths(ExtractedPathHandle extracted_paths) {
  delete static_cast<ExtractedPath*>(extracted_paths);
}

}  // namespace Explainer
}  // namespace ML
