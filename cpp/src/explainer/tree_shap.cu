/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cuml/explainer/tree_shap.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <raft/error.hpp>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <treelite/tree.h>
#include <type_traits>
#include <vector>

namespace tl = treelite;

/* All functions and classes defined in this anonymous namespace are strictly
 * for internal use by GPUTreeSHAP. */
namespace {

// A poor man's Span class.
// TODO(hcho3): Remove this class once RAFT implements a span abstraction.
template <typename T>
class Span {
 private:
  T* ptr_{nullptr};
  std::size_t size_{0};

 public:
  Span() = default;
  __host__ __device__ Span(T* ptr, std::size_t size) : ptr_(ptr), size_(size) {}
  __host__ explicit Span(std::vector<T>& vec) : ptr_(vec.data()), size_(vec.size()) {}
  __host__ explicit Span(thrust::device_vector<T>& vec)
    : ptr_(thrust::raw_pointer_cast(vec.data())), size_(vec.size())
  {
  }
  __host__ __device__ Span(const Span& other) : ptr_(other.ptr_), size_(other.size_) {}
  __host__ __device__ Span(Span&& other) : ptr_(other.ptr_), size_(other.size_)
  {
    other.ptr_  = nullptr;
    other.size_ = 0;
  }
  __host__ __device__ ~Span() {}
  __host__ __device__ Span& operator=(const Span& other)
  {
    ptr_  = other.ptr_;
    size_ = other.size_;
    return *this;
  }
  __host__ __device__ Span& operator=(Span&& other)
  {
    ptr_        = other.ptr_;
    size_       = other.size_;
    other.ptr_  = nullptr;
    other.size_ = 0;
    return *this;
  }
  __host__ __device__ std::size_t Size() const { return size_; }
  __host__ __device__ T* Data() const { return ptr_; }
  __host__ __device__ T& operator[](std::size_t offset) const { return *(ptr_ + offset); }
  __host__ __device__ Span<T> Subspan(std::size_t offset, std::size_t count)
  {
    return Span{ptr_ + offset, count};
  }
};

// A poor man's bit field, to be used to account for categorical splits in SHAP computation
// Inspired by xgboost::BitFieldContainer
template <typename T>
class BitField {
 private:
  static std::size_t constexpr kValueSize = sizeof(T) * 8;
  static std::size_t constexpr kOne       = 1;  // force correct data type

  Span<T> bits_;

 public:
  BitField() = default;
  __host__ __device__ explicit BitField(Span<T> bits) : bits_(bits) {}
  __host__ __device__ BitField(const BitField& other) : bits_(other.bits_) {}
  BitField& operator=(const BitField& other) = default;
  BitField& operator=(BitField&& other) = default;
  __host__ __device__ bool Check(std::size_t pos) const
  {
    T bitmask = kOne << (pos % kValueSize);
    return static_cast<bool>(bits_[pos / kValueSize] & bitmask);
  }
  __host__ __device__ void Set(std::size_t pos)
  {
    T bitmask = kOne << (pos % kValueSize);
    bits_[pos / kValueSize] |= bitmask;
  }
  __host__ __device__ void Intersect(const BitField other)
  {
    if (bits_.Data() == other.bits_.Data()) { return; }
    std::size_t size = min(bits_.Size(), other.bits_.Size());
    for (std::size_t i = 0; i < size; ++i) {
      bits_[i] &= other.bits_[i];
    }
    if (bits_.Size() > size) {
      for (std::size_t i = size; i < bits_.Size(); ++i) {
        bits_[i] = 0;
      }
    }
  }
  __host__ __device__ std::size_t Size() const { return kValueSize * bits_.Size(); }
  __host__ static std::size_t ComputeStorageSize(std::size_t n_cat)
  {
    return n_cat / kValueSize + (n_cat % kValueSize != 0);
  }

  static_assert(!std::is_signed<T>::value, "Must use unsiged type as underlying storage.");
};

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

template <typename ThresholdType, typename LeafType>
struct PathSegmentExtractor {
  using PathElementT = gpu_treeshap::PathElement<SplitCondition<ThresholdType>>;
  std::vector<PathElementT>& path_segments;
  std::size_t& path_idx;

  static constexpr ThresholdType inf{std::numeric_limits<ThresholdType>::infinity()};

  PathSegmentExtractor(std::vector<PathElementT>& path_segments, std::size_t& path_idx)
    : path_segments(path_segments), path_idx(path_idx)
  {
  }

  void node_handler(const tl::Tree<ThresholdType, LeafType>& tree,
                    int child_idx,
                    int parent_idx,
                    int group_id,
                    float v)
  {
    constexpr auto inf   = std::numeric_limits<ThresholdType>::infinity();
    double zero_fraction = 1.0;
    bool has_count_info  = false;
    if (tree.HasSumHess(parent_idx) && tree.HasSumHess(child_idx)) {
      zero_fraction  = static_cast<double>(tree.SumHess(child_idx) / tree.SumHess(parent_idx));
      has_count_info = true;
    }
    if (!has_count_info && tree.HasDataCount(parent_idx) && tree.HasDataCount(child_idx)) {
      zero_fraction  = static_cast<double>(tree.DataCount(child_idx)) / tree.DataCount(parent_idx);
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
    auto comparison_op        = tree.ComparisonOp(parent_idx);
    path_segments.push_back(gpu_treeshap::PathElement<SplitCondition<ThresholdType>>{
      path_idx,
      tree.SplitIndex(parent_idx),
      group_id,
      SplitCondition{lower_bound, upper_bound, comparison_op},
      zero_fraction,
      v});
  }

  void root_handler(const tl::Tree<ThresholdType, LeafType>& tree,
                    int child_idx,
                    int group_id,
                    float v)
  {
    // Root node has feature -1
    auto comparison_op = tree.ComparisonOp(child_idx);
    path_segments.push_back(gpu_treeshap::PathElement<SplitCondition<ThresholdType>>{
      path_idx, -1, group_id, SplitCondition{-inf, inf, comparison_op}, 1.0, v});
  }

  void new_path_handler() { ++path_idx; }
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
// Traverse a path from the root node to a leaf node and call the handler functions for each node.
// The fields group_id and v (leaf value) will be passed to the handler.
template <typename ThresholdType, typename LeafType, typename PathHandler>
void traverse_towards_leaf_node(const tl::Tree<ThresholdType, LeafType>& tree,
                                int leaf_node_id,
                                int group_id,
                                float v,
                                const std::vector<int>& parent_id,
                                PathHandler& path_handler)
{
  int child_idx  = leaf_node_id;
  int parent_idx = parent_id[child_idx];
  while (parent_idx != -1) {
    path_handler.node_handler(tree, child_idx, parent_idx, group_id, v);
    child_idx  = parent_idx;
    parent_idx = parent_id[child_idx];
  }
  path_handler.root_handler(tree, child_idx, group_id, v);
}

// Visit every path segments in a single tree and call handler functions for each segment.
template <typename ThresholdType, typename LeafType, typename PathHandler>
void visit_path_segments_in_tree(const std::vector<tl::Tree<ThresholdType, LeafType>>& tree_list,
                                 std::size_t tree_idx,
                                 bool use_vector_leaf,
                                 int num_groups,
                                 PathHandler& path_handler)
{
  if (num_groups < 1) { RAFT_FAIL("num_groups must be at least 1"); }

  const tl::Tree<ThresholdType, LeafType>& tree = tree_list[tree_idx];

  // Compute parent ID of each node
  std::vector<int> parent_id(tree.num_nodes, -1);
  for (int i = 0; i < tree.num_nodes; i++) {
    if (!tree.IsLeaf(i)) {
      parent_id[tree.LeftChild(i)]  = i;
      parent_id[tree.RightChild(i)] = i;
    }
  }

  for (int nid = 0; nid < tree.num_nodes; nid++) {
    if (tree.IsLeaf(nid)) {  // For each leaf node...
      // Extract path segments by traversing the path from the leaf node to the root node
      // If use_vector_leaf=True, repeat the path segments N times, where N = num_groups
      if (use_vector_leaf) {
        auto leaf_vector = tree.LeafVector(nid);
        if (leaf_vector.size() != static_cast<std::size_t>(num_groups)) {
          RAFT_FAIL("Expected leaf vector of length %d but got %d instead",
                    num_groups,
                    static_cast<int>(leaf_vector.size()));
        }
        for (int group_id = 0; group_id < num_groups; ++group_id) {
          traverse_towards_leaf_node(
            tree, nid, group_id, leaf_vector[group_id], parent_id, path_handler);
          path_handler.new_path_handler();
        }
      } else {
        int group_id    = static_cast<int>(tree_idx) % num_groups;
        auto leaf_value = tree.LeafValue(nid);
        traverse_towards_leaf_node(tree, nid, group_id, leaf_value, parent_id, path_handler);
        path_handler.new_path_handler();
      }
    }
  }
}

// Visit every path segments in the whole tree ensemble model
template <typename ThresholdType, typename LeafType, typename PathHandler>
void visit_path_segments_in_model(const tl::ModelImpl<ThresholdType, LeafType>& model,
                                  PathHandler& path_handler)
{
  int num_groups = 1;
  bool use_vector_leaf;
  if (model.task_param.num_class > 1) { num_groups = model.task_param.num_class; }
  if (model.task_type == tl::TaskType::kBinaryClfRegr ||
      model.task_type == tl::TaskType::kMultiClfGrovePerClass) {
    use_vector_leaf = false;
  } else if (model.task_type == tl::TaskType::kMultiClfProbDistLeaf) {
    use_vector_leaf = true;
  } else {
    RAFT_FAIL("Unsupported task_type: %d", static_cast<int>(model.task_type));
  }

  for (std::size_t tree_idx = 0; tree_idx < model.trees.size(); ++tree_idx) {
    visit_path_segments_in_tree(model.trees, tree_idx, use_vector_leaf, num_groups, path_handler);
  }
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

  // Each path segment will have path_idx field, which uniquely identifies the path to which the
  // segment belongs.
  std::size_t path_idx = 0;
  PathSegmentExtractor<ThresholdType, LeafType> path_extractor{path_info->paths, path_idx};
  visit_path_segments_in_model(model, path_extractor);

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
