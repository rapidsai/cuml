/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <raft/core/error.hpp>
#include <raft/core/span.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/memory.h>

#include <GPUTreeShap/gpu_treeshap.h>
#include <treelite/enum/operator.h>
#include <treelite/enum/task_type.h>
#include <treelite/enum/tree_node_type.h>
#include <treelite/tree.h>

#include <algorithm>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>
#include <variant>
#include <vector>

namespace tl = treelite;

/* All functions and classes defined in this anonymous namespace are strictly
 * for internal use by GPUTreeSHAP. */
namespace {

// A poor man's bit field, to be used to account for categorical splits in SHAP computation
// Inspired by xgboost::BitFieldContainer
template <typename T, bool is_device>
class BitField {
 private:
  static std::size_t constexpr kValueSize = sizeof(T) * 8;
  static std::size_t constexpr kOne       = 1;  // force correct data type

  raft::span<T, is_device> bits_;

 public:
  BitField() = default;
  __host__ __device__ explicit BitField(raft::span<T, is_device> bits) : bits_(bits) {}
  __host__ __device__ BitField(const BitField& other) : bits_(other.bits_) {}
  BitField& operator=(const BitField& other) = default;
  BitField& operator=(BitField&& other)      = default;
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
    if (bits_.data() == other.bits_.data()) { return; }
    std::size_t size = min(bits_.size(), other.bits_.size());
    for (std::size_t i = 0; i < size; ++i) {
      bits_[i] &= other.bits_[i];
    }
    if (bits_.size() > size) {
      for (std::size_t i = size; i < bits_.size(); ++i) {
        bits_[i] = 0;
      }
    }
  }
  __host__ __device__ std::size_t Size() const { return kValueSize * bits_.size(); }
  __host__ static std::size_t ComputeStorageSize(std::size_t n_cat)
  {
    return n_cat / kValueSize + (n_cat % kValueSize != 0);
  }
  __host__ std::string ToString(bool reverse = false) const
  {
    std::ostringstream oss;
    oss << "Bits storage size: " << bits_.size() << ", elements: ";
    for (auto i = 0; i < bits_.size(); ++i) {
      std::bitset<kValueSize> bset(bits_[i]);
      std::string s = bset.to_string();
      if (reverse) { std::reverse(s.begin(), s.end()); }
      oss << s << ", ";
    }
    return oss.str();
  }

  static_assert(!std::is_signed<T>::value, "Must use unsigned type as underlying storage.");
};

using CatBitFieldStorageT = std::uint32_t;
template <bool is_device>
using CatBitField = BitField<CatBitFieldStorageT, is_device>;
using CatT        = std::uint32_t;

template <typename ThresholdT>
struct SplitCondition {
  SplitCondition() = default;
  SplitCondition(ThresholdT feature_lower_bound,
                 ThresholdT feature_upper_bound,
                 bool is_missing_branch,
                 tl::Operator comparison_op,
                 CatBitField<false> categories)
    : feature_lower_bound(feature_lower_bound),
      feature_upper_bound(feature_upper_bound),
      is_missing_branch(is_missing_branch),
      comparison_op(comparison_op),
      categories(categories),
      d_categories()
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
  ThresholdT feature_lower_bound;
  ThresholdT feature_upper_bound;
  bool is_missing_branch;
  // Comparison operator used in the test. For now only < (kLT) and <= (kLE)
  // are supported.
  tl::Operator comparison_op;
  // List of matching categories for this path
  CatBitField<false> categories;
  CatBitField<true> d_categories;

  // Does this instance flow down this path?
  __host__ __device__ bool EvaluateSplit(ThresholdT x) const
  {
#ifdef __CUDA_ARCH__
    constexpr bool is_device = true;
#else  // __CUDA_ARCH__
    constexpr bool is_device = false;
#endif
    static_assert(std::is_floating_point<ThresholdT>::value, "x must be a floating point type");
    auto max_representable_int =
      static_cast<ThresholdT>(uint64_t(1) << std::numeric_limits<ThresholdT>::digits);
    if (isnan(x)) { return is_missing_branch; }
    if constexpr (is_device) {
      if (d_categories.Size() != 0) {
        if (x < 0 || std::fabs(x) > max_representable_int) { return false; }
        return d_categories.Check(static_cast<std::size_t>(x));
      }
    } else {
      if (categories.Size() != 0) {
        if (x < 0 || std::fabs(x) > max_representable_int) { return false; }
        return categories.Check(static_cast<std::size_t>(x));
      }
    }
    if (comparison_op == tl::Operator::kLE) {
      return x > feature_lower_bound && x <= feature_upper_bound;
    }
    return x >= feature_lower_bound && x < feature_upper_bound;
  }

  // Combine two split conditions on the same feature
  __host__ __device__ void Merge(const SplitCondition& other)
  {  // Combine duplicate features
#ifdef __CUDA_ARCH__
    constexpr bool is_device = true;
#else  // __CUDA_ARCH__
    constexpr bool is_device = false;
#endif
    bool has_category = false;
    if constexpr (is_device) {
      has_category = (d_categories.Size() != 0 || other.d_categories.Size() != 0);
    } else {
      has_category = (categories.Size() != 0 || other.categories.Size() != 0);
    }
    if (has_category) {
      if constexpr (is_device) {
        d_categories.Intersect(other.d_categories);
      } else {
        categories.Intersect(other.categories);
      }
    } else {
      feature_lower_bound = max(feature_lower_bound, other.feature_lower_bound);
      feature_upper_bound = min(feature_upper_bound, other.feature_upper_bound);
    }
    is_missing_branch = is_missing_branch && other.is_missing_branch;
  }

  static_assert(std::is_same<ThresholdT, float>::value || std::is_same<ThresholdT, double>::value,
                "ThresholdT must be a float or double");
};

template <typename ThresholdT, typename LeafT>
struct CategoricalSplitCounter {
  int n_features;
  std::vector<CatT> n_categories;
  // n_categories[K] = number of category values for feature K
  // Set to 0 for numerical features
  std::vector<std::int64_t> feature_id;
  // feature_id[I] = feature ID associated with the I-th path segment

  CategoricalSplitCounter(int n_features)
    : n_features(n_features), n_categories(n_features, 0), feature_id()
  {
  }

  void node_handler(const tl::Tree<ThresholdT, LeafT>& tree, int, int parent_idx, int, float)
  {
    const auto split_index = tree.SplitIndex(parent_idx);
    if (tree.NodeType(parent_idx) == tl::TreeNodeType::kCategoricalTestNode) {
      CatT max_cat = 0;
      for (CatT cat : tree.CategoryList(parent_idx)) {
        if (cat > max_cat) { max_cat = cat; }
      }
      n_categories[split_index] = std::max(n_categories[split_index], max_cat + 1);
    }
    feature_id.push_back(split_index);
  }

  void root_handler(const tl::Tree<ThresholdT, LeafT>&, int, int, float)
  {
    feature_id.push_back(-1);
  }

  void new_path_handler() {}
};

template <typename ThresholdT, typename LeafT>
struct PathSegmentExtractor {
  using PathElementT = gpu_treeshap::PathElement<SplitCondition<ThresholdT>>;
  std::vector<PathElementT>& path_segments;
  std::size_t& path_idx;
  std::vector<CatBitFieldStorageT>& categorical_bitfields;
  const std::vector<std::size_t>& bitfield_segments;
  std::size_t path_segment_idx;

  static constexpr ThresholdT inf{std::numeric_limits<ThresholdT>::infinity()};

  PathSegmentExtractor(std::vector<PathElementT>& path_segments,
                       std::size_t& path_idx,
                       std::vector<CatBitFieldStorageT>& categorical_bitfields,
                       const std::vector<std::size_t>& bitfield_segments)
    : path_segments(path_segments),
      path_idx(path_idx),
      categorical_bitfields(categorical_bitfields),
      bitfield_segments(bitfield_segments),
      path_segment_idx(0)
  {
  }

  void node_handler(
    const tl::Tree<ThresholdT, LeafT>& tree, int child_idx, int parent_idx, int group_id, float v)
  {
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
    bool is_left_path      = tree.LeftChild(parent_idx) == child_idx;
    bool is_missing_branch = tree.DefaultChild(parent_idx) == child_idx;
    auto node_type         = tree.NodeType(parent_idx);
    ThresholdT lower_bound, upper_bound;
    tl::Operator comparison_op;
    CatBitField<false> categories;
    if (node_type == tl::TreeNodeType::kCategoricalTestNode) {
      /* Create bit fields to store the list of categories associated with this path.
         The bit fields will be used to quickly decide whether a feature value should
         flow down down this path or not.
         The test in the test node is of form: x \in { list of category values } */
      auto n_bitfields =
        bitfield_segments[path_segment_idx + 1] - bitfield_segments[path_segment_idx];
      categories = CatBitField<false>(raft::span<CatBitFieldStorageT, false>(
                                        categorical_bitfields.data(), categorical_bitfields.size())
                                        .subspan(bitfield_segments[path_segment_idx], n_bitfields));
      for (CatT cat : tree.CategoryList(parent_idx)) {
        categories.Set(static_cast<std::size_t>(cat));
      }
      // If this path is not the path that's taken when the categorical test evaluates to be true,
      // then flip all the bits in the bit fields. This step is needed because we first built
      // the bit fields according to the list given in the categorical test.
      bool use_right = tree.CategoryListRightChild(parent_idx);
      if ((use_right && is_left_path) || (!use_right && !is_left_path)) {
        for (std::size_t i = bitfield_segments[path_segment_idx];
             i < bitfield_segments[path_segment_idx + 1];
             ++i) {
          categorical_bitfields[i] = ~categorical_bitfields[i];
        }
      }
      lower_bound   = -inf;
      upper_bound   = inf;
      comparison_op = tl::Operator::kNone;
    } else {
      if (node_type != tl::TreeNodeType::kNumericalTestNode) {
        // Assume: split is either numerical or categorical
        RAFT_FAIL("Unexpected node type: %d", static_cast<int>(node_type));
      }
      categories    = CatBitField<false>{};
      lower_bound   = is_left_path ? -inf : tree.Threshold(parent_idx);
      upper_bound   = is_left_path ? tree.Threshold(parent_idx) : inf;
      comparison_op = tree.ComparisonOp(parent_idx);
    }
    path_segments.push_back(gpu_treeshap::PathElement<SplitCondition<ThresholdT>>{
      path_idx,
      tree.SplitIndex(parent_idx),
      group_id,
      SplitCondition{lower_bound, upper_bound, is_missing_branch, comparison_op, categories},
      zero_fraction,
      v});
    ++path_segment_idx;
  }

  void root_handler(const tl::Tree<ThresholdT, LeafT>& tree, int child_idx, int group_id, float v)
  {
    // Root node has feature -1
    auto comparison_op = tree.ComparisonOp(child_idx);
    path_segments.push_back(gpu_treeshap::PathElement<SplitCondition<ThresholdT>>{
      path_idx, -1, group_id, SplitCondition{-inf, inf, false, comparison_op, {}}, 1.0, v});
    ++path_segment_idx;
  }

  void new_path_handler() { ++path_idx; }
};

};  // namespace
namespace ML {
namespace Explainer {
template <typename ThresholdT>
class TreePathInfo {
 public:
  int num_tree;
  float global_bias;
  std::size_t num_groups = 1;
  tl::TaskType task_type;
  bool average_tree_output;
  thrust::device_vector<gpu_treeshap::PathElement<SplitCondition<ThresholdT>>> path_segments;
  thrust::device_vector<CatBitFieldStorageT> categorical_bitfields;
  // bitfield_segments[I]: cumulative total count of all bit fields for path segments
  //                       0, 1, ..., I-1

  static_assert(std::is_same<ThresholdT, float>::value || std::is_same<ThresholdT, double>::value,
                "ThresholdT must be a float or double");
};
}  // namespace Explainer
}  // namespace ML

namespace {
template <typename DataT>
class DenseDatasetWrapper {
  const DataT* data;
  std::size_t num_rows;
  std::size_t num_cols;

 public:
  DenseDatasetWrapper() = default;
  DenseDatasetWrapper(const DataT* data, int num_rows, int num_cols)
    : data(data), num_rows(num_rows), num_cols(num_cols)
  {
  }
  __device__ DataT GetElement(std::size_t row_idx, std::size_t col_idx) const
  {
    return data[row_idx * num_cols + col_idx];
  }
  __host__ __device__ std::size_t NumRows() const { return num_rows; }
  __host__ __device__ std::size_t NumCols() const { return num_cols; }
};

template <typename ThresholdT, typename DataT>
void post_process(ML::Explainer::TreePathInfo<ThresholdT>* path_info,
                  std::size_t n_rows,
                  std::size_t n_cols,
                  DataT* out_preds,
                  std::size_t pred_size,
                  bool interactions)
{
  auto count_iter  = thrust::make_counting_iterator(0);
  auto num_tree    = path_info->num_tree;
  auto global_bias = path_info->global_bias;
  auto num_groups  = path_info->num_groups;
  if (path_info->average_tree_output) {
    thrust::for_each(
      thrust::device, count_iter, count_iter + pred_size, [=] __device__(std::size_t idx) {
        out_preds[idx] /= num_tree;
      });
  }
  // Set the global bias
  if (interactions) {
    thrust::for_each(thrust::device,
                     count_iter,
                     count_iter + (n_rows * num_groups),
                     [=] __device__(std::size_t idx) {
                       size_t group   = idx % num_groups;
                       size_t row_idx = idx / num_groups;
                       out_preds[gpu_treeshap::IndexPhiInteractions(
                         row_idx, num_groups, group, n_cols, n_cols, n_cols)] += global_bias;
                     });
  } else {
    thrust::for_each(
      thrust::device,
      count_iter,
      count_iter + (n_rows * num_groups),
      [=] __device__(std::size_t idx) { out_preds[(idx + 1) * (n_cols + 1) - 1] += global_bias; });
  }
}

template <typename ThresholdT, typename DataT>
void gpu_treeshap_impl(ML::Explainer::TreePathInfo<ThresholdT>* path_info,
                       const DataT* data,
                       std::size_t n_rows,
                       std::size_t n_cols,
                       DataT* out_preds,
                       std::size_t out_preds_size)
{
  DenseDatasetWrapper<DataT> X(data, n_rows, n_cols);

  std::size_t pred_size = n_rows * path_info->num_groups * (n_cols + 1);
  ASSERT(pred_size <= out_preds_size, "Predictions array is too small.");

  gpu_treeshap::GPUTreeShap(X,
                            path_info->path_segments.begin(),
                            path_info->path_segments.end(),
                            path_info->num_groups,
                            thrust::device_pointer_cast(out_preds),
                            thrust::device_pointer_cast(out_preds) + pred_size);

  // Post-processing
  post_process(path_info, n_rows, n_cols, out_preds, pred_size, false);
}

template <typename ThresholdT, typename DataT>
void gpu_treeshap_interventional_impl(ML::Explainer::TreePathInfo<ThresholdT>* path_info,
                                      const DataT* data,
                                      std::size_t n_rows,
                                      std::size_t n_cols,
                                      const DataT* background_data,
                                      std::size_t background_n_rows,
                                      std::size_t background_n_cols,
                                      DataT* out_preds,
                                      std::size_t out_preds_size)
{
  DenseDatasetWrapper<DataT> X(data, n_rows, n_cols);
  DenseDatasetWrapper<DataT> R(background_data, background_n_rows, background_n_cols);
  ASSERT(n_cols == background_n_cols,
         "Dataset and background dataset have different number of columns.");

  std::size_t pred_size = n_rows * path_info->num_groups * (n_cols + 1);
  ASSERT(pred_size <= out_preds_size, "Predictions array is too small.");

  gpu_treeshap::GPUTreeShapInterventional(X,
                                          R,
                                          path_info->path_segments.begin(),
                                          path_info->path_segments.end(),
                                          path_info->num_groups,
                                          thrust::device_pointer_cast(out_preds),
                                          thrust::device_pointer_cast(out_preds) + pred_size);

  // Post-processing
  post_process(path_info, n_rows, n_cols, out_preds, pred_size, false);
}

template <typename ThresholdT, typename DataT>
void gpu_treeshap_interactions_impl(ML::Explainer::TreePathInfo<ThresholdT>* path_info,
                                    const DataT* data,
                                    std::size_t n_rows,
                                    std::size_t n_cols,
                                    DataT* out_preds,
                                    std::size_t out_preds_size)
{
  DenseDatasetWrapper<DataT> X(data, n_rows, n_cols);

  std::size_t pred_size = n_rows * path_info->num_groups * (n_cols + 1) * (n_cols + 1);
  ASSERT(pred_size <= out_preds_size, "Predictions array is too small.");

  gpu_treeshap::GPUTreeShapInteractions(X,
                                        path_info->path_segments.begin(),
                                        path_info->path_segments.end(),
                                        path_info->num_groups,
                                        thrust::device_pointer_cast(out_preds),
                                        thrust::device_pointer_cast(out_preds) + pred_size);

  // Post-processing
  post_process(path_info, n_rows, n_cols, out_preds, pred_size, true);
}

template <typename ThresholdT, typename DataT>
void gpu_treeshap_taylor_interactions_impl(ML::Explainer::TreePathInfo<ThresholdT>* path_info,
                                           const DataT* data,
                                           std::size_t n_rows,
                                           std::size_t n_cols,
                                           DataT* out_preds,
                                           std::size_t out_preds_size)
{
  DenseDatasetWrapper<DataT> X(data, n_rows, n_cols);

  std::size_t pred_size = n_rows * path_info->num_groups * (n_cols + 1) * (n_cols + 1);
  ASSERT(pred_size <= out_preds_size, "Predictions array is too small.");

  gpu_treeshap::GPUTreeShapTaylorInteractions(X,
                                              path_info->path_segments.begin(),
                                              path_info->path_segments.end(),
                                              path_info->num_groups,
                                              thrust::device_pointer_cast(out_preds),
                                              thrust::device_pointer_cast(out_preds) + pred_size);

  // Post-processing
  post_process(path_info, n_rows, n_cols, out_preds, pred_size, true);
}
}  // anonymous namespace

namespace ML {
namespace Explainer {
// Traverse a path from the root node to a leaf node and call the handler functions for each node.
// The fields group_id and v (leaf value) will be passed to the handler.
template <typename ThresholdT, typename LeafT, typename PathHandler>
void traverse_towards_leaf_node(const tl::Tree<ThresholdT, LeafT>& tree,
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
template <typename ThresholdT, typename LeafT, typename PathHandler>
void visit_path_segments_in_tree(const std::vector<tl::Tree<ThresholdT, LeafT>>& tree_list,
                                 std::size_t tree_idx,
                                 bool use_vector_leaf,
                                 int num_groups,
                                 PathHandler& path_handler)
{
  if (num_groups < 1) { RAFT_FAIL("num_groups must be at least 1"); }

  const tl::Tree<ThresholdT, LeafT>& tree = tree_list[tree_idx];

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
template <typename ThresholdT, typename LeafT, typename PathHandler>
void visit_path_segments_in_model(const tl::Model& model,
                                  const tl::ModelPreset<ThresholdT, LeafT>& model_preset,
                                  PathHandler& path_handler)
{
  int num_groups = 1;
  bool use_vector_leaf;
  ASSERT(model.num_target == 1, "TreeExplainer currently does not support multi-target models");
  if (model.num_class[0] > 1) { num_groups = model.num_class[0]; }
  if (model.leaf_vector_shape[0] == 1 && model.leaf_vector_shape[1] == 1) {
    use_vector_leaf = false;
  } else {
    use_vector_leaf = true;
  }

  for (std::size_t tree_idx = 0; tree_idx < model_preset.trees.size(); ++tree_idx) {
    visit_path_segments_in_tree(
      model_preset.trees, tree_idx, use_vector_leaf, num_groups, path_handler);
  }
}

// Traverse a path from the root node to a leaf node and return the list of the path segments
// Note: the path segments will have missing values in path_idx, group_id and v (leaf value).
//       The caller is responsible for filling in these fields.
template <typename ThresholdT, typename LeafT>
std::vector<gpu_treeshap::PathElement<SplitCondition<ThresholdT>>> traverse_towards_leaf_node(
  const tl::Tree<ThresholdT, LeafT>& tree, int leaf_node_id, const std::vector<int>& parent_id)
{
  std::vector<gpu_treeshap::PathElement<SplitCondition<ThresholdT>>> path_segments;
  int child_idx              = leaf_node_id;
  int parent_idx             = parent_id[child_idx];
  constexpr auto inf         = std::numeric_limits<ThresholdT>::infinity();
  tl::Operator comparison_op = tl::Operator::kNone;
  while (parent_idx != -1) {
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
    if (tree.NodeType(parent_idx) == tl::TreeNodeType::kCategoricalTestNode) {
      RAFT_FAIL(
        "Only trees with numerical splits are supported. "
        "Trees with categorical splits are not supported yet.");
    }
    ThresholdT lower_bound = is_left_path ? -inf : tree.Threshold(parent_idx);
    ThresholdT upper_bound = is_left_path ? tree.Threshold(parent_idx) : inf;
    comparison_op          = tree.ComparisonOp(parent_idx);
    path_segments.push_back(gpu_treeshap::PathElement<SplitCondition<ThresholdT>>{
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
  path_segments.push_back(
    gpu_treeshap::PathElement<SplitCondition<ThresholdT>>{~std::size_t(0),
                                                          -1,
                                                          -1,
                                                          SplitCondition{-inf, inf, comparison_op},
                                                          1.0,
                                                          std::numeric_limits<float>::quiet_NaN()});
  return path_segments;
}

template <typename ThresholdT, typename LeafT>
TreePathHandle extract_path_info_impl(const tl::Model& model,
                                      const tl::ModelPreset<ThresholdT, LeafT>& model_preset)
{
  auto path_info = std::make_shared<TreePathInfo<ThresholdT>>();

  /* 1. Scan the model for categorical splits and pre-allocate bit fields. */
  CategoricalSplitCounter<ThresholdT, LeafT> cat_counter{model.num_feature};
  visit_path_segments_in_model(model, model_preset, cat_counter);

  std::size_t n_path_segments = cat_counter.feature_id.size();
  std::vector<std::size_t> n_bitfields(n_path_segments, 0);
  // n_bitfields[I] : number of bit fields for path segment I

  std::transform(cat_counter.feature_id.cbegin(),
                 cat_counter.feature_id.cend(),
                 n_bitfields.begin(),
                 [&](std::int64_t fid) -> std::size_t {
                   if (fid == -1) { return 0; }
                   return CatBitField<false>::ComputeStorageSize(cat_counter.n_categories[fid]);
                 });

  std::vector<std::size_t> bitfield_segments(n_path_segments + 1, 0);
  std::inclusive_scan(n_bitfields.cbegin(), n_bitfields.cend(), bitfield_segments.begin() + 1);

  std::vector<CatBitFieldStorageT> categorical_bitfields(bitfield_segments.back(), 0);

  /* 2. Scan the model again, to extract path segments. */
  // Each path segment will have path_idx field, which uniquely identifies the path to which the
  // segment belongs.
  std::size_t path_idx = 0;
  std::vector<gpu_treeshap::PathElement<SplitCondition<ThresholdT>>> path_segments;
  PathSegmentExtractor<ThresholdT, LeafT> path_extractor{
    path_segments, path_idx, categorical_bitfields, bitfield_segments};
  visit_path_segments_in_model(model, model_preset, path_extractor);

  // Marshall bit fields to GPU memory
  path_info->categorical_bitfields = thrust::device_vector<CatBitFieldStorageT>(
    categorical_bitfields.cbegin(), categorical_bitfields.cend());
  for (std::size_t path_seg_idx = 0; path_seg_idx < path_segments.size(); ++path_seg_idx) {
    auto n_bitfields = bitfield_segments[path_seg_idx + 1] - bitfield_segments[path_seg_idx];
    path_segments[path_seg_idx].split_condition.d_categories =
      CatBitField<true>(raft::span<CatBitFieldStorageT, true>(
                          thrust::raw_pointer_cast(path_info->categorical_bitfields.data()),
                          path_info->categorical_bitfields.size())
                          .subspan(bitfield_segments[path_seg_idx], n_bitfields));
  }

  path_info->path_segments       = path_segments;
  path_info->global_bias         = model.base_scores[0];
  path_info->task_type           = model.task_type;
  path_info->average_tree_output = model.average_tree_output;
  path_info->num_tree            = static_cast<int>(model_preset.trees.size());
  path_info->num_groups          = static_cast<std::size_t>(model.num_class[0]);

  return path_info;
}

TreePathHandle extract_path_info(TreeliteModelHandle model)
{
  const tl::Model& model_ref = *static_cast<tl::Model*>(model);

  return std::visit(
    [&](auto&& model_preset) { return extract_path_info_impl(model_ref, model_preset); },
    model_ref.variant_);
}

template <typename VariantT, typename... Targs>
bool variants_hold_same_type(VariantT& first, Targs... args)
{
  bool is_same = true;
  std::visit(
    [&](auto v) {
      for (const auto& x : {args...}) {
        is_same = is_same && std::holds_alternative<decltype(v)>(x);
      }
    },
    first);
  return is_same;
}

void gpu_treeshap(TreePathHandle path_info,
                  const FloatPointer data,
                  std::size_t n_rows,
                  std::size_t n_cols,
                  FloatPointer out_preds,
                  std::size_t out_preds_size)
{
  ASSERT(variants_hold_same_type(data, out_preds),
         "Expected variant inputs to have the same data type.");
  std::visit(
    [&](auto& tree_info, auto data_) {
      gpu_treeshap_impl(tree_info.get(),
                        data_,
                        n_rows,
                        n_cols,
                        std::get<decltype(data_)>(out_preds),
                        out_preds_size);
    },
    path_info,
    data);
}

void gpu_treeshap_interventional(TreePathHandle path_info,
                                 const FloatPointer data,
                                 std::size_t n_rows,
                                 std::size_t n_cols,
                                 const FloatPointer background_data,
                                 std::size_t background_n_rows,
                                 std::size_t background_n_cols,
                                 FloatPointer out_preds,
                                 std::size_t out_preds_size)
{
  ASSERT(variants_hold_same_type(data, background_data, out_preds),
         "Expected variant inputs to have the same data type.");
  std::visit(
    [&](auto& tree_info, auto data_) {
      gpu_treeshap_interventional_impl(tree_info.get(),
                                       data_,
                                       n_rows,
                                       n_cols,
                                       std::get<decltype(data_)>(background_data),
                                       background_n_rows,
                                       background_n_cols,
                                       std::get<decltype(data_)>(out_preds),
                                       out_preds_size);
    },
    path_info,
    data);
}
void gpu_treeshap_interactions(TreePathHandle path_info,
                               const FloatPointer data,
                               std::size_t n_rows,
                               std::size_t n_cols,
                               FloatPointer out_preds,
                               std::size_t out_preds_size)
{
  ASSERT(variants_hold_same_type(data, out_preds),
         "Expected variant inputs to have the same data type.");
  std::visit(
    [&](auto& tree_info, auto data_) {
      gpu_treeshap_interactions_impl(tree_info.get(),
                                     data_,
                                     n_rows,
                                     n_cols,
                                     std::get<decltype(data_)>(out_preds),
                                     out_preds_size);
    },
    path_info,
    data);
}

void gpu_treeshap_taylor_interactions(TreePathHandle path_info,
                                      const FloatPointer data,
                                      std::size_t n_rows,
                                      std::size_t n_cols,
                                      FloatPointer out_preds,
                                      std::size_t out_preds_size)
{
  ASSERT(variants_hold_same_type(data, out_preds),
         "Expected variant inputs to have the same data type.");
  std::visit(
    [&](auto& tree_info, auto data_) {
      gpu_treeshap_taylor_interactions_impl(tree_info.get(),
                                            data_,
                                            n_rows,
                                            n_cols,
                                            std::get<decltype(data_)>(out_preds),
                                            out_preds_size);
    },
    path_info,
    data);
}
}  // namespace Explainer
}  // namespace ML
