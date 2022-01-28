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
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <cuml/explainer/tree_shap.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
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
  __host__ std::string ToString(bool reverse = false) const
  {
    std::ostringstream oss;
    oss << "Bits storage size: " << bits_.Size() << ", elements: ";
    for (auto i = 0; i < bits_.Size(); ++i) {
      std::bitset<kValueSize> bset(bits_[i]);
      std::string s = bset.to_string();
      if (reverse) { std::reverse(s.begin(), s.end()); }
      oss << s << ", ";
    }
    return oss.str();
  }

  static_assert(!std::is_signed<T>::value, "Must use unsiged type as underlying storage.");
};

using CatBitFieldStorageT = std::uint32_t;
using CatBitField         = BitField<CatBitFieldStorageT>;
using CatT                = std::uint32_t;

template <typename ThresholdType>
struct SplitCondition {
  SplitCondition() = default;
  SplitCondition(ThresholdType feature_lower_bound,
                 ThresholdType feature_upper_bound,
                 bool is_missing_branch,
                 tl::Operator comparison_op,
                 CatBitField categories)
    : feature_lower_bound(feature_lower_bound),
      feature_upper_bound(feature_upper_bound),
      is_missing_branch(is_missing_branch),
      comparison_op(comparison_op),
      categories(categories)
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
  bool is_missing_branch;
  // Comparison operator used in the test. For now only < (kLT) and <= (kLE)
  // are supported.
  tl::Operator comparison_op;
  CatBitField categories;

  // Does this instance flow down this path?
  __host__ __device__ bool EvaluateSplit(ThresholdType x) const
  {
    static_assert(std::is_floating_point<ThresholdType>::value, "x must be a floating point type");
    auto max_representable_int =
      static_cast<ThresholdType>(uint64_t(1) << std::numeric_limits<ThresholdType>::digits);
    if (isnan(x)) { return is_missing_branch; }
    if (categories.Size() != 0) {
      if (x < 0 || std::fabs(x) > max_representable_int) { return false; }
      return categories.Check(static_cast<std::size_t>(x));
    }
    if (comparison_op == tl::Operator::kLE) {
      return x > feature_lower_bound && x <= feature_upper_bound;
    }
    return x >= feature_lower_bound && x < feature_upper_bound;
  }

  // Combine two split conditions on the same feature
  __host__ __device__ void Merge(const SplitCondition& other)
  {  // Combine duplicate features
    if (categories.Size() != 0 || other.categories.Size() != 0) {
      categories.Intersect(other.categories);
    } else {
      feature_lower_bound = max(feature_lower_bound, other.feature_lower_bound);
      feature_upper_bound = min(feature_upper_bound, other.feature_upper_bound);
    }
    is_missing_branch = is_missing_branch && other.is_missing_branch;
  }
  static_assert(std::is_same<ThresholdType, float>::value ||
                  std::is_same<ThresholdType, double>::value,
                "ThresholdType must be a float or double");
};

template <typename ThresholdType, typename LeafType>
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

  void node_handler(const tl::Tree<ThresholdType, LeafType>& tree, int, int parent_idx, int, float)
  {
    const auto split_index = tree.SplitIndex(parent_idx);
    if (tree.SplitType(parent_idx) == tl::SplitFeatureType::kCategorical) {
      CatT max_cat = 0;
      for (CatT cat : tree.MatchingCategories(parent_idx)) {
        if (cat > max_cat) { max_cat = cat; }
      }
      n_categories[split_index] = std::max(n_categories[split_index], max_cat + 1);
    }
    feature_id.push_back(split_index);
  }

  void root_handler(const tl::Tree<ThresholdType, LeafType>&, int, int, float)
  {
    feature_id.push_back(-1);
  }

  void new_path_handler() {}
};

template <typename ThresholdType, typename LeafType>
struct PathSegmentExtractor {
  using PathElementT = gpu_treeshap::PathElement<SplitCondition<ThresholdType>>;
  std::vector<PathElementT>& path_segments;
  std::size_t& path_idx;
  std::vector<CatBitFieldStorageT>& cat_bitfields;
  const std::vector<std::size_t>& bitfield_segments;
  std::size_t path_segment_idx;

  static constexpr ThresholdType inf{std::numeric_limits<ThresholdType>::infinity()};

  PathSegmentExtractor(std::vector<PathElementT>& path_segments,
                       std::size_t& path_idx,
                       std::vector<CatBitFieldStorageT>& cat_bitfields,
                       const std::vector<std::size_t>& bitfield_segments)
    : path_segments(path_segments),
      path_idx(path_idx),
      cat_bitfields(cat_bitfields),
      bitfield_segments(bitfield_segments),
      path_segment_idx(0)
  {
  }

  void node_handler(const tl::Tree<ThresholdType, LeafType>& tree,
                    int child_idx,
                    int parent_idx,
                    int group_id,
                    float v)
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
    auto split_type        = tree.SplitType(parent_idx);
    ThresholdType lower_bound, upper_bound;
    tl::Operator comparison_op;
    CatBitField categories;
    if (split_type == tl::SplitFeatureType::kCategorical) {
      /* Create bit fields to store the list of categories associated with this path.
         The bit fields will be used to quickly decide whether a feature value should
         flow down down this path or not.
         The test in the test node is of form: x \in { list of category values } */
      auto n_bitfields =
        bitfield_segments[path_segment_idx + 1] - bitfield_segments[path_segment_idx];
      categories = CatBitField(Span<CatBitFieldStorageT>(cat_bitfields)
                                 .Subspan(bitfield_segments[path_segment_idx], n_bitfields));
      for (CatT cat : tree.MatchingCategories(parent_idx)) {
        categories.Set(static_cast<std::size_t>(cat));
      }
      // If this path is not the path that's taken when the categorical test evaluates to be true,
      // then flip all the bits in the bit fields. This step is needed because we first built
      // the bit fields according to the list given in the categorical test.
      bool use_right = tree.CategoriesListRightChild(parent_idx);
      if ((use_right && is_left_path) || (!use_right && !is_left_path)) {
        for (std::size_t i = bitfield_segments[path_segment_idx];
             i < bitfield_segments[path_segment_idx + 1];
             ++i) {
          cat_bitfields[i] = ~cat_bitfields[i];
        }
      }
      lower_bound   = -inf;
      upper_bound   = inf;
      comparison_op = tl::Operator::kNone;
    } else {
      if (split_type != tl::SplitFeatureType::kNumerical) {
        // Assume: split is either numerical or categorical
        RAFT_FAIL("Unexpected split type: %d", static_cast<int>(split_type));
      }
      categories    = CatBitField{};
      lower_bound   = is_left_path ? -inf : tree.Threshold(parent_idx);
      upper_bound   = is_left_path ? tree.Threshold(parent_idx) : inf;
      comparison_op = tree.ComparisonOp(parent_idx);
    }
    path_segments.push_back(gpu_treeshap::PathElement<SplitCondition<ThresholdType>>{
      path_idx,
      tree.SplitIndex(parent_idx),
      group_id,
      SplitCondition{lower_bound, upper_bound, is_missing_branch, comparison_op, categories},
      zero_fraction,
      v});
    ++path_segment_idx;
  }

  void root_handler(const tl::Tree<ThresholdType, LeafType>& tree,
                    int child_idx,
                    int group_id,
                    float v)
  {
    // Root node has feature -1
    auto comparison_op = tree.ComparisonOp(child_idx);
    path_segments.push_back(gpu_treeshap::PathElement<SplitCondition<ThresholdType>>{
      path_idx, -1, group_id, SplitCondition{-inf, inf, false, comparison_op, {}}, 1.0, v});
    ++path_segment_idx;
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
  std::vector<gpu_treeshap::PathElement<SplitCondition<ThresholdType>>> path_segments;
  std::vector<CatBitFieldStorageT> categorical_bitfields;
  std::vector<std::size_t> bitfield_segments;
  // bitfield_segments[I]: cumulative total count of all bit fields for path segments
  //                       0, 1, ..., I-1

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
void gpu_treeshap_impl(TreePathInfoImpl<ThresholdType>* path_info,
                       const float* data,
                       std::size_t n_rows,
                       std::size_t n_cols,
                       float* out_preds)
{
  // Marshall bit fields to GPU memory
  auto& categorical_bitfields = path_info->categorical_bitfields;
  auto& path_segments         = path_info->path_segments;
  auto& bitfield_segments     = path_info->bitfield_segments;
  thrust::device_vector<CatBitFieldStorageT> d_cat_bitfields(categorical_bitfields.cbegin(),
                                                             categorical_bitfields.cend());
  for (std::size_t path_seg_idx = 0; path_seg_idx < path_segments.size(); ++path_seg_idx) {
    auto n_bitfields = bitfield_segments[path_seg_idx + 1] - bitfield_segments[path_seg_idx];
    path_segments[path_seg_idx].split_condition.categories =
      CatBitField(Span<CatBitFieldStorageT>(d_cat_bitfields)
                    .Subspan(bitfield_segments[path_seg_idx], n_bitfields));
  }

  DenseDatasetWrapper X(data, n_rows, n_cols);

  std::size_t num_groups = 1;
  if (path_info->task_param.num_class > 1) {
    num_groups = static_cast<std::size_t>(path_info->task_param.num_class);
  }
  std::size_t pred_size = n_rows * num_groups * (n_cols + 1);

  thrust::device_ptr<float> out_preds_ptr = thrust::device_pointer_cast(out_preds);
  gpu_treeshap::GPUTreeShap(X,
                            path_segments.begin(),
                            path_segments.end(),
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

// Traverse a path from the root node to a leaf node and return the list of the path segments
// Note: the path segments will have missing values in path_idx, group_id and v (leaf value).
//       The callser is responsible for filling in these fields.
template <typename ThresholdType, typename LeafType>
std::vector<gpu_treeshap::PathElement<SplitCondition<ThresholdType>>> traverse_towards_leaf_node(
  const tl::Tree<ThresholdType, LeafType>& tree,
  int leaf_node_id,
  const std::vector<int>& parent_id)
{
  std::vector<gpu_treeshap::PathElement<SplitCondition<ThresholdType>>> path_segments;
  int child_idx              = leaf_node_id;
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
    comparison_op             = tree.ComparisonOp(parent_idx);
    path_segments.push_back(gpu_treeshap::PathElement<SplitCondition<ThresholdType>>{
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
  path_segments.push_back(gpu_treeshap::PathElement<SplitCondition<ThresholdType>>{
    ~std::size_t(0),
    -1,
    -1,
    SplitCondition{-inf, inf, comparison_op},
    1.0,
    std::numeric_limits<float>::quiet_NaN()});
  return path_segments;
}

// Extract the path segments from a single tree. Each path segment will have path_idx field, which
// uniquely identifies the path to which the segment belongs. The path_idx_offset parameter sets
// the path_idx field of the first path segment.
template <typename ThresholdType, typename LeafType>
std::vector<gpu_treeshap::PathElement<SplitCondition<ThresholdType>>>
extract_path_segments_from_tree(const std::vector<tl::Tree<ThresholdType, LeafType>>& tree_list,
                                std::size_t tree_idx,
                                bool use_vector_leaf,
                                int num_groups,
                                std::size_t path_idx_offset)
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

  std::size_t path_idx = path_idx_offset;
  std::vector<gpu_treeshap::PathElement<SplitCondition<ThresholdType>>> path_segments;

  for (int nid = 0; nid < tree.num_nodes; nid++) {
    if (tree.IsLeaf(nid)) {  // For each leaf node...
      // Extract path segments by traversing the path from the leaf node to the root node
      auto path_to_leaf = traverse_towards_leaf_node(tree, nid, parent_id);
      // If use_vector_leaf=True:
      // * Duplicate the path segments N times, where N = num_groups
      // * Insert the duplicated path segments into path_segments
      // If use_vector_leaf=False:
      // * Insert the path segments into path_segments
      auto path_insertor = [&path_to_leaf, &path_segments](
                             auto leaf_value, auto path_idx, int group_id) {
        for (auto& e : path_to_leaf) {
          e.path_idx = path_idx;
          e.v        = static_cast<float>(leaf_value);
          e.group    = group_id;
        }
        path_segments.insert(path_segments.end(), path_to_leaf.cbegin(), path_to_leaf.cend());
      };
      if (use_vector_leaf) {
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
        int group_id    = static_cast<int>(tree_idx) % num_groups;
        path_insertor(leaf_value, path_idx, group_id);
        path_idx++;
      }
    }
  }
  return path_segments;
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

  /* 1. Scan the model for categorical splits and pre-allocate bit fields. */
  CategoricalSplitCounter<ThresholdType, LeafType> cat_counter{model.num_feature};
  visit_path_segments_in_model(model, cat_counter);

  std::size_t n_path_segments = cat_counter.feature_id.size();
  std::vector<std::size_t> n_bitfields(n_path_segments, 0);
  // n_bitfields[I] : number of bit fields for path segment I

  std::transform(cat_counter.feature_id.cbegin(),
                 cat_counter.feature_id.cend(),
                 n_bitfields.begin(),
                 [&](std::int64_t fid) -> std::size_t {
                   if (fid == -1) { return 0; }
                   return CatBitField::ComputeStorageSize(cat_counter.n_categories[fid]);
                 });

  path_info->bitfield_segments = std::vector<std::size_t>(n_path_segments + 1, 0);
  std::inclusive_scan(
    n_bitfields.cbegin(), n_bitfields.cend(), path_info->bitfield_segments.begin() + 1);

  path_info->categorical_bitfields =
    std::vector<CatBitFieldStorageT>(path_info->bitfield_segments.back(), 0);

  /* 2. Scan the model again, to extract path segments. */
  // Each path segment will have path_idx field, which uniquely identifies the path to which the
  // segment belongs.
  std::size_t path_idx = 0;
  PathSegmentExtractor<ThresholdType, LeafType> path_extractor{path_info->path_segments,
                                                               path_idx,
                                                               path_info->categorical_bitfields,
                                                               path_info->bitfield_segments};
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

void gpu_treeshap(TreePathInfo* path_info,
                  const float* data,
                  std::size_t n_rows,
                  std::size_t n_cols,
                  float* out_preds)
{
  switch (path_info->GetThresholdType()) {
    case TreePathInfo::ThresholdTypeEnum::kDouble: {
      auto* path_info_casted = dynamic_cast<TreePathInfoImpl<double>*>(path_info);
      gpu_treeshap_impl(path_info_casted, data, n_rows, n_cols, out_preds);
    } break;
    case TreePathInfo::ThresholdTypeEnum::kFloat:
    default: {
      auto* path_info_casted = dynamic_cast<TreePathInfoImpl<float>*>(path_info);
      gpu_treeshap_impl(path_info_casted, data, n_rows, n_cols, out_preds);
    } break;
  }
}

}  // namespace Explainer
}  // namespace ML
