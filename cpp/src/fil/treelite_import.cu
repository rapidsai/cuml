/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

/** @file treelite_import.cu converts from treelite format to a FIL-centric CPU-RAM format, so that
 * fil.cu can make a `forest` object out of it. */

#include "common.cuh"    // for node_traits, num_trees, tree_num_nodes
#include "internal.cuh"  // for MAX_FIL_INT_FLOAT, BITS_PER_BYTE, cat_feature_counters, cat_sets, cat_sets_owner, categorical_sets, leaf_algo_t

#include <cuml/common/logger.hpp>  // for CUML_LOG_WARN
#include <cuml/fil/fil.h>  // for algo_t, from_treelite, storage_type_repr, storage_type_t, treelite_params_t
#include <cuml/fil/fnv_hash.h>  // for fowler_noll_vo_fingerprint64_32

#include <raft/cudart_utils.h>  // for RAFT_CUDA_TRY
#include <raft/error.hpp>       // for ASSERT
#include <raft/handle.hpp>      // for handle_t

#include <treelite/base.h>   // for Operator, SplitFeatureType, kGE, kGT, kLE, kLT, kNumerical
#include <treelite/c_api.h>  // for ModelHandle
#include <treelite/tree.h>   // for Tree, Model, ModelImpl, ModelParam

#include <omp.h>  // for omp

#include <algorithm>    // for std::max
#include <bitset>       // for std::bitset
#include <cmath>        // for NAN
#include <cstddef>      // for std::size_t
#include <cstdint>      // for uint8_t
#include <iosfwd>       // for ios, stringstream
#include <stack>        // for std::stack
#include <string>       // for std::string
#include <type_traits>  // for std::is_same

namespace ML {
namespace fil {

namespace tl = treelite;

std::ostream& operator<<(std::ostream& os, const cat_sets_owner& cso)
{
  os << "\nbits { ";
  for (uint8_t b : cso.bits) {
    os << std::bitset<BITS_PER_BYTE>(b) << " ";
  }
  os << " }\nmax_matching {";
  for (float fid_num_cats : cso.fid_num_cats) {
    os << static_cast<int>(fid_num_cats) - 1 << " ";
  }
  os << " }";
  return os;
}

template <typename T, typename L>
int tree_root(const tl::Tree<T, L>& tree)
{
  return 0;  // Treelite format assumes that the root is 0
}

// a no-op placeholder for values and callables alike
struct empty {
  template <typename... Args>
  void operator()(Args...)
  {
  }
};

/** walk a Treelite tree, visiting each inner node with visit_inner and each leaf node with
  visit_leaf. See walk_tree::element::state documentation for how TraversalState is retained
during traversal. Any per-tree state during traversal should be captured by the lambdas themselves.
  visit_inner(int node_id, TraversalState state) should return a pair of new states, one for
each child node. visit_leaf(int, TraversalState) returns nothing.
**/
template <typename T, typename L, typename InnerFunc, typename LeafFunc = empty>
inline void walk_tree(const tl::Tree<T, L>& tree,
                      InnerFunc visit_inner,
                      LeafFunc visit_leaf = empty())
{
  if constexpr (std::is_invocable<InnerFunc, int>()) {
    /// wrapper for empty path state
    walk_tree(
      tree,
      [&](int nid, empty val) {
        visit_inner(nid);
        return std::pair<empty, empty>();
      },
      [&](int nid, empty val) { visit_leaf(nid); });
  } else {
    using TraversalState = decltype(visit_inner(int(), {}).first);
    /// needed to visit a node
    struct element {
      int tl_node_id;
      /// Retained while visiting nodes on a single path from root to leaf.
      /// This generalizes the node index that's carried over during inference tree traversal.
      TraversalState state;
    };
    std::stack<element> stack;
    stack.push(element{tree_root(tree), TraversalState()});
    while (!stack.empty()) {
      element i = stack.top();
      stack.pop();
      while (!tree.IsLeaf(i.tl_node_id)) {
        auto [left_state, right_state] = visit_inner(i.tl_node_id, i.state);
        stack.push(element{tree.LeftChild(i.tl_node_id), left_state});
        i = element{tree.RightChild(i.tl_node_id), right_state};
      }
      visit_leaf(i.tl_node_id, i.state);
    }
  }
}

template <typename T, typename L>
inline int max_depth(const tl::Tree<T, L>& tree)
{
  int tree_depth = 0;
  walk_tree(
    tree,
    [](int node_id, int node_depth) {
      // trees of this depth aren't used, so it most likely means bad input data,
      // e.g. cycles in the forest
      constexpr int DEPTH_LIMIT = 500;
      ASSERT(node_depth < DEPTH_LIMIT, "node_depth limit reached, might be a cycle in the tree");
      return std::pair(node_depth + 1, node_depth + 1);
    },
    [&](int node_id, int node_depth) { tree_depth = std::max(node_depth, tree_depth); });
  return tree_depth;
}

template <typename T, typename L>
int max_depth(const tl::ModelImpl<T, L>& model)
{
  int depth         = 0;
  const auto& trees = model.trees;
#pragma omp parallel for reduction(max : depth)
  for (size_t i = 0; i < trees.size(); ++i) {
    const auto& tree = trees[i];
    depth            = std::max(depth, max_depth(tree));
  }
  return depth;
}

void elementwise_combine(std::vector<cat_feature_counters>& dst,
                         const std::vector<cat_feature_counters>& extra)
{
  std::transform(dst.begin(), dst.end(), extra.begin(), dst.begin(), cat_feature_counters::combine);
}

// constructs a vector of size n_cols (number of features, or columns) from a Treelite tree,
// where each feature has a maximum matching category and node count (from this tree alone).
template <typename T, typename L>
inline std::vector<cat_feature_counters> cat_counter_vec(const tl::Tree<T, L>& tree, int n_cols)
{
  std::vector<cat_feature_counters> res(n_cols);
  walk_tree(tree, [&](int node_id) {
    if (tree.SplitType(node_id) == tl::SplitFeatureType::kCategorical) {
      std::vector<std::uint32_t> mmv = tree.MatchingCategories(node_id);
      int max_matching_cat;
      if (mmv.size() > 0) {
        // in `struct cat_feature_counters` and GPU structures, int(max_matching_cat) is safe
        // because all precise int floats fit into ints, which are asserted to be 32 bits
        max_matching_cat = mmv.back();
        ASSERT(max_matching_cat <= MAX_FIL_INT_FLOAT,
               "FIL cannot infer on "
               "more than %d matching categories",
               MAX_FIL_INT_FLOAT);
      } else {
        max_matching_cat = -1;
      }
      cat_feature_counters& counters = res[tree.SplitIndex(node_id)];
      counters = cat_feature_counters::combine(counters, cat_feature_counters{max_matching_cat, 1});
    }
  });

  return res;
}

// computes overall categorical bit pool size for a tree imported from the Treelite tree
template <typename T, typename L>
inline std::size_t bit_pool_size(const tl::Tree<T, L>& tree, const categorical_sets& cat_sets)
{
  std::size_t size = 0;
  walk_tree(tree, [&](int node_id) {
    if (tree.SplitType(node_id) == tl::SplitFeatureType::kCategorical &&
        tree.MatchingCategories(node_id).size() > 0) {
      size += cat_sets.sizeof_mask(tree.SplitIndex(node_id));
    }
  });
  return size;
}

template <typename T, typename L>
cat_sets_owner allocate_cat_sets_owner(const tl::ModelImpl<T, L>& model)
{
#pragma omp declare reduction(cat_counter_vec_red : std::vector<cat_feature_counters> \
      : elementwise_combine(omp_out, omp_in))                 \
    initializer(omp_priv = omp_orig)
  const auto& trees = model.trees;
  cat_sets_owner cat_sets;
  std::vector<cat_feature_counters> counters(model.num_feature);
#pragma omp parallel for reduction(cat_counter_vec_red : counters)
  for (std::size_t i = 0; i < trees.size(); ++i) {
    elementwise_combine(counters, cat_counter_vec(trees[i], model.num_feature));
  }
  cat_sets.consume_counters(counters);
  std::vector<std::size_t> bit_pool_sizes(trees.size());
#pragma omp parallel for
  for (std::size_t i = 0; i < trees.size(); ++i) {
    bit_pool_sizes[i] = bit_pool_size(trees[i], cat_sets.accessor());
  }
  cat_sets.consume_bit_pool_sizes(bit_pool_sizes);
  return cat_sets;
}

void adjust_threshold(float* pthreshold, bool* swap_child_nodes, tl::Operator comparison_op)
{
  // in treelite (take left node if val [op] threshold),
  // the meaning of the condition is reversed compared to FIL;
  // thus, "<" in treelite corresonds to comparison ">=" used by FIL
  // https://github.com/dmlc/treelite/blob/master/include/treelite/tree.h#L243
  if (isnan(*pthreshold)) {
    *swap_child_nodes = !*swap_child_nodes;
    return;
  }
  switch (comparison_op) {
    case tl::Operator::kLT: break;
    case tl::Operator::kLE:
      // x <= y is equivalent to x < y', where y' is the next representable float
      *pthreshold = std::nextafterf(*pthreshold, std::numeric_limits<float>::infinity());
      break;
    case tl::Operator::kGT:
      // x > y is equivalent to x >= y', where y' is the next representable float
      // left and right still need to be swapped
      *pthreshold = std::nextafterf(*pthreshold, std::numeric_limits<float>::infinity());
    case tl::Operator::kGE:
      // swap left and right
      *swap_child_nodes = !*swap_child_nodes;
      break;
    default: ASSERT(false, "only <, >, <= and >= comparisons are supported");
  }
}

/** if the vector consists of zeros and a single one, return the position
for the one (assumed class label). Else, asserts false.
If the vector contains a NAN, asserts false */
template <typename L>
int find_class_label_from_one_hot(L* vector, int len)
{
  bool found_label = false;
  int out;
  for (int i = 0; i < len; ++i) {
    if (vector[i] == static_cast<L>(1.0)) {
      ASSERT(!found_label, "label vector contains multiple 1.0f");
      out         = i;
      found_label = true;
    } else {
      ASSERT(vector[i] == static_cast<L>(0.0),
             "label vector contains values other than 0.0 and 1.0");
    }
  }
  ASSERT(found_label, "did not find 1.0f in vector");
  return out;
}

template <typename fil_node_t, typename T, typename L>
void tl2fil_leaf_payload(fil_node_t* fil_node,
                         int fil_node_id,
                         const tl::Tree<T, L>& tl_tree,
                         int tl_node_id,
                         const forest_params_t& forest_params,
                         std::vector<float>* vector_leaf,
                         size_t* leaf_counter)
{
  auto vec = tl_tree.LeafVector(tl_node_id);
  switch (forest_params.leaf_algo) {
    case leaf_algo_t::CATEGORICAL_LEAF:
      ASSERT(vec.size() == static_cast<std::size_t>(forest_params.num_classes),
             "inconsistent number of classes in treelite leaves");
      fil_node->val.idx = find_class_label_from_one_hot(&vec[0], vec.size());
      break;
    case leaf_algo_t::VECTOR_LEAF: {
      ASSERT(vec.size() == static_cast<std::size_t>(forest_params.num_classes),
             "inconsistent number of classes in treelite leaves");
      fil_node->val.idx = *leaf_counter;
      for (int k = 0; k < forest_params.num_classes; k++) {
        (*vector_leaf)[*leaf_counter * forest_params.num_classes + k] = vec[k];
      }
      (*leaf_counter)++;
      break;
    }
    case leaf_algo_t::FLOAT_UNARY_BINARY:
    case leaf_algo_t::GROVE_PER_CLASS:
      fil_node->val.f = static_cast<float>(tl_tree.LeafValue(tl_node_id));
      ASSERT(!tl_tree.HasLeafVector(tl_node_id),
             "some but not all treelite leaves have leaf_vector()");
      break;
    default: ASSERT(false, "internal error: invalid leaf_algo");
  };
}

template <typename fil_node_t>
struct conversion_state {
  fil_node_t node;
  bool swap_child_nodes;
};

// modifies cat_sets
template <typename fil_node_t, typename T, typename L>
conversion_state<fil_node_t> tl2fil_inner_node(int fil_left_child,
                                               const tl::Tree<T, L>& tree,
                                               int tl_node_id,
                                               cat_sets_owner* cat_sets,
                                               std::size_t* bit_pool_offset)
{
  int tl_left = tree.LeftChild(tl_node_id), tl_right = tree.RightChild(tl_node_id);
  val_t<float> split  = {.f = NAN};  // yes there's a default initializer already
  int feature_id      = tree.SplitIndex(tl_node_id);
  bool is_categorical = tree.SplitType(tl_node_id) == tl::SplitFeatureType::kCategorical &&
                        tree.MatchingCategories(tl_node_id).size() > 0;
  bool swap_child_nodes = false;
  if (tree.SplitType(tl_node_id) == tl::SplitFeatureType::kNumerical) {
    split.f = static_cast<float>(tree.Threshold(tl_node_id));
    adjust_threshold(&split.f, &swap_child_nodes, tree.ComparisonOp(tl_node_id));
  } else if (tree.SplitType(tl_node_id) == tl::SplitFeatureType::kCategorical) {
    // for FIL, the list of categories is always for the right child
    swap_child_nodes = !tree.CategoriesListRightChild(tl_node_id);
    if (tree.MatchingCategories(tl_node_id).size() > 0) {
      int sizeof_mask = cat_sets->accessor().sizeof_mask(feature_id);
      split.idx       = *bit_pool_offset;
      *bit_pool_offset += sizeof_mask;
      // cat_sets->bits have been zero-initialized
      uint8_t* bits = &cat_sets->bits[split.idx];
      for (std::uint32_t category : tree.MatchingCategories(tl_node_id)) {
        bits[category / BITS_PER_BYTE] |= 1 << (category % BITS_PER_BYTE);
      }
    } else {
      // always branch left in FIL. Already accounted for Treelite branching direction above.
      split.f = NAN;
    }
  } else {
    ASSERT(false, "only numerical and categorical split nodes are supported");
  }
  bool default_left = tree.DefaultLeft(tl_node_id) ^ swap_child_nodes;
  fil_node_t node(
    val_t<float>{}, split, feature_id, default_left, false, is_categorical, fil_left_child);
  return conversion_state<fil_node_t>{node, swap_child_nodes};
}

template <typename fil_node_t, typename T, typename L>
int tree2fil(std::vector<fil_node_t>& nodes,
             int root,
             const tl::Tree<T, L>& tree,
             std::size_t tree_idx,
             const forest_params_t& forest_params,
             std::vector<float>* vector_leaf,
             std::size_t* leaf_counter,
             cat_sets_owner* cat_sets)
{
  // needed if the node is sparse, to place within memory for the FIL tree
  int sparse_index = 1;
  walk_tree(
    tree,
    [&](int node_id, int fil_node_id) {
      // reserve space for child nodes
      // left is the offset of the left child node relative to the tree root
      // in the array of all nodes of the FIL sparse forest
      int left = node_traits<fil_node_t>::IS_DENSE ? 2 * fil_node_id + 1 : sparse_index;
      sparse_index += 2;
      conversion_state<fil_node_t> cs = tl2fil_inner_node<fil_node_t>(
        left, tree, node_id, cat_sets, &cat_sets->bit_pool_offsets[tree_idx]);
      nodes[root + fil_node_id] = cs.node;

      return cs.swap_child_nodes ? std::pair(left + 1, left) : std::pair(left, left + 1);
    },
    [&](int node_id, int fil_node_id) {
      nodes[root + fil_node_id] = fil_node_t({}, {}, 0, false, true, false, 0);
      tl2fil_leaf_payload(&nodes[root + fil_node_id],
                          root + fil_node_id,
                          tree,
                          node_id,
                          forest_params,
                          vector_leaf,
                          leaf_counter);
    });
  return root;
}

struct level_entry {
  int n_branch_nodes, n_leaves;
};
// hist has branch and leaf count given depth
template <typename T, typename L>
inline void node_depth_hist(const tl::Tree<T, L>& tree, std::vector<level_entry>& hist)
{
  walk_tree(
    tree,
    [&](int node_id, std::size_t depth) {
      if (depth >= hist.size()) hist.resize(depth + 1, {0, 0});
      hist[depth].n_branch_nodes++;
      return std::pair(depth + 1, depth + 1);
    },
    [&](int node_id, std::size_t depth) {
      if (depth >= hist.size()) hist.resize(depth + 1, {0, 0});
      hist[depth].n_leaves++;
    });
}

template <typename T, typename L>
std::stringstream depth_hist_and_max(const tl::ModelImpl<T, L>& model)
{
  using namespace std;
  vector<level_entry> hist;
  for (const auto& tree : model.trees)
    node_depth_hist(tree, hist);

  int min_leaf_depth = -1, leaves_times_depth = 0, total_branches = 0, total_leaves = 0;
  stringstream forest_shape;
  ios default_state(nullptr);
  default_state.copyfmt(forest_shape);
  forest_shape << "Depth histogram:" << endl << "depth branches leaves   nodes" << endl;
  for (std::size_t level = 0; level < hist.size(); ++level) {
    level_entry e = hist[level];
    forest_shape << setw(5) << level << setw(9) << e.n_branch_nodes << setw(7) << e.n_leaves
                 << setw(8) << e.n_branch_nodes + e.n_leaves << endl;
    forest_shape.copyfmt(default_state);
    if (e.n_leaves && min_leaf_depth == -1) min_leaf_depth = level;
    leaves_times_depth += e.n_leaves * level;
    total_branches += e.n_branch_nodes;
    total_leaves += e.n_leaves;
  }
  int total_nodes = total_branches + total_leaves;
  forest_shape << "Total: branches: " << total_branches << " leaves: " << total_leaves
               << " nodes: " << total_nodes << endl;
  forest_shape << "Avg nodes per tree: " << setprecision(2)
               << total_nodes / (float)hist[0].n_branch_nodes << endl;
  forest_shape.copyfmt(default_state);
  forest_shape << "Leaf depth: min: " << min_leaf_depth << " avg: " << setprecision(2) << fixed
               << leaves_times_depth / (float)total_leaves << " max: " << hist.size() - 1 << endl;
  forest_shape.copyfmt(default_state);

  vector<char> hist_bytes(hist.size() * sizeof(hist[0]));
  memcpy(&hist_bytes[0], &hist[0], hist_bytes.size());
  // std::hash does not promise to not be identity. Xoring plain numbers which
  // add up to one another erases information, hence, std::hash is unsuitable here
  forest_shape << "Depth histogram fingerprint: " << hex
               << fowler_noll_vo_fingerprint64_32(hist_bytes.begin(), hist_bytes.end()) << endl;
  forest_shape.copyfmt(default_state);

  return forest_shape;
}

template <typename T, typename L>
size_t tl_leaf_vector_size(const tl::ModelImpl<T, L>& model)
{
  const tl::Tree<T, L>& tree = model.trees[0];
  int node_key;
  for (node_key = tree_root(tree); !tree.IsLeaf(node_key); node_key = tree.RightChild(node_key))
    ;
  if (tree.HasLeafVector(node_key)) return tree.LeafVector(node_key).size();
  return 0;
}

// tl2fil_common is the part of conversion from a treelite model
// common for dense and sparse forests
template <typename T, typename L>
void tl2fil_common(forest_params_t* params,
                   const tl::ModelImpl<T, L>& model,
                   const treelite_params_t* tl_params)
{
  // fill in forest-indendent params
  params->algo      = tl_params->algo;
  params->threshold = tl_params->threshold;

  // fill in forest-dependent params
  params->depth = max_depth(model);  // also checks for cycles

  const tl::ModelParam& param = model.param;

  // assuming either all leaves use the .leaf_vector() or all leaves use .leaf_value()
  size_t leaf_vec_size = tl_leaf_vector_size(model);
  std::string pred_transform(param.pred_transform);
  if (leaf_vec_size > 0) {
    ASSERT(leaf_vec_size == model.task_param.num_class, "treelite model inconsistent");
    params->num_classes = leaf_vec_size;
    params->leaf_algo   = leaf_algo_t::VECTOR_LEAF;

    ASSERT(pred_transform == "max_index" || pred_transform == "identity_multiclass",
           "only max_index and identity_multiclass values of pred_transform "
           "are supported for multi-class models");

  } else {
    if (model.task_param.num_class > 1) {
      params->num_classes = static_cast<int>(model.task_param.num_class);
      ASSERT(tl_params->output_class, "output_class==true is required for multi-class models");
      ASSERT(pred_transform == "identity_multiclass" || pred_transform == "max_index" ||
               pred_transform == "softmax" || pred_transform == "multiclass_ova",
             "only identity_multiclass, max_index, multiclass_ova and softmax "
             "values of pred_transform are supported for xgboost-style "
             "multi-class classification models.");
      // this function should not know how many threads per block will be used
      params->leaf_algo = leaf_algo_t::GROVE_PER_CLASS;
    } else {
      params->num_classes = tl_params->output_class ? 2 : 1;
      ASSERT(pred_transform == "sigmoid" || pred_transform == "identity",
             "only sigmoid and identity values of pred_transform "
             "are supported for binary classification and regression models.");
      params->leaf_algo = leaf_algo_t::FLOAT_UNARY_BINARY;
    }
  }

  params->num_cols = model.num_feature;

  ASSERT(param.sigmoid_alpha == 1.0f, "sigmoid_alpha not supported");
  params->global_bias = param.global_bias;
  params->output      = output_t::RAW;
  /** output_t::CLASS denotes using a threshold in FIL, when
      predict_proba == false. For all multiclass models, the best class is
      selected using argmax instead. This happens when either
      leaf_algo == CATEGORICAL_LEAF or num_classes > 2.
  **/
  if (tl_params->output_class && params->leaf_algo != CATEGORICAL_LEAF &&
      params->num_classes <= 2) {
    params->output = output_t(params->output | output_t::CLASS);
  }
  // "random forest" in treelite means tree output averaging
  if (model.average_tree_output) { params->output = output_t(params->output | output_t::AVG); }
  if (pred_transform == "sigmoid" || pred_transform == "multiclass_ova") {
    params->output = output_t(params->output | output_t::SIGMOID);
  }
  if (pred_transform == "softmax") params->output = output_t(params->output | output_t::SOFTMAX);
  params->num_trees        = model.trees.size();
  params->blocks_per_sm    = tl_params->blocks_per_sm;
  params->threads_per_tree = tl_params->threads_per_tree;
  params->n_items          = tl_params->n_items;
}

template <typename node_t>
template <typename threshold_t, typename leaf_t>
void node_traits<node_t>::check(const treelite::ModelImpl<threshold_t, leaf_t>& model)
{
  if constexpr (!std::is_same<node_t, sparse_node8>()) return;
  const int MAX_FEATURES   = 1 << sparse_node8::FID_NUM_BITS;
  const int MAX_TREE_NODES = (1 << sparse_node8::LEFT_NUM_BITS) - 1;
  // check the number of features
  int num_features = model.num_feature;
  ASSERT(num_features <= MAX_FEATURES,
         "model has %d features, "
         "but only %d supported for 8-byte sparse nodes",
         num_features,
         MAX_FEATURES);

  // check the number of tree nodes
  const std::vector<tl::Tree<threshold_t, leaf_t>>& trees = model.trees;
  for (std::size_t i = 0; i < trees.size(); ++i) {
    int num_nodes = trees[i].num_nodes;
    ASSERT(num_nodes <= MAX_TREE_NODES,
           "tree %zu has %d nodes, "
           "but only %d supported for 8-byte sparse nodes",
           i,
           num_nodes,
           MAX_TREE_NODES);
  }
}

template <typename fil_node_t, typename threshold_t, typename leaf_t>
struct tl2fil_t {
  std::vector<int> roots_;
  std::vector<fil_node_t> nodes_;
  std::vector<float> vector_leaf_;
  forest_params_t params_;
  cat_sets_owner cat_sets_;
  const tl::ModelImpl<threshold_t, leaf_t>& model_;
  const treelite_params_t& tl_params_;

  tl2fil_t(const tl::ModelImpl<threshold_t, leaf_t>& model_, const treelite_params_t& tl_params_)
    : model_(model_), tl_params_(tl_params_)
  {
  }

  void init()
  {
    static const bool IS_DENSE = node_traits<fil_node_t>::IS_DENSE;
    tl2fil_common(&params_, model_, &tl_params_);
    node_traits<fil_node_t>::check(model_);

    std::size_t num_trees = model_.trees.size();

    std::size_t total_nodes = 0;
    roots_.reserve(num_trees);
    for (auto& tree : model_.trees) {
      roots_.push_back(total_nodes);
      total_nodes += IS_DENSE ? tree_num_nodes(params_.depth) : tree.num_nodes;
    }

    if (params_.leaf_algo == VECTOR_LEAF) {
      std::size_t max_leaves = IS_DENSE ? num_trees * (tree_num_nodes(params_.depth) + 1) / 2
                                        : (total_nodes + num_trees) / 2;
      vector_leaf_.resize(max_leaves * params_.num_classes);
    }

    cat_sets_ = allocate_cat_sets_owner(model_);
    nodes_.resize(total_nodes);

// convert the nodes_
#pragma omp parallel for
    for (std::size_t tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
      // Max number of leaves processed so far
      size_t leaf_counter = (roots_[tree_idx] + tree_idx) / 2;
      tree2fil(nodes_,
               roots_[tree_idx],
               model_.trees[tree_idx],
               tree_idx,
               params_,
               &vector_leaf_,
               &leaf_counter,
               &cat_sets_);
    }

    params_.num_nodes = nodes_.size();
  }

  /// initializes FIL forest object, to be ready to infer
  void init_forest(const raft::handle_t& handle, forest_t<float>* pforest)
  {
    ML::fil::init(
      handle, pforest, cat_sets_.accessor(), vector_leaf_, roots_.data(), nodes_.data(), &params_);
    // sync is necessary as nodes_ are used in init(),
    // but destructed at the end of this function
    handle.sync_stream(handle.get_stream());
    if (tl_params_.pforest_shape_str) {
      *tl_params_.pforest_shape_str = sprintf_shape(model_, nodes_, roots_, cat_sets_);
    }
  }
};

template <typename fil_node_t, typename threshold_t, typename leaf_t>
void convert(const raft::handle_t& handle,
             forest_t<float>* pforest,
             const tl::ModelImpl<threshold_t, leaf_t>& model,
             const treelite_params_t& tl_params)
{
  tl2fil_t<fil_node_t, threshold_t, leaf_t> tl2fil(model, tl_params);
  tl2fil.init();
  tl2fil.init_forest(handle, pforest);
}

template <typename real_t>
constexpr bool type_supported()
{
  // not using std::is_floating_point because we did not instantiate fp16-based nodes/trees/forests
  return std::is_same_v<real_t, float> || std::is_same_v<real_t, double>;
}

template <typename threshold_t, typename leaf_t>
void from_treelite(const raft::handle_t& handle,
                   forest_t<float>* pforest,
                   const tl::ModelImpl<threshold_t, leaf_t>& model,
                   const treelite_params_t* tl_params)
{
  // Invariants on threshold and leaf types
  static_assert(type_supported<threshold_t>(),
                "Model must contain float32 or float64 thresholds for splits");
  ASSERT(type_supported<leaf_t>(), "Models with integer leaf output are not yet supported");
  // Display appropriate warnings when float64 values are being casted into
  // float32, as FIL only supports inferencing with float32 for the time being
  if (std::is_same<threshold_t, double>::value || std::is_same<leaf_t, double>::value) {
    CUML_LOG_WARN(
      "Casting all thresholds and leaf values to float32, as FIL currently "
      "doesn't support inferencing models with float64 values. "
      "This may lead to predictions with reduced accuracy.");
  }
  // same as std::common_type: float+double=double, float+int64_t=float
  using real_t = decltype(threshold_t(0) + leaf_t(0));

  storage_type_t storage_type = tl_params->storage_type;
  // build dense trees by default
  if (storage_type == storage_type_t::AUTO) {
    if (tl_params->algo == algo_t::ALGO_AUTO || tl_params->algo == algo_t::NAIVE) {
      int depth = max_depth(model);
      // max 2**25 dense nodes, 256 MiB dense model size. Categorical mask size is unlimited and not
      // affected by storage format.
      const int LOG2_MAX_DENSE_NODES = 25;
      int log2_num_dense_nodes       = depth + 1 + int(ceil(std::log2(model.trees.size())));
      storage_type = log2_num_dense_nodes > LOG2_MAX_DENSE_NODES ? storage_type_t::SPARSE
                                                                 : storage_type_t::DENSE;
    } else {
      // only dense storage is supported for other algorithms
      storage_type = storage_type_t::DENSE;
    }
  }

  switch (storage_type) {
    case storage_type_t::DENSE:
      convert<dense_node<float>>(handle, pforest, model, *tl_params);
      break;
    case storage_type_t::SPARSE:
      convert<sparse_node16<float>>(handle, pforest, model, *tl_params);
      break;
    case storage_type_t::SPARSE8: convert<sparse_node8>(handle, pforest, model, *tl_params); break;
    default: ASSERT(false, "tl_params->sparse must be one of AUTO, DENSE or SPARSE");
  }
}

void from_treelite(const raft::handle_t& handle,
                   forest_t<float>* pforest,
                   ModelHandle model,
                   const treelite_params_t* tl_params)
{
  const tl::Model& model_ref = *(tl::Model*)model;
  model_ref.Dispatch([&](const auto& model_inner) {
    // model_inner is of the concrete type tl::ModelImpl<threshold_t, leaf_t>
    from_treelite(handle, pforest, model_inner, tl_params);
  });
}

// allocates caller-owned char* using malloc()
template <typename threshold_t, typename leaf_t, typename node_t>
char* sprintf_shape(const tl::ModelImpl<threshold_t, leaf_t>& model,
                    const std::vector<node_t>& nodes,
                    const std::vector<int>& trees,
                    const cat_sets_owner cat_sets)
{
  std::stringstream forest_shape = depth_hist_and_max(model);
  double size_mb = (trees.size() * sizeof(trees.front()) + nodes.size() * sizeof(nodes.front()) +
                    cat_sets.bits.size()) /
                   1e6;
  forest_shape << storage_type_repr[node_traits<node_t>::storage_type_enum] << " model size "
               << std::setprecision(2) << size_mb << " MB" << std::endl;
  if (cat_sets.bits.size() > 0) {
    forest_shape << "number of categorical nodes for each feature id: {";
    std::size_t total_cat_nodes = 0;
    for (std::size_t n : cat_sets.n_nodes) {
      forest_shape << n << " ";
      total_cat_nodes += n;
    }
    forest_shape << "}" << std::endl << "total categorical nodes: " << total_cat_nodes << std::endl;
    forest_shape << "maximum matching category for each feature id: {";
    for (float fid_num_cats : cat_sets.fid_num_cats)
      forest_shape << static_cast<int>(fid_num_cats) - 1 << " ";
    forest_shape << "}" << std::endl;
  }
  // stream may be discontiguous
  std::string forest_shape_str = forest_shape.str();
  // now copy to a non-owning allocation
  char* shape_out = (char*)malloc(forest_shape_str.size() + 1);  // incl. \0
  memcpy((void*)shape_out, forest_shape_str.c_str(), forest_shape_str.size() + 1);
  return shape_out;
}

}  // namespace fil
}  // namespace ML
