/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

/** @file internal.cuh cuML-internal interface to Forest Inference Library. */

#pragma once
#include <cuml/fil/fil.h>
#include <treelite/c_api.h>
#include <treelite/tree.h>
#include <raft/cuda_utils.cuh>
#include <raft/error.hpp>
#include <vector>

namespace raft {
class handle_t;
}

namespace ML {
namespace fil {

template <typename T>
__host__ __device__ void print_rl(const T* v, std::size_t size, const char* name, int inc)
{
  return;
  auto print_until = [=](T last, size_t last_idx, size_t i) {
    if (last_idx != i - 1)
      printf("[%lu..%lu]->%u..%u ", last_idx, i - 1, last, v[i - 1]);
    else
      printf("[%lu]->%u ", last_idx, last);
  };
  printf("%s {", name);
  T last               = v[0];
  std::size_t last_idx = 0;
  for (std::size_t i = 1; i < size; ++i)
    if (v[i] != last + (i - last_idx) * inc) {
      print_until(last, last_idx, i);
      last     = v[i];
      last_idx = i;
    }
  print_until(last, last_idx, size);
  printf("}\n");
}

const int BITS_PER_BYTE = 8;

/// modpow2 returns a % b == a % pow(2, log2_b)
__host__ __device__ __forceinline__ int modpow2(int a, int log2_b)
{
  return a & ((1 << log2_b) - 1);
}

enum adjust_threshold_direction_t { FIL_TO_TREELITE = -1, TREELITE_TO_FIL = 1 };
void adjust_threshold(float* pthreshold,
                      int* tl_left,
                      int* tl_right,
                      bool* default_left,
                      treelite::Operator comparison_op,
                      adjust_threshold_direction_t dir);

/**
 * output_t are flags that define the output produced by the FIL predictor; a
 * valid output_t values consists of the following, combined using '|' (bitwise
 * or), which define stages, which operation in the next stage applied to the
 * output of the previous stage:
 * - one of RAW or AVG, indicating how to combine individual tree outputs into the forest output
 * - optional SIGMOID for applying the sigmoid transform
 * - optional CLASS, to output the class label
 */
enum output_t {
  /** raw output: the sum of the tree outputs; use for GBM models for
      regression, or for binary classification for the value before the
      transformation; note that this value is 0, and may be omitted
      when combined with other flags */
  RAW = 0x0,
  /** average output: divide the sum of the tree outputs by the number of trees
      before further transformations; use for random forests for regression
      and binary classification for the probability */
  AVG = 0x1,
  /** sigmoid transformation: apply 1/(1+exp(-x)) to the sum or average of tree
      outputs; use for GBM binary classification models for probability */
  SIGMOID = 0x10,
  /** output class label: either apply threshold to the output of the previous stage (for binary
     classification), or select the class with the most votes to get the class label (for
     multi-class classification).  */
  CLASS = 0x100,
  /** softmax: apply softmax to class margins when predicting probability
      in multiclass classification. Softmax is made robust by subtracting max
      from margins before applying. */
  SOFTMAX           = 0x1000,
  SIGMOID_CLASS     = SIGMOID | CLASS,
  AVG_CLASS         = AVG | CLASS,
  AVG_SIGMOID_CLASS = AVG | SIGMOID | CLASS,
  AVG_SOFTMAX       = AVG | SOFTMAX,
  AVG_CLASS_SOFTMAX = AVG | CLASS | SOFTMAX,
  ALL_SET           = AVG | SIGMOID | CLASS | SOFTMAX
};
std::string output2str(fil::output_t output);

/** val_t is the payload within a FIL leaf */
union val_t {
  /** threshold value for parent node or output value (e.g. class
      probability or regression summand) for leaf node */
  float f;
  /** class label, leaf vector index or categorical node set offset */
  int idx;
};

/** base_node contains common implementation details for dense and sparse nodes */
struct base_node {
  /** val, for parent nodes, is a threshold or category list offset. For leaf
      nodes, it is the tree prediction (see see leaf_output_t<leaf_algo_t>::T) */
  val_t val;
  /** bits encode various information about the node, with the exact nature of
      this information depending on the node type; it includes e.g. whether the
      node is a leaf or inner node, and for inner nodes, additional information,
      e.g. the default direction, feature id or child index */
  int bits;
  static const int IS_LEAF_OFFSET        = 31;
  static const int IS_LEAF_MASK          = 1 << IS_LEAF_OFFSET;
  static const int DEF_LEFT_OFFSET       = IS_LEAF_OFFSET - 1;
  static const int DEF_LEFT_MASK         = 1 << DEF_LEFT_OFFSET;
  static const int IS_CATEGORICAL_OFFSET = DEF_LEFT_OFFSET - 1;
  static const int IS_CATEGORICAL_MASK   = 1 << IS_CATEGORICAL_OFFSET;
  static const int FID_MASK              = (1 << IS_CATEGORICAL_OFFSET) - 1;
  template <class o_t>
  __host__ __device__ o_t output() const;
  __host__ __device__ int set() const { return val.idx; }
  __host__ __device__ float thresh() const { return val.f; }
  __host__ __device__ val_t split() const { return val; }
  __host__ __device__ int fid() const { return bits & FID_MASK; }
  __host__ __device__ bool def_left() const { return bits & DEF_LEFT_MASK; }
  __host__ __device__ bool is_leaf() const { return bits & IS_LEAF_MASK; }
  __host__ __device__ bool is_categorical() const { return bits & IS_CATEGORICAL_MASK; }
  __host__ __device__ base_node() : val{}, bits(0) {}
  base_node(val_t output, val_t split, int fid, bool def_left, bool is_leaf, bool is_categorical)
  {
    RAFT_EXPECTS((fid & FID_MASK) == fid, "internal error: feature ID doesn't fit into base_node");
    bits = (fid & FID_MASK) | (def_left ? DEF_LEFT_MASK : 0) | (is_leaf ? IS_LEAF_MASK : 0) |
           (is_categorical ? IS_CATEGORICAL_MASK : 0);
    if (is_leaf)
      val = output;
    else if (is_leaf)
      val = output;
    else
      val = split;
  }
  __host__ __device__ void print() const
  {
    return;
    const char* is[] = {"is NOT", "    IS"};
    const char* lr[] = {" left", "right"};
    printf("{.f = %9f, .idx = %11d} fid %10d, bits %8x, def %s, %s leaf, %s categorical\n",
           val.f,
           val.idx,
           fid(),
           set(),
           lr[def_left()],
           is[is_leaf()],
           is[is_categorical()]);
  }
};

template <>
__host__ __device__ __forceinline__ float base_node::output<float>() const
{
  return val.f;
}
template <>
__host__ __device__ __forceinline__ int base_node::output<int>() const
{
  return val.idx;
}
template <>
__host__ __device__ __forceinline__ val_t base_node::output<val_t>() const
{
  return val;
}

/** dense_node is a single node of a dense forest */
// template <bool can_be_categorical>
struct alignas(8) dense_node : base_node {
  dense_node() = default;
  dense_node(val_t output, val_t split, int fid, bool def_left, bool is_leaf, bool is_categorical)
    : base_node(output, split, fid, def_left, is_leaf, is_categorical)
  {
    ASSERT(is_categorical == base_node::is_categorical(), "didn't save is_categorical right");
  }
  /** index of the left child, where curr is the index of the current node */
  __host__ __device__ int left(int curr) const { return 2 * curr + 1; }
};

/** sparse_node16 is a 16-byte node in a sparse forest */
// template <bool can_be_categorical>
struct alignas(16) sparse_node16 : base_node {
  int left_idx;
  int dummy;  // make alignment explicit and reserve for future use
  __host__ __device__ sparse_node16() : left_idx(0), dummy(0) {}
  sparse_node16(val_t output,
                val_t split,
                int fid,
                bool def_left,
                bool is_leaf,
                bool is_categorical,
                int left_index)
    : base_node(output, split, fid, def_left, is_leaf, is_categorical),
      left_idx(left_index),
      dummy(0)
  {
    ASSERT(is_categorical == base_node::is_categorical(), "didn't save is_categorical right");
  }
  __host__ __device__ int left_index() const { return left_idx; }
  /** index of the left child, where curr is the index of the current node */
  __host__ __device__ int left(int curr) const { return left_idx; }
};

/** sparse_node8 is a node of reduced size (8 bytes) in a sparse forest */
struct alignas(8) sparse_node8 : base_node {
  static const int LEFT_NUM_BITS = 16;
  static const int FID_NUM_BITS  = IS_CATEGORICAL_OFFSET - LEFT_NUM_BITS;
  static const int LEFT_OFFSET   = FID_NUM_BITS;
  static const int FID_MASK      = (1 << FID_NUM_BITS) - 1;
  static const int LEFT_MASK     = ((1 << LEFT_NUM_BITS) - 1) << LEFT_OFFSET;
  __host__ __device__ int fid() const { return bits & FID_MASK; }
  __host__ __device__ int left_index() const { return (bits & LEFT_MASK) >> LEFT_OFFSET; }
  sparse_node8() = default;
  sparse_node8(val_t output,
               val_t split,
               int fid,
               bool def_left,
               bool is_leaf,
               bool is_categorical,
               int left_index)
    : base_node(output, split, fid, def_left, is_leaf, is_categorical)
  {
    RAFT_EXPECTS((fid & FID_MASK) == fid,
                 "internal error: feature ID doesn't fit into sparse_node8");
    RAFT_EXPECTS(((left_index << LEFT_OFFSET) & LEFT_MASK) == (left_index << LEFT_OFFSET),
                 "internal error: left child index doesn't fit into sparse_node8");
    bits |= left_index << LEFT_OFFSET;
  }
  /** index of the left child, where curr is the index of the current node */
  __host__ __device__ int left(int curr) const { return left_index(); }
};

/** leaf_algo_t describes what the leaves in a FIL forest store (predict)
    and how FIL aggregates them into class margins/regression result/best class
**/
enum leaf_algo_t {
  /** storing a class probability or regression summand. We add all margins
      together and determine regression result or use threshold to determine
      one of the two classes. **/
  FLOAT_UNARY_BINARY = 0,
  /** storing a class label. Trees vote on the resulting class.
      Probabilities are just normalized votes. */
  CATEGORICAL_LEAF = 1,
  /** 1-vs-rest, or tree-per-class, where trees are assigned round-robin to
      consecutive categories and predict a floating-point margin. Used in
      Gradient Boosted Decision Trees. We sum margins for each group separately
      **/
  GROVE_PER_CLASS = 2,
  /** 1-vs-rest, or tree-per-class, where trees are assigned round-robin to
      consecutive categories and predict a floating-point margin. Used in
      Gradient Boosted Decision Trees. We sum margins for each group separately
      This is a more specific version of GROVE_PER_CLASS.
      _FEW_CLASSES means fewer (or as many) classes than threads. **/
  GROVE_PER_CLASS_FEW_CLASSES = 3,
  /** 1-vs-rest, or tree-per-class, where trees are assigned round-robin to
      consecutive categories and predict a floating-point margin. Used in
      Gradient Boosted Decision Trees. We sum margins for each group separately
      This is a more specific version of GROVE_PER_CLASS.
      _MANY_CLASSES means more classes than threads. **/
  GROVE_PER_CLASS_MANY_CLASSES = 4,
  /** Leaf contains an index into a vector of class probabilities. **/
  VECTOR_LEAF = 5,
  // to be extended
};
static const char* leaf_algo_t_repr[] = {"FLOAT_UNARY_BINARY",
                                         "CATEGORICAL_LEAF",
                                         "GROVE_PER_CLASS",
                                         "GROVE_PER_CLASS_FEW_CLASSES",
                                         "GROVE_PER_CLASS_MANY_CLASSES",
                                         "VECTOR_LEAF"};

template <leaf_algo_t leaf_algo>
struct leaf_output_t {
};
template <>
struct leaf_output_t<leaf_algo_t::FLOAT_UNARY_BINARY> {
  typedef float T;
};
template <>
struct leaf_output_t<leaf_algo_t::CATEGORICAL_LEAF> {
  typedef int T;
};
template <>
struct leaf_output_t<leaf_algo_t::GROVE_PER_CLASS_FEW_CLASSES> {
  typedef float T;
};
template <>
struct leaf_output_t<leaf_algo_t::GROVE_PER_CLASS_MANY_CLASSES> {
  typedef float T;
};
template <>
struct leaf_output_t<leaf_algo_t::VECTOR_LEAF> {
  typedef int T;
};

/** forest_params_t are the trees to initialize the predictor */
struct forest_params_t {
  // total number of nodes; ignored for dense forests
  int num_nodes;
  // maximum depth; ignored for sparse forests
  int depth;
  // ntrees is the number of trees
  int num_trees;
  // num_cols is the number of columns in the data
  int num_cols;
  // leaf_algo determines what the leaves store (predict)
  leaf_algo_t leaf_algo;
  // algo is the inference algorithm;
  // sparse forests do not distinguish between NAIVE and TREE_REORG
  algo_t algo;
  // output is the desired output type
  output_t output;
  // threshold is used to for classification if leaf_algo == FLOAT_UNARY_BINARY && (output &
  // OUTPUT_CLASS) != 0 && !predict_proba, and is ignored otherwise
  float threshold;
  // global_bias is added to the sum of tree predictions
  // (after averaging, if it is used, but before any further transformations)
  float global_bias;
  // only used for CATEGORICAL_LEAF inference. since we're storing the
  // labels in leaves instead of the whole vector, this keeps track
  // of the number of classes
  int num_classes;
  // blocks_per_sm, if nonzero, works as a limit to improve cache hit rate for larger forests
  // suggested values (if nonzero) are from 2 to 7
  // if zero, launches ceildiv(num_rows, NITEMS) blocks
  int blocks_per_sm;
  // threads_per_tree determines how many threads work on a single tree
  // at once inside a block (sharing trees means splitting input rows)
  int threads_per_tree;
  // n_items is how many input samples (items) any thread processes. If 0 is given,
  // choose most (up to 4) that fit into shared memory.
  int n_items;
};

/// FIL_TPB is the number of threads per block to use with FIL kernels
const int FIL_TPB = 256;

const uint32_t MAX_PRECISE_INT_FLOAT = 1 << 24;  // 16'777'216

__host__ __device__ __forceinline__ int fetch_bit(const uint8_t* array, int bit)
{
  return (array[bit / BITS_PER_BYTE] >> (bit % BITS_PER_BYTE)) & 1;
}

struct cat_feature_counters {
  int max_matching = -1, n_nodes = 0;
};

struct categorical_sets {
  // arrays are const to use fast GPU read instructions by default
  // arrays from each node ID are concatenated first, then from all categories
  const uint8_t* bits = nullptr;
  // largest matching category in the model, per feature ID
  const int* max_matching = nullptr;
  std::size_t bits_size = 0, max_matching_size = 0;

  __host__ __device__ __forceinline__ bool cats_present() const
  {
    // If this is constructed from cat_sets_owner, will return true
    // default-initialized will return false
    // Defining edge case: there are categorical nodes, but all have max_matching == -1
    // (all categorical nodes are empty). node.thresh() would have returned 0.0f
    // and the branch condition wouldn't have always been false (i.e branched left).
    // Alternatively, we could have converted all empty categorical nodes to
    // NAN-threshold numerical nodes.
    return max_matching != nullptr;
  }

  // set count is due to tree_idx + node_within_tree_idx are both ints, hence uint32_t result
  template <typename node_t>
  __host__ __device__ __forceinline__ int category_matches(node_t node, int category) const
  {
    // standard boolean packing. This layout has better ILP
    // node.set() is global across feature IDs and is an offset (as opposed
    // to set number). If we run out of uint32_t and we have hundreds of
    // features with similar categorical feature count, we may consider
    // storing node ID within nodes with same feature ID and look up
    // {.max_matching, .first_node_offset} = ...[feature_id]
    // printf(
    //  "for fid %d and category %d, checking categories at %d:", node.fid(), category, node.set());
    print_rl(bits + node.set(), max_matching[node.fid()] + 1, "bits", 0);
    return category <= max_matching[node.fid()] && fetch_bit(bits + node.set(), category);
  }
  static int sizeof_mask_from_max_matching(int max_matching)
  {
    return raft::ceildiv(max_matching + 1, BITS_PER_BYTE);
  }
  int sizeof_mask(int feature_id) const
  {
    return sizeof_mask_from_max_matching(max_matching[feature_id]);
  }
};

struct tree_base {
  categorical_sets sets;

  template <bool CATS_SUPPORTED, typename node_t>
  __host__ __device__ __forceinline__ int child_index(const node_t& node,
                                                      int node_idx,
                                                      float val) const
  {
    const char* lr[] = {" left", "right"};
    bool cond;

    if (isnan(val)) {
      cond = !node.def_left();
      // printf("idx %d val is nan, taking %s\n", node_idx, lr[cond]);
    } else if (CATS_SUPPORTED && node.is_categorical()) {
      cond = sets.category_matches(node, (int)val);
      // printf("idx %d taking %s\n", node_idx, lr[cond]);
    } else {
      cond = val >= node.thresh();
      // printf("idx %d %f >= %f taking %s\n", node_idx, val, node.thresh(), lr[cond]);
    }
    return node.left(node_idx) + cond;
  }
};

// in internal.cuh, as opposed to fil_test.cu, because importing from treelite will require it
struct cat_sets_owner {
  // arrays from each node ID are concatenated first, then from all categories
  std::vector<uint8_t> bits;
  // largest matching category in the model, per feature ID
  std::vector<int> max_matching;

  categorical_sets accessor() const
  {
    return {
      .bits              = bits.data(),
      .max_matching      = max_matching.data(),
      .bits_size         = bits.size(),
      .max_matching_size = max_matching.size(),
    };
  }
  cat_sets_owner() {}
  cat_sets_owner(std::vector<uint8_t> bits_, std::vector<int> max_matching_)
    : bits(bits_), max_matching(max_matching_)
  {
  }
  cat_sets_owner(const std::vector<cat_feature_counters>& cf);
};

std::ostream& operator<<(std::ostream& os, const cat_sets_owner& cso);

/** init_dense uses params and nodes to initialize the dense forest stored in pf
 *  @param h cuML handle used by this function
 *  @param pf pointer to where to store the newly created forest
 *  @param nodes nodes for the forest, of length
      (2**(params->depth + 1) - 1) * params->ntrees
 *  @param params pointer to parameters used to initialize the forest
 *  @param vector_leaf optional vector leaves
 */
void init_dense(const raft::handle_t& h,
                forest_t* pf,
                const categorical_sets& cat_sets,
                const std::vector<float>& vector_leaf,
                const dense_node* nodes,
                const forest_params_t* params);

/** init_sparse uses params, trees and nodes to initialize the sparse forest
 *  with sparse nodes stored in pf
 *  @tparam fil_node_t node type to use with the sparse forest;
 *    must be sparse_node16 or sparse_node8
 *  @param h cuML handle used by this function
 *  @param pf pointer to where to store the newly created forest
 *  @param trees indices of tree roots in the nodes arrray, of length params->ntrees
 *  @param nodes nodes for the forest, of length params->num_nodes
 *  @param params pointer to parameters used to initialize the forest
 *  @param vector_leaf optional vector leaves
 */
template <typename fil_node_t>
void init_sparse(const raft::handle_t& h,
                 forest_t* pf,
                 const categorical_sets& cat_sets,
                 const std::vector<float>& vector_leaf,
                 const int* trees,
                 const fil_node_t* nodes,
                 const forest_params_t* params);

}  // namespace fil
}  // namespace ML
