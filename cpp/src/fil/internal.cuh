/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <bitset>
#include <cstdint>
#include <cuml/fil/fil.h>
#include <iostream>
#include <numeric>
#include <raft/core/error.hpp>
#include <raft/cuda_utils.cuh>
#include <rmm/device_uvector.hpp>
#include <treelite/c_api.h>
#include <treelite/tree.h>
#include <utility>
#include <vector>

namespace raft {
class handle_t;
}

// needed for node_traits<...>
namespace treelite {
template <typename, typename>
struct ModelImpl;
}

namespace ML {
namespace fil {

const int BITS_PER_BYTE = 8;

/// modpow2 returns a % b == a % pow(2, log2_b)
__host__ __device__ __forceinline__ int modpow2(int a, int log2_b)
{
  return a & ((1 << log2_b) - 1);
}

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

/** val_t is the payload within a FIL leaf */
template <typename real_t>
union val_t {
  /** floating-point threshold value for parent node or output value
      (e.g. class probability or regression summand) for leaf node */
  real_t f = NAN;
  /** class label, leaf vector index or categorical node set offset */
  int idx;
};

/** base_node contains common implementation details for dense and sparse nodes */
template <typename real_t>
struct alignas(2 * sizeof(real_t)) base_node {
  using real_type = real_t;  // floating-point type
  /** val, for parent nodes, is a threshold or category list offset. For leaf
      nodes, it is the tree prediction (see see leaf_output_t<leaf_algo_t>::T) */
  val_t<real_t> val;
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
  __host__ __device__ o_t output() const
  {
    static_assert(
      std::is_same_v<o_t, int> || std::is_same_v<o_t, real_t> || std::is_same_v<o_t, val_t<real_t>>,
      "invalid o_t type parameter in node.output()");
    if constexpr (std::is_same_v<o_t, int>) {
      return val.idx;
    } else if constexpr (std::is_same_v<o_t, real_t>) {
      return val.f;
    } else if constexpr (std::is_same_v<o_t, val_t<real_t>>) {
      return val;
    }
    // control flow should not reach here
    return o_t();
  }
  __host__ __device__ int set() const { return val.idx; }
  __host__ __device__ real_t thresh() const { return val.f; }
  __host__ __device__ val_t<real_t> split() const { return val; }
  __host__ __device__ int fid() const { return bits & FID_MASK; }
  __host__ __device__ bool def_left() const { return bits & DEF_LEFT_MASK; }
  __host__ __device__ bool is_leaf() const { return bits & IS_LEAF_MASK; }
  __host__ __device__ bool is_categorical() const { return bits & IS_CATEGORICAL_MASK; }
  __host__ __device__ base_node() : val{}, bits(0) {}
  base_node(val_t<real_t> output,
            val_t<real_t> split,
            int fid,
            bool def_left,
            bool is_leaf,
            bool is_categorical)
  {
    RAFT_EXPECTS((fid & FID_MASK) == fid, "internal error: feature ID doesn't fit into base_node");
    bits = (fid & FID_MASK) | (def_left ? DEF_LEFT_MASK : 0) | (is_leaf ? IS_LEAF_MASK : 0) |
           (is_categorical ? IS_CATEGORICAL_MASK : 0);
    if (is_leaf)
      val = output;
    else
      val = split;
  }
};

/** dense_node is a single node of a dense forest */
template <typename real_t>
struct alignas(2 * sizeof(real_t)) dense_node : base_node<real_t> {
  dense_node() = default;
  /// ignoring left_index, this is useful to unify import from treelite
  dense_node(val_t<real_t> output,
             val_t<real_t> split,
             int fid,
             bool def_left,
             bool is_leaf,
             bool is_categorical,
             int left_index = -1)
    : base_node<real_t>(output, split, fid, def_left, is_leaf, is_categorical)
  {
  }
  /** index of the left child, where curr is the index of the current node */
  __host__ __device__ int left(int curr) const { return 2 * curr + 1; }
};

/** sparse_node16 is a 16-byte node in a sparse forest */
template <typename real_t>
struct alignas(16) sparse_node16 : base_node<real_t> {
  int left_idx;
  __host__ __device__ sparse_node16() : left_idx(0) {}
  sparse_node16(val_t<real_t> output,
                val_t<real_t> split,
                int fid,
                bool def_left,
                bool is_leaf,
                bool is_categorical,
                int left_index)
    : base_node<real_t>(output, split, fid, def_left, is_leaf, is_categorical), left_idx(left_index)
  {
  }
  __host__ __device__ int left_index() const { return left_idx; }
  /** index of the left child, where curr is the index of the current node */
  __host__ __device__ int left(int curr) const { return left_idx; }
};

/** sparse_node8 is a node of reduced size (8 bytes) in a sparse forest */
struct alignas(8) sparse_node8 : base_node<float> {
  static const int LEFT_NUM_BITS = 16;
  static const int FID_NUM_BITS  = IS_CATEGORICAL_OFFSET - LEFT_NUM_BITS;
  static const int LEFT_OFFSET   = FID_NUM_BITS;
  static const int FID_MASK      = (1 << FID_NUM_BITS) - 1;
  static const int LEFT_MASK     = ((1 << LEFT_NUM_BITS) - 1) << LEFT_OFFSET;
  __host__ __device__ int fid() const { return bits & FID_MASK; }
  __host__ __device__ int left_index() const { return (bits & LEFT_MASK) >> LEFT_OFFSET; }
  sparse_node8() = default;
  sparse_node8(val_t<float> output,
               val_t<float> split,
               int fid,
               bool def_left,
               bool is_leaf,
               bool is_categorical,
               int left_index)
    : base_node<float>(output, split, fid, def_left, is_leaf, is_categorical)
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

template <typename node_t>
struct storage;

template <typename node_t>
struct dense_forest;
template <typename node_t>
struct sparse_forest;

template <typename node_t>
struct node_traits {
  using real_type            = typename node_t::real_type;
  using storage              = ML::fil::storage<node_t>;
  using forest               = sparse_forest<node_t>;
  static const bool IS_DENSE = false;
  static constexpr storage_type_t storage_type_enum =
    std::is_same_v<sparse_node16<real_type>, node_t> ? SPARSE : SPARSE8;
  template <typename threshold_t, typename leaf_t>
  static void check(const treelite::ModelImpl<threshold_t, leaf_t>& model);
};

template <typename real_t>
struct node_traits<dense_node<real_t>> {
  using storage                                 = storage<dense_node<real_t>>;
  using forest                                  = dense_forest<dense_node<real_t>>;
  static const bool IS_DENSE                    = true;
  static const storage_type_t storage_type_enum = DENSE;
  template <typename threshold_t, typename leaf_t>
  static void check(const treelite::ModelImpl<threshold_t, leaf_t>& model)
  {
  }
};

/** leaf_algo_t describes what the leaves in a FIL forest store (predict)
    and how FIL aggregates them into class margins/regression result/best class
**/
enum leaf_algo_t {
  /** For iteration purposes */
  MIN_LEAF_ALGO = 0,
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
  MAX_LEAF_ALGO = 5
};

template <typename node_t>
struct tree;

template <leaf_algo_t leaf_algo, typename real_t>
struct leaf_output_t {
};
template <typename real_t>
struct leaf_output_t<leaf_algo_t::FLOAT_UNARY_BINARY, real_t> {
  typedef real_t T;
};
template <typename real_t>
struct leaf_output_t<leaf_algo_t::CATEGORICAL_LEAF, real_t> {
  typedef int T;
};
template <typename real_t>
struct leaf_output_t<leaf_algo_t::GROVE_PER_CLASS_FEW_CLASSES, real_t> {
  typedef real_t T;
};
template <typename real_t>
struct leaf_output_t<leaf_algo_t::GROVE_PER_CLASS_MANY_CLASSES, real_t> {
  typedef real_t T;
};
template <typename real_t>
struct leaf_output_t<leaf_algo_t::VECTOR_LEAF, real_t> {
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
  double threshold;
  // global_bias is added to the sum of tree predictions
  // (after averaging, if it is used, but before any further transformations)
  double global_bias;
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
  // choose most (up to MAX_N_ITEMS) that fit into shared memory.
  int n_items;
};

/// FIL_TPB is the number of threads per block to use with FIL kernels
const int FIL_TPB = 256;

// 1 << 24 is the largest integer representable exactly as a float.
// To avoid edge cases, 16'777'214 is the most FIL will use.
constexpr std::int32_t MAX_FIL_INT_FLOAT = (1 << 24) - 2;

__host__ __device__ __forceinline__ int fetch_bit(const uint8_t* array, uint32_t bit)
{
  return (array[bit / BITS_PER_BYTE] >> (bit % BITS_PER_BYTE)) & 1;
}

struct categorical_sets {
  // arrays are const to use fast GPU read instructions by default
  // arrays from each node ID are concatenated first, then from all categories
  const uint8_t* bits = nullptr;
  // number of matching categories FIL stores in the bit array, per feature ID
  const float* fid_num_cats = nullptr;
  std::size_t bits_size     = 0;
  // either 0 or num_cols. When 0, indicates intended empty array.
  std::size_t fid_num_cats_size = 0;

  __host__ __device__ __forceinline__ bool cats_present() const
  {
    // If this is constructed from cat_sets_owner, will return true; but false by default
    // We have converted all empty categorical nodes to NAN-threshold numerical nodes.
    return fid_num_cats != nullptr;
  }

  // set count is due to tree_idx + node_within_tree_idx are both ints, hence uint32_t result
  template <typename node_t>
  __host__ __device__ __forceinline__ int category_matches(
    node_t node, typename node_t::real_type category) const
  {
    // standard boolean packing. This layout has better ILP
    // node.set() is global across feature IDs and is an offset (as opposed
    // to set number). If we run out of uint32_t and we have hundreds of
    // features with similar categorical feature count, we may consider
    // storing node ID within nodes with same feature ID and look up
    // {.fid_num_cats, .first_node_offset} = ...[feature_id]

    /* category < 0.0f or category > INT_MAX is equivalent to out-of-dictionary category
    (not matching, branch left). -0.0f represents category 0.
    If (float)(int)category != category, we will discard the fractional part.
    E.g. 3.8f represents category 3 regardless of fid_num_cats value.
    FIL will reject a model where an integer within [0, fid_num_cats] cannot be represented
    precisely as a 32-bit float.
    */
    using real_t = typename node_t::real_type;
    return category < static_cast<real_t>(fid_num_cats[node.fid()]) && category >= real_t(0) &&
           fetch_bit(bits + node.set(), static_cast<uint32_t>(static_cast<int>(category)));
  }
  static int sizeof_mask_from_num_cats(int num_cats)
  {
    return raft::ceildiv(num_cats, BITS_PER_BYTE);
  }
  int sizeof_mask(int feature_id) const
  {
    return sizeof_mask_from_num_cats(static_cast<int>(fid_num_cats[feature_id]));
  }
};

// lets any tree determine a child index for a node in a generic fasion
// used in fil_test.cu fot its child_index() in CPU predicting
struct tree_base {
  categorical_sets cat_sets;

  template <bool CATS_SUPPORTED, typename node_t>
  __host__ __device__ __forceinline__ int child_index(const node_t& node,
                                                      int node_idx,
                                                      typename node_t::real_type val) const
  {
    bool cond;

    if (isnan(val)) {
      cond = !node.def_left();
    } else if (CATS_SUPPORTED && node.is_categorical()) {
      cond = cat_sets.category_matches(node, val);
    } else {
      cond = val >= node.thresh();
    }
    return node.left(node_idx) + cond;
  }
};

// -1 means no matching categories
struct cat_feature_counters {
  int max_matching = -1;
  int n_nodes      = 0;
  static cat_feature_counters combine(cat_feature_counters a, cat_feature_counters b)
  {
    return {.max_matching = std::max(a.max_matching, b.max_matching),
            .n_nodes      = a.n_nodes + b.n_nodes};
  }
};

// used only during model import. For inference, trimmed down using cat_sets_owner::accessor()
// in internal.cuh, as opposed to fil_test.cu, because importing from treelite will require it
struct cat_sets_owner {
  // arrays from each node ID are concatenated first, then from all categories
  std::vector<uint8_t> bits;
  // largest matching category in the model, per feature ID. uses int because GPU code can only fit
  // int
  std::vector<float> fid_num_cats;
  // how many categorical nodes use a given feature id. Used for model shape string.
  std::vector<std::size_t> n_nodes;
  // per tree, size and offset of bit pool within the overall bit pool
  std::vector<std::size_t> bit_pool_offsets;

  categorical_sets accessor() const
  {
    return {
      .bits              = bits.data(),
      .fid_num_cats      = fid_num_cats.data(),
      .bits_size         = bits.size(),
      .fid_num_cats_size = fid_num_cats.size(),
    };
  }

  void consume_counters(const std::vector<cat_feature_counters>& counters)
  {
    for (cat_feature_counters cf : counters) {
      fid_num_cats.push_back(static_cast<float>(cf.max_matching + 1));
      n_nodes.push_back(cf.n_nodes);
    }
  }

  void consume_bit_pool_sizes(const std::vector<std::size_t>& bit_pool_sizes)
  {
    bit_pool_offsets.push_back(0);
    for (std::size_t i = 0; i < bit_pool_sizes.size() - 1; ++i) {
      bit_pool_offsets.push_back(bit_pool_offsets.back() + bit_pool_sizes[i]);
    }
    bits.resize(bit_pool_offsets.back() + bit_pool_sizes.back());
  }

  cat_sets_owner() {}
  cat_sets_owner(std::vector<uint8_t> bits_, std::vector<float> fid_num_cats_)
    : bits(bits_), fid_num_cats(fid_num_cats_)
  {
  }
};

std::ostream& operator<<(std::ostream& os, const cat_sets_owner& cso);

struct cat_sets_device_owner {
  // arrays from each node ID are concatenated first, then from all categories
  rmm::device_uvector<uint8_t> bits;
  // largest matching category in the model, per feature ID
  rmm::device_uvector<float> fid_num_cats;

  categorical_sets accessor() const
  {
    return {
      .bits              = bits.data(),
      .fid_num_cats      = fid_num_cats.data(),
      .bits_size         = bits.size(),
      .fid_num_cats_size = fid_num_cats.size(),
    };
  }
  cat_sets_device_owner(cudaStream_t stream) : bits(0, stream), fid_num_cats(0, stream) {}
  cat_sets_device_owner(categorical_sets cat_sets, cudaStream_t stream)
    : bits(cat_sets.bits_size, stream), fid_num_cats(cat_sets.fid_num_cats_size, stream)
  {
    ASSERT(bits.size() <= std::size_t(INT_MAX) + std::size_t(1),
           "too many categories/categorical nodes: cannot store bits offset in node");
    if (cat_sets.fid_num_cats_size > 0) {
      ASSERT(cat_sets.fid_num_cats != nullptr, "internal error: cat_sets.fid_num_cats is nil");
      RAFT_CUDA_TRY(cudaMemcpyAsync(fid_num_cats.data(),
                                    cat_sets.fid_num_cats,
                                    fid_num_cats.size() * sizeof(float),
                                    cudaMemcpyDefault,
                                    stream));
    }
    if (cat_sets.bits_size > 0) {
      ASSERT(cat_sets.bits != nullptr, "internal error: cat_sets.bits is nil");
      RAFT_CUDA_TRY(cudaMemcpyAsync(
        bits.data(), cat_sets.bits, bits.size() * sizeof(uint8_t), cudaMemcpyDefault, stream));
    }
  }
  void release()
  {
    bits.release();
    fid_num_cats.release();
  }
};

/** init uses params, trees and nodes to initialize the forest
 *  with nodes stored in pf
 *  @tparam fil_node_t node type to use with the forest;
 *    must be sparse_node16, sparse_node8 or dense_node
 *  @param h cuML handle used by this function
 *  @param pf pointer to where to store the newly created forest
 *  @param trees for sparse forests, indices of tree roots in the nodes arrray, of length
 params->ntrees; ignored for dense forests
 *  @param nodes nodes for the forest, of length params->num_nodes for sparse
      or (2**(params->depth + 1) - 1) * params->ntrees for dense forests
 *  @param params pointer to parameters used to initialize the forest
 *  @param vector_leaf optional vector leaves
 */
template <typename fil_node_t, typename real_t = typename fil_node_t::real_type>
void init(const raft::handle_t& h,
          forest_t<real_t>* pf,
          const categorical_sets& cat_sets,
          const std::vector<real_t>& vector_leaf,
          const int* trees,
          const fil_node_t* nodes,
          const forest_params_t* params);

struct predict_params;

}  // namespace fil

static const int MAX_SHM_STD = 48 * 1024;  // maximum architecture-independent size

std::string output2str(fil::output_t output);
}  // namespace ML
