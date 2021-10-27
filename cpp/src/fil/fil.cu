/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

/** @file fil.cu implements forest inference */

#include "common.cuh"

#include <cuml/fil/fil.h>
#include <cuml/fil/fnv_hash.h>
#include <cuml/common/logger.hpp>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>

#include <treelite/c_api.h>
#include <treelite/tree.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <omp.h>

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <stack>
#include <utility>

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
  for (int mm : cso.max_matching) {
    os << mm << " ";
  }
  os << " }";
  return os;
}

__host__ __device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/** performs additional transformations on the array of forest predictions
    (preds) of size n; the transformations are defined by output, and include
    averaging (multiplying by inv_num_trees), adding global_bias (always done),
    sigmoid and applying threshold. in case of complement_proba,
    fills in the complement probability */
__global__ void transform_k(float* preds,
                            size_t n,
                            output_t output,
                            float inv_num_trees,
                            float threshold,
                            float global_bias,
                            bool complement_proba)
{
  size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x;
  if (i >= n) return;
  if (complement_proba && i % 2 != 0) return;

  float result = preds[i];
  if ((output & output_t::AVG) != 0) result *= inv_num_trees;
  result += global_bias;
  if ((output & output_t::SIGMOID) != 0) result = sigmoid(result);
  // will not be done on CATEGORICAL_LEAF because the whole kernel will not run
  if ((output & output_t::CLASS) != 0) { result = result > threshold ? 1.0f : 0.0f; }
  // sklearn outputs numpy array in 'C' order, with the number of classes being last dimension
  // that is also the default order, so we should use the same one
  if (complement_proba) {
    preds[i]     = 1.0f - result;
    preds[i + 1] = result;
  } else
    preds[i] = result;
}

// needed to avoid expanding the dispatch template into unresolved
// compute_smem_footprint::run() calls. In infer.cu, we don't export those symbols,
// but rather one symbol for the whole template specialization, as below.
extern template int dispatch_on_fil_template_params(compute_smem_footprint, predict_params);

struct forest {
  forest(const raft::handle_t& h) : vector_leaf_(0, h.get_stream()), cat_sets_(h.get_stream()) {}

  void init_n_items(int device)
  {
    int max_shm_std = 48 * 1024;  // 48 KiB
    /// the most shared memory a kernel can request on the GPU in question
    int max_shm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shm, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    /* Our GPUs have been growing the shared memory size generation after
       generation. Eventually, a CUDA GPU might come by that supports more
       shared memory that would fit into unsigned 16-bit int. For such a GPU,
       we would have otherwise silently overflowed the index calculation due
       to short division. It would have failed cpp tests, but we might forget
       about this source of bugs, if not for the failing assert. */
    ASSERT(max_shm < 262144,
           "internal error: please use a larger type inside"
           " infer_k for column count");
    // TODO(canonizer): use >48KiB shared memory if available
    max_shm = std::min(max_shm, max_shm_std);

    // searching for the most items per block while respecting the shared
    // memory limits creates a full linear programming problem.
    // solving it in a single equation looks less tractable than this
    for (bool predict_proba : {false, true}) {
      shmem_size_params& ssp_ = predict_proba ? proba_ssp_ : class_ssp_;
      ssp_.predict_proba      = predict_proba;
      shmem_size_params ssp   = ssp_;
      // if n_items was not provided, try from 1 to MAX_N_ITEMS. Otherwise, use as-is.
      int min_n_items = ssp.n_items == 0 ? 1 : ssp.n_items;
      int max_n_items =
        ssp.n_items == 0 ? (algo_ == algo_t::BATCH_TREE_REORG ? MAX_N_ITEMS : 1) : ssp.n_items;
      for (bool cols_in_shmem : {false, true}) {
        ssp.cols_in_shmem = cols_in_shmem;
        for (ssp.n_items = min_n_items; ssp.n_items <= max_n_items; ++ssp.n_items) {
          ssp.shm_sz = dispatch_on_fil_template_params(compute_smem_footprint(), ssp);
          if (ssp.shm_sz < max_shm) ssp_ = ssp;
        }
      }
      ASSERT(max_shm >= ssp_.shm_sz,
             "FIL out of shared memory. Perhaps the maximum number of \n"
             "supported classes is exceeded? 5'000 would still be safe.");
    }
  }

  void init_fixed_block_count(int device, int blocks_per_sm)
  {
    int max_threads_per_sm, sm_count;
    CUDA_CHECK(
      cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    blocks_per_sm = std::min(blocks_per_sm, max_threads_per_sm / FIL_TPB);
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    fixed_block_count_ = blocks_per_sm * sm_count;
  }

  void init_common(const raft::handle_t& h,
                   const categorical_sets& cat_sets,
                   const std::vector<float>& vector_leaf,
                   const forest_params_t* params)
  {
    depth_                           = params->depth;
    num_trees_                       = params->num_trees;
    algo_                            = params->algo;
    output_                          = params->output;
    threshold_                       = params->threshold;
    global_bias_                     = params->global_bias;
    proba_ssp_.n_items               = params->n_items;
    proba_ssp_.log2_threads_per_tree = log2(params->threads_per_tree);
    proba_ssp_.leaf_algo             = params->leaf_algo;
    proba_ssp_.num_cols              = params->num_cols;
    proba_ssp_.num_classes           = params->num_classes;
    proba_ssp_.cats_present          = cat_sets.cats_present();
    class_ssp_                       = proba_ssp_;

    int device          = h.get_device();
    cudaStream_t stream = h.get_stream();
    init_n_items(device);  // n_items takes priority over blocks_per_sm
    init_fixed_block_count(device, params->blocks_per_sm);

    // vector leaf
    if (!vector_leaf.empty()) {
      vector_leaf_.resize(vector_leaf.size(), stream);

      CUDA_CHECK(cudaMemcpyAsync(vector_leaf_.data(),
                                 vector_leaf.data(),
                                 vector_leaf.size() * sizeof(float),
                                 cudaMemcpyHostToDevice,
                                 stream));
    }

    // categorical features
    cat_sets_ = cat_sets_device_owner(cat_sets, stream);
  }

  virtual void infer(predict_params params, cudaStream_t stream) = 0;

  void predict(
    const raft::handle_t& h, float* preds, const float* data, size_t num_rows, bool predict_proba)
  {
    // Initialize prediction parameters.
    predict_params params(predict_proba ? proba_ssp_ : class_ssp_);
    params.algo     = algo_;
    params.preds    = preds;
    params.data     = data;
    params.num_rows = num_rows;
    // ignored unless predict_proba is true and algo is GROVE_PER_CLASS
    params.transform = output_;
    // fixed_block_count_ == 0 means the number of thread blocks is
    // proportional to the number of rows
    params.num_blocks = fixed_block_count_;

    /**
    The binary classification / regression (FLOAT_UNARY_BINARY) predict_proba() works as follows
      (always 2 outputs):
    RAW: output the sum of tree predictions
    AVG is set: divide by the number of trees (averaging)
    SIGMOID is set: apply sigmoid
    CLASS is set: ignored
    SOFTMAX is set: error
    write the output of the previous stages and its complement

    The binary classification / regression (FLOAT_UNARY_BINARY) predict() works as follows
      (always 1 output):
    RAW (no values set): output the sum of tree predictions
    AVG is set: divide by the number of trees (averaging)
    SIGMOID is set: apply sigmoid
    CLASS is set: apply threshold (equivalent to choosing best class)
    SOFTMAX is set: error

    The multi-class classification / regression (CATEGORICAL_LEAF) predict_proba() works as follows
      (always num_classes outputs):
    RAW (no values set): output class votes
    AVG is set: divide by the number of trees (averaging, output class probability)
    SIGMOID is set: apply sigmoid
    CLASS is set: ignored
    SOFTMAX is set: error

    The multi-class classification / regression (CATEGORICAL_LEAF) predict() works as follows
      (always 1 output):
    RAW (no values set): output the label of the class with highest probability, else output label
    0. SOFTMAX is set: error All other flags (AVG, SIGMOID, CLASS) are ignored

    The multi-class classification / regression (GROVE_PER_CLASS) predict_proba() works as follows
      (always num_classes outputs):
    RAW (no values set): output class votes
    AVG is set: divide by the number of trees (averaging, output class probability)
    SIGMOID is set: apply sigmoid; if SOFTMAX is also set: error
    CLASS is set: ignored
    SOFTMAX is set: softmax is applied after averaging and global_bias

    The multi-class classification / regression (GROVE_PER_CLASS) predict() works as follows
      (always 1 output):
    RAW (no values set): output the label of the class with highest margin,
      equal margins resolved in favor of smaller label integer
    All other flags (AVG, SIGMOID, CLASS, SOFTMAX) are ignored

    The multi-class classification / regression (VECTOR_LEAF) predict_proba() works as follows
      (always num_classes outputs):
    RAW (no values set): output class votes
    AVG is set: divide by the number of trees (averaging, output class probability)
    SIGMOID is set: apply sigmoid; if SOFTMAX is also set: error
    CLASS is set: ignored
    SOFTMAX is set: softmax is applied after averaging and global_bias
    All other flags (SIGMOID, CLASS, SOFTMAX) are ignored

    The multi-class classification / regression (VECTOR_LEAF) predict() works as follows
      (always 1 output):
    RAW (no values set): output the label of the class with highest margin,
      equal margins resolved in favor of smaller label integer
    All other flags (AVG, SIGMOID, CLASS, SOFTMAX) are ignored
    */
    output_t ot = output_;
    // Treelite applies bias before softmax, but we do after.
    // Simulating treelite order, which cancels out bias.
    // If non-proba prediction used, it still will not matter
    // for the same reason softmax will not.
    float global_bias     = (ot & output_t::SOFTMAX) != 0 ? 0.0f : global_bias_;
    bool complement_proba = false, do_transform;

    if (predict_proba) {
      // no threshold on probabilities
      ot = output_t(ot & ~output_t::CLASS);

      switch (params.leaf_algo) {
        case leaf_algo_t::FLOAT_UNARY_BINARY:
          params.num_outputs = 2;
          complement_proba   = true;
          do_transform       = true;
          break;
        case leaf_algo_t::GROVE_PER_CLASS:
          // for GROVE_PER_CLASS, averaging happens in infer_k
          ot                 = output_t(ot & ~output_t::AVG);
          params.num_outputs = params.num_classes;
          do_transform = (ot != output_t::RAW && ot != output_t::SOFTMAX) || global_bias != 0.0f;
          break;
        case leaf_algo_t::CATEGORICAL_LEAF:
          params.num_outputs = params.num_classes;
          do_transform       = ot != output_t::RAW || global_bias_ != 0.0f;
          break;
        case leaf_algo_t::VECTOR_LEAF:
          // for VECTOR_LEAF, averaging happens in infer_k
          ot                 = output_t(ot & ~output_t::AVG);
          params.num_outputs = params.num_classes;
          do_transform = (ot != output_t::RAW && ot != output_t::SOFTMAX) || global_bias != 0.0f;
          break;
        default: ASSERT(false, "internal error: predict: invalid leaf_algo %d", params.leaf_algo);
      }
    } else {
      if (params.leaf_algo == leaf_algo_t::FLOAT_UNARY_BINARY) {
        do_transform = ot != output_t::RAW || global_bias_ != 0.0f;
      } else {
        // GROVE_PER_CLASS, CATEGORICAL_LEAF: moot since choosing best class and
        // all transforms are monotonic. also, would break current code
        do_transform = false;
      }
      params.num_outputs = 1;
    }

    // Predict using the forest.
    cudaStream_t stream = h.get_stream();
    infer(params, stream);

    if (do_transform) {
      size_t num_values_to_transform = (size_t)num_rows * (size_t)params.num_outputs;
      transform_k<<<raft::ceildiv(num_values_to_transform, (size_t)FIL_TPB), FIL_TPB, 0, stream>>>(
        preds,
        num_values_to_transform,
        ot,
        num_trees_ > 0 ? (1.0f / num_trees_) : 1.0f,
        threshold_,
        global_bias,
        complement_proba);
      CUDA_CHECK(cudaPeekAtLastError());
    }
  }

  virtual void free(const raft::handle_t& h)
  {
    cat_sets_.release();
    vector_leaf_.release();
  }

  virtual ~forest() {}

  int num_trees_     = 0;
  int depth_         = 0;
  algo_t algo_       = algo_t::NAIVE;
  output_t output_   = output_t::RAW;
  float threshold_   = 0.5;
  float global_bias_ = 0;
  shmem_size_params class_ssp_, proba_ssp_;
  int fixed_block_count_ = 0;
  // Optionally used
  rmm::device_uvector<float> vector_leaf_;
  cat_sets_device_owner cat_sets_;
};

struct dense_forest : forest {
  dense_forest(const raft::handle_t& h) : forest(h), nodes_(0, h.get_stream()) {}

  void transform_trees(const dense_node* nodes)
  {
    /* Populate node information:
       For each tree, the nodes are still stored in the breadth-first,
       left-to-right order. However, instead of storing the nodes of the same
       tree adjacently, it uses a different layout. In this layout, the roots
       of all trees (node 0) are stored first, followed by left children of
       the roots of all trees (node 1), followed by the right children of the
       roots of all trees (node 2), and so on.
    */
    int global_node = 0;
    for (int tree = 0; tree < num_trees_; ++tree) {
      int tree_node = 0;
      // the counters `level` and `branch` are not used for computing node
      // indices, they are only here to highlight the node ordering within
      // each tree
      for (int level = 0; level <= depth_; ++level) {
        for (int branch = 0; branch < 1 << level; ++branch) {
          h_nodes_[tree_node * num_trees_ + tree] = nodes[global_node];
          ++tree_node;
          ++global_node;
        }
      }
    }
  }

  void init(const raft::handle_t& h,
            const categorical_sets& cat_sets,
            const std::vector<float>& vector_leaf,
            const dense_node* nodes,
            const forest_params_t* params)
  {
    init_common(h, cat_sets, vector_leaf, params);
    if (algo_ == algo_t::NAIVE) algo_ = algo_t::BATCH_TREE_REORG;

    int num_nodes = forest_num_nodes(num_trees_, depth_);
    nodes_.resize(num_nodes, h.get_stream());
    h_nodes_.resize(num_nodes);
    if (algo_ == algo_t::NAIVE) {
      std::copy(nodes, nodes + num_nodes, h_nodes_.begin());
    } else {
      transform_trees(nodes);
    }
    CUDA_CHECK(cudaMemcpyAsync(nodes_.data(),
                               h_nodes_.data(),
                               num_nodes * sizeof(dense_node),
                               cudaMemcpyHostToDevice,
                               h.get_stream()));
    // copy must be finished before freeing the host data
    CUDA_CHECK(cudaStreamSynchronize(h.get_stream()));
    h_nodes_.clear();
    h_nodes_.shrink_to_fit();
  }

  virtual void infer(predict_params params, cudaStream_t stream) override
  {
    dense_storage forest(cat_sets_.accessor(),
                         vector_leaf_.data(),
                         nodes_.data(),
                         num_trees_,
                         algo_ == algo_t::NAIVE ? tree_num_nodes(depth_) : 1,
                         algo_ == algo_t::NAIVE ? 1 : num_trees_);
    fil::infer(forest, params, stream);
  }

  virtual void free(const raft::handle_t& h) override
  {
    nodes_.release();
    forest::free(h);
  }

  rmm::device_uvector<dense_node> nodes_;
  thrust::host_vector<dense_node> h_nodes_;
};

template <typename node_t>
struct sparse_forest : forest {
  sparse_forest(const raft::handle_t& h)
    : forest(h), trees_(0, h.get_stream()), nodes_(0, h.get_stream())
  {
  }

  void init(const raft::handle_t& h,
            const categorical_sets& cat_sets,
            const std::vector<float>& vector_leaf,
            const int* trees,
            const node_t* nodes,
            const forest_params_t* params)
  {
    init_common(h, cat_sets, vector_leaf, params);
    if (algo_ == algo_t::ALGO_AUTO) algo_ = algo_t::NAIVE;
    depth_     = 0;  // a placeholder value
    num_nodes_ = params->num_nodes;

    // trees
    trees_.resize(num_trees_, h.get_stream());
    CUDA_CHECK(cudaMemcpyAsync(
      trees_.data(), trees, sizeof(int) * num_trees_, cudaMemcpyHostToDevice, h.get_stream()));

    // nodes
    nodes_.resize(num_nodes_, h.get_stream());
    CUDA_CHECK(cudaMemcpyAsync(
      nodes_.data(), nodes, sizeof(node_t) * num_nodes_, cudaMemcpyHostToDevice, h.get_stream()));
  }

  virtual void infer(predict_params params, cudaStream_t stream) override
  {
    sparse_storage<node_t> forest(
      cat_sets_.accessor(), vector_leaf_.data(), trees_.data(), nodes_.data(), num_trees_);
    fil::infer(forest, params, stream);
  }

  void free(const raft::handle_t& h) override
  {
    forest::free(h);
    trees_.release();
    nodes_.release();
  }

  int num_nodes_ = 0;
  rmm::device_uvector<int> trees_;
  rmm::device_uvector<node_t> nodes_;
};

void check_params(const forest_params_t* params, bool dense)
{
  if (dense) {
    ASSERT(params->depth >= 0, "depth must be non-negative for dense forests");
  } else {
    ASSERT(params->num_nodes >= 0, "num_nodes must be non-negative for sparse forests");
    ASSERT(params->algo == algo_t::NAIVE || params->algo == algo_t::ALGO_AUTO,
           "only ALGO_AUTO and NAIVE algorithms are supported "
           "for sparse forests");
  }
  ASSERT(params->num_trees >= 0, "num_trees must be non-negative");
  ASSERT(params->num_cols >= 0, "num_cols must be non-negative");
  switch (params->algo) {
    case algo_t::ALGO_AUTO:
    case algo_t::NAIVE:
    case algo_t::TREE_REORG:
    case algo_t::BATCH_TREE_REORG: break;
    default: ASSERT(false, "algo should be ALGO_AUTO, NAIVE, TREE_REORG or BATCH_TREE_REORG");
  }
  switch (params->leaf_algo) {
    case leaf_algo_t::FLOAT_UNARY_BINARY:
      if ((params->output & output_t::CLASS) != 0) {
        ASSERT(params->num_classes == 2,
               "only supporting binary"
               " classification using FLOAT_UNARY_BINARY");
      } else {
        ASSERT(params->num_classes == 1,
               "num_classes must be 1 for "
               "regression");
      }
      ASSERT((params->output & output_t::SOFTMAX) == 0,
             "softmax does not make sense for leaf_algo == FLOAT_UNARY_BINARY");
      break;
    case leaf_algo_t::GROVE_PER_CLASS:
      ASSERT(params->threads_per_tree == 1, "multiclass not supported with threads_per_tree > 1");
      ASSERT(params->num_classes > 2,
             "num_classes > 2 is required for leaf_algo == GROVE_PER_CLASS");
      ASSERT(params->num_trees % params->num_classes == 0,
             "num_classes must divide num_trees evenly for GROVE_PER_CLASS");
      break;
    case leaf_algo_t::CATEGORICAL_LEAF:
      ASSERT(params->threads_per_tree == 1, "multiclass not supported with threads_per_tree > 1");
      ASSERT(params->num_classes >= 2,
             "num_classes >= 2 is required for "
             "leaf_algo == CATEGORICAL_LEAF");
      ASSERT((params->output & output_t::SOFTMAX) == 0,
             "softmax not supported for leaf_algo == CATEGORICAL_LEAF");
      break;
    case leaf_algo_t::VECTOR_LEAF:
      ASSERT(params->num_classes >= 2,
             "num_classes >= 2 is required for "
             "leaf_algo == VECTOR_LEAF");
      break;
    default:
      ASSERT(false,
             "leaf_algo must be FLOAT_UNARY_BINARY, CATEGORICAL_LEAF"
             " or GROVE_PER_CLASS");
  }
  // output_t::RAW == 0, and doesn't have a separate flag
  if ((params->output & ~output_t::ALL_SET) != 0) {
    ASSERT(false, "output should be a combination of RAW, AVG, SIGMOID, CLASS and SOFTMAX");
  }
  ASSERT(~params->output & (output_t::SIGMOID | output_t::SOFTMAX),
         "combining softmax and sigmoid is not supported");
  ASSERT(params->blocks_per_sm >= 0, "blocks_per_sm must be nonnegative");
  ASSERT(params->n_items >= 0, "n_items must be non-negative");
  ASSERT(params->threads_per_tree > 0, "threads_per_tree must be positive");
  ASSERT(thrust::detail::is_power_of_2(params->threads_per_tree),
         "threads_per_tree must be a power of 2");
  ASSERT(params->threads_per_tree <= FIL_TPB,
         "threads_per_tree must not "
         "exceed block size %d",
         FIL_TPB);
}

template <typename T, typename L>
int tree_root(const tl::Tree<T, L>& tree)
{
  return 0;  // Treelite format assumes that the root is 0
}

template <typename T, typename L>
inline int max_depth(const tl::Tree<T, L>& tree)
{
  // trees of this depth aren't used, so it most likely means bad input data,
  // e.g. cycles in the forest
  const int DEPTH_LIMIT = 500;
  int root_index        = tree_root(tree);
  typedef std::pair<int, int> pair_t;
  std::stack<pair_t> stack;
  stack.push(pair_t(root_index, 0));
  int max_depth = 0;
  while (!stack.empty()) {
    const pair_t& pair = stack.top();
    int node_id        = pair.first;
    int depth          = pair.second;
    stack.pop();
    while (!tree.IsLeaf(node_id)) {
      stack.push(pair_t(tree.LeftChild(node_id), depth + 1));
      node_id = tree.RightChild(node_id);
      depth++;
      ASSERT(depth < DEPTH_LIMIT, "depth limit reached, might be a cycle in the tree");
    }
    // only need to update depth for leaves
    max_depth = std::max(max_depth, depth);
  }
  return max_depth;
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
  std::stack<int> stack;
  stack.push(tree_root(tree));
  while (!stack.empty()) {
    int node_id = stack.top();
    stack.pop();
    while (!tree.IsLeaf(node_id)) {
      if (tree.SplitType(node_id) == tl::SplitFeatureType::kCategorical) {
        std::vector<std::uint32_t> mmv = tree.MatchingCategories(node_id);
        int max_matching_cat;
        if (mmv.size() > 0) {
          // in `struct cat_feature_counters` and GPU structures, max matching category is an int
          // cast is safe because all precise int floats fit into ints, which are asserted to be 32
          // bits
          max_matching_cat = mmv.back();
          ASSERT(max_matching_cat <= MAX_PRECISE_INT_FLOAT,
                 "FIL cannot infer on "
                 "more than %d matching categories",
                 MAX_PRECISE_INT_FLOAT);
        } else {
          max_matching_cat = -1;
        }
        cat_feature_counters& counters = res[tree.SplitIndex(node_id)];
        counters =
          cat_feature_counters::combine(counters, cat_feature_counters{max_matching_cat, 1});
      }
      stack.push(tree.LeftChild(node_id));
      node_id = tree.RightChild(node_id);
    }
  }
  return res;
}

// computes overall categorical bit pool size for a tree imported from the Treelite tree
template <typename T, typename L>
inline std::size_t bit_pool_size(const tl::Tree<T, L>& tree, const categorical_sets& cat_sets)
{
  std::size_t size = 0;
  std::stack<int> stack;
  stack.push(tree_root(tree));
  while (!stack.empty()) {
    int node_id = stack.top();
    stack.pop();
    while (!tree.IsLeaf(node_id)) {
      if (tree.SplitType(node_id) == tl::SplitFeatureType::kCategorical) {
        int fid = tree.SplitIndex(node_id);
        size += cat_sets.sizeof_mask(fid);
      }
      stack.push(tree.LeftChild(node_id));
      node_id = tree.RightChild(node_id);
    }
  }
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

void adjust_threshold(
  float* pthreshold, int* tl_left, int* tl_right, bool* default_left, tl::Operator comparison_op)
{
  // in treelite (take left node if val [op] threshold),
  // the meaning of the condition is reversed compared to FIL;
  // thus, "<" in treelite corresonds to comparison ">=" used by FIL
  // https://github.com/dmlc/treelite/blob/master/include/treelite/tree.h#L243
  if (isnan(*pthreshold)) {
    std::swap(*tl_left, *tl_right);
    *default_left = !*default_left;
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
      std::swap(*tl_left, *tl_right);
      *default_left = !*default_left;
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
  int tl_left;
  int tl_right;
};

// modifies cat_sets
template <typename fil_node_t, typename T, typename L>
conversion_state<fil_node_t> tl2fil_inner_node(int fil_left_child,
                                               const tl::Tree<T, L>& tree,
                                               int tl_node_id,
                                               const forest_params_t& forest_params,
                                               cat_sets_owner* cat_sets,
                                               std::size_t* bit_pool_offset)
{
  int tl_left = tree.LeftChild(tl_node_id), tl_right = tree.RightChild(tl_node_id);
  val_t split         = {.f = NAN};  // yes there's a default initializer already
  int feature_id      = tree.SplitIndex(tl_node_id);
  bool is_categorical = tree.SplitType(tl_node_id) == tl::SplitFeatureType::kCategorical;
  bool default_left   = tree.DefaultLeft(tl_node_id);
  if (tree.SplitType(tl_node_id) == tl::SplitFeatureType::kNumerical) {
    split.f = static_cast<float>(tree.Threshold(tl_node_id));
    adjust_threshold(&split.f, &tl_left, &tl_right, &default_left, tree.ComparisonOp(tl_node_id));
  } else if (tree.SplitType(tl_node_id) == tl::SplitFeatureType::kCategorical) {
    // for FIL, the list of categories is always for the right child
    if (!tree.CategoriesListRightChild(tl_node_id)) {
      std::swap(tl_left, tl_right);
      default_left = !default_left;
    }
    int sizeof_mask = cat_sets->accessor().sizeof_mask(feature_id);
    split.idx       = *bit_pool_offset;
    *bit_pool_offset += sizeof_mask;
    // cat_sets->bits have been zero-initialized
    uint8_t* bits = &cat_sets->bits[split.idx];
    for (std::uint32_t category : tree.MatchingCategories(tl_node_id)) {
      bits[category / BITS_PER_BYTE] |= 1 << (category % BITS_PER_BYTE);
    }
  } else {
    ASSERT(false, "only numerical and categorical split nodes are supported");
  }
  fil_node_t node;
  if constexpr (std::is_same<fil_node_t, dense_node>()) {
    node = fil_node_t({}, split, feature_id, default_left, false, is_categorical);
  } else {
    node = fil_node_t({}, split, feature_id, default_left, false, is_categorical, fil_left_child);
  }
  return conversion_state<fil_node_t>{node, tl_left, tl_right};
}

template <typename T, typename L>
void node2fil_dense(std::vector<dense_node>* pnodes,
                    int root,
                    int cur,
                    const tl::Tree<T, L>& tree,
                    int node_id,
                    const forest_params_t& forest_params,
                    std::vector<float>* vector_leaf,
                    std::size_t* leaf_counter,
                    cat_sets_owner* cat_sets,
                    std::size_t* bit_pool_offset)
{
  if (tree.IsLeaf(node_id)) {
    (*pnodes)[root + cur] = dense_node({}, {}, 0, false, true, false);
    tl2fil_leaf_payload(
      &(*pnodes)[root + cur], root + cur, tree, node_id, forest_params, vector_leaf, leaf_counter);
    return;
  }

  // inner node
  int left = 2 * cur + 1;
  conversion_state<dense_node> cs =
    tl2fil_inner_node<dense_node>(left, tree, node_id, forest_params, cat_sets, bit_pool_offset);
  (*pnodes)[root + cur] = cs.node;
  node2fil_dense(pnodes,
                 root,
                 left,
                 tree,
                 cs.tl_left,
                 forest_params,
                 vector_leaf,
                 leaf_counter,
                 cat_sets,
                 bit_pool_offset);
  node2fil_dense(pnodes,
                 root,
                 left + 1,
                 tree,
                 cs.tl_right,
                 forest_params,
                 vector_leaf,
                 leaf_counter,
                 cat_sets,
                 bit_pool_offset);
}

template <typename T, typename L>
void tree2fil_dense(std::vector<dense_node>* pnodes,
                    int root,
                    const tl::Tree<T, L>& tree,
                    std::size_t tree_idx,
                    const forest_params_t& forest_params,
                    std::vector<float>* vector_leaf,
                    std::size_t* leaf_counter,
                    cat_sets_owner* cat_sets)
{
  node2fil_dense(pnodes,
                 root,
                 0,
                 tree,
                 tree_root(tree),
                 forest_params,
                 vector_leaf,
                 leaf_counter,
                 cat_sets,
                 &cat_sets->bit_pool_offsets[tree_idx]);
}

template <typename fil_node_t, typename T, typename L>
int tree2fil_sparse(std::vector<fil_node_t>& nodes,
                    int root,
                    const tl::Tree<T, L>& tree,
                    std::size_t tree_idx,
                    const forest_params_t& forest_params,
                    std::vector<float>* vector_leaf,
                    std::size_t* leaf_counter,
                    cat_sets_owner* cat_sets)
{
  typedef std::pair<int, int> pair_t;
  std::stack<pair_t> stack;
  int built_index = root + 1;
  stack.push(pair_t(tree_root(tree), 0));
  while (!stack.empty()) {
    const pair_t& top = stack.top();
    int node_id       = top.first;
    int cur           = top.second;
    stack.pop();

    while (!tree.IsLeaf(node_id)) {
      // reserve space for child nodes
      // left is the offset of the left child node relative to the tree root
      // in the array of all nodes of the FIL sparse forest
      int left = built_index - root;
      built_index += 2;
      conversion_state<fil_node_t> cs = tl2fil_inner_node<fil_node_t>(
        left, tree, node_id, forest_params, cat_sets, &cat_sets->bit_pool_offsets[tree_idx]);
      nodes[root + cur] = cs.node;
      // push child nodes into the stack
      stack.push(pair_t(cs.tl_right, left + 1));
      // stack.push(pair_t(tl_left, left));
      node_id = cs.tl_left;
      cur     = left;
    }

    // leaf node
    nodes[root + cur] = fil_node_t({}, {}, 0, false, true, false, 0);
    tl2fil_leaf_payload(
      &nodes[root + cur], root + cur, tree, node_id, forest_params, vector_leaf, leaf_counter);
  }

  return root;
}

struct level_entry {
  int n_branch_nodes, n_leaves;
};
typedef std::pair<int, int> pair_t;
// hist has branch and leaf count given depth
template <typename T, typename L>
inline void tree_depth_hist(const tl::Tree<T, L>& tree, std::vector<level_entry>& hist)
{
  std::stack<pair_t> stack;  // {tl_id, depth}
  stack.push({tree_root(tree), 0});
  while (!stack.empty()) {
    const pair_t& top = stack.top();
    int node_id       = top.first;
    int depth         = top.second;
    stack.pop();

    while (!tree.IsLeaf(node_id)) {
      if (static_cast<std::size_t>(depth) >= hist.size()) hist.resize(depth + 1, {0, 0});
      hist[depth].n_branch_nodes++;
      stack.push({tree.LeftChild(node_id), depth + 1});
      node_id = tree.RightChild(node_id);
      depth++;
    }

    if (static_cast<std::size_t>(depth) >= hist.size()) hist.resize(depth + 1, {0, 0});
    hist[depth].n_leaves++;
  }
}

template <typename T, typename L>
std::stringstream depth_hist_and_max(const tl::ModelImpl<T, L>& model)
{
  using namespace std;
  vector<level_entry> hist;
  for (const auto& tree : model.trees)
    tree_depth_hist(tree, hist);

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

// uses treelite model with additional tl_params to initialize FIL params
// and dense nodes (stored in *pnodes)
template <typename threshold_t, typename leaf_t>
void tl2fil_dense(std::vector<dense_node>* pnodes,
                  forest_params_t* params,
                  const tl::ModelImpl<threshold_t, leaf_t>& model,
                  const treelite_params_t* tl_params,
                  cat_sets_owner* cat_sets,
                  std::vector<float>* vector_leaf)
{
  tl2fil_common(params, model, tl_params);

  // convert the nodes
  int num_nodes           = forest_num_nodes(params->num_trees, params->depth);
  int max_leaves_per_tree = (tree_num_nodes(params->depth) + 1) / 2;
  if (params->leaf_algo == VECTOR_LEAF) {
    vector_leaf->resize(max_leaves_per_tree * params->num_trees * params->num_classes);
  }
  *cat_sets = allocate_cat_sets_owner(model);
  pnodes->resize(num_nodes, dense_node());
  for (std::size_t i = 0; i < model.trees.size(); ++i) {
    size_t leaf_counter = max_leaves_per_tree * i;
    tree2fil_dense(pnodes,
                   i * tree_num_nodes(params->depth),
                   model.trees[i],
                   i,
                   *params,
                   vector_leaf,
                   &leaf_counter,
                   cat_sets);
  }
}

template <typename fil_node_t>
struct tl2fil_sparse_check_t {
  template <typename threshold_t, typename leaf_t>
  static void check(const tl::ModelImpl<threshold_t, leaf_t>& model)
  {
    ASSERT(false,
           "internal error: "
           "only a specialization of this template should be used");
  }
};

template <>
struct tl2fil_sparse_check_t<sparse_node16> {
  // no extra check for 16-byte sparse nodes
  template <typename threshold_t, typename leaf_t>
  static void check(const tl::ModelImpl<threshold_t, leaf_t>& model)
  {
  }
};

template <>
struct tl2fil_sparse_check_t<sparse_node8> {
  static const int MAX_FEATURES   = 1 << sparse_node8::FID_NUM_BITS;
  static const int MAX_TREE_NODES = (1 << sparse_node8::LEFT_NUM_BITS) - 1;
  template <typename threshold_t, typename leaf_t>
  static void check(const tl::ModelImpl<threshold_t, leaf_t>& model)
  {
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
};

// uses treelite model with additional tl_params to initialize FIL params,
// trees (stored in *ptrees) and sparse nodes (stored in *pnodes)
template <typename fil_node_t, typename threshold_t, typename leaf_t>
void tl2fil_sparse(std::vector<int>* ptrees,
                   std::vector<fil_node_t>* pnodes,
                   forest_params_t* params,
                   const tl::ModelImpl<threshold_t, leaf_t>& model,
                   const treelite_params_t* tl_params,
                   cat_sets_owner* cat_sets,
                   std::vector<float>* vector_leaf)
{
  tl2fil_common(params, model, tl_params);
  tl2fil_sparse_check_t<fil_node_t>::check(model);

  size_t num_trees = model.trees.size();

  ptrees->reserve(num_trees);
  ptrees->push_back(0);
  for (size_t i = 0; i < num_trees - 1; ++i) {
    ptrees->push_back(model.trees[i].num_nodes + ptrees->back());
  }
  size_t total_nodes = ptrees->back() + model.trees.back().num_nodes;

  if (params->leaf_algo == VECTOR_LEAF) {
    size_t max_leaves = (total_nodes + num_trees) / 2;
    vector_leaf->resize(max_leaves * params->num_classes);
  }

  *cat_sets = allocate_cat_sets_owner(model);
  pnodes->resize(total_nodes);

// convert the nodes
#pragma omp parallel for
  for (std::size_t i = 0; i < num_trees; ++i) {
    // Max number of leaves processed so far
    size_t leaf_counter = ((*ptrees)[i] + i) / 2;
    tree2fil_sparse(
      *pnodes, (*ptrees)[i], model.trees[i], i, *params, vector_leaf, &leaf_counter, cat_sets);
  }

  params->num_nodes = pnodes->size();
}

void init_dense(const raft::handle_t& h,
                forest_t* pf,
                const categorical_sets& cat_sets,
                const std::vector<float>& vector_leaf,
                const dense_node* nodes,
                const forest_params_t* params)
{
  check_params(params, true);
  dense_forest* f = new dense_forest(h);
  f->init(h, cat_sets, vector_leaf, nodes, params);
  *pf = f;
}

template <typename fil_node_t>
void init_sparse(const raft::handle_t& h,
                 forest_t* pf,
                 const categorical_sets& cat_sets,
                 const std::vector<float>& vector_leaf,
                 const int* trees,
                 const fil_node_t* nodes,
                 const forest_params_t* params)
{
  check_params(params, false);
  sparse_forest<fil_node_t>* f = new sparse_forest<fil_node_t>(h);
  f->init(h, cat_sets, vector_leaf, trees, nodes, params);
  *pf = f;
}

// explicit instantiations for init_sparse()
template void init_sparse<sparse_node16>(const raft::handle_t& h,
                                         forest_t* pf,
                                         const categorical_sets& cat_sets,
                                         const std::vector<float>& vector_leaf,
                                         const int* trees,
                                         const sparse_node16* nodes,
                                         const forest_params_t* params);

template void init_sparse<sparse_node8>(const raft::handle_t& h,
                                        forest_t* pf,
                                        const categorical_sets& cat_sets,
                                        const std::vector<float>& vector_leaf,
                                        const int* trees,
                                        const sparse_node8* nodes,
                                        const forest_params_t* params);

template <typename threshold_t, typename leaf_t>
void from_treelite(const raft::handle_t& handle,
                   forest_t* pforest,
                   const tl::ModelImpl<threshold_t, leaf_t>& model,
                   const treelite_params_t* tl_params)
{
  // Invariants on threshold and leaf types
  static_assert(std::is_same<threshold_t, float>::value || std::is_same<threshold_t, double>::value,
                "Model must contain float32 or float64 thresholds for splits");
  ASSERT((std::is_same<leaf_t, float>::value || std::is_same<leaf_t, double>::value),
         "Models with integer leaf output are not yet supported");
  // Display appropriate warnings when float64 values are being casted into
  // float32, as FIL only supports inferencing with float32 for the time being
  if (std::is_same<threshold_t, double>::value || std::is_same<leaf_t, double>::value) {
    CUML_LOG_WARN(
      "Casting all thresholds and leaf values to float32, as FIL currently "
      "doesn't support inferencing models with float64 values. "
      "This may lead to predictions with reduced accuracy.");
  }

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

  forest_params_t params;
  cat_sets_owner cat_sets;
  switch (storage_type) {
    case storage_type_t::DENSE: {
      std::vector<dense_node> nodes;
      std::vector<float> vector_leaf;
      tl2fil_dense(&nodes, &params, model, tl_params, &cat_sets, &vector_leaf);
      init_dense(handle, pforest, cat_sets.accessor(), vector_leaf, nodes.data(), &params);
      // sync is necessary as nodes is used in init_dense(),
      // but destructed at the end of this function
      CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
      if (tl_params->pforest_shape_str) {
        *tl_params->pforest_shape_str = sprintf_shape(model, storage_type, nodes, {}, cat_sets);
      }
      break;
    }
    case storage_type_t::SPARSE: {
      std::vector<int> trees;
      std::vector<sparse_node16> nodes;
      std::vector<float> vector_leaf;
      tl2fil_sparse(&trees, &nodes, &params, model, tl_params, &cat_sets, &vector_leaf);
      init_sparse(
        handle, pforest, cat_sets.accessor(), vector_leaf, trees.data(), nodes.data(), &params);
      CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
      if (tl_params->pforest_shape_str) {
        *tl_params->pforest_shape_str = sprintf_shape(model, storage_type, nodes, trees, cat_sets);
      }
      break;
    }
    case storage_type_t::SPARSE8: {
      std::vector<int> trees;
      std::vector<sparse_node8> nodes;
      std::vector<float> vector_leaf;
      tl2fil_sparse(&trees, &nodes, &params, model, tl_params, &cat_sets, &vector_leaf);
      init_sparse(
        handle, pforest, cat_sets.accessor(), vector_leaf, trees.data(), nodes.data(), &params);
      CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
      if (tl_params->pforest_shape_str) {
        *tl_params->pforest_shape_str = sprintf_shape(model, storage_type, nodes, trees, cat_sets);
      }
      break;
    }
    default: ASSERT(false, "tl_params->sparse must be one of AUTO, DENSE or SPARSE");
  }
}

void from_treelite(const raft::handle_t& handle,
                   forest_t* pforest,
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
                    storage_type_t storage,
                    const std::vector<node_t>& nodes,
                    const std::vector<int>& trees,
                    const cat_sets_owner cat_sets)
{
  std::stringstream forest_shape = depth_hist_and_max(model);
  double size_mb = (trees.size() * sizeof(trees.front()) + nodes.size() * sizeof(nodes.front()) +
                    cat_sets.bits.size()) /
                   1e6;
  forest_shape << storage_type_repr[storage] << " model size " << std::setprecision(2) << size_mb
               << " MB" << std::endl;
  if (cat_sets.bits.size() > 0) {
    forest_shape << "number of categorical nodes for each feature id: {";
    std::size_t total_cat_nodes = 0;
    for (std::size_t n : cat_sets.n_nodes) {
      forest_shape << n << " ";
      total_cat_nodes += n;
    }
    forest_shape << "}" << std::endl << "total categorical nodes: " << total_cat_nodes << std::endl;
    forest_shape << "maximum matching category for each feature id: {";
    for (int mm : cat_sets.max_matching)
      forest_shape << mm << " ";
    forest_shape << "}" << std::endl;
  }
  // stream may be discontiguous
  std::string forest_shape_str = forest_shape.str();
  // now copy to a non-owning allocation
  char* shape_out = (char*)malloc(forest_shape_str.size() + 1);  // incl. \0
  memcpy((void*)shape_out, forest_shape_str.c_str(), forest_shape_str.size() + 1);
  return shape_out;
}

void free(const raft::handle_t& h, forest_t f)
{
  f->free(h);
  delete f;
}

void predict(const raft::handle_t& h,
             forest_t f,
             float* preds,
             const float* data,
             size_t num_rows,
             bool predict_proba)
{
  f->predict(h, preds, data, num_rows, predict_proba);
}

}  // namespace fil
}  // namespace ML
