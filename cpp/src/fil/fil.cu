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

#include <omp.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <treelite/c_api.h>
#include <treelite/tree.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stack>
#include <utility>

#include <cuml/fil/fil.h>
#include <raft/cudart_utils.h>
#include <cuml/common/cuml_allocator.hpp>
#include "common.cuh"

namespace ML {
namespace fil {

using namespace MLCommon;
namespace tl = treelite;

__host__ __device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/** performs additional transformations on the array of forest predictions
    (preds) of size n; the transformations are defined by output, and include
    averaging (multiplying by inv_num_trees), adding global_bias (always done),
    sigmoid and applying threshold. in case of complement_proba,
    fills in the complement probability */
__global__ void transform_k(float* preds, size_t n, output_t output,
                            float inv_num_trees, float threshold,
                            float global_bias, bool complement_proba) {
  size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x;
  if (i >= n) return;
  if (complement_proba && i % 2 != 0) return;

  float result = preds[i];
  if ((output & output_t::AVG) != 0) result *= inv_num_trees;
  result += global_bias;
  if ((output & output_t::SIGMOID) != 0) result = sigmoid(result);
  // will not be done on CATEGORICAL_LEAF because the whole kernel will not run
  if ((output & output_t::CLASS) != 0) {
    result = result > threshold ? 1.0f : 0.0f;
  }
  // sklearn outputs numpy array in 'C' order, with the number of classes being last dimension
  // that is also the default order, so we should use the same one
  if (complement_proba) {
    preds[i] = 1.0f - result;
    preds[i + 1] = result;
  } else
    preds[i] = result;
}

struct forest {
  void init_n_items(int device) {
    int max_shm_std = 48 * 1024;  // 48 KiB
    /// the most shared memory a kernel can request on the GPU in question
    int max_shm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shm, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    // TODO(canonizer): use >48KiB shared memory if available
    max_shm = std::min(max_shm, max_shm_std);

    // searching for the most items per block while respecting the shared
    // memory limits creates a full linear programming problem.
    // solving it in a single equation looks less tractable than this
    for (bool predict_proba : {false, true}) {
      shmem_size_params& ssp_ = predict_proba ? proba_ssp_ : class_ssp_;
      ssp_.predict_proba = predict_proba;
      shmem_size_params ssp = ssp_;
      for (bool cols_in_shmem : {false, true}) {
        ssp.cols_in_shmem = cols_in_shmem;
        for (ssp.n_items = 1;
             ssp.n_items <= (algo_ == algo_t::BATCH_TREE_REORG ? 4 : 1);
             ++ssp.n_items) {
          ssp.compute_smem_footprint();
          if (ssp.shm_sz < max_shm) ssp_ = ssp;
        }
      }
      ASSERT(max_shm >= ssp_.shm_sz,
             "FIL out of shared memory. Perhaps the maximum number of \n"
             "supported classes is exceeded? 5'000 would still be safe.");
    }
  }

  void init_fixed_block_count(int device, int blocks_per_sm) {
    int max_threads_per_sm, sm_count;
    CUDA_CHECK(cudaDeviceGetAttribute(
      &max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    int max_blocks_per_sm = max_threads_per_sm / FIL_TPB;
    ASSERT(blocks_per_sm <= max_blocks_per_sm,
           "on this GPU, FIL blocks_per_sm cannot exceed %d",
           max_blocks_per_sm);
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                      device));
    fixed_block_count_ = blocks_per_sm * sm_count;
  }

  void init_common(const raft::handle_t& h, const forest_params_t* params) {
    depth_ = params->depth;
    num_trees_ = params->num_trees;
    algo_ = params->algo;
    output_ = params->output;
    threshold_ = params->threshold;
    global_bias_ = params->global_bias;
    proba_ssp_.leaf_algo = params->leaf_algo;
    proba_ssp_.num_cols = params->num_cols;
    proba_ssp_.num_classes = params->num_classes;
    class_ssp_ = proba_ssp_;

    int device = h.get_device();
    init_n_items(device);  // n_items takes priority over blocks_per_sm
    init_fixed_block_count(device, params->blocks_per_sm);
  }

  virtual void infer(predict_params params, cudaStream_t stream) = 0;

  void predict(const raft::handle_t& h, float* preds, const float* data,
               size_t num_rows, bool predict_proba) {
    // Initialize prediction parameters.
    predict_params params(predict_proba ? proba_ssp_ : class_ssp_);
    params.algo = algo_;
    params.preds = preds;
    params.data = data;
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
    RAW (no values set): output the label of the class with highest probability, else output label 0.
    SOFTMAX is set: error
    All other flags (AVG, SIGMOID, CLASS) are ignored
    
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
    */
    output_t ot = output_;
    // Treelite applies bias before softmax, but we do after.
    // Simulating treelite order, which cancels out bias.
    // If non-proba prediction used, it still will not matter
    // for the same reason softmax will not.
    float global_bias = (ot & output_t::SOFTMAX) != 0 ? 0.0f : global_bias_;
    bool complement_proba = false, do_transform;

    if (predict_proba) {
      // no threshold on probabilities
      ot = output_t(ot & ~output_t::CLASS);

      switch (params.leaf_algo) {
        case leaf_algo_t::FLOAT_UNARY_BINARY:
          params.num_outputs = 2;
          complement_proba = true;
          do_transform = true;
          break;
        case leaf_algo_t::GROVE_PER_CLASS:
          // for GROVE_PER_CLASS, averaging happens in infer_k
          ot = output_t(ot & ~output_t::AVG);
          params.num_outputs = params.num_classes;
          do_transform = ot != output_t::RAW && ot != output_t::SOFTMAX ||
                         global_bias != 0.0f;
          break;
        case leaf_algo_t::CATEGORICAL_LEAF:
          params.num_outputs = params.num_classes;
          do_transform = ot != output_t::RAW || global_bias_ != 0.0f;
          break;
        default:
          ASSERT(false, "internal error: invalid leaf_algo_");
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
      size_t num_values_to_transform =
        (size_t)num_rows * (size_t)params.num_outputs;
      transform_k<<<raft::ceildiv(num_values_to_transform, (size_t)FIL_TPB),
                    FIL_TPB, 0, stream>>>(
        preds, num_values_to_transform, ot,
        num_trees_ > 0 ? (1.0f / num_trees_) : 1.0f, threshold_, global_bias,
        complement_proba);
      CUDA_CHECK(cudaPeekAtLastError());
    }
  }

  virtual void free(const raft::handle_t& h) = 0;
  virtual ~forest() {}

  int num_trees_ = 0;
  int depth_ = 0;
  algo_t algo_ = algo_t::NAIVE;
  output_t output_ = output_t::RAW;
  float threshold_ = 0.5;
  float global_bias_ = 0;
  shmem_size_params class_ssp_, proba_ssp_;
  int fixed_block_count_ = 0;
};

struct dense_forest : forest {
  void transform_trees(const dense_node* nodes) {
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

  void init(const raft::handle_t& h, const dense_node* nodes,
            const forest_params_t* params) {
    init_common(h, params);
    if (algo_ == algo_t::NAIVE) algo_ = algo_t::BATCH_TREE_REORG;

    int num_nodes = forest_num_nodes(num_trees_, depth_);
    nodes_ = (dense_node*)h.get_device_allocator()->allocate(
      sizeof(dense_node) * num_nodes, h.get_stream());
    h_nodes_.resize(num_nodes);
    if (algo_ == algo_t::NAIVE) {
      std::copy(nodes, nodes + num_nodes, h_nodes_.begin());
    } else {
      transform_trees(nodes);
    }
    CUDA_CHECK(cudaMemcpyAsync(nodes_, h_nodes_.data(),
                               num_nodes * sizeof(dense_node),
                               cudaMemcpyHostToDevice, h.get_stream()));
    // copy must be finished before freeing the host data
    CUDA_CHECK(cudaStreamSynchronize(h.get_stream()));
    h_nodes_.clear();
    h_nodes_.shrink_to_fit();
  }

  virtual void infer(predict_params params, cudaStream_t stream) override {
    dense_storage forest(nodes_, num_trees_,
                         algo_ == algo_t::NAIVE ? tree_num_nodes(depth_) : 1,
                         algo_ == algo_t::NAIVE ? 1 : num_trees_);
    fil::infer(forest, params, stream);
  }

  virtual void free(const raft::handle_t& h) override {
    int num_nodes = forest_num_nodes(num_trees_, depth_);
    h.get_device_allocator()->deallocate(nodes_, sizeof(dense_node) * num_nodes,
                                         h.get_stream());
  }

  dense_node* nodes_ = nullptr;
  thrust::host_vector<dense_node> h_nodes_;
};

template <typename node_t>
struct sparse_forest : forest {
  void init(const raft::handle_t& h, const int* trees, const node_t* nodes,
            const forest_params_t* params) {
    init_common(h, params);
    if (algo_ == algo_t::ALGO_AUTO) algo_ = algo_t::NAIVE;
    depth_ = 0;  // a placeholder value
    num_nodes_ = params->num_nodes;

    // trees
    trees_ = (int*)h.get_device_allocator()->allocate(sizeof(int) * num_trees_,
                                                      h.get_stream());
    CUDA_CHECK(cudaMemcpyAsync(trees_, trees, sizeof(int) * num_trees_,
                               cudaMemcpyHostToDevice, h.get_stream()));

    // nodes
    nodes_ = (node_t*)h.get_device_allocator()->allocate(
      sizeof(node_t) * num_nodes_, h.get_stream());
    CUDA_CHECK(cudaMemcpyAsync(nodes_, nodes, sizeof(node_t) * num_nodes_,
                               cudaMemcpyHostToDevice, h.get_stream()));
  }

  virtual void infer(predict_params params, cudaStream_t stream) override {
    sparse_storage<node_t> forest(trees_, nodes_, num_trees_);
    fil::infer(forest, params, stream);
  }

  void free(const raft::handle_t& h) override {
    h.get_device_allocator()->deallocate(trees_, sizeof(int) * num_trees_,
                                         h.get_stream());
    h.get_device_allocator()->deallocate(nodes_, sizeof(node_t) * num_nodes_,
                                         h.get_stream());
  }

  int num_nodes_ = 0;
  int* trees_ = nullptr;
  node_t* nodes_ = nullptr;
};

void check_params(const forest_params_t* params, bool dense) {
  if (dense) {
    ASSERT(params->depth >= 0, "depth must be non-negative for dense forests");
  } else {
    ASSERT(params->num_nodes >= 0,
           "num_nodes must be non-negative for sparse forests");
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
    case algo_t::BATCH_TREE_REORG:
      break;
    default:
      ASSERT(false,
             "algo should be ALGO_AUTO, NAIVE, TREE_REORG or BATCH_TREE_REORG");
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
      ASSERT(params->num_classes > 2,
             "num_classes > 2 is required for leaf_algo == GROVE_PER_CLASS");
      ASSERT(params->num_trees % params->num_classes == 0,
             "num_classes must divide num_trees evenly for GROVE_PER_CLASS");
      break;
    case leaf_algo_t::CATEGORICAL_LEAF:
      ASSERT(params->num_classes >= 2,
             "num_classes >= 2 is required for "
             "leaf_algo == CATEGORICAL_LEAF");
      ASSERT((params->output & output_t::SOFTMAX) == 0,
             "softmax not supported for leaf_algo == CATEGORICAL_LEAF");
      break;
    default:
      ASSERT(false,
             "leaf_algo must be FLOAT_UNARY_BINARY, CATEGORICAL_LEAF"
             " or GROVE_PER_CLASS");
  }
  if ((params->output & ~output_t::ALL_SET) != 0) {
    ASSERT(
      false,
      "output should be a combination of RAW, AVG, SIGMOID, CLASS and SOFTMAX");
  }
  ASSERT(~params->output & (output_t::SIGMOID | output_t::SOFTMAX),
         "combining softmax and sigmoid is not supported");
  ASSERT(params->blocks_per_sm >= 0, "blocks_per_sm must be nonnegative");
}

template <typename T, typename L>
int tree_root(const tl::Tree<T, L>& tree) {
  return 0;  // Treelite format assumes that the root is 0
}

template <typename T, typename L>
inline int max_depth(const tl::Tree<T, L>& tree) {
  // trees of this depth aren't used, so it most likely means bad input data,
  // e.g. cycles in the forest
  const int DEPTH_LIMIT = 500;
  int root_index = tree_root(tree);
  typedef std::pair<int, int> pair_t;
  std::stack<pair_t> stack;
  stack.push(pair_t(root_index, 0));
  int max_depth = 0;
  while (!stack.empty()) {
    const pair_t& pair = stack.top();
    int node_id = pair.first;
    int depth = pair.second;
    stack.pop();
    while (!tree.IsLeaf(node_id)) {
      stack.push(pair_t(tree.LeftChild(node_id), depth + 1));
      node_id = tree.RightChild(node_id);
      depth++;
      ASSERT(depth < DEPTH_LIMIT,
             "depth limit reached, might be a cycle in the tree");
    }
    // only need to update depth for leaves
    max_depth = std::max(max_depth, depth);
  }
  return max_depth;
}

template <typename T, typename L>
int max_depth(const tl::ModelImpl<T, L>& model) {
  int depth = 0;
  const auto& trees = model.trees;
#pragma omp parallel for reduction(max : depth)
  for (size_t i = 0; i < trees.size(); ++i) {
    const auto& tree = trees[i];
    depth = std::max(depth, max_depth(tree));
  }
  return depth;
}

inline void adjust_threshold(float* pthreshold, int* tl_left, int* tl_right,
                             bool* default_left, tl::Operator comparison_op) {
  // in treelite (take left node if val [op] threshold),
  // the meaning of the condition is reversed compared to FIL;
  // thus, "<" in treelite corresonds to comparison ">=" used by FIL
  // https://github.com/dmlc/treelite/blob/master/include/treelite/tree.h#L243
  switch (comparison_op) {
    case tl::Operator::kLT:
      break;
    case tl::Operator::kLE:
      // x <= y is equivalent to x < y', where y' is the next representable float
      *pthreshold =
        std::nextafterf(*pthreshold, std::numeric_limits<float>::infinity());
      break;
    case tl::Operator::kGT:
      // x > y is equivalent to x >= y', where y' is the next representable float
      // left and right still need to be swapped
      *pthreshold =
        std::nextafterf(*pthreshold, std::numeric_limits<float>::infinity());
    case tl::Operator::kGE:
      // swap left and right
      std::swap(*tl_left, *tl_right);
      *default_left = !*default_left;
      break;
    default:
      ASSERT(false, "only <, >, <= and >= comparisons are supported");
  }
}

/** if the vector consists of zeros and a single one, return the position
for the one (assumed class label). Else, asserts false.
If the vector contains a NAN, asserts false */
template <typename L>
int find_class_label_from_one_hot(L* vector, int len) {
  bool found_label = false;
  int out;
  for (int i = 0; i < len; ++i) {
    if (vector[i] == static_cast<L>(1.0)) {
      ASSERT(!found_label, "label vector contains multiple 1.0f");
      out = i;
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
void tl2fil_leaf_payload(fil_node_t* fil_node, const tl::Tree<T, L>& tl_tree,
                         int tl_node_id, const forest_params_t& forest_params) {
  auto vec = tl_tree.LeafVector(tl_node_id);
  switch (forest_params.leaf_algo) {
    case leaf_algo_t::CATEGORICAL_LEAF:
      ASSERT(vec.size() == forest_params.num_classes,
             "inconsistent number of classes in treelite leaves");
      fil_node->val.idx = find_class_label_from_one_hot(&vec[0], vec.size());
      break;
    case leaf_algo_t::FLOAT_UNARY_BINARY:
    case leaf_algo_t::GROVE_PER_CLASS:
      fil_node->val.f = static_cast<float>(tl_tree.LeafValue(tl_node_id));
      ASSERT(!tl_tree.HasLeafVector(tl_node_id),
             "some but not all treelite leaves have leaf_vector()");
      break;
    default:
      ASSERT(false, "internal error: invalid leaf_algo");
  };
}

template <typename T, typename L>
void node2fil_dense(std::vector<dense_node>* pnodes, int root, int cur,
                    const tl::Tree<T, L>& tree, int node_id,
                    const forest_params_t& forest_params) {
  if (tree.IsLeaf(node_id)) {
    (*pnodes)[root + cur] = dense_node(val_t{.f = NAN}, NAN, 0, false, true);
    tl2fil_leaf_payload(&(*pnodes)[root + cur], tree, node_id, forest_params);
    return;
  }

  // inner node
  ASSERT(tree.SplitType(node_id) == tl::SplitFeatureType::kNumerical,
         "only numerical split nodes are supported");
  int tl_left = tree.LeftChild(node_id), tl_right = tree.RightChild(node_id);
  bool default_left = tree.DefaultLeft(node_id);
  float threshold = static_cast<float>(tree.Threshold(node_id));
  adjust_threshold(&threshold, &tl_left, &tl_right, &default_left,
                   tree.ComparisonOp(node_id));
  (*pnodes)[root + cur] = dense_node(
    val_t{.f = 0}, threshold, tree.SplitIndex(node_id), default_left, false);
  int left = 2 * cur + 1;
  node2fil_dense(pnodes, root, left, tree, tl_left, forest_params);
  node2fil_dense(pnodes, root, left + 1, tree, tl_right, forest_params);
}

template <typename T, typename L>
void tree2fil_dense(std::vector<dense_node>* pnodes, int root,
                    const tl::Tree<T, L>& tree,
                    const forest_params_t& forest_params) {
  node2fil_dense(pnodes, root, 0, tree, tree_root(tree), forest_params);
}

template <typename fil_node_t, typename T, typename L>
int tree2fil_sparse(std::vector<fil_node_t>& nodes, int root,
                    const tl::Tree<T, L>& tree,
                    const forest_params_t& forest_params) {
  typedef std::pair<int, int> pair_t;
  std::stack<pair_t> stack;
  int built_index = root + 1;
  stack.push(pair_t(tree_root(tree), 0));
  while (!stack.empty()) {
    const pair_t& top = stack.top();
    int node_id = top.first;
    int cur = top.second;
    stack.pop();

    while (!tree.IsLeaf(node_id)) {
      // inner node
      ASSERT(tree.SplitType(node_id) == tl::SplitFeatureType::kNumerical,
             "only numerical split nodes are supported");
      // tl_left and tl_right are indices of the children in the treelite tree
      // (stored  as an array of nodes)
      int tl_left = tree.LeftChild(node_id),
          tl_right = tree.RightChild(node_id);
      bool default_left = tree.DefaultLeft(node_id);
      float threshold = static_cast<float>(tree.Threshold(node_id));
      adjust_threshold(&threshold, &tl_left, &tl_right, &default_left,
                       tree.ComparisonOp(node_id));

      // reserve space for child nodes
      // left is the offset of the left child node relative to the tree root
      // in the array of all nodes of the FIL sparse forest
      int left = built_index - root;
      built_index += 2;
      nodes[root + cur] =
        fil_node_t(val_t{.f = 0}, threshold, tree.SplitIndex(node_id),
                   default_left, false, left);

      // push child nodes into the stack
      stack.push(pair_t(tl_right, left + 1));
      //stack.push(pair_t(tl_left, left));
      node_id = tl_left;
      cur = left;
    }

    // leaf node
    nodes[root + cur] = fil_node_t(val_t{.f = NAN}, NAN, 0, false, true, 0);
    tl2fil_leaf_payload(&nodes[root + cur], tree, node_id, forest_params);
  }

  return root;
}

template <typename T, typename L>
size_t tl_leaf_vector_size(const tl::ModelImpl<T, L>& model) {
  const tl::Tree<T, L>& tree = model.trees[0];
  int node_key;
  for (node_key = tree_root(tree); !tree.IsLeaf(node_key);
       node_key = tree.RightChild(node_key))
    ;
  if (tree.HasLeafVector(node_key)) return tree.LeafVector(node_key).size();
  return 0;
}

// tl2fil_common is the part of conversion from a treelite model
// common for dense and sparse forests
template <typename T, typename L>
void tl2fil_common(forest_params_t* params, const tl::ModelImpl<T, L>& model,
                   const treelite_params_t* tl_params) {
  // fill in forest-indendent params
  params->algo = tl_params->algo;
  params->threshold = tl_params->threshold;

  // fill in forest-dependent params
  params->depth = max_depth(model);  // also checks for cycles

  const tl::ModelParam& param = model.param;

  // assuming either all leaves use the .leaf_vector() or all leaves use .leaf_value()
  size_t leaf_vec_size = tl_leaf_vector_size(model);
  std::string pred_transform(param.pred_transform);
  if (leaf_vec_size > 0) {
    ASSERT(leaf_vec_size == model.task_param.num_class,
           "treelite model inconsistent");
    params->num_classes = leaf_vec_size;
    params->leaf_algo = leaf_algo_t::CATEGORICAL_LEAF;

    ASSERT(tl_params->output_class,
           "output_class==true is required for multi-class models");

    ASSERT(
      pred_transform == "max_index" || pred_transform == "identity_multiclass",
      "only max_index and identity_multiclass values of pred_transform "
      "are supported for multi-class models");

  } else {
    if (model.task_param.num_class > 1) {
      params->num_classes = static_cast<int>(model.task_param.num_class);
      ASSERT(tl_params->output_class,
             "output_class==true is required for multi-class models");
      ASSERT(pred_transform == "identity_multiclass" ||
               pred_transform == "max_index" || pred_transform == "softmax" ||
               pred_transform == "multiclass_ova",
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
  params->output = output_t::RAW;
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
  if (model.average_tree_output) {
    params->output = output_t(params->output | output_t::AVG);
  }
  if (pred_transform == "sigmoid" || pred_transform == "multiclass_ova") {
    params->output = output_t(params->output | output_t::SIGMOID);
  }
  if (pred_transform == "softmax")
    params->output = output_t(params->output | output_t::SOFTMAX);
  params->num_trees = model.trees.size();
  params->blocks_per_sm = tl_params->blocks_per_sm;
}

// uses treelite model with additional tl_params to initialize FIL params
// and dense nodes (stored in *pnodes)
template <typename T, typename L>
void tl2fil_dense(std::vector<dense_node>* pnodes, forest_params_t* params,
                  const tl::ModelImpl<T, L>& model,
                  const treelite_params_t* tl_params) {
  tl2fil_common(params, model, tl_params);

  // convert the nodes
  int num_nodes = forest_num_nodes(params->num_trees, params->depth);
  pnodes->resize(num_nodes, dense_node());
  for (int i = 0; i < model.trees.size(); ++i) {
    tree2fil_dense(pnodes, i * tree_num_nodes(params->depth), model.trees[i],
                   *params);
  }
}

template <typename fil_node_t>
struct tl2fil_sparse_check_t {
  template <typename T, typename L>
  static void check(const tl::ModelImpl<T, L>& model) {
    ASSERT(false,
           "internal error: "
           "only a specialization of this template should be used");
  }
};

template <>
struct tl2fil_sparse_check_t<sparse_node16> {
  // no extra check for 16-byte sparse nodes
  template <typename T, typename L>
  static void check(const tl::ModelImpl<T, L>& model) {}
};

template <>
struct tl2fil_sparse_check_t<sparse_node8> {
  static const int MAX_FEATURES = 1 << sparse_node8::FID_NUM_BITS;
  static const int MAX_TREE_NODES = (1 << sparse_node8::LEFT_NUM_BITS) - 1;
  template <typename T, typename L>
  static void check(const tl::ModelImpl<T, L>& model) {
    // check the number of features
    int num_features = model.num_feature;
    ASSERT(num_features <= MAX_FEATURES,
           "model has %d features, "
           "but only %d supported for 8-byte sparse nodes",
           num_features, MAX_FEATURES);

    // check the number of tree nodes
    const std::vector<tl::Tree<T, L>>& trees = model.trees;
    for (int i = 0; i < trees.size(); ++i) {
      int num_nodes = trees[i].num_nodes;
      ASSERT(num_nodes <= MAX_TREE_NODES,
             "tree %d has %d nodes, "
             "but only %d supported for 8-byte sparse nodes",
             i, num_nodes, MAX_TREE_NODES);
    }
  }
};

// uses treelite model with additional tl_params to initialize FIL params,
// trees (stored in *ptrees) and sparse nodes (stored in *pnodes)
template <typename fil_node_t, typename T, typename L>
void tl2fil_sparse(std::vector<int>* ptrees, std::vector<fil_node_t>* pnodes,
                   forest_params_t* params, const tl::ModelImpl<T, L>& model,
                   const treelite_params_t* tl_params) {
  tl2fil_common(params, model, tl_params);
  tl2fil_sparse_check_t<fil_node_t>::check(model);

  size_t num_trees = model.trees.size();

  ptrees->reserve(num_trees);
  ptrees->push_back(0);
  for (size_t i = 0; i < num_trees - 1; ++i) {
    ptrees->push_back(model.trees[i].num_nodes + ptrees->back());
  }
  size_t total_nodes = ptrees->back() + model.trees.back().num_nodes;

  pnodes->resize(total_nodes);

  // convert the nodes
#pragma omp parallel for
  for (int i = 0; i < num_trees; ++i) {
    tree2fil_sparse(*pnodes, (*ptrees)[i], model.trees[i], *params);
  }

  params->num_nodes = pnodes->size();
}

void init_dense(const raft::handle_t& h, forest_t* pf, const dense_node* nodes,
                const forest_params_t* params) {
  check_params(params, true);
  dense_forest* f = new dense_forest;
  f->init(h, nodes, params);
  *pf = f;
}

template <typename fil_node_t>
void init_sparse(const raft::handle_t& h, forest_t* pf, const int* trees,
                 const fil_node_t* nodes, const forest_params_t* params) {
  check_params(params, false);
  sparse_forest<fil_node_t>* f = new sparse_forest<fil_node_t>;
  f->init(h, trees, nodes, params);
  *pf = f;
}

// explicit instantiations for init_sparse()
template void init_sparse<sparse_node16>(const raft::handle_t& h, forest_t* pf,
                                         const int* trees,
                                         const sparse_node16* nodes,
                                         const forest_params_t* params);

template void init_sparse<sparse_node8>(const raft::handle_t& h, forest_t* pf,
                                        const int* trees,
                                        const sparse_node8* nodes,
                                        const forest_params_t* params);

template <typename T, typename L>
void from_treelite(const raft::handle_t& handle, forest_t* pforest,
                   const tl::ModelImpl<T, L>& model,
                   const treelite_params_t* tl_params) {
  // Invariants on threshold and leaf types
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "Model must contain float32 or float64 thresholds for splits");
  ASSERT((std::is_same<L, float>::value || std::is_same<L, double>::value),
         "Models with integer leaf output are not yet supported");
  // Display appropriate warnings when float64 values are being casted into
  // float32, as FIL only supports inferencing with float32 for the time being
  if (std::is_same<T, double>::value || std::is_same<L, double>::value) {
    CUML_LOG_WARN(
      "Casting all thresholds and leaf values to float32, as FIL currently "
      "doesn't support inferencing models with float64 values. "
      "This may lead to predictions with reduced accuracy.");
  }

  storage_type_t storage_type = tl_params->storage_type;
  // build dense trees by default
  if (storage_type == storage_type_t::AUTO) {
    if (tl_params->algo == algo_t::ALGO_AUTO ||
        tl_params->algo == algo_t::NAIVE) {
      int depth = max_depth(model);
      // max 2**25 dense nodes, 256 MiB dense model size
      const int LOG2_MAX_DENSE_NODES = 25;
      int log2_num_dense_nodes =
        depth + 1 + int(ceil(std::log2(model.trees.size())));
      storage_type = log2_num_dense_nodes > LOG2_MAX_DENSE_NODES
                       ? storage_type_t::SPARSE
                       : storage_type_t::DENSE;
    } else {
      // only dense storage is supported for other algorithms
      storage_type = storage_type_t::DENSE;
    }
  }

  forest_params_t params;
  switch (storage_type) {
    case storage_type_t::DENSE: {
      std::vector<dense_node> nodes;
      tl2fil_dense(&nodes, &params, model, tl_params);
      init_dense(handle, pforest, nodes.data(), &params);
      // sync is necessary as nodes is used in init_dense(),
      // but destructed at the end of this function
      CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
      break;
    }
    case storage_type_t::SPARSE: {
      std::vector<int> trees;
      std::vector<sparse_node16> nodes;
      tl2fil_sparse(&trees, &nodes, &params, model, tl_params);
      init_sparse(handle, pforest, trees.data(), nodes.data(), &params);
      CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
      break;
    }
    case storage_type_t::SPARSE8: {
      std::vector<int> trees;
      std::vector<sparse_node8> nodes;
      tl2fil_sparse(&trees, &nodes, &params, model, tl_params);
      init_sparse(handle, pforest, trees.data(), nodes.data(), &params);
      CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));
      break;
    }
    default:
      ASSERT(false, "tl_params->sparse must be one of AUTO, DENSE or SPARSE");
  }
}

void from_treelite(const raft::handle_t& handle, forest_t* pforest,
                   ModelHandle model, const treelite_params_t* tl_params) {
  const tl::Model& model_ref = *(tl::Model*)model;
  model_ref.Dispatch([&handle, pforest, tl_params](const auto& model_inner) {
    // model_inner is of the concrete type tl::ModelImpl<T, L>
    from_treelite(handle, pforest, model_inner, tl_params);
  });
}

void free(const raft::handle_t& h, forest_t f) {
  f->free(h);
  delete f;
}

void predict(const raft::handle_t& h, forest_t f, float* preds,
             const float* data, size_t num_rows, bool predict_proba) {
  f->predict(h, preds, data, num_rows, predict_proba);
}

}  // namespace fil
}  // namespace ML
