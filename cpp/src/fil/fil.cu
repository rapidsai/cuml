/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <treelite/c_api.h>
#include <treelite/tree.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <cuml/fil/fil.h>
#include <cuml/common/cuml_allocator.hpp>
#include "common.cuh"

namespace ML {
namespace fil {

using namespace MLCommon;
namespace tl = treelite;

void dense_node_init(dense_node_t* n, float output, float thresh, int fid,
                     bool def_left, bool is_leaf) {
  dense_node dn(output, thresh, fid, def_left, is_leaf);
  n->bits = dn.bits;
  n->val = dn.val;
}

void dense_node_decode(const dense_node_t* n, float* output, float* thresh,
                       int* fid, bool* def_left, bool* is_leaf) {
  dense_node dn(*n);
  *output = dn.output();
  *thresh = dn.thresh();
  *fid = dn.fid();
  *def_left = dn.def_left();
  *is_leaf = dn.is_leaf();
}

void sparse_node_init(sparse_node_t* node, float output, float thresh, int fid,
                      bool def_left, bool is_leaf, int left_index) {
  sparse_node n(output, thresh, fid, def_left, is_leaf, left_index);
  node->bits = n.bits;
  node->val = n.val;
  node->left_idx = n.left_idx;
}

/** sparse_node_decode extracts individual members from node */
void sparse_node_decode(const sparse_node_t* node, float* output, float* thresh,
                        int* fid, bool* def_left, bool* is_leaf,
                        int* left_index) {
  sparse_node n(*node);
  *output = n.output();
  *thresh = n.thresh();
  *fid = n.fid();
  *def_left = n.def_left();
  *is_leaf = n.is_leaf();
  *left_index = n.left_index();
}

__host__ __device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/** performs additional transformations on the array of forest predictions
    (preds) of size n; the transformations are defined by output, and include
    averaging (multiplying by inv_num_trees), adding global_bias (always done),
    sigmoid and applying threshold. in case of predict_proba, skips threshold
    and fills in the converse probability */
__global__ void transform_k(float* preds, size_t n, output_t output,
                            float inv_num_trees, float threshold,
                            float global_bias, bool predict_proba) {
  size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x;
  if (i >= n) return;

  float result = preds[predict_proba ? i * 2 : i];
  if ((output & output_t::AVG) != 0) result *= inv_num_trees;
  result += global_bias;
  if ((output & output_t::SIGMOID) != 0) result = sigmoid(result);
  if ((output & output_t::THRESHOLD) && !predict_proba) {
    result = result > threshold ? 1.0f : 0.0f;
  }
  // sklearn outputs numpy array in 'C' order, with the number of classes being last dimension
  // that is also the default order, so we should use the same one
  if (predict_proba) {
    preds[i * 2] = 1.f - result;
    preds[i * 2 + 1] = result;
  } else
    preds[i] = result;
}

struct forest {
  void init_max_shm() {
    int max_shm_std = 48 * 1024;  // 48 KiB
    int device = 0;
    // TODO(canonizer): use cumlHandle for this
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shm_, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    // TODO(canonizer): use >48KiB shared memory if available
    max_shm_ = std::min(max_shm_, max_shm_std);
  }

  void init_common(const forest_params_t* params) {
    depth_ = params->depth;
    num_trees_ = params->num_trees;
    num_cols_ = params->num_cols;
    algo_ = params->algo;
    output_ = params->output;
    threshold_ = params->threshold;
    global_bias_ = params->global_bias;
    init_max_shm();
  }

  virtual void infer(predict_params params, cudaStream_t stream) = 0;

  void predict(const cumlHandle& h, float* preds, const float* data,
               size_t num_rows, bool predict_proba) {
    // Initialize prediction parameters.
    predict_params params;
    params.num_cols = num_cols_;
    params.algo = algo_;
    params.preds = preds;
    params.data = data;
    params.num_rows = num_rows;
    params.max_shm = max_shm_;
    params.num_output_classes = predict_proba ? 2 : 1;

    // Predict using the forest.
    cudaStream_t stream = h.getStream();
    infer(params, stream);

    // Transform the output if necessary.
    if (output_ != output_t::RAW || global_bias_ != 0.0f || predict_proba) {
      transform_k<<<ceildiv(int(num_rows), FIL_TPB), FIL_TPB, 0, stream>>>(
        preds, num_rows, output_, num_trees_ > 0 ? (1.0f / num_trees_) : 1.0f,
        threshold_, global_bias_, predict_proba);
      CUDA_CHECK(cudaPeekAtLastError());
    }
  }

  virtual void free(const cumlHandle& h) = 0;
  virtual ~forest() {}

  int num_trees_ = 0;
  int depth_ = 0;
  int num_cols_ = 0;
  algo_t algo_ = algo_t::NAIVE;
  int max_shm_ = 0;
  output_t output_ = output_t::RAW;
  float threshold_ = 0.5;
  float global_bias_ = 0;
};

struct dense_forest : forest {
  void transform_trees(const dense_node_t* nodes) {
    // populate node information
    for (int i = 0, gid = 0; i < num_trees_; ++i) {
      for (int j = 0, nid = 0; j <= depth_; ++j) {
        for (int k = 0; k < 1 << j; ++k, ++nid, ++gid) {
          h_nodes_[nid * num_trees_ + i] = dense_node(nodes[gid]);
        }
      }
    }
  }

  void init(const cumlHandle& h, const dense_node_t* nodes,
            const forest_params_t* params) {
    init_common(params);
    if (algo_ == algo_t::NAIVE) algo_ = algo_t::BATCH_TREE_REORG;

    int num_nodes = forest_num_nodes(num_trees_, depth_);
    nodes_ = (dense_node*)h.getDeviceAllocator()->allocate(
      sizeof(dense_node) * num_nodes, h.getStream());
    h_nodes_.resize(num_nodes);
    if (algo_ == algo_t::NAIVE) {
      std::copy(nodes, nodes + num_nodes, h_nodes_.begin());
    } else {
      transform_trees(nodes);
    }
    CUDA_CHECK(cudaMemcpyAsync(nodes_, h_nodes_.data(),
                               num_nodes * sizeof(dense_node),
                               cudaMemcpyHostToDevice, h.getStream()));
    // copy must be finished before freeing the host data
    CUDA_CHECK(cudaStreamSynchronize(h.getStream()));
    h_nodes_.clear();
    h_nodes_.shrink_to_fit();
  }

  virtual void infer(predict_params params, cudaStream_t stream) override {
    dense_storage forest(nodes_, num_trees_,
                         algo_ == algo_t::NAIVE ? tree_num_nodes(depth_) : 1,
                         algo_ == algo_t::NAIVE ? 1 : num_trees_);
    fil::infer(forest, params, stream);
  }

  virtual void free(const cumlHandle& h) override {
    int num_nodes = forest_num_nodes(num_trees_, depth_);
    h.getDeviceAllocator()->deallocate(nodes_, sizeof(dense_node) * num_nodes,
                                       h.getStream());
  }

  dense_node* nodes_ = nullptr;
  thrust::host_vector<dense_node> h_nodes_;
};

struct sparse_forest : forest {
  void init(const cumlHandle& h, const int* trees, const sparse_node_t* nodes,
            const forest_params_t* params) {
    init_common(params);
    if (algo_ == algo_t::ALGO_AUTO) algo_ = algo_t::NAIVE;
    depth_ = 0;  // a placeholder value
    num_nodes_ = params->num_nodes;

    // trees
    trees_ = (int*)h.getDeviceAllocator()->allocate(sizeof(int) * num_trees_,
                                                    h.getStream());
    CUDA_CHECK(cudaMemcpyAsync(trees_, trees, sizeof(int) * num_trees_,
                               cudaMemcpyHostToDevice, h.getStream()));

    // nodes
    nodes_ = (sparse_node*)h.getDeviceAllocator()->allocate(
      sizeof(sparse_node) * num_nodes_, h.getStream());
    CUDA_CHECK(cudaMemcpyAsync(nodes_, nodes, sizeof(sparse_node) * num_nodes_,
                               cudaMemcpyHostToDevice, h.getStream()));
  }

  virtual void infer(predict_params params, cudaStream_t stream) override {
    sparse_storage forest(trees_, nodes_, num_trees_);
    fil::infer(forest, params, stream);
  }

  void free(const cumlHandle& h) override {
    h.getDeviceAllocator()->deallocate(trees_, sizeof(int) * num_trees_,
                                       h.getStream());
    h.getDeviceAllocator()->deallocate(nodes_, sizeof(sparse_node) * num_nodes_,
                                       h.getStream());
  }

  int num_nodes_ = 0;
  int* trees_ = nullptr;
  sparse_node* nodes_ = nullptr;
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
  // output_t::RAW == 0, and doesn't have a separate flag
  output_t all_set =
    output_t(output_t::AVG | output_t::SIGMOID | output_t::THRESHOLD);
  if ((params->output & ~all_set) != 0) {
    ASSERT(false,
           "output should be a combination of RAW, AVG, SIGMOID and THRESHOLD");
  }
}

// tl_node_at is a checked version of tree[i]
const tl::Tree::Node& tl_node_at(const tl::Tree& tree, size_t i) {
  ASSERT(i < tree.num_nodes, "node index out of range");
  return tree[i];
}

int tree_root(const tl::Tree& tree) {
  // find the root
  int root = -1;
  for (int i = 0; i < tree.num_nodes; ++i) {
    if (tl_node_at(tree, i).is_root()) {
      ASSERT(root == -1, "multi-root trees not supported");
      root = i;
    }
  }
  ASSERT(root != -1, "a tree must have a root");
  return root;
}

int max_depth_helper(const tl::Tree& tree, const tl::Tree::Node& node,
                     int limit) {
  if (node.is_leaf()) return 0;
  ASSERT(limit > 0,
         "recursion depth limit reached, might be a cycle in the tree");
  return 1 +
         std::max(
           max_depth_helper(tree, tl_node_at(tree, node.cleft()), limit - 1),
           max_depth_helper(tree, tl_node_at(tree, node.cright()), limit - 1));
}

int max_depth(const tl::Tree& tree) {
  // trees of this depth aren't used, so it most likely means bad input data,
  // e.g. cycles in the forest
  const int RECURSION_LIMIT = 500;
  return max_depth_helper(tree, tl_node_at(tree, tree_root(tree)),
                          RECURSION_LIMIT);
}

int max_depth(const tl::Model& model) {
  int depth = 0;
  for (const auto& tree : model.trees) depth = std::max(depth, max_depth(tree));
  return depth;
}

void adjust_threshold(float* pthreshold, int* tl_left, int* tl_right,
                      bool* default_left, const tl::Tree::Node& node) {
  // in treelite (take left node if val [op] threshold),
  // the meaning of the condition is reversed compared to FIL;
  // thus, "<" in treelite corresonds to comparison ">=" used by FIL
  // https://github.com/dmlc/treelite/blob/master/include/treelite/tree.h#L243
  switch (node.comparison_op()) {
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

void node2fil_dense(std::vector<dense_node_t>* pnodes, int root, int cur,
                    const tl::Tree& tree, const tl::Tree::Node& node) {
  if (node.is_leaf()) {
    dense_node_init(&(*pnodes)[root + cur], node.leaf_value(), 0, 0, false,
                    true);
    return;
  }

  // inner node
  ASSERT(node.split_type() == tl::SplitFeatureType::kNumerical,
         "only numerical split nodes are supported");
  int tl_left = node.cleft(), tl_right = node.cright();
  bool default_left = node.default_left();
  float threshold = node.threshold();
  adjust_threshold(&threshold, &tl_left, &tl_right, &default_left, node);
  dense_node_init(&(*pnodes)[root + cur], 0, threshold, node.split_index(),
                  default_left, false);
  int left = 2 * cur + 1;
  node2fil_dense(pnodes, root, left, tree, tl_node_at(tree, tl_left));
  node2fil_dense(pnodes, root, left + 1, tree, tl_node_at(tree, tl_right));
}

void node2fil_sparse(std::vector<sparse_node_t>* pnodes, int root, int cur,
                     const tl::Tree& tree, const tl::Tree::Node& node) {
  if (node.is_leaf()) {
    sparse_node_init(&(*pnodes)[root + cur], node.leaf_value(), 0, 0, false,
                     true, 0);
    return;
  }

  // inner node
  ASSERT(node.split_type() == tl::SplitFeatureType::kNumerical,
         "only numerical split nodes are supported");
  // tl_left and tl_right are indices of the children in the treelite tree
  // (stored  as an array of nodes)
  int tl_left = node.cleft(), tl_right = node.cright();
  bool default_left = node.default_left();
  float threshold = node.threshold();
  adjust_threshold(&threshold, &tl_left, &tl_right, &default_left, node);

  // reserve space for child nodes
  // left is the offset of the left child node relative to the tree root
  // in the array of all nodes of the FIL sparse forest
  int left = pnodes->size() - root;
  pnodes->push_back(sparse_node_t());
  pnodes->push_back(sparse_node_t());
  sparse_node_init(&(*pnodes)[root + cur], 0, threshold, node.split_index(),
                   default_left, false, left);

  // init child nodes
  node2fil_sparse(pnodes, root, left, tree, tl_node_at(tree, tl_left));
  node2fil_sparse(pnodes, root, left + 1, tree, tl_node_at(tree, tl_right));
}

void tree2fil_dense(std::vector<dense_node_t>* pnodes, int root,
                    const tl::Tree& tree) {
  node2fil_dense(pnodes, root, 0, tree, tl_node_at(tree, tree_root(tree)));
}

int tree2fil_sparse(std::vector<sparse_node_t>* pnodes, const tl::Tree& tree) {
  int root = pnodes->size();
  pnodes->push_back(sparse_node_t());
  node2fil_sparse(pnodes, root, 0, tree, tl_node_at(tree, tree_root(tree)));
  return root;
}

// tl2fil_common is the part of conversion from a treelite model
// common for dense and sparse forests
void tl2fil_common(forest_params_t* params, const tl::Model& model,
                   const treelite_params_t* tl_params) {
  // fill in forest-indendent params
  params->algo = tl_params->algo;
  params->threshold = tl_params->threshold;

  // fill in forest-dependent params
  params->num_cols = model.num_feature;
  ASSERT(model.num_output_group == 1,
         "multi-class classification not supported");
  const tl::ModelParam& param = model.param;
  ASSERT(param.sigmoid_alpha == 1.0f, "sigmoid_alpha not supported");
  params->global_bias = param.global_bias;
  params->output = output_t::RAW;
  if (tl_params->output_class) {
    params->output = output_t(params->output | output_t::THRESHOLD);
  }
  // "random forest" in treelite means tree output averaging
  if (model.random_forest_flag) {
    params->output = output_t(params->output | output_t::AVG);
  }
  if (param.pred_transform == "sigmoid") {
    params->output = output_t(params->output | output_t::SIGMOID);
  } else if (param.pred_transform != "identity") {
    ASSERT(false, "%s: unsupported treelite prediction transform",
           param.pred_transform.c_str());
  }
  params->num_trees = model.trees.size();
  params->depth = max_depth(model);
}

// uses treelite model with additional tl_params to initialize FIL params
// and dense nodes (stored in *pnodes)
void tl2fil_dense(std::vector<dense_node_t>* pnodes, forest_params_t* params,
                  const tl::Model& model, const treelite_params_t* tl_params) {
  tl2fil_common(params, model, tl_params);

  // convert the nodes
  int num_nodes = forest_num_nodes(params->num_trees, params->depth);
  pnodes->resize(num_nodes, dense_node_t{0, 0});
  for (int i = 0; i < model.trees.size(); ++i) {
    tree2fil_dense(pnodes, i * tree_num_nodes(params->depth), model.trees[i]);
  }
}

// uses treelite model with additional tl_params to initialize FIL params,
// trees (stored in *ptrees) and sparse nodes (stored in *pnodes)
void tl2fil_sparse(std::vector<int>* ptrees, std::vector<sparse_node_t>* pnodes,
                   forest_params_t* params, const tl::Model& model,
                   const treelite_params_t* tl_params) {
  tl2fil_common(params, model, tl_params);

  // convert the nodes
  for (int i = 0; i < model.trees.size(); ++i) {
    int root = tree2fil_sparse(pnodes, model.trees[i]);
    ptrees->push_back(root);
  }
  params->num_nodes = pnodes->size();
}

void init_dense(const cumlHandle& h, forest_t* pf, const dense_node_t* nodes,
                const forest_params_t* params) {
  check_params(params, true);
  dense_forest* f = new dense_forest;
  f->init(h, nodes, params);
  *pf = f;
}

void init_sparse(const cumlHandle& h, forest_t* pf, const int* trees,
                 const sparse_node_t* nodes, const forest_params_t* params) {
  check_params(params, false);
  sparse_forest* f = new sparse_forest;
  f->init(h, trees, nodes, params);
  *pf = f;
}

void from_treelite(const cumlHandle& handle, forest_t* pforest,
                   ModelHandle model, const treelite_params_t* tl_params) {
  storage_type_t storage_type = tl_params->storage_type;
  // build dense trees by default
  const tl::Model& model_ref = *(tl::Model*)model;
  if (storage_type == storage_type_t::AUTO) {
    if (tl_params->algo == algo_t::ALGO_AUTO ||
        tl_params->algo == algo_t::NAIVE) {
      int depth = max_depth(model_ref);
      // max 2**25 dense nodes, 256 MiB dense model size
      const int LOG2_MAX_DENSE_NODES = 25;
      int log2_num_dense_nodes =
        depth + 1 + int(ceil(std::log2(model_ref.trees.size())));
      storage_type = log2_num_dense_nodes > LOG2_MAX_DENSE_NODES
                       ? storage_type_t::SPARSE
                       : storage_type_t::DENSE;
    } else {
      // only dense storage is supported for other algorithms
      storage_type = storage_type_t::DENSE;
    }
  }

  switch (storage_type) {
    case storage_type_t::DENSE: {
      forest_params_t params;
      std::vector<dense_node_t> nodes;
      tl2fil_dense(&nodes, &params, model_ref, tl_params);
      init_dense(handle, pforest, nodes.data(), &params);
      // sync is necessary as nodes is used in init_dense(),
      // but destructed at the end of this function
      CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
      break;
    }
    case storage_type_t::SPARSE: {
      forest_params_t params;
      std::vector<int> trees;
      std::vector<sparse_node_t> nodes;
      tl2fil_sparse(&trees, &nodes, &params, model_ref, tl_params);
      init_sparse(handle, pforest, trees.data(), nodes.data(), &params);
      // sync is necessary as nodes is used in init_dense(),
      // but destructed at the end of this function
      CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
      break;
    }
    default:
      ASSERT(false, "tl_params->sparse must be one of AUTO, DENSE or SPARSE");
  }
}

void free(const cumlHandle& h, forest_t f) {
  f->free(h);
  delete f;
}

void predict(const cumlHandle& h, forest_t f, float* preds, const float* data,
             size_t num_rows, bool predict_proba) {
  f->predict(h, preds, data, num_rows, predict_proba);
}

}  // namespace fil
}  // namespace ML
