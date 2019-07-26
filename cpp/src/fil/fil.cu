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
#include <treelite/tree.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "common.cuh"
#include "fil.h"

namespace ML {
namespace fil {

using namespace MLCommon;
namespace tl = treelite;

void naive(const predict_params& ps, cudaStream_t stream);
void tree_reorg(const predict_params& ps, cudaStream_t stream);
void batch_tree_reorg(const predict_params& ps, cudaStream_t stream);

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

__host__ __device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void transform_k(float* preds, size_t n, bool output_class,
                            float threshold) {
  size_t i = threadIdx.x + size_t(blockIdx.x) * blockDim.x;
  if (i >= n) return;
  float out = preds[i];
  out = sigmoid(out);
  if (output_class) out = out > threshold ? 1.0f : 0.0f;
  preds[i] = out;
}

struct forest {
  forest()
    : depth_(0),
      ntrees_(0),
      cols_(0),
      algo_(algo_t::NAIVE),
      output_(output_t::RAW),
      threshold_(0.5) {}

  void transform_trees(const dense_node_t* nodes) {
    // populate node information
    for (int i = 0, gid = 0; i < ntrees_; ++i) {
      for (int j = 0, nid = 0; j <= depth_; ++j) {
        for (int k = 0; k < 1 << j; ++k, ++nid, ++gid) {
          h_nodes_[nid * ntrees_ + i] = dense_node(nodes[gid]);
        }
      }
    }
  }

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

  void init(const cumlHandle& h, const forest_params_t* params) {
    depth_ = params->depth;
    ntrees_ = params->ntrees;
    cols_ = params->cols;
    algo_ = params->algo;
    output_ = params->output;
    threshold_ = params->threshold;
    init_max_shm();

    int nnodes = forest_num_nodes(ntrees_, depth_);
    nodes_ = (dense_node*)h.getDeviceAllocator()->allocate(
      sizeof(dense_node) * nnodes, h.getStream());
    h_nodes_.resize(nnodes);
    if (algo_ == algo_t::NAIVE) {
      std::copy(params->nodes, params->nodes + nnodes, h_nodes_.begin());
    } else {
      transform_trees(params->nodes);
    }
    CUDA_CHECK(cudaMemcpy(nodes_, h_nodes_.data(), nnodes * sizeof(dense_node),
                          cudaMemcpyHostToDevice));
    h_nodes_.clear();
    h_nodes_.shrink_to_fit();
  }

  void predict(const cumlHandle& h, float* preds, const float* data,
               size_t rows) {
    // Initialize prediction parameters.
    predict_params ps;
    ps.nodes = nodes_;
    ps.ntrees = ntrees_;
    ps.depth = depth_;
    ps.cols = cols_;
    ps.preds = preds;
    ps.data = data;
    ps.rows = rows;
    ps.max_shm = max_shm_;

    // Predict using the forest.
    cudaStream_t stream = h.getStream();
    switch (algo_) {
      case algo_t::NAIVE:
        naive(ps, stream);
        break;
      case algo_t::TREE_REORG:
        tree_reorg(ps, stream);
        break;
      case algo_t::BATCH_TREE_REORG:
        batch_tree_reorg(ps, stream);
        break;
      default:
        ASSERT(false, "internal error: invalid algorithm");
    }

    // Transform the output if necessary (sigmoid + thresholding if necessary).
    if (output_ != output_t::RAW) {
      transform_k<<<ceildiv(int(rows), FIL_TPB), FIL_TPB, 0, stream>>>(
        preds, rows, output_ == output_t::CLASS, threshold_);
      CUDA_CHECK(cudaPeekAtLastError());
    }
  }

  void free(const cumlHandle& h) {
    int num_nodes = forest_num_nodes(ntrees_, depth_);
    h.getDeviceAllocator()->deallocate(nodes_, sizeof(dense_node) * num_nodes,
                                       h.getStream());
  }

  int ntrees_;
  int depth_;
  int cols_;
  algo_t algo_;
  int max_shm_;
  output_t output_;
  float threshold_;
  dense_node* nodes_ = nullptr;
  thrust::host_vector<dense_node> h_nodes_;
};

void check_params(const forest_params_t* params) {
  ASSERT(params->depth >= 0, "depth must be non-negative");
  ASSERT(params->ntrees >= 0, "ntrees must be non-negative");
  ASSERT(params->cols >= 0, "cols must be non-negative");
  switch (params->algo) {
    case algo_t::NAIVE:
    case algo_t::TREE_REORG:
    case algo_t::BATCH_TREE_REORG:
      break;
    default:
      ASSERT(false, "aglo should be NAIVE, TREE_REORG or BATCH_TREE_REORG");
  }
  switch (params->output) {
    case output_t::RAW:
    case output_t::PROB:
    case output_t::CLASS:
      break;
    default:
      ASSERT(false, "output should be RAW, PROB or CLASS");
  }
}

int tree_root(const tl::Tree& t) {
  // find the root
  int root = -1;
  for (int i = 0; i < t.num_nodes; ++i) {
    if (t[i].is_root()) {
      ASSERT(root == -1, "multi-root trees not supported");
      root = i;
    }
  }
  ASSERT(root != -1, "a tree must have a root");
  return root;
}

int max_depth(const tl::Tree& t, const tl::Tree::Node& n) {
  // TODO: add detection of infinite loops
  if (n.is_leaf()) return 0;
  // TODO: check for out-of-range
  return std::max(max_depth(t, t[n.cleft()]), max_depth(t, t[n.cright()])) + 1;
}

int max_depth(const tl::Tree& t) { return max_depth(t, t[tree_root(t)]); }

void node2fil(std::vector<dense_node_t>* pnodes, int root, int cur,
              const tl::Tree& t, const tl::Tree::Node& n) {
  std::vector<dense_node_t>& nodes = *pnodes;
  if (n.is_leaf()) {
    dense_node_init(&nodes[root + cur], n.leaf_value(), 0, 0, false, true);
    return;
  }

  // inner node
  ASSERT(n.split_type() == tl::SplitFeatureType::kNumerical,
         "only numerical split nodes are supported");
  int left = n.cleft(), right = n.cright();
  bool def_left = n.default_left();
  float threshold = n.threshold();
  // in treelite (take left node if val [op] threshold),
  // the meaning of the condition is reversed compared to FIL;
  // thus, "<" in treelite corresonds to comparison ">=" used by FIL
  // https://github.com/dmlc/treelite/blob/master/include/treelite/tree.h#L243
  switch (n.comparison_op()) {
    case tl::Operator::kLT:
      break;
    case tl::Operator::kLE:
      // x <= y is equivalent to x < y', where y' is the next representable float
      threshold =
        std::nextafterf(threshold, std::numeric_limits<float>::infinity());
      break;
    case tl::Operator::kGT:
      // x > y is equivalent to x >= y', where y' is the next representable float
      // left and right still need to be swapped
      threshold =
        std::nextafterf(threshold, std::numeric_limits<float>::infinity());
    case tl::Operator::kGE:
      // swap left and right
      std::swap(left, right);
      def_left = !def_left;
      break;
    default:
      ASSERT(false, "only <, >, <= and >= comparisons are supported");
  }
  dense_node_init(&nodes[root + cur], 0, n.threshold(), n.split_index(),
                  def_left, false);
  node2fil(pnodes, root, 2 * cur + 1, t, t[left]);
  node2fil(pnodes, root, 2 * cur + 2, t, t[right]);
}

void tree2fil(std::vector<dense_node_t>* pnodes, int root, const tl::Tree& t) {
  node2fil(pnodes, root, 0, t, t[tree_root(t)]);
}

void tl2fil(forest_params_t* ps, std::vector<dense_node_t>* pnodes,
            const tl::Model& tl_model) {
  // fill in forest-indendent params
  ps->algo = algo_t::BATCH_TREE_REORG;
  ps->threshold = 0.5;

  // fill in forest-dependent params
  ps->cols = tl_model.num_feature;
  ASSERT(tl_model.num_output_group == 1,
         "multi-class classification not supported");
  const tl::ModelParam& param = tl_model.param;
  ASSERT(param.sigmoid_alpha == 1.0f, "sigmoid_alpha not supported");
  ASSERT(param.global_bias == 0.0f, "bias not supported");
  if (param.pred_transform == "identity") {
    ps->output = output_t::RAW;
  } else if (param.pred_transform == "sigmoid") {
    ps->output = output_t::PROB;
  } else {
    ASSERT(false, "%s: unsupported treelite prediction transform",
           param.pred_transform.c_str());
  }
  ps->ntrees = tl_model.trees.size();

  int depth = 0;
  for (const auto& t : tl_model.trees) depth = std::max(depth, max_depth(t));
  ps->depth = depth;

  // convert the nodes
  std::vector<dense_node_t>& nodes = *pnodes;
  int nnodes = forest_num_nodes(ps->ntrees, ps->depth);
  nodes.resize(nnodes, dense_node_t{0, 0});
  for (int i = 0; i < tl_model.trees.size(); ++i) {
    tree2fil(pnodes, i * tree_num_nodes(ps->depth), tl_model.trees[i]);
  }
  ps->nodes = nodes.data();
}

void init_dense(const cumlHandle& h, forest_t* pf,
                const forest_params_t* params) {
  check_params(params);
  forest* f = new forest;
  f->init(h, params);
  *pf = f;
}

void from_treelite(const cumlHandle& handle, forest_t* pforest,
                   ModelHandle model) {
  forest_params_t ps;
  std::vector<dense_node_t> nodes;
  tl2fil(&ps, &nodes, *(tl::Model*)model);
  init_dense(handle, pforest, &ps);
  // sync is necessary as nodes is used in init_dense(),
  // but destructed at the end of this function
  CUDA_CHECK(cudaStreamSynchronize(handle.getStream()));
}

void free(const cumlHandle& h, forest_t f) {
  f->free(h);
  delete f;
}

void predict(const cumlHandle& h, forest_t f, float* preds, const float* data,
             size_t n) {
  f->predict(h, preds, data, n);
}

}  // namespace fil
}  // namespace ML
