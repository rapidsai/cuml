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
#include <algorithm>

#include "common.cuh"
#include "fil.h"

namespace ML {
namespace fil {

using namespace MLCommon;

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
    max_shm_ = 48 * 1024;  // 48 KiB
    int device = 0;
    // TODO: use cumlHandle for this
    CUDA_CHECK(cudaGetDevice(&device));
    int max_shm_device;
    CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shm_device, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    max_shm_ = std::min(max_shm_, max_shm_device);
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
        ASSERT(false, "should not reach here");
    }

    // Transform the output if necessary (sigmoid + thresholding if necessary).
    if (output_ != output_t::RAW) {
      transform_k<<<ceildiv(int(rows), TPB), TPB, 0, stream>>>(
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

int init_dense(const cumlHandle& h, forest_t* pf,
               const forest_params_t* params) {
  forest* f = new forest;
  f->init(h, params);
  *pf = f;
  return 0;
}

int free(const cumlHandle& h, forest_t f) {
  f->free(h);
  delete f;
  return 0;
}

int predict(const cumlHandle& h, forest_t f, float* preds, const float* data,
            size_t n) {
  f->predict(h, preds, data, n);
  return 0;
}

}  // namespace fil
}  // namespace ML
