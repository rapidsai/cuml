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

#include <cuda_utils.h>
#include <gtest/gtest.h>
#include <test_utils.h>
#include "fil/fil.h"
#include "ml_utils.h"
#include "random/rng.h"
#include "test_utils.h"

namespace ML {

using namespace MLCommon;

struct FilTestParams {
  // input data parameters
  int rows;
  int cols;
  float nan_prob;
  // forest parameters
  int depth;
  int num_trees;
  float leaf_prob;
  // output parameters
  fil::output_t output;
  float threshold;
  // runtime parameters
  fil::algo_t algo;
  int seed;
  float tolerance;
};

std::ostream& operator<<(std::ostream& os, const FilTestParams& ps) {
  os << "rows = " << ps.rows << ", cols = " << ps.cols
     << ", nan_prob = " << ps.nan_prob << ", depth = " << ps.depth
     << ", num_trees = " << ps.num_trees << ", leaf_prob = " << ps.leaf_prob
     << ", output = " << ps.output << ", threshold = " << ps.threshold
     << ", algo = " << ps.algo << ", seed = " << ps.seed
     << ", tolerance = " << ps.tolerance;
  return os;
}

__global__ void nan_kernel(float* data, const bool* mask, int len, float nan) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= len) return;
  if (!mask[tid]) data[tid] = nan;
}

float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

class FilTest : public testing::TestWithParam<FilTestParams> {
 protected:
  void SetUp() override {
    // setup
    ps = testing::TestWithParam<FilTestParams>::GetParam();
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.setStream(stream);

    generate_forest();
    generate_data();
    predict_on_cpu();
    predict_on_gpu();
  }

  void TearDown() override {
    fil::free(handle, forest);

    delete[] def_lefts_h;
    delete[] is_leafs_h;

    CUDA_CHECK(cudaFree(preds_d));
    CUDA_CHECK(cudaFree(want_preds_d));
    CUDA_CHECK(cudaFree(mask_d));
    CUDA_CHECK(cudaFree(data_d));
    CUDA_CHECK(cudaFree(is_leafs_d));
    CUDA_CHECK(cudaFree(def_lefts_d));
    CUDA_CHECK(cudaFree(fids_d));
    CUDA_CHECK(cudaFree(thresholds_d));
    CUDA_CHECK(cudaFree(weights_d));
  }

  void generate_forest() {
    size_t num_nodes = forest_num_nodes();

    // allocate GPU data
    allocate(weights_d, num_nodes);
    allocate(thresholds_d, num_nodes);
    allocate(fids_d, num_nodes);
    allocate(def_lefts_d, num_nodes);
    allocate(is_leafs_d, num_nodes);

    // generate on-GPU random data
    Random::Rng r(ps.seed);
    r.uniform(weights_d, num_nodes, -1.0f, 1.0f, stream);
    r.uniform(thresholds_d, num_nodes, -1.0f, 1.0f, stream);
    r.uniformInt(fids_d, num_nodes, 0, ps.cols, stream);
    r.bernoulli(def_lefts_d, num_nodes, 0.5f, stream);
    r.bernoulli(is_leafs_d, num_nodes, 1.0f - ps.leaf_prob, stream);

    // copy data to host
    std::vector<float> weights_h(num_nodes), thresholds_h(num_nodes);
    std::vector<int> fids_h(num_nodes);
    def_lefts_h = new bool[num_nodes];
    is_leafs_h = new bool[num_nodes];

    updateHost(weights_h.data(), weights_d, num_nodes, stream);
    updateHost(thresholds_h.data(), thresholds_d, num_nodes, stream);
    updateHost(fids_h.data(), fids_d, num_nodes, stream);
    updateHost(def_lefts_h, def_lefts_d, num_nodes, stream);
    updateHost(is_leafs_h, is_leafs_d, num_nodes, stream);

    // mark leaves
    for (size_t i = 0; i < ps.num_trees; ++i) {
      int num_tree_nodes = tree_num_nodes();
      size_t leaf_start = num_tree_nodes * i + num_tree_nodes / 2;
      size_t leaf_end = num_tree_nodes * (i + 1);
      for (size_t j = leaf_start; j < leaf_end; ++j) {
        is_leafs_h[j] = true;
      }
    }

    // initialize nodes
    nodes.resize(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
      fil::dense_node_init(&nodes[i], weights_h[i], thresholds_h[i], fids_h[i],
                           def_lefts_h[i], is_leafs_h[i]);
    }
  }

  void generate_data() {
    size_t num_data = ps.rows * ps.cols;
    allocate(data_d, num_data);
    allocate(mask_d, num_data);

    Random::Rng r(ps.seed);
    r.uniform(data_d, num_data, -1.0f, 1.0f, stream);
    r.bernoulli(mask_d, num_data, ps.nan_prob, stream);
    int tpb = 256;
    nan_kernel<<<ceildiv(int(num_data), tpb), tpb, 0, stream>>>(
      data_d, mask_d, num_data, std::numeric_limits<float>::quiet_NaN());
    CUDA_CHECK(cudaPeekAtLastError());

    data_h.resize(num_data);
    updateHost(data_h.data(), data_d, num_data, stream);
  }

  void predict_on_cpu() {
    // predict on host
    want_preds_h.resize(ps.rows);
    int num_nodes = tree_num_nodes();
    for (int i = 0; i < ps.rows; ++i) {
      float pred = 0.0f;
      for (int j = 0; j < ps.num_trees; ++j) {
        pred += infer_one_tree(&nodes[j * num_nodes], &data_h[i * ps.cols]);
      }
      if (ps.output != fil::output_t::RAW) pred = sigmoid(pred);
      if (ps.output == fil::output_t::CLASS) {
        pred = pred > ps.threshold ? 1.0f : 0.0f;
      }
      want_preds_h[i] = pred;
    }

    // copy to GPU
    allocate(want_preds_d, ps.rows);
    updateDevice(want_preds_d, want_preds_h.data(), ps.rows, stream);
  }

  void predict_on_gpu() {
    // init FIL model
    fil::forest_params_t fil_ps;
    fil_ps.nodes = nodes.data();
    fil_ps.depth = ps.depth;
    fil_ps.ntrees = ps.num_trees;
    fil_ps.cols = ps.cols;
    fil_ps.algo = ps.algo;
    fil_ps.output = ps.output;
    fil_ps.threshold = ps.threshold;
    fil::init_dense(handle, &forest, &fil_ps);

    // predict
    allocate(preds_d, ps.rows);
    fil::predict(handle, forest, preds_d, data_d, ps.rows);
  }

  void compare() {
    ASSERT_TRUE(devArrMatch(want_preds_d, preds_d, ps.rows,
                            CompareApprox<float>(ps.tolerance), stream));
  }

  float infer_one_tree(fil::dense_node_t* root, float* data) {
    int curr = 0;
    float output = 0.0f, threshold = 0.0f;
    int fid = 0;
    bool def_left = false, is_leaf = false;
    for (;;) {
      fil::dense_node_decode(&root[curr], &output, &threshold, &fid, &def_left,
                             &is_leaf);
      if (is_leaf) break;
      float val = data[fid];
      bool cond = isnan(val) ? !def_left : val >= threshold;
      curr = (curr << 1) + 1 + (cond ? 1 : 0);
    }
    return output;
  }

  int tree_num_nodes() { return (1 << (ps.depth + 1)) - 1; }

  int forest_num_nodes() { return tree_num_nodes() * ps.num_trees; }

  // FIL
  fil::forest_t forest = nullptr;

  // predictions
  float* preds_d = nullptr;
  float* want_preds_d = nullptr;
  std::vector<float> want_preds_h;

  // input data
  float* data_d = nullptr;
  std::vector<float> data_h;

  // forest data
  std::vector<fil::dense_node_t> nodes;

  // helper data
  bool* mask_d = nullptr;
  float* weights_d = nullptr;
  float* thresholds_d = nullptr;
  int* fids_d = nullptr;
  bool* def_lefts_d = nullptr;
  bool* is_leafs_d = nullptr;
  bool* def_lefts_h = nullptr;
  bool* is_leafs_h = nullptr;

  // parameters
  cudaStream_t stream;
  cumlHandle handle;
  FilTestParams ps;
};

// rows, cols, nan_prob, depth, num_trees, leaf_prob, output, threshold, algo,
// seed, tolerance
std::vector<FilTestParams> inputs = {
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, fil::algo_t::NAIVE, 42,
   2e-3f},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, fil::algo_t::TREE_REORG,
   42, 2e-3f},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::PROB, 0, fil::algo_t::NAIVE, 42,
   2e-3f},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::PROB, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::PROB, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::CLASS, 0, fil::algo_t::NAIVE,
   42, 2e-3f},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::CLASS, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::CLASS, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f},
};

TEST_P(FilTest, Predict) { compare(); }

INSTANTIATE_TEST_CASE_P(FilTests, FilTest, testing::ValuesIn(inputs));

}  // namespace ML
