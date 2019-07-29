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
#include <treelite/frontend.h>
#include <treelite/tree.h>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include "fil/fil.h"
#include "ml_utils.h"
#include "random/rng.h"
#include "test_utils.h"

#define TL_CPP_CHECK(call) ASSERT(int(call) >= 0, "treelite call error")

namespace ML {

using namespace MLCommon;
namespace tl = treelite;
namespace tlf = treelite::frontend;

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
  // treelite parameters, only used for treelite tests
  tl::Operator op;
};

std::ostream& operator<<(std::ostream& os, const FilTestParams& ps) {
  os << "rows = " << ps.rows << ", cols = " << ps.cols
     << ", nan_prob = " << ps.nan_prob << ", depth = " << ps.depth
     << ", num_trees = " << ps.num_trees << ", leaf_prob = " << ps.leaf_prob
     << ", output = " << ps.output << ", threshold = " << ps.threshold
     << ", algo = " << ps.algo << ", seed = " << ps.seed
     << ", tolerance = " << ps.tolerance << ", op = " << tl::OpName(ps.op);
  return os;
}

__global__ void nan_kernel(float* data, const bool* mask, int len, float nan) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= len) return;
  if (!mask[tid]) data[tid] = nan;
}

float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

class BaseFilTest : public testing::TestWithParam<FilTestParams> {
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
    CUDA_CHECK(cudaFree(preds_d));
    CUDA_CHECK(cudaFree(want_preds_d));
    CUDA_CHECK(cudaFree(data_d));
  }

  void generate_forest() {
    size_t num_nodes = forest_num_nodes();

    // helper data
    float* weights_d = nullptr;
    float* thresholds_d = nullptr;
    int* fids_d = nullptr;
    bool* def_lefts_d = nullptr;
    bool* is_leafs_d = nullptr;
    bool* def_lefts_h = nullptr;
    bool* is_leafs_h = nullptr;

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
    CUDA_CHECK(cudaStreamSynchronize(stream));

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

    // clean up
    delete[] def_lefts_h;
    delete[] is_leafs_h;
    CUDA_CHECK(cudaFree(is_leafs_d));
    CUDA_CHECK(cudaFree(def_lefts_d));
    CUDA_CHECK(cudaFree(fids_d));
    CUDA_CHECK(cudaFree(thresholds_d));
    CUDA_CHECK(cudaFree(weights_d));
  }

  void generate_data() {
    // allocate arrays
    size_t num_data = ps.rows * ps.cols;
    allocate(data_d, num_data);
    bool* mask_d = nullptr;
    allocate(mask_d, num_data);

    // generate random data
    Random::Rng r(ps.seed);
    r.uniform(data_d, num_data, -1.0f, 1.0f, stream);
    r.bernoulli(mask_d, num_data, ps.nan_prob, stream);
    int tpb = 256;
    nan_kernel<<<ceildiv(int(num_data), tpb), tpb, 0, stream>>>(
      data_d, mask_d, num_data, std::numeric_limits<float>::quiet_NaN());
    CUDA_CHECK(cudaPeekAtLastError());

    // copy to host
    data_h.resize(num_data);
    updateHost(data_h.data(), data_d, num_data, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // clean up
    CUDA_CHECK(cudaFree(mask_d));
  }

  void predict_on_cpu() {
    // predict on host
    std::vector<float> want_preds_h(ps.rows);
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
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  virtual void init_forest(fil::forest_t* pforest) = 0;

  void predict_on_gpu() {
    fil::forest_t forest = nullptr;
    init_forest(&forest);

    // predict
    allocate(preds_d, ps.rows);
    fil::predict(handle, forest, preds_d, data_d, ps.rows);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // cleanup
    fil::free(handle, forest);
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

  // predictions
  float* preds_d = nullptr;
  float* want_preds_d = nullptr;

  // input data
  float* data_d = nullptr;
  std::vector<float> data_h;

  // forest data
  std::vector<fil::dense_node_t> nodes;

  // parameters
  cudaStream_t stream;
  cumlHandle handle;
  FilTestParams ps;
};

class PredictFilTest : public BaseFilTest {
 protected:
  void init_forest(fil::forest_t* pforest) override {
    // init FIL model
    fil::forest_params_t fil_ps;
    fil_ps.nodes = nodes.data();
    fil_ps.depth = ps.depth;
    fil_ps.ntrees = ps.num_trees;
    fil_ps.cols = ps.cols;
    fil_ps.algo = ps.algo;
    fil_ps.output = ps.output;
    fil_ps.threshold = ps.threshold;
    fil::init_dense(handle, pforest, &fil_ps);
  }
};

class TreeliteFilTest : public BaseFilTest {
 protected:
  /** adds nodes[node] of tree starting at index root to builder 
      at index at *pkey, increments *pkey, 
      and returns the treelite key of the node */
  int node_to_treelite(tlf::TreeBuilder* builder, int* pkey, int root,
                       int node) {
    int key = (*pkey)++;
    TL_CPP_CHECK(builder->CreateNode(key));
    int feature;
    float threshold, output;
    bool is_leaf, default_left;
    fil::dense_node_decode(&nodes[node], &output, &threshold, &feature,
                           &default_left, &is_leaf);
    if (is_leaf) {
      TL_CPP_CHECK(builder->SetLeafNode(key, output));
    } else {
      int left = root + 2 * (node - root) + 1;
      int right = root + 2 * (node - root) + 2;
      switch (ps.op) {
        case tl::Operator::kLT:
          break;
        case tl::Operator::kLE:
          // adjust the threshold
          threshold =
            std::nextafterf(threshold, -std::numeric_limits<float>::infinity());
          break;
        case tl::Operator::kGT:
          // adjust the threshold; left and right still need to be swapped
          threshold =
            std::nextafterf(threshold, -std::numeric_limits<float>::infinity());
        case tl::Operator::kGE:
          // swap left and right
          std::swap(left, right);
          default_left = !default_left;
          break;
        default:
          ASSERT(false, "comparison operator must be <, >, <= or >=");
      }
      int left_key = node_to_treelite(builder, pkey, root, left);
      int right_key = node_to_treelite(builder, pkey, root, right);
      TL_CPP_CHECK(builder->SetNumericalTestNode(
        key, feature, ps.op, threshold, default_left, left_key, right_key));
    }
    return key;
  }

  void init_forest(fil::forest_t* pforest) override {
    std::unique_ptr<tlf::ModelBuilder> model_builder(
      new tlf::ModelBuilder(ps.cols, 1, false));

    // model metadata
    switch (ps.output) {
      case fil::output_t::RAW:
        break;
      case fil::output_t::PROB:
      case fil::output_t::CLASS:
        model_builder->SetModelParam("pred_transform", "sigmoid");
        break;
      default:
        ASSERT(false, "invalid output type");
    }

    // build the trees
    for (int i_tree = 0; i_tree < ps.num_trees; ++i_tree) {
      tlf::TreeBuilder* tree_builder = new tlf::TreeBuilder();
      int key_counter = 0;
      int root = i_tree * tree_num_nodes();
      int root_key = node_to_treelite(tree_builder, &key_counter, root, root);
      TL_CPP_CHECK(tree_builder->SetRootNode(root_key));
      // InsertTree() consumes tree_builder
      TL_CPP_CHECK(model_builder->InsertTree(tree_builder));
    }

    // commit the model
    std::unique_ptr<tl::Model> model(new tl::Model);
    TL_CPP_CHECK(model_builder->CommitModel(model.get()));

    // init FIL forest with the model
    fil::treelite_params_t params;
    params.algo = ps.algo;
    params.threshold = ps.threshold;
    params.output_class = ps.output == fil::output_t::CLASS;
    fil::from_treelite(handle, pforest, (ModelHandle)model.get(), &params);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
};

// rows, cols, nan_prob, depth, num_trees, leaf_prob, output, threshold, algo,
// seed, tolerance
std::vector<FilTestParams> predict_inputs = {
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
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f},
};

TEST_P(PredictFilTest, Predict) { compare(); }

INSTANTIATE_TEST_CASE_P(FilTests, PredictFilTest,
                        testing::ValuesIn(predict_inputs));

std::vector<FilTestParams> import_inputs = {
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, fil::algo_t::NAIVE, 42,
   2e-3f, tl::Operator::kLT},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::PROB, 0, fil::algo_t::NAIVE, 42,
   2e-3f, tl::Operator::kLE},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::CLASS, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator::kGT},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, fil::algo_t::TREE_REORG,
   42, 2e-3f, tl::Operator::kGE},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::PROB, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator::kLT},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::CLASS, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator::kLE},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLT},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::PROB, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLT},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLE},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::PROB, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLE},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kGT},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::PROB, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kGT},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kGE},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::PROB, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kGE},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::CLASS, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLT},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::CLASS, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLE},
};

TEST_P(TreeliteFilTest, Import) { compare(); }

INSTANTIATE_TEST_CASE_P(FilTests, TreeliteFilTest,
                        testing::ValuesIn(import_inputs));

}  // namespace ML
