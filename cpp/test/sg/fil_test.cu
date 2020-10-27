/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>
#include <cuml/fil/fil.h>
#include <gtest/gtest.h>
#include <test_utils.h>
#include <treelite/c_api.h>
#include <treelite/frontend.h>
#include <treelite/tree.h>
#include <cmath>
#include <cstdio>
#include <raft/cuda_utils.cuh>
#include <limits>
#include <memory>
#include <raft/random/rng.cuh>
#include <utility>

#define TL_CPP_CHECK(call) ASSERT(int(call) >= 0, "treelite call error")

namespace ML {

using namespace MLCommon;
namespace tl = treelite;
namespace tlf = treelite::frontend;

struct FilTestParams {
  // input data parameters
  int num_rows;
  int num_cols;
  float nan_prob;
  // forest parameters
  int depth;
  int num_trees;
  float leaf_prob;
  // output parameters
  fil::output_t output;
  float threshold;
  float global_bias;
  // runtime parameters
  fil::algo_t algo;
  int seed;
  float tolerance;
  // treelite parameters, only used for treelite tests
  tl::Operator op;
  fil::leaf_algo_t leaf_algo;
  // when FLOAT_UNARY_BINARY == leaf_algo:
  // num_classes = 1 means it's regression
  // num_classes = 2 means it's binary classification
  // (complement probabilities, then use threshold)
  // when GROVE_PER_CLASS == leaf_algo:
  // it's multiclass classification (num_classes must be > 2),
  // done by splitting the forest in num_classes groups,
  // each of which computes one-vs-all probability for its class.
  // when CATEGORICAL_LEAF == leaf_algo:
  // num_classes must be > 1 and it's multiclass classification.
  // done by storing the class label in each leaf and voting.
  // it's used in treelite ModelBuilder initialization
  int num_classes;

  size_t num_proba_outputs() { return num_rows * std::max(num_classes, 2); }
  size_t num_preds_outputs() { return num_rows; }
};

std::string output2str(fil::output_t output) {
  if (output == fil::RAW) return "RAW";
  std::string s = "";
  if (output & fil::AVG) s += "| AVG";
  if (output & fil::CLASS) s += "| CLASS";
  if (output & fil::SIGMOID) s += "| SIGMOID";
  return s;
}

std::ostream& operator<<(std::ostream& os, const FilTestParams& ps) {
  os << "num_rows = " << ps.num_rows << ", num_cols = " << ps.num_cols
     << ", nan_prob = " << ps.nan_prob << ", depth = " << ps.depth
     << ", num_trees = " << ps.num_trees << ", leaf_prob = " << ps.leaf_prob
     << ", output = " << output2str(ps.output)
     << ", threshold = " << ps.threshold << ", algo = " << ps.algo
     << ", seed = " << ps.seed << ", tolerance = " << ps.tolerance
     << ", op = " << tl::OpName(ps.op) << ", global_bias = " << ps.global_bias
     << ", leaf_algo = " << ps.leaf_algo
     << ", num_classes = " << ps.num_classes;
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
  void setup_helper() {
    // setup
    ps = testing::TestWithParam<FilTestParams>::GetParam();
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.set_stream(stream);

    generate_forest();
    generate_data();
    predict_on_cpu();
    predict_on_gpu();
  }

  void SetUp() override { setup_helper(); }

  void TearDown() override {
    CUDA_CHECK(cudaFree(preds_d));
    CUDA_CHECK(cudaFree(want_preds_d));
    CUDA_CHECK(cudaFree(data_d));
    CUDA_CHECK(cudaFree(want_proba_d));
    CUDA_CHECK(cudaFree(proba_d));
  }

  void generate_forest() {
    size_t num_nodes = forest_num_nodes();

    // helper data
    /// weights, used as float* or int*
    int* weights_d = nullptr;
    float* thresholds_d = nullptr;
    int* fids_d = nullptr;
    bool* def_lefts_d = nullptr;
    bool* is_leafs_d = nullptr;
    bool* def_lefts_h = nullptr;
    bool* is_leafs_h = nullptr;

    // allocate GPU data
    raft::allocate(weights_d, num_nodes);
    // sizeof(float) == sizeof(int)
    raft::allocate(thresholds_d, num_nodes);
    raft::allocate(fids_d, num_nodes);
    raft::allocate(def_lefts_d, num_nodes);
    raft::allocate(is_leafs_d, num_nodes);

    // generate on-GPU random data
    raft::random::Rng r(ps.seed);
    if (ps.leaf_algo != fil::leaf_algo_t::CATEGORICAL_LEAF) {
      r.uniform((float*)weights_d, num_nodes, -1.0f, 1.0f, stream);
    } else {
      // [0..num_classes)
      r.uniformInt((int*)weights_d, num_nodes, 0, ps.num_classes, stream);
    }
    r.uniform(thresholds_d, num_nodes, -1.0f, 1.0f, stream);
    r.uniformInt(fids_d, num_nodes, 0, ps.num_cols, stream);
    r.bernoulli(def_lefts_d, num_nodes, 0.5f, stream);
    r.bernoulli(is_leafs_d, num_nodes, 1.0f - ps.leaf_prob, stream);

    // copy data to host
    std::vector<float> thresholds_h(num_nodes);
    std::vector<int> weights_h(num_nodes), fids_h(num_nodes);
    def_lefts_h = new bool[num_nodes];
    is_leafs_h = new bool[num_nodes];

    raft::update_host(weights_h.data(), (int*)weights_d, num_nodes, stream);
    raft::update_host(thresholds_h.data(), thresholds_d, num_nodes, stream);
    raft::update_host(fids_h.data(), fids_d, num_nodes, stream);
    raft::update_host(def_lefts_h, def_lefts_d, num_nodes, stream);
    raft::update_host(is_leafs_h, is_leafs_d, num_nodes, stream);
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
      fil::val_t w;
      switch (ps.leaf_algo) {
        case fil::leaf_algo_t::CATEGORICAL_LEAF:
          w.idx = weights_h[i];
          break;
        case fil::leaf_algo_t::FLOAT_UNARY_BINARY:
        case fil::leaf_algo_t::GROVE_PER_CLASS:
          // not relying on fil::val_t internals
          // merely that we copied floats into weights_h earlier
          std::memcpy(&w.f, &weights_h[i], sizeof w.f);
          break;
        default:
          ASSERT(false, "internal error: invalid ps.leaf_algo");
      }
      fil::node_init(&nodes[i], w, thresholds_h[i], fids_h[i], def_lefts_h[i],
                     is_leafs_h[i]);
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
    size_t num_data = ps.num_rows * ps.num_cols;
    raft::allocate(data_d, num_data);
    bool* mask_d = nullptr;
    raft::allocate(mask_d, num_data);

    // generate random data
    raft::random::Rng r(ps.seed);
    r.uniform(data_d, num_data, -1.0f, 1.0f, stream);
    r.bernoulli(mask_d, num_data, ps.nan_prob, stream);
    int tpb = 256;
    nan_kernel<<<raft::ceildiv(int(num_data), tpb), tpb, 0, stream>>>(
      data_d, mask_d, num_data, std::numeric_limits<float>::quiet_NaN());
    CUDA_CHECK(cudaPeekAtLastError());

    // copy to host
    data_h.resize(num_data);
    raft::update_host(data_h.data(), data_d, num_data, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // clean up
    CUDA_CHECK(cudaFree(mask_d));
  }

  void transform(float f, float& proba, float& output) {
    if ((ps.output & fil::output_t::AVG) != 0) {
      f *= (1.0f / ps.num_trees);
    }
    f += ps.global_bias;
    if ((ps.output & fil::output_t::SIGMOID) != 0) {
      f = sigmoid(f);
    }
    proba = f;
    if ((ps.output & fil::output_t::CLASS) != 0) {
      f = f > ps.threshold ? 1.0f : 0.0f;
    }
    output = f;
  }

  void complement(float* proba) { proba[0] = 1.0f - proba[1]; }

  void predict_on_cpu() {
    // predict on host
    std::vector<float> want_preds_h(ps.num_preds_outputs());
    std::vector<float> want_proba_h(ps.num_proba_outputs());
    int num_nodes = tree_num_nodes();
    std::vector<float> class_scores(ps.num_classes);
    switch (ps.leaf_algo) {
      case fil::leaf_algo_t::FLOAT_UNARY_BINARY:
        for (int i = 0; i < ps.num_rows; ++i) {
          float pred = 0.0f;
          for (int j = 0; j < ps.num_trees; ++j) {
            pred +=
              infer_one_tree(&nodes[j * num_nodes], &data_h[i * ps.num_cols]).f;
          }
          transform(pred, want_proba_h[i * 2 + 1], want_preds_h[i]);
          complement(&(want_proba_h[i * 2]));
        }
        break;
      case fil::leaf_algo_t::GROVE_PER_CLASS:
        for (int row = 0; row < ps.num_rows; ++row) {
          std::fill(class_scores.begin(), class_scores.end(), 0.0f);
          for (int tree = 0; tree < ps.num_trees; ++tree) {
            class_scores[tree % ps.num_classes] +=
              infer_one_tree(&nodes[tree * num_nodes],
                             &data_h[row * ps.num_cols])
                .f;
          }
          // not supporting predict_proba() with GROVE_PER_CLASS (xgboost-style models)
          want_preds_h[row] =
            std::max_element(class_scores.begin(), class_scores.end()) -
            class_scores.begin();
        }
        break;
      case fil::leaf_algo_t::CATEGORICAL_LEAF:
        std::vector<int> class_votes(ps.num_classes);
        for (int r = 0; r < ps.num_rows; ++r) {
          std::fill(class_votes.begin(), class_votes.end(), 0);
          for (int j = 0; j < ps.num_trees; ++j) {
            int class_label =
              infer_one_tree(&nodes[j * num_nodes], &data_h[r * ps.num_cols])
                .idx;
            ++class_votes[class_label];
          }
          for (int c = 0; c < ps.num_classes; ++c) {
            float thresholded_proba;  // not used; do argmax instead
            transform(class_votes[c], want_proba_h[r * ps.num_classes + c],
                      thresholded_proba);
          }
          want_preds_h[r] =
            std::max_element(class_votes.begin(), class_votes.end()) -
            class_votes.begin();
        }
        break;
    }

    // copy to GPU
    raft::allocate(want_preds_d, ps.num_preds_outputs());
    raft::allocate(want_proba_d, ps.num_proba_outputs());
    raft::update_device(want_preds_d, want_preds_h.data(),
                        ps.num_preds_outputs(), stream);
    raft::update_device(want_proba_d, want_proba_h.data(),
                        ps.num_proba_outputs(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  virtual void init_forest(fil::forest_t* pforest) = 0;

  void predict_on_gpu() {
    fil::forest_t forest = nullptr;
    init_forest(&forest);

    // predict
    raft::allocate(preds_d, ps.num_preds_outputs());
    raft::allocate(proba_d, ps.num_proba_outputs());
    fil::predict(handle, forest, preds_d, data_d, ps.num_rows);
    // not supporting predict_proba() with GROVE_PER_CLASS (xgboost-style models)
    if (ps.leaf_algo != fil::leaf_algo_t::GROVE_PER_CLASS)
      fil::predict(handle, forest, proba_d, data_d, ps.num_rows, true);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // cleanup
    fil::free(handle, forest);
  }

  void compare() {
    // not supporting predict_proba() with GROVE_PER_CLASS (xgboost-style models)
    if (ps.leaf_algo != fil::leaf_algo_t::GROVE_PER_CLASS) {
      ASSERT_TRUE(
        raft::devArrMatch(want_proba_d, proba_d, ps.num_proba_outputs(),
                          raft::CompareApprox<float>(ps.tolerance), stream));
    }
    float tolerance = ps.leaf_algo == fil::leaf_algo_t::FLOAT_UNARY_BINARY
                        ? ps.tolerance
                        : std::numeric_limits<float>::epsilon();
    // in multi-class prediction, floats represent the most likely class
    // and would be generated by converting an int to float
    ASSERT_TRUE(raft::devArrMatch(want_preds_d, preds_d, ps.num_rows,
                                  raft::CompareApprox<float>(tolerance),
                                  stream));
  }

  fil::val_t infer_one_tree(fil::dense_node_t* root, float* data) {
    int curr = 0;
    float threshold = 0.0f;
    fil::val_t output{.f = 0.0f};
    int fid = 0;
    bool def_left = false, is_leaf = false;
    for (;;) {
      fil::node_decode(&root[curr], &output, &threshold, &fid, &def_left,
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
  float* proba_d = nullptr;
  float* want_preds_d = nullptr;
  float* want_proba_d = nullptr;

  // input data
  float* data_d = nullptr;
  std::vector<float> data_h;

  // forest data
  std::vector<fil::dense_node_t> nodes;

  // parameters
  cudaStream_t stream;
  raft::handle_t handle;
  FilTestParams ps;
};

class PredictDenseFilTest : public BaseFilTest {
 protected:
  void init_forest(fil::forest_t* pforest) override {
    // init FIL model
    fil::forest_params_t fil_ps;
    fil_ps.depth = ps.depth;
    fil_ps.num_trees = ps.num_trees;
    fil_ps.num_cols = ps.num_cols;
    fil_ps.algo = ps.algo;
    fil_ps.output = ps.output;
    fil_ps.threshold = ps.threshold;
    fil_ps.global_bias = ps.global_bias;
    fil_ps.leaf_algo = ps.leaf_algo;
    fil_ps.num_classes = ps.num_classes;

    fil::init_dense(handle, pforest, nodes.data(), &fil_ps);
  }
};

template <typename fil_node_t>
class BasePredictSparseFilTest : public BaseFilTest {
 protected:
  void dense2sparse_node(const fil::dense_node_t* dense_root, int i_dense,
                         int i_sparse_root, int i_sparse) {
    float threshold;
    fil::val_t output;
    int feature;
    bool def_left, is_leaf;
    fil::node_decode(&dense_root[i_dense], &output, &threshold, &feature,
                     &def_left, &is_leaf);
    if (is_leaf) {
      // leaf sparse node
      node_init(&sparse_nodes[i_sparse], output, threshold, feature, def_left,
                is_leaf, 0);
      return;
    }
    // inner sparse node
    // reserve space for children
    int left_index = sparse_nodes.size();
    sparse_nodes.push_back(fil_node_t());
    sparse_nodes.push_back(fil_node_t());
    node_init(&sparse_nodes[i_sparse], output, threshold, feature, def_left,
              is_leaf, left_index - i_sparse_root);
    dense2sparse_node(dense_root, 2 * i_dense + 1, i_sparse_root, left_index);
    dense2sparse_node(dense_root, 2 * i_dense + 2, i_sparse_root,
                      left_index + 1);
  }

  void dense2sparse_tree(const fil::dense_node_t* dense_root) {
    int i_sparse_root = sparse_nodes.size();
    sparse_nodes.push_back(fil_node_t());
    dense2sparse_node(dense_root, 0, i_sparse_root, i_sparse_root);
    trees.push_back(i_sparse_root);
  }

  void dense2sparse() {
    for (int tree = 0; tree < ps.num_trees; ++tree) {
      dense2sparse_tree(&nodes[tree * tree_num_nodes()]);
    }
  }

  void init_forest(fil::forest_t* pforest) override {
    // init FIL model
    fil::forest_params_t fil_params;
    fil_params.num_trees = ps.num_trees;
    fil_params.num_cols = ps.num_cols;
    fil_params.algo = ps.algo;
    fil_params.output = ps.output;
    fil_params.threshold = ps.threshold;
    fil_params.global_bias = ps.global_bias;
    fil_params.leaf_algo = ps.leaf_algo;
    fil_params.num_classes = ps.num_classes;

    dense2sparse();
    fil_params.num_nodes = sparse_nodes.size();
    fil::init_sparse(handle, pforest, trees.data(), sparse_nodes.data(),
                     &fil_params);
  }
  std::vector<fil_node_t> sparse_nodes;
  std::vector<int> trees;
};

typedef BasePredictSparseFilTest<fil::sparse_node16_t> PredictSparse16FilTest;
typedef BasePredictSparseFilTest<fil::sparse_node8_t> PredictSparse8FilTest;

class TreeliteFilTest : public BaseFilTest {
 protected:
  /** adds nodes[node] of tree starting at index root to builder
      at index at *pkey, increments *pkey,
      and returns the treelite key of the node */
  int node_to_treelite(tlf::TreeBuilder* builder, int* pkey, int root,
                       int node) {
    int key = (*pkey)++;
    builder->CreateNode(key);
    int feature;
    float threshold;
    fil::val_t output;
    bool is_leaf, default_left;
    fil::node_decode(&nodes[node], &output, &threshold, &feature, &default_left,
                     &is_leaf);
    if (is_leaf) {
      switch (ps.leaf_algo) {
        case fil::leaf_algo_t::FLOAT_UNARY_BINARY:
        case fil::leaf_algo_t::GROVE_PER_CLASS:
          // default is fil::FLOAT_UNARY_BINARY
          builder->SetLeafNode(key, output.f);
          break;
        case fil::leaf_algo_t::CATEGORICAL_LEAF:
          std::vector<tl::tl_float> vec(ps.num_classes);
          for (int i = 0; i < ps.num_classes; ++i)
            vec[i] = i == output.idx ? 1.0f : 0.0f;
          builder->SetLeafVectorNode(key, vec);
      }
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
      builder->SetNumericalTestNode(key, feature, ps.op, threshold,
                                    default_left, left_key, right_key);
    }
    return key;
  }

  void init_forest_impl(fil::forest_t* pforest,
                        fil::storage_type_t storage_type) {
    bool random_forest_flag = (ps.output & fil::output_t::AVG) != 0;
    int treelite_num_classes =
      ps.leaf_algo == fil::leaf_algo_t::FLOAT_UNARY_BINARY ? 1 : ps.num_classes;
    std::unique_ptr<tlf::ModelBuilder> model_builder(new tlf::ModelBuilder(
      ps.num_cols, treelite_num_classes, random_forest_flag));

    // prediction transform
    if ((ps.output & fil::output_t::SIGMOID) != 0) {
      model_builder->SetModelParam("pred_transform", "sigmoid");
    } else if (ps.leaf_algo != fil::leaf_algo_t::FLOAT_UNARY_BINARY) {
      model_builder->SetModelParam("pred_transform", "max_index");
      ps.output = fil::output_t(ps.output | fil::output_t::CLASS);
    } else {
      model_builder->SetModelParam("pred_transform", "identity");
    }

    // global bias
    char* global_bias_str = nullptr;
    ASSERT(asprintf(&global_bias_str, "%f", double(ps.global_bias)) > 0,
           "cannot convert global_bias into a string");
    model_builder->SetModelParam("global_bias", global_bias_str);
    free(global_bias_str);

    // build the trees
    for (int i_tree = 0; i_tree < ps.num_trees; ++i_tree) {
      tlf::TreeBuilder* tree_builder = new tlf::TreeBuilder();
      int key_counter = 0;
      int root = i_tree * tree_num_nodes();
      int root_key = node_to_treelite(tree_builder, &key_counter, root, root);
      tree_builder->SetRootNode(root_key);
      // InsertTree() consumes tree_builder
      TL_CPP_CHECK(model_builder->InsertTree(tree_builder));
    }

    // commit the model
    std::unique_ptr<tl::Model> model(new tl::Model);
    model_builder->CommitModel(model.get());

    // init FIL forest with the model
    fil::treelite_params_t params;
    params.algo = ps.algo;
    params.threshold = ps.threshold;
    params.output_class = (ps.output & fil::output_t::CLASS) != 0;
    params.storage_type = storage_type;
    fil::from_treelite(handle, pforest, (ModelHandle)model.get(), &params);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
};

class TreeliteDenseFilTest : public TreeliteFilTest {
 protected:
  void init_forest(fil::forest_t* pforest) override {
    init_forest_impl(pforest, fil::storage_type_t::DENSE);
  }
};

class TreeliteSparse16FilTest : public TreeliteFilTest {
 protected:
  void init_forest(fil::forest_t* pforest) override {
    init_forest_impl(pforest, fil::storage_type_t::SPARSE);
  }
};

class TreeliteSparse8FilTest : public TreeliteFilTest {
 protected:
  void init_forest(fil::forest_t* pforest) override {
    init_forest_impl(pforest, fil::storage_type_t::SPARSE8);
  }
};

class TreeliteAutoFilTest : public TreeliteFilTest {
 protected:
  void init_forest(fil::forest_t* pforest) override {
    init_forest_impl(pforest, fil::storage_type_t::AUTO);
  }
};

// test for failures; currently only supported for sparse8 nodes
class TreeliteThrowSparse8FilTest : public TreeliteSparse8FilTest {
 protected:
  // model import happens in check(), so this function is empty
  void SetUp() override {}

  void check() { ASSERT_THROW(setup_helper(), raft::exception); }
};

// rows, cols, nan_prob, depth, num_trees, leaf_prob, output, threshold,
// global_bias, algo, seed, tolerance, branch comparison operator, FIL implementation, number of classes
std::vector<FilTestParams> predict_dense_inputs = {
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator(0), fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::SIGMOID | fil::output_t::CLASS), 0, 0,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::SIGMOID | fil::output_t::CLASS), 0, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::SIGMOID | fil::output_t::CLASS), 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator(0), fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 0, 0,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 0, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0.5,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0.5,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0.5, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator(0), fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 1.0, 0.5,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0,
   fil::algo_t::ALGO_AUTO, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::CATEGORICAL_LEAF, 5},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 0, 0,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::CATEGORICAL_LEAF, 5},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::CATEGORICAL_LEAF, 7},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0.5,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::CATEGORICAL_LEAF, 4},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0.5, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator(0), fil::leaf_algo_t::CATEGORICAL_LEAF, 4},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::GROVE_PER_CLASS, 5},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::GROVE_PER_CLASS, 5},
  {20000, 50, 0.05, 8, 49, 0.05, fil::output_t::SIGMOID, 0, 0,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::GROVE_PER_CLASS, 7},
  {20000, 50, 0.05, 8, 52, 0.05, fil::output_t::RAW, 0, 0.5,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::GROVE_PER_CLASS, 4},
  {20000, 50, 0.05, 8, 52, 0.05, fil::output_t::AVG, 0, 0.5, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator(0), fil::leaf_algo_t::GROVE_PER_CLASS, 4},
};

TEST_P(PredictDenseFilTest, Predict) { compare(); }

INSTANTIATE_TEST_CASE_P(FilTests, PredictDenseFilTest,
                        testing::ValuesIn(predict_dense_inputs));

// rows, cols, nan_prob, depth, num_trees, leaf_prob, output, threshold,
// global_bias, algo, seed, tolerance, branch comparison operator, FIL implementation, number of classes
std::vector<FilTestParams> predict_sparse_inputs = {
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator(0), fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::SIGMOID | fil::output_t::CLASS), 0, 0,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator(0), fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 0, 0.5,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0.5, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator(0), fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0.5,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0.5, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator(0), fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 1.0, 0.5,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::SIGMOID | fil::output_t::CLASS), 0, 0,
   fil::algo_t::ALGO_AUTO, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 1.0, 0.5,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::CATEGORICAL_LEAF, 5000},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0.5, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator(0), fil::leaf_algo_t::CATEGORICAL_LEAF, 6},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::CLASS, 0, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator(0), fil::leaf_algo_t::CATEGORICAL_LEAF, 3},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator(0), fil::leaf_algo_t::CATEGORICAL_LEAF, 3},
  {20000, 50, 0.05, 2, 5000, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 1.0, 0.5,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator(0),
   fil::leaf_algo_t::GROVE_PER_CLASS, 5000},
  {20000, 50, 0.05, 8, 60, 0.05, fil::output_t::RAW, 0, 0.5, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator(0), fil::leaf_algo_t::GROVE_PER_CLASS, 6},
  {20000, 50, 0.05, 8, 51, 0.05, fil::output_t::CLASS, 0, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator(0), fil::leaf_algo_t::GROVE_PER_CLASS, 3},
  {20000, 50, 0.05, 8, 51, 0.05, fil::output_t::RAW, 0, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator(0), fil::leaf_algo_t::GROVE_PER_CLASS, 3},
};

TEST_P(PredictSparse16FilTest, Predict) { compare(); }

INSTANTIATE_TEST_CASE_P(FilTests, PredictSparse16FilTest,
                        testing::ValuesIn(predict_sparse_inputs));

TEST_P(PredictSparse8FilTest, Predict) { compare(); }

INSTANTIATE_TEST_CASE_P(FilTests, PredictSparse8FilTest,
                        testing::ValuesIn(predict_sparse_inputs));

// rows, cols, nan_prob, depth, num_trees, leaf_prob, output, threshold,
// global_bias, algo, seed, tolerance, branch comparison operator, FIL implementation, number of classes
std::vector<FilTestParams> import_dense_inputs = {
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator::kLT, fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::SIGMOID | fil::output_t::CLASS), 0, 0,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator::kGT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator::kGE, fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 0, 0,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator::kGT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::SIGMOID | fil::output_t::CLASS), 0, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator::kGE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 0, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kGT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kGT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kGE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kGE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::SIGMOID | fil::output_t::CLASS), 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::SIGMOID | fil::output_t::CLASS), 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kGT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kGE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0.5,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0.5,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0.5, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator::kGT, fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 1.0, 0.5,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator::kGE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0,
   fil::algo_t::ALGO_AUTO, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0,
   fil::algo_t::ALGO_AUTO, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kGE,
   fil::leaf_algo_t::CATEGORICAL_LEAF, 5},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kGT,
   fil::leaf_algo_t::CATEGORICAL_LEAF, 6},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::CATEGORICAL_LEAF, 3},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::CATEGORICAL_LEAF, 5},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 0, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::CATEGORICAL_LEAF, 5},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::CATEGORICAL_LEAF, 7},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator::kLT, fil::leaf_algo_t::CATEGORICAL_LEAF, 6},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::CLASS, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kGE,
   fil::leaf_algo_t::GROVE_PER_CLASS, 5},
  {20000, 50, 0.05, 8, 48, 0.05, fil::output_t::CLASS, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kGT,
   fil::leaf_algo_t::GROVE_PER_CLASS, 6},
  {20000, 50, 0.05, 8, 51, 0.05, fil::output_t::CLASS, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::GROVE_PER_CLASS, 3},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::CLASS, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::GROVE_PER_CLASS, 5},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::CLASS, 0, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::GROVE_PER_CLASS, 5},
  {20000, 50, 0.05, 8, 49, 0.05, fil::output_t::CLASS, 0, 0,
   fil::algo_t::TREE_REORG, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::GROVE_PER_CLASS, 7},
  {20000, 50, 0.05, 8, 48, 0.05, fil::output_t::CLASS, 0, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator::kLT, fil::leaf_algo_t::GROVE_PER_CLASS, 6},
};

TEST_P(TreeliteDenseFilTest, Import) { compare(); }

INSTANTIATE_TEST_CASE_P(FilTests, TreeliteDenseFilTest,
                        testing::ValuesIn(import_dense_inputs));

// rows, cols, nan_prob, depth, num_trees, leaf_prob, output, threshold,
// global_bias, algo, seed, tolerance, branch comparison operator, FIL implementation, number of classes
std::vector<FilTestParams> import_sparse_inputs = {
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator::kLT, fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::SIGMOID | fil::output_t::CLASS), 0, 0,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator::kGT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator::kGE, fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 0, 0,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0.5, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator::kLT, fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::SIGMOID, 0, 0.5,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator::kLE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0.5, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator::kGT, fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 1.0, 0.5,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator::kGE,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 2},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::RAW, 0, 0,
   fil::algo_t::ALGO_AUTO, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 8, 50, 0.05,
   fil::output_t(fil::output_t::AVG | fil::output_t::CLASS), 1.0, 0.5,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator::kGE,
   fil::leaf_algo_t::CATEGORICAL_LEAF, 10},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0,
   fil::algo_t::ALGO_AUTO, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::CATEGORICAL_LEAF, 4},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator::kLE, fil::leaf_algo_t::CATEGORICAL_LEAF, 5},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::AVG, 0, 0.5, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator::kLT, fil::leaf_algo_t::CATEGORICAL_LEAF, 3},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::CLASS, 1.0, 0.5,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator::kGE,
   fil::leaf_algo_t::GROVE_PER_CLASS, 10},
  {20000, 50, 0.05, 8, 52, 0.05, fil::output_t::CLASS, 0, 0,
   fil::algo_t::ALGO_AUTO, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::GROVE_PER_CLASS, 4},
  {20000, 50, 0.05, 8, 50, 0.05, fil::output_t::CLASS, 0, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator::kLE, fil::leaf_algo_t::GROVE_PER_CLASS, 5},
  {20000, 50, 0.05, 8, 51, 0.05, fil::output_t::CLASS, 0, 0.5,
   fil::algo_t::NAIVE, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::GROVE_PER_CLASS, 3},
};

TEST_P(TreeliteSparse16FilTest, Import) { compare(); }

INSTANTIATE_TEST_CASE_P(FilTests, TreeliteSparse16FilTest,
                        testing::ValuesIn(import_sparse_inputs));

TEST_P(TreeliteSparse8FilTest, Import) { compare(); }

INSTANTIATE_TEST_CASE_P(FilTests, TreeliteSparse8FilTest,
                        testing::ValuesIn(import_sparse_inputs));

// rows, cols, nan_prob, depth, num_trees, leaf_prob, output, threshold,
// global_bias, algo, seed, tolerance, branch comparison operator, FIL implementation, number of classes
std::vector<FilTestParams> import_auto_inputs = {
  {20000, 50, 0.05, 10, 50, 0.05, fil::output_t::RAW, 0, 0,
   fil::algo_t::ALGO_AUTO, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 15, 50, 0.05, fil::output_t::RAW, 0, 0,
   fil::algo_t::ALGO_AUTO, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 19, 50, 0.05, fil::output_t::RAW, 0, 0,
   fil::algo_t::ALGO_AUTO, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 19, 50, 0.05, fil::output_t::RAW, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  {20000, 50, 0.05, 10, 50, 0.05, fil::output_t::AVG, 0, 0,
   fil::algo_t::ALGO_AUTO, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::CATEGORICAL_LEAF, 3},
  {20000, 50, 0.05, 10, 51, 0.05, fil::output_t::CLASS, 0, 0,
   fil::algo_t::ALGO_AUTO, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::GROVE_PER_CLASS, 3},
#if 0  
  {20000, 50, 0.05, 19, 50, 0.05, fil::output_t::AVG, 0, 0,
   fil::algo_t::BATCH_TREE_REORG, 42, 2e-3f, tl::Operator::kLT,
   fil::leaf_algo_t::CATEGORICAL_LEAF, 6},
#endif
};

TEST_P(TreeliteAutoFilTest, Import) { compare(); }

INSTANTIATE_TEST_CASE_P(FilTests, TreeliteAutoFilTest,
                        testing::ValuesIn(import_auto_inputs));

// rows, cols, nan_prob, depth, num_trees, leaf_prob, output, threshold,
// global_bias, algo, seed, tolerance, branch comparison operator,
// FIL implementation, number of classes
// adjust test parameters if the sparse8 format changes
std::vector<FilTestParams> import_throw_sparse8_inputs = {
  // to many features
  {100, 20000, 0.05, 10, 50, 0.05, fil::output_t::RAW, 0, 0, fil::algo_t::NAIVE,
   42, 2e-3f, tl::Operator::kLT, fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
  // too many tree nodes
  {20000, 50, 0.05, 16, 5, 0, fil::output_t::RAW, 0, 0, fil::algo_t::NAIVE, 42,
   2e-3f, tl::Operator::kLT, fil::leaf_algo_t::FLOAT_UNARY_BINARY, 1},
};

TEST_P(TreeliteThrowSparse8FilTest, Import) { check(); }

INSTANTIATE_TEST_CASE_P(FilTests, TreeliteThrowSparse8FilTest,
                        testing::ValuesIn(import_throw_sparse8_inputs));

}  // namespace ML
