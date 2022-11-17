/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include "../../src/fil/internal.cuh"

#include <test_utils.h>

#include <cuml/fil/fil.h>

#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <test_utils.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <treelite/c_api.h>
#include <treelite/frontend.h>
#include <treelite/tree.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <utility>

#define TL_CPP_CHECK(call) ASSERT(int(call) >= 0, "treelite call error")

namespace ML {

namespace tl  = treelite;
namespace tlf = treelite::frontend;
using namespace fil;

struct FilTestParams {
  // input data parameters
  int num_rows   = 20'000;
  int num_cols   = 50;
  float nan_prob = 0.05;
  // forest parameters
  int depth       = 8;
  int num_trees   = 50;
  float leaf_prob = 0.05;
  // below, categorical nodes means categorical inner nodes
  // probability that a node is categorical (given that its feature is categorical)
  float node_categorical_prob = 0.0f;
  // probability that a feature is categorical (pertains to data generation, can
  // still be interpreted as numerical by a node)
  float feature_categorical_prob = 0.0f;
  // during model creation, how often categories < fid_num_cats are marked as matching?
  float cat_match_prob = 0.5f;
  // Order Of Magnitude for maximum matching category for categorical nodes
  float max_magnitude_of_matching_cat = 1.0f;
  // output parameters
  output_t output    = output_t::RAW;
  double threshold   = 0.0f;
  double global_bias = 0.0f;
  // runtime parameters
  int blocks_per_sm       = 0;
  int threads_per_tree    = 1;
  int n_items             = 0;
  algo_t algo             = algo_t::NAIVE;
  int seed                = 42;
  float tolerance         = 2e-3f;
  bool print_forest_shape = false;
  // treelite parameters, only used for treelite tests
  tl::Operator op       = tl::Operator::kLT;
  leaf_algo_t leaf_algo = leaf_algo_t::FLOAT_UNARY_BINARY;
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
  int num_classes = 1;

  size_t num_proba_outputs() { return num_rows * std::max(num_classes, 2); }
  size_t num_preds_outputs() { return num_rows; }
};

std::string output2str(fil::output_t output)
{
  if (output == fil::RAW) return "RAW";
  std::string s = "";
  if (output & fil::AVG) s += "| AVG";
  if (output & fil::CLASS) s += "| CLASS";
  if (output & fil::SIGMOID) s += "| SIGMOID";
  if (output & fil::SOFTMAX) s += "| SOFTMAX";
  return s;
}

std::ostream& operator<<(std::ostream& os, const FilTestParams& ps)
{
  os << "num_rows = " << ps.num_rows << ", num_cols = " << ps.num_cols
     << ", nan_prob = " << ps.nan_prob << ", depth = " << ps.depth
     << ", num_trees = " << ps.num_trees << ", leaf_prob = " << ps.leaf_prob
     << ", output = " << output2str(ps.output) << ", threshold = " << ps.threshold
     << ", threads_per_tree = " << ps.threads_per_tree << ", n_items = " << ps.n_items
     << ", blocks_per_sm = " << ps.blocks_per_sm << ", algo = " << ps.algo << ", seed = " << ps.seed
     << ", tolerance = " << ps.tolerance << ", op = " << tl::OpName(ps.op)
     << ", global_bias = " << ps.global_bias << ", leaf_algo = " << ps.leaf_algo
     << ", num_classes = " << ps.num_classes
     << ", node_categorical_prob = " << ps.node_categorical_prob
     << ", feature_categorical_prob = " << ps.feature_categorical_prob
     << ", cat_match_prob = " << ps.cat_match_prob
     << ", max_magnitude_of_matching_cat = " << ps.max_magnitude_of_matching_cat;
  return os;
}

template <typename real_t>
__global__ void nan_kernel(real_t* data, const bool* mask, int len, real_t nan)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= len) return;
  if (!mask[tid]) data[tid] = nan;
}

template <typename real_t>
real_t sigmoid(real_t x)
{
  return real_t(1) / (real_t(1) + exp(-x));
}

void hard_clipped_bernoulli(
  raft::random::Rng rng, float* d, std::size_t n_vals, float prob_of_zero, cudaStream_t stream)
{
  rng.uniform(d, n_vals, 0.0f, 1.0f, stream);
  thrust::transform(
    thrust::cuda::par.on(stream), d, d + n_vals, d, [=] __device__(float uniform_0_1) -> float {
      // if prob_of_zero == 0.0f, we should never generate a zero
      if (prob_of_zero == 0.0f) return 1.0f;
      float truly_0_1 = fmax(fmin(uniform_0_1, 1.0f), 0.0f);
      // if prob_of_zero == 1.0f, we should never generate a one, hence ">"
      return truly_0_1 > prob_of_zero ? 1.0f : 0.0f;
    });
}

template <typename real_t>
struct replace_some_floating_with_categorical {
  float* fid_num_cats_d;
  int num_cols;
  __device__ real_t operator()(real_t data, int data_idx)
  {
    auto fid_num_cats = static_cast<real_t>(fid_num_cats_d[data_idx % num_cols]);
    if (fid_num_cats == real_t(0)) return data;
    // Transform `data` from (uniform on) [-1.0, 1.0] into [-fid_num_cats-3, fid_num_cats+3].
    real_t tmp = data * (fid_num_cats + real_t(3));
    // Also test invalid (negative and above fid_num_cats) categories: samples within
    // [fid_num_cats+2.5, fid_num_cats+3) and opposite will test infinite floats as categorical.
    if (tmp + fid_num_cats < real_t(-2.5f)) return -std::numeric_limits<real_t>::infinity();
    if (tmp - fid_num_cats > real_t(+2.5f)) return +std::numeric_limits<real_t>::infinity();
    // Samples within [fid_num_cats+2, fid_num_cats+2.5) (and their negative counterparts) will
    // test huge invalid categories.
    if (tmp + fid_num_cats < real_t(-2.0f)) tmp -= real_t(MAX_FIL_INT_FLOAT);
    if (tmp - fid_num_cats > real_t(+2.0f)) tmp += real_t(MAX_FIL_INT_FLOAT);
    // Samples within [0, fid_num_cats+2) will be valid categories, rounded towards 0 with a cast.
    // Negative categories are always invalid. For correct interpretation, see
    // cpp/src/fil/internal.cuh `int category_matches(node_t node, float category)`
    return tmp;
  }
};

template <typename real_t>
__global__ void floats_to_bit_stream_k(uint8_t* dst, real_t* src, std::size_t size)
{
  std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= size) return;
  int byte = 0;
#pragma unroll
  for (int i = 0; i < BITS_PER_BYTE; ++i)
    byte |= (int)src[idx * BITS_PER_BYTE + i] << i;
  dst[idx] = byte;
}

template <typename real_t>
void adjust_threshold_to_treelite(
  real_t* pthreshold, int* tl_left, int* tl_right, bool* default_left, tl::Operator comparison_op)
{
  // in treelite (take left node if val [op] threshold),
  // the meaning of the condition is reversed compared to FIL;
  // thus, "<" in treelite corresonds to comparison ">=" used by FIL
  // https://github.com/dmlc/treelite/blob/master/include/treelite/tree.h#L243
  // TODO(levsnv): remove workaround once confirmed to work with empty category lists in Treelite
  if (isnan(*pthreshold)) {
    std::swap(*tl_left, *tl_right);
    *default_left = !*default_left;
    return;
  }
  switch (comparison_op) {
    case tl::Operator::kLT: break;
    case tl::Operator::kLE:
      // x <= y is equivalent to x < y', where y' is the next representable float
      *pthreshold = std::nextafterf(*pthreshold, -std::numeric_limits<real_t>::infinity());
      break;
    case tl::Operator::kGT:
      // x > y is equivalent to x >= y', where y' is the next representable float
      // left and right still need to be swapped
      *pthreshold = std::nextafterf(*pthreshold, -std::numeric_limits<real_t>::infinity());
    case tl::Operator::kGE:
      // swap left and right
      std::swap(*tl_left, *tl_right);
      *default_left = !*default_left;
      break;
    default: ASSERT(false, "only <, >, <= and >= comparisons are supported");
  }
}

template <typename real_t>
class BaseFilTest : public testing::TestWithParam<FilTestParams> {
 public:
  BaseFilTest()
    : ps(::testing::TestWithParam<FilTestParams>::GetParam()),
      stream(handle.get_stream()),
      preds_d(0, stream),
      want_preds_d(0, stream),
      want_proba_d(0, stream),
      data_d(ps.num_rows * ps.num_cols, stream),
      proba_d(0, stream)
  {
  }

 protected:
  void setup_helper()
  {
    generate_forest();
    generate_data();
    predict_on_cpu();
    predict_on_gpu();
  }

  void SetUp() override { setup_helper(); }

  void generate_forest()
  {
    auto stream = handle.get_stream();

    size_t num_nodes = forest_num_nodes();

    // helper data
    rmm::device_uvector<int> weights_int_d(num_nodes, stream);
    rmm::device_uvector<real_t> weights_real_d(num_nodes, stream);
    rmm::device_uvector<real_t> thresholds_d(num_nodes, stream);
    rmm::device_uvector<bool> def_lefts_d(num_nodes, stream);
    rmm::device_uvector<bool> is_leafs_d(num_nodes, stream);
    rmm::device_uvector<float> is_categoricals_d(num_nodes, stream);

    fids_d.resize(num_nodes, stream);
    fid_num_cats_d.resize(ps.num_cols, stream);

    // generate on-GPU random data
    raft::random::Rng r(ps.seed);
    if (ps.leaf_algo == fil::leaf_algo_t::CATEGORICAL_LEAF) {
      // [0..num_classes)
      r.uniformInt(weights_int_d.data(), num_nodes, 0, ps.num_classes, stream);
    } else if (ps.leaf_algo == fil::leaf_algo_t::VECTOR_LEAF) {
      std::mt19937 gen(3);
      std::uniform_real_distribution<real_t> dist(0, 1);
      vector_leaf.resize(num_nodes * ps.num_classes);
      for (size_t i = 0; i < vector_leaf.size(); i++) {
        vector_leaf[i] = dist(gen);
      }
      // Normalise probabilities to 1
      for (size_t i = 0; i < vector_leaf.size(); i += ps.num_classes) {
        auto sum = std::accumulate(&vector_leaf[i], &vector_leaf[i + ps.num_classes], real_t(0));
        for (size_t j = i; j < i + ps.num_classes; j++) {
          vector_leaf[j] /= sum;
        }
      }
    } else {
      r.uniform(weights_real_d.data(), num_nodes, real_t(-1), real_t(1), stream);
    }
    r.uniform(thresholds_d.data(), num_nodes, real_t(-1), real_t(1), stream);
    r.uniformInt(fids_d.data(), num_nodes, 0, ps.num_cols, stream);
    r.bernoulli(def_lefts_d.data(), num_nodes, 0.5f, stream);
    r.bernoulli(is_leafs_d.data(), num_nodes, ps.leaf_prob, stream);
    hard_clipped_bernoulli(
      r, is_categoricals_d.data(), num_nodes, 1.0f - ps.node_categorical_prob, stream);

    // copy data to host
    std::vector<real_t> thresholds_h(num_nodes), weights_real_h(num_nodes);
    std::vector<float> is_categoricals_h(num_nodes);
    std::vector<int> weights_int_h(num_nodes), fids_h(num_nodes), node_cat_set(num_nodes);
    std::vector<float> fid_num_cats_h(ps.num_cols);
    std::vector<bool> feature_categorical(ps.num_cols);
    // bool vectors are not guaranteed to be stored byte-per-value
    bool* def_lefts_h = new bool[num_nodes];
    bool* is_leafs_h  = new bool[num_nodes];

    // uniformily distributed in orders of magnitude: smaller models which
    // still stress large bitfields.
    // up to 10**ps.max_magnitude_of_matching_cat (only if feature is categorical, else -1)
    std::mt19937 gen(ps.seed);
    std::uniform_real_distribution mmc(-1.0f, ps.max_magnitude_of_matching_cat);
    std::bernoulli_distribution fc(ps.feature_categorical_prob);
    cat_sets_h.fid_num_cats.resize(ps.num_cols);
    for (int fid = 0; fid < ps.num_cols; ++fid) {
      feature_categorical[fid] = fc(gen);
      if (feature_categorical[fid]) {
        // categorical features will never have fid_num_cats == 0
        float mm = ceil(pow(10, mmc(gen)));
        ASSERT(mm < float(MAX_FIL_INT_FLOAT),
               "internal error: max_magnitude_of_matching_cat %f is too large",
               ps.max_magnitude_of_matching_cat);
        cat_sets_h.fid_num_cats[fid] = mm;
      } else {
        cat_sets_h.fid_num_cats[fid] = 0.0f;
      }
    }
    raft::update_host(weights_int_h.data(), weights_int_d.data(), num_nodes, stream);
    raft::update_host(weights_real_h.data(), weights_real_d.data(), num_nodes, stream);
    raft::update_host(thresholds_h.data(), thresholds_d.data(), num_nodes, stream);
    raft::update_host(fids_h.data(), fids_d.data(), num_nodes, stream);
    raft::update_host(def_lefts_h, def_lefts_d.data(), num_nodes, stream);
    raft::update_host(is_leafs_h, is_leafs_d.data(), num_nodes, stream);
    raft::update_host(is_categoricals_h.data(), is_categoricals_d.data(), num_nodes, stream);
    handle.sync_stream();

    // mark leaves
    for (int i = 0; i < ps.num_trees; ++i) {
      int num_tree_nodes = tree_num_nodes();
      size_t leaf_start  = num_tree_nodes * i + num_tree_nodes / 2;
      size_t leaf_end    = num_tree_nodes * (i + 1);
      for (size_t j = leaf_start; j < leaf_end; ++j) {
        is_leafs_h[j] = true;
      }
    }

    // count nodes for each feature id, while splitting the sets between nodes
    std::size_t bit_pool_size = 0;
    cat_sets_h.n_nodes        = std::vector<std::size_t>(ps.num_cols, 0);
    for (std::size_t node_id = 0; node_id < num_nodes; ++node_id) {
      int fid = fids_h[node_id];

      if (!feature_categorical[fid] || is_leafs_h[node_id]) is_categoricals_h[node_id] = 0.0f;

      if (is_categoricals_h[node_id] == 1.0f) {
        // might allocate a categorical set for an unreachable inner node. That's OK.
        ++cat_sets_h.n_nodes[fid];
        node_cat_set[node_id] = bit_pool_size;
        bit_pool_size += cat_sets_h.accessor().sizeof_mask(fid);
      }
    }
    cat_sets_h.bits.resize(bit_pool_size);
    raft::update_device(fid_num_cats_d.data(), cat_sets_h.fid_num_cats.data(), ps.num_cols, stream);
    // calculate sizes and allocate arrays for category sets
    // fill category sets
    // there is a faster trick with a 256-byte LUT, but we can implement it later if the tests
    // become too slow
    rmm::device_uvector<float> bits_precursor_d(cat_sets_h.bits.size() * BITS_PER_BYTE, stream);
    rmm::device_uvector<uint8_t> bits_d(cat_sets_h.bits.size(), stream);
    if (cat_sets_h.bits.size() != 0) {
      hard_clipped_bernoulli(r,
                             bits_precursor_d.data(),
                             cat_sets_h.bits.size() * BITS_PER_BYTE,
                             1.0f - ps.cat_match_prob,
                             stream);
      floats_to_bit_stream_k<<<raft::ceildiv(cat_sets_h.bits.size(), (std::size_t)FIL_TPB),
                               FIL_TPB,
                               0,
                               stream>>>(
        bits_d.data(), bits_precursor_d.data(), cat_sets_h.bits.size());
      raft::update_host(cat_sets_h.bits.data(), bits_d.data(), cat_sets_h.bits.size(), stream);
    }

    // initialize nodes
    nodes.resize(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
      fil::val_t<real_t> w;
      switch (ps.leaf_algo) {
        case fil::leaf_algo_t::CATEGORICAL_LEAF: w.idx = weights_int_h[i]; break;
        case fil::leaf_algo_t::FLOAT_UNARY_BINARY:
        case fil::leaf_algo_t::GROVE_PER_CLASS:
          // not relying on fil::val_t<float> internals
          // merely that we copied floats into weights_h earlier
          w.f = weights_real_h[i];
          break;
        case fil::leaf_algo_t::VECTOR_LEAF: w.idx = i; break;
        default: ASSERT(false, "internal error: invalid ps.leaf_algo");
      }
      // make sure nodes are categorical only when their feature ID is categorical
      bool is_categorical = is_categoricals_h[i] == 1.0f;
      val_t<real_t> split;
      if (is_categorical)
        split.idx = node_cat_set[i];
      else
        split.f = thresholds_h[i];
      nodes[i] =
        fil::dense_node<real_t>(w, split, fids_h[i], def_lefts_h[i], is_leafs_h[i], is_categorical);
    }

    // clean up
    delete[] def_lefts_h;
    delete[] is_leafs_h;
    // cat_sets_h.bits and fid_num_cats_d are now visible to host
  }

  void generate_data()
  {
    auto stream = handle.get_stream();
    // allocate arrays
    size_t num_data = ps.num_rows * ps.num_cols;
    rmm::device_uvector<bool> mask_d(num_data, stream);

    // generate random data
    raft::random::Rng r(ps.seed);
    r.uniform(data_d.data(), num_data, real_t(-1), real_t(1), stream);
    thrust::transform(
      thrust::cuda::par.on(stream),
      data_d.data(),
      data_d.data() + num_data,
      thrust::counting_iterator(0),
      data_d.data(),
      replace_some_floating_with_categorical<real_t>{fid_num_cats_d.data(), ps.num_cols});
    r.bernoulli(mask_d.data(), num_data, 1 - ps.nan_prob, stream);
    int tpb = 256;
    nan_kernel<<<raft::ceildiv(int(num_data), tpb), tpb, 0, stream>>>(
      data_d.data(), mask_d.data(), num_data, std::numeric_limits<real_t>::quiet_NaN());
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // copy to host
    data_h.resize(num_data);
    raft::update_host(data_h.data(), data_d.data(), num_data, stream);
    handle.sync_stream();
  }

  void apply_softmax(real_t* class_scores)
  {
    real_t max = *std::max_element(class_scores, &class_scores[ps.num_classes]);
    for (int i = 0; i < ps.num_classes; ++i)
      class_scores[i] = exp(class_scores[i] - max);
    real_t sum = std::accumulate(class_scores, &class_scores[ps.num_classes], real_t(0));
    for (int i = 0; i < ps.num_classes; ++i)
      class_scores[i] /= sum;
  }

  void transform(real_t f, real_t& proba, real_t& output)
  {
    if ((ps.output & fil::output_t::AVG) != 0) {
      if (ps.leaf_algo == fil::leaf_algo_t::GROVE_PER_CLASS) {
        f /= ps.num_trees / ps.num_classes;
      } else {
        f *= real_t(1) / ps.num_trees;
      }
    }
    f += ps.global_bias;
    if ((ps.output & fil::output_t::SIGMOID) != 0) { f = sigmoid(f); }
    proba = f;
    if ((ps.output & fil::output_t::CLASS) != 0) { f = f > ps.threshold ? real_t(1) : real_t(0); }
    output = f;
  }

  void complement(real_t* proba) { proba[0] = real_t(1) - proba[1]; }

  void predict_on_cpu()
  {
    auto stream = handle.get_stream();
    // predict on host
    std::vector<real_t> want_preds_h(ps.num_preds_outputs());
    want_proba_h.resize(ps.num_proba_outputs());
    int num_nodes = tree_num_nodes();
    std::vector<real_t> class_scores(ps.num_classes);
    // we use tree_base::child_index() on CPU
    tree_base base{cat_sets_h.accessor()};
    switch (ps.leaf_algo) {
      case fil::leaf_algo_t::FLOAT_UNARY_BINARY:
        for (int i = 0; i < ps.num_rows; ++i) {
          real_t pred = 0;
          for (int j = 0; j < ps.num_trees; ++j) {
            pred += infer_one_tree(&nodes[j * num_nodes], &data_h[i * ps.num_cols], base).f;
          }
          transform(pred, want_proba_h[i * 2 + 1], want_preds_h[i]);
          complement(&(want_proba_h[i * 2]));
        }
        break;
      case fil::leaf_algo_t::GROVE_PER_CLASS:
        for (int row = 0; row < ps.num_rows; ++row) {
          std::fill(class_scores.begin(), class_scores.end(), real_t(0));
          for (int tree = 0; tree < ps.num_trees; ++tree) {
            class_scores[tree % ps.num_classes] +=
              infer_one_tree(&nodes[tree * num_nodes], &data_h[row * ps.num_cols], base).f;
          }
          want_preds_h[row] =
            std::max_element(class_scores.begin(), class_scores.end()) - class_scores.begin();
          for (int c = 0; c < ps.num_classes; ++c) {
            real_t thresholded_proba;  // not used;
            transform(class_scores[c], want_proba_h[row * ps.num_classes + c], thresholded_proba);
          }
          if ((ps.output & fil::output_t::SOFTMAX) != 0)
            apply_softmax(&want_proba_h[row * ps.num_classes]);
        }
        break;
      case fil::leaf_algo_t::CATEGORICAL_LEAF: {
        std::vector<int> class_votes(ps.num_classes);
        for (int r = 0; r < ps.num_rows; ++r) {
          std::fill(class_votes.begin(), class_votes.end(), 0);
          for (int j = 0; j < ps.num_trees; ++j) {
            int class_label =
              infer_one_tree(&nodes[j * num_nodes], &data_h[r * ps.num_cols], base).idx;
            ++class_votes[class_label];
          }
          for (int c = 0; c < ps.num_classes; ++c) {
            real_t thresholded_proba;  // not used; do argmax instead
            transform(class_votes[c], want_proba_h[r * ps.num_classes + c], thresholded_proba);
          }
          want_preds_h[r] =
            std::max_element(class_votes.begin(), class_votes.end()) - class_votes.begin();
        }
        break;
      }
      case fil::leaf_algo_t::VECTOR_LEAF:
        for (int r = 0; r < ps.num_rows; ++r) {
          std::vector<real_t> class_probabilities(ps.num_classes);
          for (int j = 0; j < ps.num_trees; ++j) {
            int vector_index =
              infer_one_tree(&nodes[j * num_nodes], &data_h[r * ps.num_cols], base).idx;
            real_t sum = 0;
            for (int k = 0; k < ps.num_classes; k++) {
              class_probabilities[k] += vector_leaf[vector_index * ps.num_classes + k];
              sum += vector_leaf[vector_index * ps.num_classes + k];
            }
            ASSERT_LE(std::abs(sum - real_t(1)), real_t(1e-5));
          }

          for (int c = 0; c < ps.num_classes; ++c) {
            want_proba_h[r * ps.num_classes + c] = class_probabilities[c];
          }
          want_preds_h[r] =
            std::max_element(class_probabilities.begin(), class_probabilities.end()) -
            class_probabilities.begin();
        }
        break;
      case fil::leaf_algo_t::GROVE_PER_CLASS_FEW_CLASSES:
      case fil::leaf_algo_t::GROVE_PER_CLASS_MANY_CLASSES: break;
    }

    // copy to GPU
    want_preds_d.resize(ps.num_preds_outputs(), stream);
    want_proba_d.resize(ps.num_proba_outputs(), stream);
    raft::update_device(want_preds_d.data(), want_preds_h.data(), ps.num_preds_outputs(), stream);
    raft::update_device(want_proba_d.data(), want_proba_h.data(), ps.num_proba_outputs(), stream);
    handle.sync_stream();
  }

  virtual void init_forest(fil::forest_t<real_t>* pforest) = 0;

  void predict_on_gpu()
  {
    auto stream                  = handle.get_stream();
    fil::forest_t<real_t> forest = nullptr;
    init_forest(&forest);

    // predict
    preds_d.resize(ps.num_preds_outputs(), stream);
    proba_d.resize(ps.num_proba_outputs(), stream);
    fil::predict(handle, forest, preds_d.data(), data_d.data(), ps.num_rows);
    fil::predict(handle, forest, proba_d.data(), data_d.data(), ps.num_rows, true);
    handle.sync_stream();

    // cleanup
    fil::free(handle, forest);
  }

  void compare()
  {
    ASSERT_TRUE(raft::devArrMatch(want_proba_d.data(),
                                  proba_d.data(),
                                  ps.num_proba_outputs(),
                                  raft::CompareApprox<real_t>(ps.tolerance),
                                  stream));
    float tolerance = ps.leaf_algo == fil::leaf_algo_t::FLOAT_UNARY_BINARY
                        ? ps.tolerance
                        : std::numeric_limits<real_t>::epsilon();
    // in multi-class prediction, floats represent the most likely class
    // and would be generated by converting an int to float
    ASSERT_TRUE(raft::devArrMatch(want_preds_d.data(),
                                  preds_d.data(),
                                  ps.num_rows,
                                  raft::CompareApprox<real_t>(tolerance),
                                  stream));
  }

  fil::val_t<real_t> infer_one_tree(fil::dense_node<real_t>* root,
                                    real_t* data,
                                    const tree_base& tree)
  {
    int curr = 0;
    fil::val_t<real_t> output{.f = 0.0f};
    for (;;) {
      const fil::dense_node<real_t>& node = root[curr];
      if (node.is_leaf()) return node.template output<val_t<real_t>>();
      real_t val = data[node.fid()];
      curr       = tree.child_index<true>(node, curr, val);
    }
    return output;
  }

  int tree_num_nodes() { return (1 << (ps.depth + 1)) - 1; }

  int forest_num_nodes() { return tree_num_nodes() * ps.num_trees; }

  // parameters
  FilTestParams ps;
  raft::handle_t handle;
  cudaStream_t stream = 0;

  // predictions
  rmm::device_uvector<real_t> preds_d;
  rmm::device_uvector<real_t> proba_d;
  rmm::device_uvector<real_t> want_preds_d;
  rmm::device_uvector<real_t> want_proba_d;

  // input data
  rmm::device_uvector<real_t> data_d;
  std::vector<real_t> data_h;
  std::vector<real_t> want_proba_h;

  // forest data
  std::vector<fil::dense_node<real_t>> nodes;
  std::vector<real_t> vector_leaf;
  cat_sets_owner cat_sets_h;
  rmm::device_uvector<int> fids_d           = rmm::device_uvector<int>(0, cudaStream_t());
  rmm::device_uvector<float> fid_num_cats_d = rmm::device_uvector<float>(0, cudaStream_t());
};

template <typename fil_node_t>
class BasePredictFilTest : public BaseFilTest<typename fil_node_t::real_type> {
  using real_t = typename fil_node_t::real_type;

 protected:
  void dense2sparse_node(const fil::dense_node<real_t>* dense_root,
                         int i_dense,
                         int i_sparse_root,
                         int i_sparse)
  {
    const fil::dense_node<real_t>& node = dense_root[i_dense];
    if (node.is_leaf()) {
      // leaf sparse node
      sparse_nodes[i_sparse] = fil_node_t(node.template output<fil::val_t<real_t>>(),
                                          {},
                                          node.fid(),
                                          node.def_left(),
                                          node.is_leaf(),
                                          false,
                                          0);
      return;
    }
    // inner sparse node
    // reserve space for children
    int left_index = sparse_nodes.size();
    sparse_nodes.push_back(fil_node_t());
    sparse_nodes.push_back(fil_node_t());
    sparse_nodes[i_sparse] = fil_node_t({},
                                        node.split(),
                                        node.fid(),
                                        node.def_left(),
                                        node.is_leaf(),
                                        node.is_categorical(),
                                        left_index - i_sparse_root);
    dense2sparse_node(dense_root, 2 * i_dense + 1, i_sparse_root, left_index);
    dense2sparse_node(dense_root, 2 * i_dense + 2, i_sparse_root, left_index + 1);
  }

  void dense2sparse_tree(const fil::dense_node<real_t>* dense_root)
  {
    int i_sparse_root = sparse_nodes.size();
    sparse_nodes.push_back(fil_node_t());
    dense2sparse_node(dense_root, 0, i_sparse_root, i_sparse_root);
    trees.push_back(i_sparse_root);
  }

  void dense2sparse()
  {
    for (int tree = 0; tree < this->ps.num_trees; ++tree) {
      dense2sparse_tree(&this->nodes[tree * this->tree_num_nodes()]);
    }
  }

  void init_forest(fil::forest_t<real_t>* pforest) override
  {
    constexpr bool IS_DENSE = node_traits<fil_node_t>::IS_DENSE;
    std::vector<fil_node_t> init_nodes;
    if constexpr (!IS_DENSE) {
      dense2sparse();
      init_nodes = sparse_nodes;
    } else {
      init_nodes = this->nodes;
    }
    ASSERT(init_nodes.size() < std::size_t(INT_MAX), "generated too many nodes");

    // init FIL model
    fil::forest_params_t fil_params = {
      .num_nodes        = static_cast<int>(init_nodes.size()),
      .depth            = this->ps.depth,
      .num_trees        = this->ps.num_trees,
      .num_cols         = this->ps.num_cols,
      .leaf_algo        = this->ps.leaf_algo,
      .algo             = this->ps.algo,
      .output           = this->ps.output,
      .threshold        = this->ps.threshold,
      .global_bias      = this->ps.global_bias,
      .num_classes      = this->ps.num_classes,
      .blocks_per_sm    = this->ps.blocks_per_sm,
      .threads_per_tree = this->ps.threads_per_tree,
      .n_items          = this->ps.n_items,
    };

    fil::init(this->handle,
              pforest,
              this->cat_sets_h.accessor(),
              this->vector_leaf,
              trees.data(),
              init_nodes.data(),
              &fil_params);
  }
  std::vector<fil_node_t> sparse_nodes;
  std::vector<int> trees;
};

using PredictDenseFloat32FilTest    = BasePredictFilTest<fil::dense_node<float>>;
using PredictDenseFloat64FilTest    = BasePredictFilTest<fil::dense_node<double>>;
using PredictSparse16Float32FilTest = BasePredictFilTest<fil::sparse_node16<float>>;
using PredictSparse16Float64FilTest = BasePredictFilTest<fil::sparse_node16<double>>;
using PredictSparse8FilTest         = BasePredictFilTest<fil::sparse_node8>;

template <typename real_t>
class TreeliteFilTest : public BaseFilTest<real_t> {
 protected:
  /** adds nodes[node] of tree starting at index root to builder
      at index at *pkey, increments *pkey,
      and returns the treelite key of the node */
  int node_to_treelite(tlf::TreeBuilder* builder, int* pkey, int root, int node)
  {
    int key = (*pkey)++;
    builder->CreateNode(key);
    const fil::dense_node<real_t>& dense_node = this->nodes[node];
    std::vector<std::uint32_t> left_categories;
    if (dense_node.is_leaf()) {
      switch (this->ps.leaf_algo) {
        case fil::leaf_algo_t::FLOAT_UNARY_BINARY:
        case fil::leaf_algo_t::GROVE_PER_CLASS:
          // default is fil::FLOAT_UNARY_BINARY
          builder->SetLeafNode(key, tlf::Value::Create(dense_node.template output<real_t>()));
          break;
        case fil::leaf_algo_t::CATEGORICAL_LEAF: {
          std::vector<tlf::Value> vec(this->ps.num_classes);
          for (int i = 0; i < this->ps.num_classes; ++i) {
            vec[i] =
              tlf::Value::Create(i == dense_node.template output<int>() ? real_t(1) : real_t(0));
          }
          builder->SetLeafVectorNode(key, vec);
          break;
        }
        case fil::leaf_algo_t::VECTOR_LEAF: {
          std::vector<tlf::Value> vec(this->ps.num_classes);
          for (int i = 0; i < this->ps.num_classes; ++i) {
            auto idx = dense_node.template output<int>();
            vec[i]   = tlf::Value::Create(this->vector_leaf[idx * this->ps.num_classes + i]);
          }
          builder->SetLeafVectorNode(key, vec);
          break;
        }
        case fil::leaf_algo_t::GROVE_PER_CLASS_FEW_CLASSES:
        case fil::leaf_algo_t::GROVE_PER_CLASS_MANY_CLASSES: break;
      }
    } else {
      int left          = root + 2 * (node - root) + 1;
      int right         = root + 2 * (node - root) + 2;
      bool default_left = dense_node.def_left();
      real_t threshold  = dense_node.is_categorical() ? std::numeric_limits<real_t>::quiet_NaN()
                                                      : dense_node.thresh();
      if (dense_node.is_categorical()) {
        uint8_t byte = 0;
        for (int category = 0;
             category < static_cast<int>(this->cat_sets_h.fid_num_cats[dense_node.fid()]);
             ++category) {
          if (category % BITS_PER_BYTE == 0) {
            byte = this->cat_sets_h.bits[dense_node.set() + category / BITS_PER_BYTE];
          }
          if ((byte & (1 << (category % BITS_PER_BYTE))) != 0) {
            left_categories.push_back(category);
          }
        }
      }
      int left_key  = node_to_treelite(builder, pkey, root, left);
      int right_key = node_to_treelite(builder, pkey, root, right);
      // TODO(levsnv): remove workaround once confirmed to work with empty category lists in
      // Treelite
      if (!left_categories.empty() && dense_node.is_categorical()) {
        // Treelite builder APIs don't allow to set categorical_split_right_child
        // (which child the categories pertain to). Only the Tree API allows that.
        // in FIL, categories always pertain to the right child, and the default in treelite
        // is left categories in SetCategoricalTestNode
        std::swap(left_key, right_key);
        default_left = !default_left;
        builder->SetCategoricalTestNode(
          key, dense_node.fid(), left_categories, default_left, left_key, right_key);
      } else {
        adjust_threshold_to_treelite(&threshold, &left_key, &right_key, &default_left, this->ps.op);
        builder->SetNumericalTestNode(key,
                                      dense_node.fid(),
                                      this->ps.op,
                                      tlf::Value::Create(threshold),
                                      default_left,
                                      left_key,
                                      right_key);
      }
    }
    return key;
  }

  void init_forest_impl(fil::forest_t<real_t>* pforest, fil::storage_type_t storage_type)
  {
    auto stream             = this->handle.get_stream();
    bool random_forest_flag = (this->ps.output & fil::output_t::AVG) != 0;
    tl::TypeInfo tl_type_info =
      std::is_same_v<real_t, float> ? tl::TypeInfo::kFloat32 : tl::TypeInfo::kFloat64;
    int treelite_num_classes =
      this->ps.leaf_algo == fil::leaf_algo_t::FLOAT_UNARY_BINARY ? 1 : this->ps.num_classes;
    std::unique_ptr<tlf::ModelBuilder> model_builder(new tlf::ModelBuilder(
      this->ps.num_cols, treelite_num_classes, random_forest_flag, tl_type_info, tl_type_info));

    // prediction transform
    if ((this->ps.output & fil::output_t::SIGMOID) != 0) {
      if (this->ps.num_classes > 2)
        model_builder->SetModelParam("pred_transform", "multiclass_ova");
      else
        model_builder->SetModelParam("pred_transform", "sigmoid");
    } else if (this->ps.leaf_algo != fil::leaf_algo_t::FLOAT_UNARY_BINARY) {
      model_builder->SetModelParam("pred_transform", "max_index");
      this->ps.output = fil::output_t(this->ps.output | fil::output_t::CLASS);
    } else if (this->ps.leaf_algo == GROVE_PER_CLASS) {
      model_builder->SetModelParam("pred_transform", "identity_multiclass");
    } else {
      model_builder->SetModelParam("pred_transform", "identity");
    }

    // global bias
    char* global_bias_str = nullptr;
    ASSERT(asprintf(&global_bias_str, "%f", double(this->ps.global_bias)) > 0,
           "cannot convert global_bias into a string");
    model_builder->SetModelParam("global_bias", global_bias_str);
    ::free(global_bias_str);

    // build the trees
    for (int i_tree = 0; i_tree < this->ps.num_trees; ++i_tree) {
      tlf::TreeBuilder* tree_builder = new tlf::TreeBuilder(tl_type_info, tl_type_info);
      int key_counter                = 0;
      int root                       = i_tree * this->tree_num_nodes();
      int root_key                   = node_to_treelite(tree_builder, &key_counter, root, root);
      tree_builder->SetRootNode(root_key);
      // InsertTree() consumes tree_builder
      TL_CPP_CHECK(model_builder->InsertTree(tree_builder));
    }

    // commit the model
    std::unique_ptr<tl::Model> model = model_builder->CommitModel();

    // init FIL forest with the model
    char* forest_shape_str = nullptr;
    fil::treelite_params_t params;
    params.algo              = this->ps.algo;
    params.threshold         = this->ps.threshold;
    params.output_class      = (this->ps.output & fil::output_t::CLASS) != 0;
    params.storage_type      = storage_type;
    params.blocks_per_sm     = this->ps.blocks_per_sm;
    params.threads_per_tree  = this->ps.threads_per_tree;
    params.n_items           = this->ps.n_items;
    params.pforest_shape_str = this->ps.print_forest_shape ? &forest_shape_str : nullptr;
    params.precision         = fil::PRECISION_NATIVE;
    fil::forest_variant forest_variant;
    fil::from_treelite(this->handle, &forest_variant, (ModelHandle)model.get(), &params);
    *pforest = std::get<fil::forest_t<real_t>>(forest_variant);
    this->handle.sync_stream(stream);
    if (this->ps.print_forest_shape) {
      std::string str(forest_shape_str);
      for (const char* substr : {"model size",
                                 " MB",
                                 "Depth histogram:",
                                 "Avg nodes per tree",
                                 "Leaf depth",
                                 "Depth histogram fingerprint"}) {
        ASSERT(str.find(substr) != std::string::npos,
               "\"%s\" not found in forest shape :\n%s",
               substr,
               str.c_str());
      }
    }
    ::free(forest_shape_str);
  }
};

template <typename real_t>
class TreeliteDenseFilTest : public TreeliteFilTest<real_t> {
 protected:
  void init_forest(fil::forest_t<real_t>* pforest) override
  {
    this->init_forest_impl(pforest, fil::storage_type_t::DENSE);
  }
};

template <typename real_t>
class TreeliteSparse16FilTest : public TreeliteFilTest<real_t> {
 protected:
  void init_forest(fil::forest_t<real_t>* pforest) override
  {
    this->init_forest_impl(pforest, fil::storage_type_t::SPARSE);
  }
};

class TreeliteSparse8FilTest : public TreeliteFilTest<float> {
 protected:
  void init_forest(fil::forest_t<float>* pforest) override
  {
    this->init_forest_impl(pforest, fil::storage_type_t::SPARSE8);
  }
};

template <typename real_t>
class TreeliteAutoFilTest : public TreeliteFilTest<real_t> {
 protected:
  void init_forest(fil::forest_t<real_t>* pforest) override
  {
    this->init_forest_impl(pforest, fil::storage_type_t::AUTO);
  }
};

using TreeliteDenseFloat32FilTest    = TreeliteDenseFilTest<float>;
using TreeliteDenseFloat64FilTest    = TreeliteDenseFilTest<double>;
using TreeliteSparse16Float32FilTest = TreeliteDenseFilTest<float>;
using TreeliteSparse16Float64FilTest = TreeliteDenseFilTest<double>;
using TreeliteAutoFloat32FilTest     = TreeliteAutoFilTest<float>;
using TreeliteAutoFloat64FilTest     = TreeliteAutoFilTest<double>;

// test for failures; currently only supported for sparse8 nodes
class TreeliteThrowSparse8FilTest : public TreeliteSparse8FilTest {
 protected:
  // model import happens in check(), so this function is empty
  void SetUp() override {}

  void check() { ASSERT_THROW(setup_helper(), raft::exception); }
};

/** mechanism to use named aggregate initialization before C++20, and also use
    the struct defaults. Using it directly only works if all defaulted
    members come after ones explicitly mentioned.
**/
#define FIL_TEST_PARAMS(...)                                \
  []() {                                                    \
    struct NonDefaultFilTestParams : public FilTestParams { \
      NonDefaultFilTestParams() { __VA_ARGS__; }            \
    };                                                      \
    return FilTestParams(NonDefaultFilTestParams());        \
  }()

// kEQ is intentionally unused, and kLT is default
static const tl::Operator kLE = tl::Operator::kLE;
static const tl::Operator kGT = tl::Operator::kGT;
static const tl::Operator kGE = tl::Operator::kGE;

std::vector<FilTestParams> predict_dense_inputs = {
  FIL_TEST_PARAMS(),
  FIL_TEST_PARAMS(algo = TREE_REORG),
  FIL_TEST_PARAMS(algo = BATCH_TREE_REORG),
  FIL_TEST_PARAMS(output = SIGMOID),
  FIL_TEST_PARAMS(output = SIGMOID, algo = TREE_REORG),
  FIL_TEST_PARAMS(output = SIGMOID, algo = BATCH_TREE_REORG),
  FIL_TEST_PARAMS(output = SIGMOID_CLASS, num_classes = 2),
  FIL_TEST_PARAMS(output = SIGMOID_CLASS, algo = TREE_REORG, num_classes = 2),
  FIL_TEST_PARAMS(output = SIGMOID_CLASS, algo = BATCH_TREE_REORG, num_classes = 2),
  FIL_TEST_PARAMS(output = AVG),
  FIL_TEST_PARAMS(output = AVG, algo = TREE_REORG),
  FIL_TEST_PARAMS(output = AVG, algo = BATCH_TREE_REORG),
  FIL_TEST_PARAMS(output = AVG_CLASS, num_classes = 2),
  FIL_TEST_PARAMS(output = AVG_CLASS, algo = TREE_REORG, num_classes = 2),
  FIL_TEST_PARAMS(output = AVG_CLASS, algo = BATCH_TREE_REORG, num_classes = 2),
  FIL_TEST_PARAMS(global_bias = 0.5, algo = TREE_REORG),
  FIL_TEST_PARAMS(output = SIGMOID, global_bias = 0.5, algo = BATCH_TREE_REORG),
  FIL_TEST_PARAMS(output = AVG, global_bias = 0.5),
  FIL_TEST_PARAMS(
    output = AVG_CLASS, threshold = 1.0, global_bias = 0.5, algo = TREE_REORG, num_classes = 2),
  FIL_TEST_PARAMS(output = SIGMOID, algo = ALGO_AUTO),
  FIL_TEST_PARAMS(
    output = AVG_CLASS, algo = BATCH_TREE_REORG, leaf_algo = CATEGORICAL_LEAF, num_classes = 5),
  FIL_TEST_PARAMS(output = AVG_CLASS, num_classes = 2),
  FIL_TEST_PARAMS(algo = TREE_REORG, leaf_algo = CATEGORICAL_LEAF, num_classes = 5),
  FIL_TEST_PARAMS(output = SIGMOID, leaf_algo = CATEGORICAL_LEAF, num_classes = 7),
  FIL_TEST_PARAMS(
    global_bias = 0.5, algo = TREE_REORG, leaf_algo = CATEGORICAL_LEAF, num_classes = 4),
  FIL_TEST_PARAMS(output = AVG, global_bias = 0.5, leaf_algo = CATEGORICAL_LEAF, num_classes = 4),
  FIL_TEST_PARAMS(
    output = AVG_CLASS, algo = BATCH_TREE_REORG, leaf_algo = GROVE_PER_CLASS, num_classes = 5),
  FIL_TEST_PARAMS(algo = TREE_REORG, leaf_algo = GROVE_PER_CLASS, num_classes = 5),
  FIL_TEST_PARAMS(num_trees = 49, output = SIGMOID, leaf_algo = GROVE_PER_CLASS, num_classes = 7),
  FIL_TEST_PARAMS(num_trees   = 52,
                  global_bias = 0.5,
                  algo        = TREE_REORG,
                  leaf_algo   = GROVE_PER_CLASS,
                  num_classes = 4),
  FIL_TEST_PARAMS(
    num_trees = 52, output = AVG, global_bias = 0.5, leaf_algo = GROVE_PER_CLASS, num_classes = 4),
  FIL_TEST_PARAMS(blocks_per_sm = 1),
  FIL_TEST_PARAMS(blocks_per_sm = 4),
  FIL_TEST_PARAMS(num_classes = 3, blocks_per_sm = 1, leaf_algo = CATEGORICAL_LEAF),
  FIL_TEST_PARAMS(num_classes = 3, blocks_per_sm = 4, leaf_algo = CATEGORICAL_LEAF),
  FIL_TEST_PARAMS(num_classes = 5, blocks_per_sm = 1, leaf_algo = GROVE_PER_CLASS),
  FIL_TEST_PARAMS(num_classes = 5, blocks_per_sm = 4, leaf_algo = GROVE_PER_CLASS),
  FIL_TEST_PARAMS(
    leaf_algo = GROVE_PER_CLASS, blocks_per_sm = 1, num_trees = 512, num_classes = 512),
  FIL_TEST_PARAMS(
    leaf_algo = GROVE_PER_CLASS, blocks_per_sm = 4, num_trees = 512, num_classes = 512),
  FIL_TEST_PARAMS(num_trees = 52, output = SOFTMAX, leaf_algo = GROVE_PER_CLASS, num_classes = 4),
  FIL_TEST_PARAMS(
    num_trees = 52, output = AVG_SOFTMAX, leaf_algo = GROVE_PER_CLASS, num_classes = 4),
  FIL_TEST_PARAMS(num_trees   = 3 * (FIL_TPB + 1),
                  output      = SOFTMAX,
                  leaf_algo   = GROVE_PER_CLASS,
                  num_classes = FIL_TPB + 1),
  FIL_TEST_PARAMS(num_trees   = 3 * (FIL_TPB + 1),
                  output      = AVG_SOFTMAX,
                  leaf_algo   = GROVE_PER_CLASS,
                  num_classes = FIL_TPB + 1),
  FIL_TEST_PARAMS(num_rows  = 10'000,
                  num_cols  = 100'000,
                  depth     = 5,
                  num_trees = 1,
                  leaf_algo = FLOAT_UNARY_BINARY),
  FIL_TEST_PARAMS(num_rows    = 101,
                  num_cols    = 100'000,
                  depth       = 5,
                  num_trees   = 9,
                  algo        = BATCH_TREE_REORG,
                  leaf_algo   = GROVE_PER_CLASS,
                  num_classes = 3),
  FIL_TEST_PARAMS(num_rows    = 102,
                  num_cols    = 100'000,
                  depth       = 5,
                  num_trees   = 3 * (FIL_TPB + 1),
                  algo        = BATCH_TREE_REORG,
                  leaf_algo   = GROVE_PER_CLASS,
                  num_classes = FIL_TPB + 1),
  FIL_TEST_PARAMS(num_rows    = 103,
                  num_cols    = 100'000,
                  depth       = 5,
                  num_trees   = 1,
                  algo        = BATCH_TREE_REORG,
                  leaf_algo   = CATEGORICAL_LEAF,
                  num_classes = 3),
  // use shared memory opt-in carveout if available, or infer out of L1 cache
  FIL_TEST_PARAMS(num_rows = 103, num_cols = MAX_SHM_STD / sizeof(float) + 1024, algo = NAIVE),
  FIL_TEST_PARAMS(num_rows    = 103,
                  num_cols    = MAX_SHM_STD / sizeof(float) + 1024,
                  leaf_algo   = GROVE_PER_CLASS,
                  num_classes = 5),
  FIL_TEST_PARAMS(num_rows    = 103,
                  num_cols    = MAX_SHM_STD / sizeof(float) + 1024,
                  num_trees   = FIL_TPB + 1,
                  leaf_algo   = GROVE_PER_CLASS,
                  num_classes = FIL_TPB + 1),
  FIL_TEST_PARAMS(num_rows    = 103,
                  num_cols    = MAX_SHM_STD / sizeof(float) + 1024,
                  leaf_algo   = CATEGORICAL_LEAF,
                  num_classes = 3),
  FIL_TEST_PARAMS(algo = BATCH_TREE_REORG, threads_per_tree = 2),
  FIL_TEST_PARAMS(algo = NAIVE, threads_per_tree = 4),
  FIL_TEST_PARAMS(algo = TREE_REORG, threads_per_tree = 8),
  FIL_TEST_PARAMS(algo = ALGO_AUTO, threads_per_tree = 16),
  FIL_TEST_PARAMS(algo = BATCH_TREE_REORG, threads_per_tree = 32),
  FIL_TEST_PARAMS(algo = NAIVE, threads_per_tree = 64),
  FIL_TEST_PARAMS(algo = BATCH_TREE_REORG, threads_per_tree = 128, n_items = 3),
  FIL_TEST_PARAMS(algo = BATCH_TREE_REORG, threads_per_tree = 256),
  FIL_TEST_PARAMS(algo = TREE_REORG, threads_per_tree = 32, n_items = 1),
  FIL_TEST_PARAMS(algo = BATCH_TREE_REORG, threads_per_tree = 16, n_items = 4),
  FIL_TEST_PARAMS(algo = NAIVE, threads_per_tree = 32, n_items = 4),
  FIL_TEST_PARAMS(
    num_rows = 500, num_cols = 2000, algo = BATCH_TREE_REORG, threads_per_tree = 64, n_items = 4),
  FIL_TEST_PARAMS(leaf_algo = VECTOR_LEAF, num_classes = 2),
  FIL_TEST_PARAMS(leaf_algo = VECTOR_LEAF, num_trees = 9, num_classes = 20),
  FIL_TEST_PARAMS(num_rows    = 103,
                  num_cols    = 100'000,
                  depth       = 5,
                  num_trees   = 1,
                  algo        = BATCH_TREE_REORG,
                  leaf_algo   = VECTOR_LEAF,
                  num_classes = 3),
  FIL_TEST_PARAMS(num_rows    = 103,
                  num_cols    = 5,
                  depth       = 5,
                  num_trees   = 3,
                  leaf_algo   = VECTOR_LEAF,
                  num_classes = 4000),
  FIL_TEST_PARAMS(node_categorical_prob = 0.5, feature_categorical_prob = 0.5),
  FIL_TEST_PARAMS(
    node_categorical_prob = 1.0, feature_categorical_prob = 1.0, cat_match_prob = 1.0),
  FIL_TEST_PARAMS(
    node_categorical_prob = 1.0, feature_categorical_prob = 1.0, cat_match_prob = 0.0),
  FIL_TEST_PARAMS(depth                         = 3,
                  node_categorical_prob         = 0.5,
                  feature_categorical_prob      = 0.5,
                  max_magnitude_of_matching_cat = 5),
};

TEST_P(PredictDenseFloat32FilTest, Predict) { compare(); }
TEST_P(PredictDenseFloat64FilTest, Predict) { compare(); }

INSTANTIATE_TEST_CASE_P(FilTests,
                        PredictDenseFloat32FilTest,
                        testing::ValuesIn(predict_dense_inputs));
INSTANTIATE_TEST_CASE_P(FilTests,
                        PredictDenseFloat64FilTest,
                        testing::ValuesIn(predict_dense_inputs));

std::vector<FilTestParams> predict_sparse_inputs = {
  FIL_TEST_PARAMS(),
  FIL_TEST_PARAMS(output = SIGMOID),
  FIL_TEST_PARAMS(output = SIGMOID_CLASS, num_classes = 2),
  FIL_TEST_PARAMS(output = AVG),
  FIL_TEST_PARAMS(output = AVG_CLASS, global_bias = 0.5, num_classes = 2),
  FIL_TEST_PARAMS(global_bias = 0.5),
  FIL_TEST_PARAMS(output = SIGMOID, global_bias = 0.5),
  FIL_TEST_PARAMS(output = AVG, global_bias = 0.5),
  FIL_TEST_PARAMS(output = AVG_CLASS, threshold = 1.0, global_bias = 0.5, num_classes = 2),
  FIL_TEST_PARAMS(output = SIGMOID_CLASS, algo = ALGO_AUTO, num_classes = 2),
  FIL_TEST_PARAMS(output      = AVG_CLASS,
                  threshold   = 1.0,
                  global_bias = 0.5,
                  leaf_algo   = CATEGORICAL_LEAF,
                  num_classes = 5000),
  FIL_TEST_PARAMS(global_bias = 0.5, leaf_algo = CATEGORICAL_LEAF, num_classes = 6),
  FIL_TEST_PARAMS(output = CLASS, leaf_algo = CATEGORICAL_LEAF, num_classes = 3),
  FIL_TEST_PARAMS(leaf_algo = CATEGORICAL_LEAF, num_classes = 3),
  FIL_TEST_PARAMS(depth       = 2,
                  num_trees   = 5000,
                  output      = AVG_CLASS,
                  threshold   = 1.0,
                  global_bias = 0.5,
                  leaf_algo   = GROVE_PER_CLASS,
                  num_classes = 5000),
  FIL_TEST_PARAMS(num_trees = 60, global_bias = 0.5, leaf_algo = GROVE_PER_CLASS, num_classes = 6),
  FIL_TEST_PARAMS(num_trees = 51, output = CLASS, leaf_algo = GROVE_PER_CLASS, num_classes = 3),
  FIL_TEST_PARAMS(num_trees = 51, leaf_algo = GROVE_PER_CLASS, num_classes = 3),
  FIL_TEST_PARAMS(algo = NAIVE, threads_per_tree = 2),
  FIL_TEST_PARAMS(algo = NAIVE, threads_per_tree = 8, n_items = 1),
  FIL_TEST_PARAMS(algo = ALGO_AUTO, threads_per_tree = 16, n_items = 1),
  FIL_TEST_PARAMS(algo = ALGO_AUTO, threads_per_tree = 32),
  FIL_TEST_PARAMS(num_cols = 1, num_trees = 1, algo = NAIVE, threads_per_tree = 64, n_items = 1),
  FIL_TEST_PARAMS(num_rows = 500, num_cols = 2000, algo = NAIVE, threads_per_tree = 64),
  FIL_TEST_PARAMS(
    num_rows = 500, num_cols = 2000, algo = ALGO_AUTO, threads_per_tree = 256, n_items = 1),
  FIL_TEST_PARAMS(num_trees = 51, leaf_algo = VECTOR_LEAF, num_classes = 15),
  FIL_TEST_PARAMS(leaf_algo = VECTOR_LEAF, num_trees = 9, num_classes = 20),
  FIL_TEST_PARAMS(num_rows    = 103,
                  num_cols    = 1000,
                  depth       = 5,
                  num_trees   = 1,
                  leaf_algo   = VECTOR_LEAF,
                  num_classes = 3),
  FIL_TEST_PARAMS(num_rows    = 103,
                  num_cols    = 5,
                  depth       = 5,
                  num_trees   = 3,
                  leaf_algo   = VECTOR_LEAF,
                  num_classes = 4000),
  FIL_TEST_PARAMS(num_rows    = 103,
                  num_cols    = 5,
                  depth       = 5,
                  num_trees   = 530,
                  leaf_algo   = VECTOR_LEAF,
                  num_classes = 11),
  FIL_TEST_PARAMS(num_rows    = 103,
                  num_cols    = 5,
                  depth       = 5,
                  num_trees   = 530,
                  leaf_algo   = VECTOR_LEAF,
                  num_classes = 1111),
  FIL_TEST_PARAMS(node_categorical_prob = 0.5, feature_categorical_prob = 0.5),
  FIL_TEST_PARAMS(
    node_categorical_prob = 1.0, feature_categorical_prob = 1.0, cat_match_prob = 1.0),
  FIL_TEST_PARAMS(
    node_categorical_prob = 1.0, feature_categorical_prob = 1.0, cat_match_prob = 0.0),
  FIL_TEST_PARAMS(depth                         = 3,
                  node_categorical_prob         = 0.5,
                  feature_categorical_prob      = 0.5,
                  max_magnitude_of_matching_cat = 5),
};

TEST_P(PredictSparse16Float32FilTest, Predict) { compare(); }
TEST_P(PredictSparse16Float64FilTest, Predict) { compare(); }

INSTANTIATE_TEST_CASE_P(FilTests,
                        PredictSparse16Float32FilTest,
                        testing::ValuesIn(predict_sparse_inputs));
INSTANTIATE_TEST_CASE_P(FilTests,
                        PredictSparse16Float64FilTest,
                        testing::ValuesIn(predict_sparse_inputs));

TEST_P(PredictSparse8FilTest, Predict) { compare(); }

INSTANTIATE_TEST_CASE_P(FilTests, PredictSparse8FilTest, testing::ValuesIn(predict_sparse_inputs));

std::vector<FilTestParams> import_dense_inputs = {
  FIL_TEST_PARAMS(),
  FIL_TEST_PARAMS(output = SIGMOID, op = kLE),
  FIL_TEST_PARAMS(output = SIGMOID_CLASS, op = kGT, num_classes = 2),
  FIL_TEST_PARAMS(output = AVG, op = kGE),
  FIL_TEST_PARAMS(output = AVG_CLASS, num_classes = 2),
  FIL_TEST_PARAMS(algo = TREE_REORG, op = kLE),
  FIL_TEST_PARAMS(output = SIGMOID, algo = TREE_REORG, op = kGT),
  FIL_TEST_PARAMS(output = SIGMOID_CLASS, algo = TREE_REORG, op = kGE, num_classes = 2),
  FIL_TEST_PARAMS(output = AVG, algo = TREE_REORG),
  FIL_TEST_PARAMS(output = AVG_CLASS, algo = TREE_REORG, op = kLE, num_classes = 2),
  FIL_TEST_PARAMS(algo = BATCH_TREE_REORG),
  FIL_TEST_PARAMS(output = SIGMOID, algo = BATCH_TREE_REORG),
  FIL_TEST_PARAMS(algo = BATCH_TREE_REORG, op = kLE),
  FIL_TEST_PARAMS(output = SIGMOID, algo = BATCH_TREE_REORG, op = kLE),
  FIL_TEST_PARAMS(algo = BATCH_TREE_REORG, op = kGT),
  FIL_TEST_PARAMS(output = SIGMOID, algo = BATCH_TREE_REORG, op = kGT),
  FIL_TEST_PARAMS(algo = BATCH_TREE_REORG, op = kGE),
  FIL_TEST_PARAMS(output = SIGMOID, algo = BATCH_TREE_REORG, op = kGE),
  FIL_TEST_PARAMS(output = SIGMOID_CLASS, algo = BATCH_TREE_REORG, num_classes = 2),
  FIL_TEST_PARAMS(output = SIGMOID_CLASS, algo = BATCH_TREE_REORG, op = kLE, num_classes = 2),
  FIL_TEST_PARAMS(output = AVG, algo = BATCH_TREE_REORG),
  FIL_TEST_PARAMS(output = AVG, algo = BATCH_TREE_REORG, op = kLE),
  FIL_TEST_PARAMS(output = AVG_CLASS, algo = BATCH_TREE_REORG, op = kGT, num_classes = 2),
  FIL_TEST_PARAMS(output = AVG_CLASS, algo = BATCH_TREE_REORG, op = kGE, num_classes = 2),
  FIL_TEST_PARAMS(global_bias = 0.5, algo = TREE_REORG),
  FIL_TEST_PARAMS(output = SIGMOID, global_bias = 0.5, algo = BATCH_TREE_REORG, op = kLE),
  FIL_TEST_PARAMS(output = AVG, global_bias = 0.5, op = kGT),
  FIL_TEST_PARAMS(output      = AVG_CLASS,
                  threshold   = 1.0,
                  global_bias = 0.5,
                  algo        = TREE_REORG,
                  op          = kGE,
                  num_classes = 2),
  FIL_TEST_PARAMS(output = SIGMOID, algo = ALGO_AUTO, op = kLE),
  FIL_TEST_PARAMS(output = SIGMOID, algo = ALGO_AUTO, op = kLE),
  FIL_TEST_PARAMS(
    output = AVG, algo = BATCH_TREE_REORG, op = kGE, leaf_algo = CATEGORICAL_LEAF, num_classes = 5),
  FIL_TEST_PARAMS(
    output = AVG, algo = BATCH_TREE_REORG, op = kGT, leaf_algo = CATEGORICAL_LEAF, num_classes = 6),
  FIL_TEST_PARAMS(
    output = AVG, algo = BATCH_TREE_REORG, op = kLE, leaf_algo = CATEGORICAL_LEAF, num_classes = 3),
  FIL_TEST_PARAMS(
    output = AVG, algo = BATCH_TREE_REORG, op = kLE, leaf_algo = CATEGORICAL_LEAF, num_classes = 5),
  FIL_TEST_PARAMS(
    output = AVG_CLASS, algo = TREE_REORG, op = kLE, leaf_algo = CATEGORICAL_LEAF, num_classes = 5),
  FIL_TEST_PARAMS(
    output = AVG, algo = TREE_REORG, op = kLE, leaf_algo = CATEGORICAL_LEAF, num_classes = 7),
  FIL_TEST_PARAMS(output = AVG, leaf_algo = CATEGORICAL_LEAF, num_classes = 6),
  FIL_TEST_PARAMS(output      = CLASS,
                  algo        = BATCH_TREE_REORG,
                  op          = kGE,
                  leaf_algo   = GROVE_PER_CLASS,
                  num_classes = 5),
  FIL_TEST_PARAMS(num_trees   = 48,
                  output      = CLASS,
                  algo        = BATCH_TREE_REORG,
                  op          = kGT,
                  leaf_algo   = GROVE_PER_CLASS,
                  num_classes = 6),
  FIL_TEST_PARAMS(num_trees   = 51,
                  output      = CLASS,
                  algo        = BATCH_TREE_REORG,
                  op          = kLE,
                  leaf_algo   = GROVE_PER_CLASS,
                  num_classes = 3),
  FIL_TEST_PARAMS(output      = CLASS,
                  algo        = BATCH_TREE_REORG,
                  op          = kLE,
                  leaf_algo   = GROVE_PER_CLASS,
                  num_classes = 5),
  FIL_TEST_PARAMS(
    output = CLASS, algo = TREE_REORG, op = kLE, leaf_algo = GROVE_PER_CLASS, num_classes = 5),
  FIL_TEST_PARAMS(num_trees   = 49,
                  output      = CLASS,
                  algo        = TREE_REORG,
                  op          = kLE,
                  leaf_algo   = GROVE_PER_CLASS,
                  num_classes = 7),
  FIL_TEST_PARAMS(num_trees = 48, output = CLASS, leaf_algo = GROVE_PER_CLASS, num_classes = 6),
  FIL_TEST_PARAMS(print_forest_shape = true),
  FIL_TEST_PARAMS(leaf_algo = VECTOR_LEAF, num_classes = 2),
  FIL_TEST_PARAMS(leaf_algo = VECTOR_LEAF, num_trees = 19, num_classes = 20),
  FIL_TEST_PARAMS(node_categorical_prob = 0.5, feature_categorical_prob = 0.5),
  FIL_TEST_PARAMS(
    node_categorical_prob = 1.0, feature_categorical_prob = 1.0, cat_match_prob = 1.0),
  FIL_TEST_PARAMS(
    node_categorical_prob = 1.0, feature_categorical_prob = 1.0, cat_match_prob = 0.0),
  FIL_TEST_PARAMS(depth                         = 3,
                  node_categorical_prob         = 0.5,
                  feature_categorical_prob      = 0.5,
                  max_magnitude_of_matching_cat = 5),
};

TEST_P(TreeliteDenseFloat32FilTest, Import) { compare(); }
TEST_P(TreeliteDenseFloat64FilTest, Import) { compare(); }

INSTANTIATE_TEST_CASE_P(FilTests,
                        TreeliteDenseFloat32FilTest,
                        testing::ValuesIn(import_dense_inputs));
INSTANTIATE_TEST_CASE_P(FilTests,
                        TreeliteDenseFloat64FilTest,
                        testing::ValuesIn(import_dense_inputs));

std::vector<FilTestParams> import_sparse_inputs = {
  FIL_TEST_PARAMS(),
  FIL_TEST_PARAMS(output = SIGMOID, op = kLE),
  FIL_TEST_PARAMS(output = SIGMOID_CLASS, op = kGT, num_classes = 2),
  FIL_TEST_PARAMS(output = AVG, op = kGE),
  FIL_TEST_PARAMS(output = AVG_CLASS, num_classes = 2),
  FIL_TEST_PARAMS(global_bias = 0.5),
  FIL_TEST_PARAMS(output = SIGMOID, global_bias = 0.5, op = kLE),
  FIL_TEST_PARAMS(output = AVG, global_bias = 0.5, op = kGT),
  FIL_TEST_PARAMS(
    output = AVG_CLASS, threshold = 1.0, global_bias = 0.5, op = kGE, num_classes = 2),
  FIL_TEST_PARAMS(algo = ALGO_AUTO),
  FIL_TEST_PARAMS(
    output = AVG_CLASS, threshold = 1.0, op = kGE, leaf_algo = CATEGORICAL_LEAF, num_classes = 10),
  FIL_TEST_PARAMS(output = AVG, algo = ALGO_AUTO, leaf_algo = CATEGORICAL_LEAF, num_classes = 4),
  FIL_TEST_PARAMS(output = AVG, op = kLE, leaf_algo = CATEGORICAL_LEAF, num_classes = 5),
  FIL_TEST_PARAMS(output = AVG, leaf_algo = CATEGORICAL_LEAF, num_classes = 3),
  FIL_TEST_PARAMS(output      = CLASS,
                  threshold   = 1.0,
                  global_bias = 0.5,
                  op          = kGE,
                  leaf_algo   = GROVE_PER_CLASS,
                  num_classes = 10),
  FIL_TEST_PARAMS(
    num_trees = 52, output = CLASS, algo = ALGO_AUTO, leaf_algo = GROVE_PER_CLASS, num_classes = 4),
  FIL_TEST_PARAMS(output = CLASS, op = kLE, leaf_algo = GROVE_PER_CLASS, num_classes = 5),
  FIL_TEST_PARAMS(num_trees   = 51,
                  output      = CLASS,
                  global_bias = 0.5,
                  leaf_algo   = GROVE_PER_CLASS,
                  num_classes = 3),
  FIL_TEST_PARAMS(num_trees   = 51,
                  output      = SIGMOID_CLASS,
                  global_bias = 0.5,
                  leaf_algo   = GROVE_PER_CLASS,
                  num_classes = 3),
  FIL_TEST_PARAMS(leaf_algo = VECTOR_LEAF, num_classes = 2),
  FIL_TEST_PARAMS(leaf_algo = VECTOR_LEAF, num_trees = 19, num_classes = 20),
  FIL_TEST_PARAMS(node_categorical_prob = 0.5, feature_categorical_prob = 0.5),
  FIL_TEST_PARAMS(
    node_categorical_prob = 1.0, feature_categorical_prob = 1.0, cat_match_prob = 1.0),
  FIL_TEST_PARAMS(
    node_categorical_prob = 1.0, feature_categorical_prob = 1.0, cat_match_prob = 0.0),
  FIL_TEST_PARAMS(depth                         = 3,
                  node_categorical_prob         = 0.5,
                  feature_categorical_prob      = 0.5,
                  max_magnitude_of_matching_cat = 5),
};

TEST_P(TreeliteSparse16Float32FilTest, Import) { compare(); }
TEST_P(TreeliteSparse16Float64FilTest, Import) { compare(); }

INSTANTIATE_TEST_CASE_P(FilTests,
                        TreeliteSparse16Float32FilTest,
                        testing::ValuesIn(import_sparse_inputs));
INSTANTIATE_TEST_CASE_P(FilTests,
                        TreeliteSparse16Float64FilTest,
                        testing::ValuesIn(import_sparse_inputs));

TEST_P(TreeliteSparse8FilTest, Import) { compare(); }

INSTANTIATE_TEST_CASE_P(FilTests, TreeliteSparse8FilTest, testing::ValuesIn(import_sparse_inputs));

std::vector<FilTestParams> import_auto_inputs = {
  FIL_TEST_PARAMS(depth = 10, algo = ALGO_AUTO),
  FIL_TEST_PARAMS(depth = 15, algo = ALGO_AUTO),
  FIL_TEST_PARAMS(depth = 19, algo = ALGO_AUTO),
  FIL_TEST_PARAMS(depth = 19, algo = BATCH_TREE_REORG),
  FIL_TEST_PARAMS(
    depth = 10, output = AVG, algo = ALGO_AUTO, leaf_algo = CATEGORICAL_LEAF, num_classes = 3),
  FIL_TEST_PARAMS(depth       = 10,
                  num_trees   = 51,
                  output      = CLASS,
                  algo        = ALGO_AUTO,
                  leaf_algo   = GROVE_PER_CLASS,
                  num_classes = 3),
  FIL_TEST_PARAMS(leaf_algo = VECTOR_LEAF, num_classes = 3, algo = ALGO_AUTO),
#if 0
 FIL_TEST_PARAMS(depth = 19, output = AVG, algo = BATCH_TREE_REORG,
                 leaf_algo = CATEGORICAL_LEAF, num_classes = 6),
#endif
};

TEST_P(TreeliteAutoFloat32FilTest, Import) { compare(); }
TEST_P(TreeliteAutoFloat64FilTest, Import) { compare(); }

INSTANTIATE_TEST_CASE_P(FilTests,
                        TreeliteAutoFloat32FilTest,
                        testing::ValuesIn(import_auto_inputs));
INSTANTIATE_TEST_CASE_P(FilTests,
                        TreeliteAutoFloat64FilTest,
                        testing::ValuesIn(import_auto_inputs));

// adjust test parameters if the sparse8 format changes
std::vector<FilTestParams> import_throw_sparse8_inputs = {
  // too many features
  FIL_TEST_PARAMS(num_rows = 100, num_cols = 20000, depth = 10),
  // too many tree nodes
  FIL_TEST_PARAMS(depth = 16, num_trees = 5, leaf_prob = 0),
};

TEST_P(TreeliteThrowSparse8FilTest, Import) { check(); }

INSTANTIATE_TEST_CASE_P(FilTests,
                        TreeliteThrowSparse8FilTest,
                        testing::ValuesIn(import_throw_sparse8_inputs));
}  // namespace ML
