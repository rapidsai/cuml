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

/** @file fil.cu fil.cu implements the forest data types (dense and sparse), including their
creation and prediction (the main inference kernel is defined in infer.cu). */

#include "common.cuh"    // for predict_params, sparse_storage, dense_storage
#include "internal.cuh"  // for cat_sets_device_owner, categorical_sets, output_t,

#include <cuml/fil/fil.h>  // for algo_t,

#include <raft/cudart_utils.h>     // for CUDA_CHECK, cudaStream_t,
#include <thrust/host_vector.h>    // for host_vector
#include <raft/error.hpp>          // for ASSERT
#include <raft/handle.hpp>         // for handle_t
#include <rmm/device_uvector.hpp>  // for device_uvector

#include <cmath>    // for expf
#include <cstddef>  // for size_t

namespace ML {
namespace fil {

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
