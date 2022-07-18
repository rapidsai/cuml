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

/** @file common.cuh Common GPU functionality */
#pragma once

#include <cub/cub.cuh>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <cuml/fil/fil.h>
#include <raft/cuda_utils.cuh>

#include "internal.cuh"

namespace ML {
namespace fil {

__host__ __device__ __forceinline__ int tree_num_nodes(int depth) { return (1 << (depth + 1)) - 1; }

__host__ __device__ __forceinline__ int forest_num_nodes(int num_trees, int depth)
{
  return num_trees * tree_num_nodes(depth);
}

template <typename real_t>
struct storage_base {
  categorical_sets sets_;
  real_t* vector_leaf_;
  bool cats_present() const { return sets_.cats_present(); }
};

/** represents a dense tree */
template <typename real_t>
struct tree<dense_node<real_t>> : tree_base {
  using real_type = real_t;
  __host__ __device__ tree(categorical_sets cat_sets, dense_node<real_t>* nodes, int node_pitch)
    : tree_base{cat_sets}, nodes_(nodes), node_pitch_(node_pitch)
  {
  }
  __host__ __device__ const dense_node<real_t>& operator[](int i) const
  {
    return nodes_[i * node_pitch_];
  }
  dense_node<real_t>* nodes_ = nullptr;
  int node_pitch_            = 0;
};

/** partial specialization of storage. Stores the forest on GPU as a collection of dense nodes */
template <typename real_t>
struct storage<dense_node<real_t>> : storage_base<real_t> {
  using real_type = real_t;
  using node_t    = dense_node<real_t>;
  __host__ __device__ storage(categorical_sets cat_sets,
                              real_t* vector_leaf,
                              node_t* nodes,
                              int num_trees,
                              int tree_stride,
                              int node_pitch)
    : storage_base<real_t>{cat_sets, vector_leaf},
      nodes_(nodes),
      num_trees_(num_trees),
      tree_stride_(tree_stride),
      node_pitch_(node_pitch)
  {
  }
  __host__ __device__ int num_trees() const { return num_trees_; }
  __host__ __device__ tree<node_t> operator[](int i) const
  {
    // sets_ is a dependent name (in template sense)
    return tree<node_t>(this->sets_, nodes_ + i * tree_stride_, node_pitch_);
  }
  node_t* nodes_   = nullptr;
  int num_trees_   = 0;
  int tree_stride_ = 0;
  int node_pitch_  = 0;
};

/** sparse tree */
template <typename node_t>
struct tree : tree_base {
  using real_type = typename node_t::real_type;
  __host__ __device__ tree(categorical_sets cat_sets, node_t* nodes)
    : tree_base{cat_sets}, nodes_(nodes)
  {
  }
  __host__ __device__ const node_t& operator[](int i) const { return nodes_[i]; }
  node_t* nodes_ = nullptr;
};

/** storage stores the forest on GPU as a collection of sparse nodes */
template <typename node_t_>
struct storage : storage_base<typename node_t_::real_type> {
  using node_t    = node_t_;
  using real_type = typename node_t::real_type;
  int* trees_     = nullptr;
  node_t* nodes_  = nullptr;
  int num_trees_  = 0;
  __host__ __device__ storage(
    categorical_sets cat_sets, real_type* vector_leaf, int* trees, node_t* nodes, int num_trees)
    : storage_base<real_type>{cat_sets, vector_leaf},
      trees_(trees),
      nodes_(nodes),
      num_trees_(num_trees)
  {
  }
  __host__ __device__ int num_trees() const { return num_trees_; }
  __host__ __device__ tree<node_t> operator[](int i) const
  {
    // sets_ is a dependent name (in template sense)
    return tree<node_t>(this->sets_, &nodes_[trees_[i]]);
  }
};

using dense_storage_f32    = storage<dense_node<float>>;
using dense_storage_f64    = storage<dense_node<double>>;
using sparse_storage16_f32 = storage<sparse_node16<float>>;
using sparse_storage16_f64 = storage<sparse_node16<double>>;
using sparse_storage8      = storage<sparse_node8>;

/// all model parameters mostly required to compute shared memory footprint,
/// also the footprint itself
struct shmem_size_params {
  /// for class probabilities, this is the number of classes considered;
  /// num_classes is ignored otherwise
  int num_classes = 1;
  // leaf_algo determines what the leaves store (predict) and how FIL
  // aggregates them into class margins/predicted class/regression answer
  leaf_algo_t leaf_algo = leaf_algo_t::FLOAT_UNARY_BINARY;
  /// how many columns an input row has
  int num_cols = 0;
  /// whether to predict class probabilities or classes (or regress)
  bool predict_proba = false;
  /// are the input columns are prefetched into shared
  /// memory before inferring the row in question
  bool cols_in_shmem = true;
  // are there categorical inner nodes? doesn't currently affect shared memory size,
  // but participates in template dispatch and may affect it later
  bool cats_present = false;
  /// log2_threads_per_tree determines how many threads work on a single tree
  /// at once inside a block (sharing trees means splitting input rows)
  int log2_threads_per_tree = 0;
  /// n_items is how many input samples (items) any thread processes. If 0 is given,
  /// choose the reasonable most (<= MAX_N_ITEMS) that fit into shared memory. See init_n_items()
  int n_items = 0;
  // block_dim_x is the CUDA block size. Set by dispatch_on_leaf_algo(...)
  int block_dim_x = 0;
  /// shm_sz is the associated shared memory footprint
  int shm_sz = INT_MAX;
  /// sizeof_real is the size in bytes of all floating-point variables during inference
  std::size_t sizeof_real = 4;

  __host__ __device__ int sdata_stride()
  {
    return num_cols | 1;  // pad to odd
  }
  __host__ __device__ int cols_shmem_size()
  {
    return cols_in_shmem ? sizeof_real * sdata_stride() * n_items << log2_threads_per_tree : 0;
  }
  template <int NITEMS, typename real_t, leaf_algo_t leaf_algo>
  size_t get_smem_footprint();
};

// predict_params are parameters for prediction
struct predict_params : shmem_size_params {
  predict_params(shmem_size_params ssp) : shmem_size_params(ssp) {}
  // Model parameters.
  algo_t algo;
  // number of outputs for the forest per each data row
  int num_outputs;

  // Data parameters; preds and data are pointers to either float or double.
  void* preds;
  const void* data;
  // number of data rows (instances) to predict on
  int64_t num_rows;

  // to signal infer kernel to apply softmax and also average prior to that
  // for GROVE_PER_CLASS for predict_proba
  output_t transform;
  // number of blocks to launch
  int num_blocks;
};

constexpr leaf_algo_t next_leaf_algo(leaf_algo_t algo)
{
  return static_cast<leaf_algo_t>(algo + 1);
}

template <bool COLS_IN_SHMEM_    = false,
          bool CATS_SUPPORTED_   = false,
          leaf_algo_t LEAF_ALGO_ = MIN_LEAF_ALGO,
          int N_ITEMS_           = 1>
struct KernelTemplateParams {
  static const bool COLS_IN_SHMEM    = COLS_IN_SHMEM_;
  static const bool CATS_SUPPORTED   = CATS_SUPPORTED_;
  static const leaf_algo_t LEAF_ALGO = LEAF_ALGO_;
  static const int N_ITEMS           = N_ITEMS_;

  template <bool _cats_supported>
  using ReplaceCatsSupported =
    KernelTemplateParams<COLS_IN_SHMEM, _cats_supported, LEAF_ALGO, N_ITEMS>;
  using NextLeafAlgo =
    KernelTemplateParams<COLS_IN_SHMEM, CATS_SUPPORTED, next_leaf_algo(LEAF_ALGO), N_ITEMS>;
  template <leaf_algo_t NEW_LEAF_ALGO>
  using ReplaceLeafAlgo =
    KernelTemplateParams<COLS_IN_SHMEM, CATS_SUPPORTED, NEW_LEAF_ALGO, N_ITEMS>;
  using IncNItems = KernelTemplateParams<COLS_IN_SHMEM, CATS_SUPPORTED, LEAF_ALGO, N_ITEMS + 1>;
};

// inherit from this struct to pass the functor to dispatch_on_fil_template_params()
// compiler will prevent defining a .run() method with a different output type
template <typename T>
struct dispatch_functor {
  typedef T return_t;
  template <class KernelParams = KernelTemplateParams<>>
  T run(predict_params);
};

namespace dispatch {

template <class KernelParams, class Func, class T = typename Func::return_t>
T dispatch_on_n_items(Func func, predict_params params)
{
  if (params.n_items == KernelParams::N_ITEMS) {
    return func.template run<KernelParams>(params);
  } else if constexpr (KernelParams::N_ITEMS < MAX_N_ITEMS) {
    return dispatch_on_n_items<class KernelParams::IncNItems>(func, params);
  } else {
    ASSERT(false, "n_items > %d or < 1", MAX_N_ITEMS);
  }
  return T();  // appeasing the compiler
}

template <class KernelParams, class Func, class T = typename Func::return_t>
T dispatch_on_leaf_algo(Func func, predict_params params)
{
  if (params.leaf_algo == KernelParams::LEAF_ALGO) {
    if constexpr (KernelParams::LEAF_ALGO == GROVE_PER_CLASS) {
      if (params.num_classes <= FIL_TPB) {
        params.block_dim_x = FIL_TPB - FIL_TPB % params.num_classes;
        using Next         = typename KernelParams::ReplaceLeafAlgo<GROVE_PER_CLASS_FEW_CLASSES>;
        return dispatch_on_n_items<Next>(func, params);
      } else {
        params.block_dim_x = FIL_TPB;
        using Next         = typename KernelParams::ReplaceLeafAlgo<GROVE_PER_CLASS_MANY_CLASSES>;
        return dispatch_on_n_items<Next>(func, params);
      }
    } else {
      params.block_dim_x = FIL_TPB;
      return dispatch_on_n_items<KernelParams>(func, params);
    }
  } else if constexpr (next_leaf_algo(KernelParams::LEAF_ALGO) <= MAX_LEAF_ALGO) {
    return dispatch_on_leaf_algo<class KernelParams::NextLeafAlgo>(func, params);
  } else {
    ASSERT(false, "internal error: dispatch: invalid leaf_algo %d", params.leaf_algo);
  }
  return T();  // appeasing the compiler
}

template <class KernelParams, class Func, class T = typename Func::return_t>
T dispatch_on_cats_supported(Func func, predict_params params)
{
  return params.cats_present
           ? dispatch_on_leaf_algo<typename KernelParams::ReplaceCatsSupported<true>>(func, params)
           : dispatch_on_leaf_algo<typename KernelParams::ReplaceCatsSupported<false>>(func,
                                                                                       params);
}

template <class Func, class T = typename Func::return_t>
T dispatch_on_cols_in_shmem(Func func, predict_params params)
{
  return params.cols_in_shmem
           ? dispatch_on_cats_supported<KernelTemplateParams<true>>(func, params)
           : dispatch_on_cats_supported<KernelTemplateParams<false>>(func, params);
}

}  // namespace dispatch

template <class Func, class T = typename Func::return_t>
T dispatch_on_fil_template_params(Func func, predict_params params)
{
  return dispatch::dispatch_on_cols_in_shmem(func, params);
}

// For an example of Func declaration, see this.
// the .run(predict_params) method will be defined in infer.cu
struct compute_smem_footprint : dispatch_functor<int> {
  template <class KernelParams = KernelTemplateParams<>>
  int run(predict_params);
};

template <int NITEMS,
          leaf_algo_t leaf_algo,
          bool cols_in_shmem,
          bool CATS_SUPPORTED,
          class storage_type>
__global__ void infer_k(storage_type forest, predict_params params);

// infer() calls the inference kernel with the parameters on the stream
template <typename storage_type>
void infer(storage_type forest, predict_params params, cudaStream_t stream);

}  // namespace fil
}  // namespace ML
