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

#include <algorithm>
#include <cmath>
#include "common.cuh"

namespace ML {
namespace fil {

using namespace MLCommon;

// vec wraps float[N] for cub::BlockReduce
template <int N, typename T>
struct vec;

template <typename BinaryOp>
struct Vectorized {
  template <int NITEMS, typename T>
  constexpr __host__ __device__ __forceinline__ vec<NITEMS, T> operator()(
    vec<NITEMS, T> a, vec<NITEMS, T> b) const {
    vec<NITEMS, T> c;
#pragma unroll
    for (int i = 0; i < NITEMS; i++) c[i] = BinaryOp()(a[i], b[i]);
    return c;
  }
};
template <float (*BinaryOp)(float, float), int N, typename T>
constexpr __host__ __device__ __forceinline__ vec<N, T> vectorized(
  vec<N, T> a, vec<N, T> b) {
  vec<N, T> c;
#pragma unroll
  for (int i = 0; i < N; i++) c[i] = BinaryOp(a[i], b[i]);
  return c;
}

template <int N, typename T>
struct vec {
  T data[N];
  explicit __host__ __device__ vec(T t = T()) {
#pragma unroll
    for (int i = 0; i < N; ++i) data[i] = t;
  }
  __host__ __device__ T& operator[](int i) { return data[i]; }
  __host__ __device__ T operator[](int i) const { return data[i]; }
  friend __host__ __device__ void operator+=(vec<N, T>& a, const vec<N, T>& b) {
    a = Vectorized<cub::Sum>()(a, b);
  }
  friend __host__ __device__ vec<N, T> operator+(const vec<N, T>& a,
                                                 const vec<N, T>& b) {
    return Vectorized<cub::Sum>()(a, b);
  }
  template <typename Vec>
  friend __host__ __device__ void operator/=(vec<N, T>& a, const Vec& b) {
    a = Vectorized<thrust::divides<T>>()(a, vec<N, T>(b));
  }
};

typedef cub::KeyValuePair<int, float> best_margin_label;

template <int NITEMS>
__device__ __forceinline__ vec<NITEMS, best_margin_label> to_vec(
  int c, vec<NITEMS, float> margin) {
  vec<NITEMS, best_margin_label> ret;
#pragma unroll
  for (int i = 0; i < NITEMS; i++) ret[i] = best_margin_label(c, margin[i]);
  return ret;
}

template <int NITEMS, typename output_type, typename tree_type>
__device__ __forceinline__ vec<NITEMS, output_type> infer_one_tree(
  tree_type tree, float* sdata, int cols) {
  int curr[NITEMS];
  int mask = (1 << NITEMS) - 1;  // all active
  for (int j = 0; j < NITEMS; ++j) curr[j] = 0;
  do {
#pragma unroll
    for (int j = 0; j < NITEMS; ++j) {
      auto n = tree[curr[j]];
      if (n.is_leaf()) {
        mask &= ~(1 << j);
        continue;
      }
      float val = sdata[j * cols + n.fid()];
      bool cond = isnan(val) ? !n.def_left() : val >= n.thresh();
      curr[j] = n.left(curr[j]) + cond;
    }
  } while (mask != 0);
  vec<NITEMS, output_type> out;
#pragma unroll
  for (int j = 0; j < NITEMS; ++j) {
    /** dependent names are not considered templates by default,
        unless it's a member of a current [template] instantiation.
        alternatively, could have used .base_node::output<... */
    out[j] = tree[curr[j]].template output<output_type>();
  }
  return out;
}

template <typename output_type, typename tree_type>
__device__ __forceinline__ vec<1, output_type> infer_one_tree(tree_type tree,
                                                              float* sdata,
                                                              int cols) {
  int curr = 0;
  for (;;) {
    auto n = tree[curr];
    if (n.is_leaf()) break;
    float val = sdata[n.fid()];
    bool cond = isnan(val) ? !n.def_left() : val >= n.thresh();
    curr = n.left(curr) + cond;
  }
  vec<1, output_type> out;
  out[0] = tree[curr].base_node::output<output_type>();
  return out;
}

/**
The shared memory requirements for finalization stage may differ based
on the set of PTX architectures the kernels were compiled for, as well as 
the CUDA compute capability of the device chosen for computation.

TODO (levsnv): run a test kernel during forest init to determine the compute capability
chosen for the inference, for an accurate sizeof(BlockReduce::TempStorage),
which is used in determining max NITEMS or max input data columns.

600 is the __CUDA_ARCH__ for Pascal (6.0) GPUs, which is not defined in
host code.
6.0 is the earliest compute capability supported by FIL and RAPIDS in general.
See https://rapids.ai/start.html as well as cmake defaults.
*/
// values below are defaults as of this change.
template <int NITEMS>
size_t block_reduce_footprint_host() {
  return sizeof(typename cub::BlockReduce<vec<NITEMS, float>, FIL_TPB,
                                          cub::BLOCK_REDUCE_WARP_REDUCTIONS, 1,
                                          1, 600>::TempStorage);
}

template <int NITEMS>
size_t block_reduce_best_class_footprint_host() {
  return sizeof(
    typename cub::BlockReduce<vec<NITEMS, best_margin_label>, FIL_TPB,
                              cub::BLOCK_REDUCE_WARP_REDUCTIONS, 1, 1,
                              600>::TempStorage);
}

// the device template should achieve the best performance, using up-to-date
// CUB defaults
template <typename BinaryOp, typename T>
__device__ __forceinline__ T block_reduce(T value,
                                          const unsigned int valid_threads,
                                          void* storage) {
  typedef cub::BlockReduce<T, FIL_TPB> BlockReduceT;
  return BlockReduceT(*(typename BlockReduceT::TempStorage*)storage)
    .Reduce(value, Vectorized<BinaryOp>(), valid_threads);
}

template <int NITEMS,
          leaf_algo_t leaf_algo>  // = FLOAT_UNARY_BINARY
struct tree_aggregator_t {
  vec<NITEMS, float> acc;
  void* tmp_storage;

  /** shared memory footprint of the accumulator during
  the finalization of forest inference kernel, when infer_k output
  value is computed.
  num_classes is used for other template parameters */
  static size_t smem_finalize_footprint(size_t data_row_size, int num_classes,
                                        bool predict_proba) {
    return block_reduce_footprint_host<NITEMS>();
  }

  /** shared memory footprint of the accumulator during
  the accumulation of forest inference, when individual trees
  are inferred and partial aggregates are accumulated.
  num_classes is used for other template parameters */
  static size_t smem_accumulate_footprint(int num_classes) { return 0; }

  /** 
  num_classes is used for other template parameters */
  __device__ __forceinline__ tree_aggregator_t(int num_classes,
                                               void* shared_workspace, size_t)
    : tmp_storage(shared_workspace) {}

  __device__ __forceinline__ void accumulate(
    vec<NITEMS, float> single_tree_prediction, int tree) {
    acc += single_tree_prediction;
  }

  __device__ __forceinline__ void finalize(float* out, int num_rows,
                                           int output_stride,
                                           output_t transform, int num_trees) {
    __syncthreads();
    acc = block_reduce<cub::Sum>(acc, blockDim.x, tmp_storage);
    if (threadIdx.x == 0) {
      for (int i = 0; i < NITEMS; ++i) {
        int row = blockIdx.x * NITEMS + i;
        if (row < num_rows) out[row * output_stride] = acc[i];
      }
    }
  }
};

/// needed for softmax
__device__ float shifted_exp(float margin, float max) {
  return expf(margin - max);
}

template <typename BinaryOp, int NITEMS>
__device__ __forceinline__ vec<NITEMS, float> block_allreduce(
  vec<NITEMS, float> value, int valid_threads, void* storage) {
  auto result = block_reduce<BinaryOp>(value, valid_threads, storage);
  // broadcast sum to all threads
  __syncthreads();  // free up tmp_storage
  if (threadIdx.x == 0) *(vec<NITEMS, float>*)storage = result;
  __syncthreads();
  return *(vec<NITEMS, float>*)storage;
}

template <int NITEMS>
__device__ __forceinline__ void write_best_class_in_block(
  vec<NITEMS, best_margin_label> best, void* tmp_storage, int valid_threads,
  float* out, int num_rows) {
  // find best class per block (for each of the NITEMS rows)
  best = block_reduce<cub::ArgMax>(best, valid_threads, tmp_storage);
  // write it out to global memory
  if (threadIdx.x == 0) {
    for (int i = 0; i < NITEMS; ++i) {
      int row = blockIdx.x * NITEMS + i;
      if (row < num_rows) out[row] = best[i].key;
    }
  }
}

template<typename T>
struct BlockStridedIt: thrust::forward_device_iterator_tag {
  //BlockStridedItArray& a;
  //BlockStridedIt(BlockStridedItArray& a_, int size): a(a_) {}
  T* p;
  
  BlockStridedIt(T* p_): p(p) { }
  typedef BlockStridedIt iterator;
  
  __device__ iterator& operator++ () { p += blockDim.x; return *this; }
  __device__ iterator operator++ (int) {
    iterator tmp = *this; ++*this; return tmp; }
  __device__ iterator operator+= (int d) { p += d * blockDim.x; return *this; }
  __device__ iterator operator+ (iterator it, int d) {
    iterator res = it; res += d; return res;
  }
  __device__ T& operator*() const { return *p; }
};

template<typename T>
struct BlockStridedItArray {
  friend class BlockStridedIt<T>;
  T* v;
  int size;
  BlockStridedItArray(T* v_, int size_): v(v_), size(size_) {}

  typedef BlockStridedIt<T> iterator;
  typedef ptrdiff_t difference_type;
  typedef size_t size_type;
  typedef T value_type;
  typedef T * pointer;
  typedef T & reference;
  __device__ iterator begin() { return iterator(v + threadIdx.x); }
  __device__ iterator end() { return iterator(v + size); }
};

template <typename BinaryOp, typename ItA>
__device__ __forceinline__ ItA::value_type allreduce_shmem(ItA values, void* tmp_storage) {
  ItA::value_type partial = thrust::reduce(
    thrust::seq, values.begin(), values.end(),
    ItA::value_type(), Vectorized<BinaryOp>());

  return block_allreduce<BinaryOp>(partial, blockDim.x, tmp_storage);
}

template<typename ItA>
__device__ __forceinline__ void block_softmax(ItA per_class_value, void* tmp_storage) {
  // subtract max before exponentiating for numerical stability
  ItA::value_type max =
    allreduce_shmem<cub::Max>(per_class_value, tmp_storage);
  auto exp = thrust::make_transform_iterator(per_class_value.begin(),
    [max](ItA::value_type v){ return vectorized<shifted_exp>(v, max); });
  // sum of exponents
  ItA::value_type soe = allreduce_shmem<cub::Sum>(exp, tmp_storage);
  // softmax phase 2: normalization
  thrust::for_each(per_class_value.begin(), per_class_value.end(),
    [soe](ItA::reference v){ v /= soe; });
}

template <int NITEMS>
__device__ __forceinline__ void write_softmax(
  vec<NITEMS, float>* per_class_value, output_t transform, void* tmp_storage, int num_classes,
  float* out, int num_rows) {

  BlockStridedItArray per_class_a(per_class_value, num_classes);
  if ((transform & output_t::AVG) != 0) {
    thrust::for_each(per_class_a.begin(), per_class_a.end(),
      [num_trees](vec<NITEMS, float>& v){ v /= num_trees; });
  }
  if ((transform & output_t::SOFTMAX) != 0)
    block_softmax(per_class_a, tmp_storage);
  // write result to global memory
  for (int i = 0; i < NITEMS; ++i) {
    int row = blockIdx.x * NITEMS + i;
    if (row >= num_rows) return;
    BlockStridedItArray out_a(out + num_classes * row, num_classes);
    auto unvectorized = thrust::make_transform_iterator(
      per_class_a.begin(),
      [](vec<NITEMS, float> v){ return v[i]; });
    thrust::copy(unvectorized.begin(), unvectorized.end(), out_a.begin());
  }
}


template <int NITEMS>
struct tree_aggregator_t<NITEMS, GROVE_PER_CLASS_FEW_CLASSES> {
  vec<NITEMS, float> acc;
  void* tmp_storage;
  int num_classes;

  static size_t smem_finalize_footprint(size_t data_row_size, int num_classes,
                                        bool predict_proba) {
    size_t phase1 =
      (FIL_TPB - FIL_TPB % num_classes) * sizeof(vec<NITEMS, float>);
    size_t phase2 = predict_proba
                      ? block_reduce_footprint_host<NITEMS>()
                      : block_reduce_best_class_footprint_host<NITEMS>();
    return std::max(phase1, phase2);
  }

  static size_t smem_accumulate_footprint(int num_classes) { return 0; }

  __device__ __forceinline__ tree_aggregator_t(int num_classes_,
                                               void* shared_workspace, size_t)
    : tmp_storage(shared_workspace), num_classes(num_classes_) {}

  __device__ __forceinline__ void accumulate(
    vec<NITEMS, float> single_tree_prediction, int tree) {
    acc += single_tree_prediction;
  }

  __device__ __forceinline__ void finalize(float* out, int num_rows,
                                           int num_outputs, output_t transform,
                                           int num_trees) {
    __syncthreads();  // free up input row
    // load margin into shared memory
    auto per_thread = (vec<NITEMS, float>*)tmp_storage;
    if (threadIdx.x >= num_classes) per_thread[threadIdx.x] = acc;

    __syncthreads();
    // reduce per-thread margin summand into per-class complete margin
    // (for each of the NITEMS rows)
    // TODO(levsnv): use CUB/tree reduction when num_classes is small
    if (threadIdx.x < num_classes) {
      for (int c = threadIdx.x + num_classes; c < blockDim.x; c += num_classes)
        acc += per_thread[c];
    }
    __syncthreads();  // free up per_thread[] margin

    int c = threadIdx.x;
    if (num_outputs == 1) {  // will output class
      write_best_class_in_block(to_vec(c, acc), tmp_storage, num_classes, out,
                                num_rows);
    } else {  // output softmax-ed margin
      if(threadIdx.x < num_classes) per_thread[threadIdx.x] = acc;
      write_softmax(per_thread, transform, tmp_storage, num_classes, out, num_rows);
    }
  }
};

template <int NITEMS>
struct tree_aggregator_t<NITEMS, GROVE_PER_CLASS_MANY_CLASSES> {
  vec<NITEMS, float> acc;
  /// at first, per class margin, then, possibly, different softmax partials
  vec<NITEMS, float>* per_class_value;
  void* tmp_storage;
  int num_classes;

  static size_t smem_finalize_footprint(size_t data_row_size, int num_classes,
                                        bool predict_proba) {
    size_t phase1 = data_row_size + num_classes * sizeof(vec<NITEMS, float>);
    return predict_proba
             ? phase1 + block_reduce_footprint_host<NITEMS>()
             : std::max(phase1,
                        block_reduce_best_class_footprint_host<NITEMS>());
  }

  static size_t smem_accumulate_footprint(int num_classes) {
    return num_classes * sizeof(vec<NITEMS, float>);
  }

  __device__ __forceinline__ tree_aggregator_t(int num_classes_,
                                               void* shared_workspace,
                                               size_t data_row_size)
    : tmp_storage(shared_workspace),
      num_classes(num_classes_),
      per_class_value(
        (vec<NITEMS, float>*)((char*)shared_workspace + data_row_size)) {
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x)
      per_class_value[c] = vec<NITEMS, float>();  // initialize to 0.0f
    // __syncthreads() is called in infer_k
  }

  __device__ __forceinline__ void accumulate(
    vec<NITEMS, float> single_tree_prediction, int tree) {
    // since threads are assigned to consecutive classes, no need for atomics
    per_class_value[tree % num_classes] += single_tree_prediction;
    // __syncthreads() is called in infer_k
  }

  __device__ __forceinline__ void finalize(float* out, int num_rows,
                                           int num_outputs, output_t transform,
                                           int num_trees) {
    if (num_outputs == 1) {  // will output class
      // reduce per-class candidate margins to one best class candidate
      // per thread (for each of the NITEMS rows)
      vec<NITEMS, best_margin_label> best({-1, -INFINITY});

      for (int c = threadIdx.x; c < num_classes; c += blockDim.x)
        best = Vectorized<cub::ArgMax>()(best, to_vec(c, per_class_value[c]));

      __syncthreads();  // free up per_class_value[]
      write_best_class_in_block(best, tmp_storage, blockDim.x, out, num_rows);
    } else {  // will output softmax-ed margins
      write_softmax(per_class_value, transform, tmp_storage, num_classes, out, num_rows);
    }
  }
};

template <int NITEMS>
struct tree_aggregator_t<NITEMS, CATEGORICAL_LEAF> {
  // could switch to unsigned short to save shared memory
  // provided raft::myAtomicAdd(short*) simulated with appropriate shifts
  int* votes;
  int num_classes;

  static size_t smem_finalize_footprint(int data_row_size, int num_classes,
                                        bool predict_proba) {
    return data_row_size + sizeof(int) * num_classes * NITEMS;
  }
  static size_t smem_accumulate_footprint(int num_classes) {
    return sizeof(int) * num_classes * NITEMS;
  }

  __device__ __forceinline__ tree_aggregator_t(int num_classes_,
                                               void* shared_workspace,
                                               size_t data_row_size)
    : num_classes(num_classes_),
      votes((int*)(data_row_size + (char*)shared_workspace)) {
    for (int c = threadIdx.x; c < num_classes; c += FIL_TPB * NITEMS)
#pragma unroll
      for (int item = 0; item < NITEMS; ++item) votes[c * NITEMS + item] = 0;
    // __syncthreads() is called in infer_k
  }
  __device__ __forceinline__ void accumulate(
    vec<NITEMS, int> single_tree_prediction, int tree) {
#pragma unroll
    for (int item = 0; item < NITEMS; ++item)
      raft::myAtomicAdd(votes + single_tree_prediction[item] * NITEMS + item,
                        1);
  }
  // class probabilities or regression. for regression, num_classes
  // is just the number of outputs for each data instance
  __device__ __forceinline__ void finalize_multiple_outputs(float* out,
                                                            int num_rows) {
    __syncthreads();
    int item = threadIdx.x;
    int row = blockIdx.x * NITEMS + item;
    if (item < NITEMS && row < num_rows) {
#pragma unroll
      for (int c = 0; c < num_classes; ++c)
        out[row * num_classes + c] = votes[c * NITEMS + item];
    }
  }
  // using this when predicting a single class label, as opposed to sparse class vector
  // or class probabilities or regression
  __device__ __forceinline__ void finalize_class_label(float* out,
                                                       int num_rows) {
    __syncthreads();
    int item = threadIdx.x;
    int row = blockIdx.x * NITEMS + item;
    if (item < NITEMS && row < num_rows) {
      int max_votes = 0;
      int best_class = 0;
      for (int c = 0; c < num_classes; ++c) {
        if (votes[c * NITEMS + item] > max_votes) {
          max_votes = votes[c * NITEMS + item];
          best_class = c;
        }
      }
      out[row] = best_class;
    }
  }
  __device__ __forceinline__ void finalize(float* out, int num_rows,
                                           int num_outputs, output_t transform,
                                           int num_trees) {
    if (num_outputs > 1) {
      // only supporting num_outputs == num_classes
      finalize_multiple_outputs(out, num_rows);
    } else {
      finalize_class_label(out, num_rows);
    }
  }
};

template <int NITEMS, leaf_algo_t leaf_algo, class storage_type>
__global__ void infer_k(storage_type forest, predict_params params) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * NITEMS;
  for (int j = 0; j < NITEMS; ++j) {
    for (int i = threadIdx.x; i < params.num_cols; i += blockDim.x) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] =
        row < params.num_rows ? params.data[row * params.num_cols + i] : 0.0f;
    }
  }

  tree_aggregator_t<NITEMS, leaf_algo> acc(
    params.num_classes, sdata, params.num_cols * NITEMS * sizeof(float));

  __syncthreads();  // for both row cache init and acc init

  // one block works on NITEMS rows and the whole forest
  for (int j = threadIdx.x; j - threadIdx.x < forest.num_trees();
       j += blockDim.x) {
    /* j - threadIdx.x < forest.num_trees() is a necessary but block-uniform
       condition for "j < forest.num_trees()". It lets use __syncthreads()
       and is made exact below.
    */
    if (j < forest.num_trees()) {
      acc.accumulate(infer_one_tree<NITEMS, leaf_output_t<leaf_algo>::T>(
                       forest[j], sdata, params.num_cols),
                     j);
    }
    if (leaf_algo == GROVE_PER_CLASS_MANY_CLASSES) __syncthreads();
  }
  acc.finalize(params.preds, params.num_rows, params.num_outputs,
               params.transform, forest.num_trees());
}

template <int NITEMS, leaf_algo_t leaf_algo>
size_t get_smem_footprint(predict_params params) {
  size_t finalize_footprint =
    tree_aggregator_t<NITEMS, leaf_algo>::smem_finalize_footprint(
      sizeof(float) * params.num_cols * NITEMS, params.num_classes,
      params.num_outputs == params.num_classes);
  size_t accumulate_footprint =
    sizeof(float) * params.num_cols * NITEMS +
    tree_aggregator_t<NITEMS, leaf_algo>::smem_accumulate_footprint(
      params.num_classes);

  return std::max(accumulate_footprint, finalize_footprint);
}

template <leaf_algo_t leaf_algo, typename storage_type>
void infer_k_launcher(storage_type forest, predict_params params,
                      cudaStream_t stream, int blockdim_x) {
  const int MAX_BATCH_ITEMS = 4;
  params.max_items =
    params.algo == algo_t::BATCH_TREE_REORG ? MAX_BATCH_ITEMS : 1;

  /** searching for the most items per block while respecting the shared
  * memory limits creates a full linear programming problem.
  * solving it in a single equation looks less tractable than this */
  int num_items = 0;
  size_t shm_sz = 0;
  for (int nitems = 1; nitems <= params.max_items; ++nitems) {
    size_t peak_footprint;
    switch (nitems) {
      case 1:
        peak_footprint = get_smem_footprint<1, leaf_algo>(params);
        break;
      case 2:
        peak_footprint = get_smem_footprint<2, leaf_algo>(params);
        break;
      case 3:
        peak_footprint = get_smem_footprint<3, leaf_algo>(params);
        break;
      case 4:
        peak_footprint = get_smem_footprint<4, leaf_algo>(params);
        break;
      default:
        ASSERT(false, "internal error: nitems > 4");
    }
    // for data row
    if (peak_footprint <= params.max_shm) {
      num_items = nitems;
      shm_sz = peak_footprint;
    }
  }
  if (num_items == 0) {
    int given_num_cols = params.num_cols;
    // starting with maximum that might fit in shared memory, in case
    // given_num_cols is a random large int
    params.num_cols = params.max_shm / sizeof(float);
    // since we're crashing, this will not take too long
    while (params.num_cols > 0 &&
           get_smem_footprint<1, leaf_algo>(params) > params.max_shm) {
      --params.num_cols;
    }
    ASSERT(false, "p.num_cols == %d: too many features, only %d allowed",
           given_num_cols, params.num_cols);
  }
  int num_blocks = raft::ceildiv(int(params.num_rows), num_items);
  switch (num_items) {
    case 1:
      infer_k<1, leaf_algo>
        <<<num_blocks, blockdim_x, shm_sz, stream>>>(forest, params);
      break;
    case 2:
      infer_k<2, leaf_algo>
        <<<num_blocks, blockdim_x, shm_sz, stream>>>(forest, params);
      break;
    case 3:
      infer_k<3, leaf_algo>
        <<<num_blocks, blockdim_x, shm_sz, stream>>>(forest, params);
      break;
    case 4:
      infer_k<4, leaf_algo>
        <<<num_blocks, blockdim_x, shm_sz, stream>>>(forest, params);
      break;
    default:
      ASSERT(false, "internal error: nitems > 4");
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename storage_type>
void infer(storage_type forest, predict_params params, cudaStream_t stream) {
  switch (params.leaf_algo) {
    case FLOAT_UNARY_BINARY:
      infer_k_launcher<FLOAT_UNARY_BINARY>(forest, params, stream, FIL_TPB);
      break;
    case GROVE_PER_CLASS:
      if (params.num_classes > FIL_TPB) {
        params.leaf_algo = GROVE_PER_CLASS_MANY_CLASSES;
        infer_k_launcher<GROVE_PER_CLASS_MANY_CLASSES>(forest, params, stream,
                                                       FIL_TPB);
      } else {
        params.leaf_algo = GROVE_PER_CLASS_FEW_CLASSES;
        infer_k_launcher<GROVE_PER_CLASS_FEW_CLASSES>(
          forest, params, stream, FIL_TPB - FIL_TPB % params.num_classes);
      }
      break;
    case CATEGORICAL_LEAF:
      infer_k_launcher<CATEGORICAL_LEAF>(forest, params, stream, FIL_TPB);
      break;
    default:
      ASSERT(false, "internal error: invalid leaf_algo");
  }
}

template void infer<dense_storage>(dense_storage forest, predict_params params,
                                   cudaStream_t stream);
template void infer<sparse_storage16>(sparse_storage16 forest,
                                      predict_params params,
                                      cudaStream_t stream);
template void infer<sparse_storage8>(sparse_storage8 forest,
                                     predict_params params,
                                     cudaStream_t stream);

}  // namespace fil
}  // namespace ML
