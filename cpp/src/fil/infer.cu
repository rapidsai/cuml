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

#include "common.cuh"

#include "internal.cuh"

#include <cuml/fil/multi_sum.cuh>

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>

#include <thrust/functional.h>

#include <algorithm>
#include <cmath>

#ifndef CUDA_PRAGMA_UNROLL
#ifdef __CUDA_ARCH__
#define CUDA_PRAGMA_UNROLL _Pragma("unroll")
#else
#define CUDA_PRAGMA_UNROLL
#endif  // __CUDA_ARCH__
#endif  // CUDA_PRAGMA_UNROLL

#define INLINE_CONFIG __forceinline__

namespace ML {
namespace fil {

// vec wraps float[N], int[N] or double[N] for cub::BlockReduce
template <int N, typename T>
struct vec;

template <typename BinaryOp>
struct Vectorized {
  BinaryOp op;
  __host__ __device__ Vectorized(BinaryOp op_) : op(op_) {}
  template <int NITEMS, typename T>
  constexpr __host__ __device__ __forceinline__ vec<NITEMS, T> operator()(vec<NITEMS, T> a,
                                                                          vec<NITEMS, T> b) const
  {
    vec<NITEMS, T> c;
    CUDA_PRAGMA_UNROLL
    for (int i = 0; i < NITEMS; i++)
      c[i] = op(a[i], b[i]);
    return c;
  }
};
template <typename BinaryOp>
constexpr __host__ __device__ Vectorized<BinaryOp> vectorized(BinaryOp op)
{
  return Vectorized<BinaryOp>(op);
}

template <int N, typename T>
struct vec {
  static const int NITEMS = N;
  T data[N];
  explicit __host__ __device__ vec(T t)
  {
    CUDA_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i)
      data[i] = t;
  }
  __host__ __device__ vec() : vec(T()) {}
  __host__ __device__ T& operator[](int i) { return data[i]; }
  __host__ __device__ T operator[](int i) const { return data[i]; }
  friend __host__ __device__ vec<N, T> operator+(const vec<N, T>& a, const vec<N, T>& b)
  {
    return vectorized(cub::Sum())(a, b);
  }
  friend __host__ __device__ void operator+=(vec<N, T>& a, const vec<N, T>& b) { a = a + b; }
  template <typename Vec>
  friend __host__ __device__ vec<N, T> operator/(vec<N, T>& a, const Vec& b)
  {
    return vectorized(thrust::divides<T>())(a, vec<N, T>(b));
  }
  template <typename Vec>
  friend __host__ __device__ void operator/=(vec<N, T>& a, const Vec& b)
  {
    a = a / b;
  }
};

template <typename real_t>
struct best_margin_label : cub::KeyValuePair<int, real_t> {
  __host__ __device__ best_margin_label(cub::KeyValuePair<int, real_t> pair)
    : cub::KeyValuePair<int, real_t>(pair)
  {
  }
  __host__ __device__ best_margin_label(int c = 0, real_t f = -INFINITY)
    : cub::KeyValuePair<int, real_t>({c, f})
  {
  }
};

template <int NITEMS, typename real_t>
__device__ __forceinline__ vec<NITEMS, best_margin_label<real_t>> to_vec(int c,
                                                                         vec<NITEMS, real_t> margin)
{
  vec<NITEMS, best_margin_label<real_t>> ret;
  CUDA_PRAGMA_UNROLL
  for (int i = 0; i < NITEMS; ++i)
    ret[i] = best_margin_label<real_t>(c, margin[i]);
  return ret;
}

struct ArgMax {
  template <int NITEMS, typename real_t>
  __host__ __device__ __forceinline__ vec<NITEMS, best_margin_label<real_t>> operator()(
    vec<NITEMS, best_margin_label<real_t>> a, vec<NITEMS, best_margin_label<real_t>> b) const
  {
    vec<NITEMS, best_margin_label<real_t>> c;
    CUDA_PRAGMA_UNROLL
    for (int i = 0; i < NITEMS; i++)
      c[i] = cub::ArgMax()(a[i], b[i]);
    return c;
  }
};

/** tree_leaf_output returns the leaf outputs from the tree with leaf indices
    given by leaves for n_rows items. FULL_ITEMS indicates whether n_rows ==
    NITEMS, to allow the compiler to skip the conditional when unrolling the
    loop. */
template <typename output_type, bool FULL_NITEMS, int NITEMS, typename tree_type>
__device__ __forceinline__ vec<NITEMS, output_type> tree_leaf_output(tree_type tree,
                                                                     int n_rows,
                                                                     int (&leaves)[NITEMS])
{
  vec<NITEMS, output_type> out(0);
  CUDA_PRAGMA_UNROLL
  for (int j = 0; j < NITEMS; ++j) {
    if (FULL_NITEMS || j < n_rows) {
      /** dependent names are not considered templates by default, unless it's a
          member of a current [template] instantiation. As output<>() is a
          member function inherited from the base class, template
          output<output_type>() is required. */
      out[j] = tree[leaves[j]].template output<output_type>();
    }
  }
  return out;
}

template <int NITEMS, bool CATS_SUPPORTED, typename output_type, typename tree_type>
__device__ __forceinline__ vec<NITEMS, output_type> infer_one_tree(
  tree_type tree, const typename tree_type::real_type* input, int cols, int n_rows)
{
  // find the leaf nodes for each row
  int curr[NITEMS];
  // the first n_rows are active
  int mask = (1 << n_rows) - 1;
  for (int j = 0; j < NITEMS; ++j)
    curr[j] = 0;
  do {
    CUDA_PRAGMA_UNROLL
    for (int j = 0; j < NITEMS; ++j) {
      auto n = tree[curr[j]];
      mask &= ~(n.is_leaf() << j);
      if ((mask & (1 << j)) != 0) {
        curr[j] = tree.child_index<CATS_SUPPORTED>(n, curr[j], input[j * cols + n.fid()]);
      }
    }
  } while (mask != 0);

  // get the output from the leaves
  if (n_rows == NITEMS) {
    return tree_leaf_output<output_type, true>(tree, n_rows, curr);
  } else {
    return tree_leaf_output<output_type, false>(tree, n_rows, curr);
  }
}

template <typename output_type, typename tree_type>
__device__ __forceinline__ vec<1, output_type> infer_one_tree(
  tree_type tree, const typename tree_type::real_type* input, int cols, int rows)
{
  int curr = 0;
  for (;;) {
    auto n = tree[curr];
    if (n.is_leaf()) break;
    bool cond = tree.child_index<true>(n, curr, input[n.fid()]);
    curr      = n.left(curr) + cond;
  }
  vec<1, output_type> out;
  /** dependent names are not considered templates by default,
      unless it's a member of a current [template] instantiation.**/
  out[0] = tree[curr].template output<output_type>();
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
template <int NITEMS, typename real_t>
size_t block_reduce_footprint_host()
{
  return sizeof(
    typename cub::
      BlockReduce<vec<NITEMS, real_t>, FIL_TPB, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 1, 1, 600>::
        TempStorage);
}

template <int NITEMS, typename real_t>
size_t block_reduce_best_class_footprint_host()
{
  return sizeof(typename cub::BlockReduce<vec<NITEMS, best_margin_label<real_t>>,
                                          FIL_TPB,
                                          cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                                          1,
                                          1,
                                          600>::TempStorage);
}

// the device template should achieve the best performance, using up-to-date
// CUB defaults
template <typename T, typename BinaryOp>
__device__ __forceinline__ T block_reduce(T value, BinaryOp op, void* storage)
{
  typedef cub::BlockReduce<T, FIL_TPB> BlockReduceT;
  return BlockReduceT(*(typename BlockReduceT::TempStorage*)storage).Reduce(value, op, blockDim.x);
}

template <int NITEMS,
          typename real_t,
          leaf_algo_t leaf_algo>  // = FLOAT_UNARY_BINARY
struct tree_aggregator_t {
  vec<NITEMS, real_t> acc;
  void* tmp_storage;

  /** shared memory footprint of the accumulator during
  the finalization of forest inference kernel, when infer_k output
  value is computed.
  num_classes is used for other template parameters */
  static size_t smem_finalize_footprint(size_t data_row_size,
                                        int num_classes,
                                        int log2_threads_per_tree,
                                        bool predict_proba)
  {
    return log2_threads_per_tree != 0 ? FIL_TPB * NITEMS * sizeof(real_t)
                                      : block_reduce_footprint_host<NITEMS, real_t>();
  }

  /** shared memory footprint of the accumulator during
  the accumulation of forest inference, when individual trees
  are inferred and partial aggregates are accumulated.
  num_classes is used for other template parameters */
  static size_t smem_accumulate_footprint(int num_classes) { return 0; }

  /**
  num_classes is used for other template parameters */
  __device__ __forceinline__ tree_aggregator_t(predict_params params,
                                               void* accumulate_workspace,
                                               void* finalize_workspace,
                                               real_t* vector_leaf)
    : tmp_storage(finalize_workspace)
  {
  }

  __device__ __forceinline__ void accumulate(vec<NITEMS, real_t> single_tree_prediction,
                                             int tree,
                                             int thread_num_rows)
  {
    acc += single_tree_prediction;
  }

  __device__ INLINE_CONFIG void finalize(real_t* block_out,
                                         int block_num_rows,
                                         int output_stride,
                                         output_t transform,
                                         int num_trees,
                                         int log2_threads_per_tree)
  {
    if (FIL_TPB != 1 << log2_threads_per_tree) {  // anything to reduce?
      // ensure input columns can be overwritten (no threads traversing trees)
      __syncthreads();
      if (log2_threads_per_tree == 0) {
        acc = block_reduce(acc, vectorized(cub::Sum()), tmp_storage);
      } else {
        auto per_thread         = (vec<NITEMS, real_t>*)tmp_storage;
        per_thread[threadIdx.x] = acc;
        __syncthreads();
        // We have two pertinent cases for splitting FIL_TPB == 256 values:
        // 1. 2000 columns, which fit few threads/tree in shared memory,
        // so ~256 groups. These are the models that will run the slowest.
        // multi_sum performance is not sensitive to the radix here.
        // 2. 50 columns, so ~32 threads/tree, so ~8 groups. These are the most
        // popular.
        acc =
          multi_sum<5>(per_thread, 1 << log2_threads_per_tree, FIL_TPB >> log2_threads_per_tree);
      }
    }

    if (threadIdx.x * NITEMS >= block_num_rows) return;
    CUDA_PRAGMA_UNROLL
    for (int row = 0; row < NITEMS; ++row) {
      int out_preds_i = threadIdx.x * NITEMS + row;
      if (out_preds_i < block_num_rows) block_out[out_preds_i * output_stride] = acc[row];
    }
  }
};

// tmp_storage may overlap shared memory addressed by [begin, end)
// allreduce_shmem ensures no race conditions
template <typename Iterator, typename BinaryOp>
__device__ __forceinline__ auto allreduce_shmem(Iterator begin,
                                                Iterator end,
                                                BinaryOp op,
                                                void* tmp_storage)
{
  typedef typename std::iterator_traits<Iterator>::value_type value_type;
  value_type thread_partial;
  for (Iterator it = begin + threadIdx.x; it < end; it += blockDim.x)
    thread_partial = op(thread_partial, *it);
  __syncthreads();  // free shared memory [begin, end)
  auto res = block_reduce(thread_partial, op, tmp_storage);
  // broadcast sum to all threads
  __syncthreads();  // free up tmp_storage
  if (threadIdx.x == 0) *(value_type*)tmp_storage = res;
  __syncthreads();
  return *(value_type*)tmp_storage;
}

// *begin and *end shall be struct vec
// tmp_storage may overlap shared memory addressed by [begin, end)
template <typename Iterator, typename real_t>
__device__ __forceinline__ void write_best_class(
  Iterator begin, Iterator end, void* tmp_storage, real_t* out, int num_rows)
{
  // reduce per-class candidate margins to one best class candidate
  // per thread (for each of the NITEMS rows)
  auto best = vec<begin->NITEMS, best_margin_label<real_t>>();
  for (int c = threadIdx.x; c < end - begin; c += blockDim.x)
    best = vectorized(cub::ArgMax())(best, to_vec(c, begin[c]));
  // [begin, end) may overlap tmp_storage
  __syncthreads();
  // find best class per block (for each of the NITEMS rows)
  best = block_reduce(best, vectorized(cub::ArgMax()), tmp_storage);
  // write it out to global memory
  if (threadIdx.x > 0) return;
  CUDA_PRAGMA_UNROLL
  for (int row = 0; row < best.NITEMS; ++row)
    if (row < num_rows) out[row] = best[row].key;
}

/// needed for softmax
struct shifted_exp {
  template <typename real_t>
  __device__ double operator()(real_t margin, real_t max) const
  {
    return exp(margin - max);
  }
};

// *begin and *end shall be struct vec
// tmp_storage may NOT overlap shared memory addressed by [begin, end)
template <typename Iterator>
__device__ __forceinline__ void block_softmax(Iterator begin, Iterator end, void* tmp_storage)
{
  // subtract max before exponentiating for numerical stability
  using value_type = typename std::iterator_traits<Iterator>::value_type;
  value_type max   = allreduce_shmem(begin, end, vectorized(cub::Max()), tmp_storage);
  for (Iterator it = begin + threadIdx.x; it < end; it += blockDim.x)
    *it = vectorized(shifted_exp())(*it, max);
  // sum of exponents
  value_type soe = allreduce_shmem(begin, end, vectorized(cub::Sum()), tmp_storage);
  // softmax phase 2: normalization
  for (Iterator it = begin + threadIdx.x; it < end; it += blockDim.x)
    *it /= soe;
}

// *begin and *end shall be struct vec
// tmp_storage may NOT overlap shared memory addressed by [begin, end)
template <typename Iterator, typename real_t>
__device__ __forceinline__ void normalize_softmax_and_write(Iterator begin,
                                                            Iterator end,
                                                            output_t transform,
                                                            int trees_per_class,
                                                            void* tmp_storage,
                                                            real_t* out,
                                                            int num_rows)
{
  if ((transform & output_t::AVG) != 0) {
    for (Iterator it = begin + threadIdx.x; it < end; it += blockDim.x)
      *it /= trees_per_class;
  }
  if ((transform & output_t::SOFTMAX) != 0) block_softmax(begin, end, tmp_storage);
  // write result to global memory
  CUDA_PRAGMA_UNROLL
  for (int row = 0; row < begin->NITEMS; ++row) {
    for (int c = threadIdx.x; c < end - begin; c += blockDim.x)
      if (row < num_rows) out[row * (end - begin) + c] = begin[c][row];
  }
}

// *begin and *end shall be struct vec
// tmp_storage may NOT overlap shared memory addressed by [begin, end)
// in case num_outputs > 1
template <typename Iterator, typename real_t>
__device__ __forceinline__ void class_margins_to_global_memory(Iterator begin,
                                                               Iterator end,
                                                               output_t transform,
                                                               int trees_per_class,
                                                               void* tmp_storage,
                                                               real_t* out,
                                                               int num_rows,
                                                               int num_outputs)
{
  if (num_outputs == 1) {  // will output class
    // reduce per-class candidate margins to one best class candidate
    // per thread (for each of the NITEMS rows)
    write_best_class(begin, end, tmp_storage, out, num_rows);
  } else {  // output softmax-ed margin
    normalize_softmax_and_write(begin, end, transform, trees_per_class, tmp_storage, out, num_rows);
  }
}

template <int NITEMS, typename real_t>
struct tree_aggregator_t<NITEMS, real_t, GROVE_PER_CLASS_FEW_CLASSES> {
  vec<NITEMS, real_t> acc;
  int num_classes;
  vec<NITEMS, real_t>* per_thread;
  void* tmp_storage;

  static size_t smem_finalize_footprint(size_t data_row_size,
                                        int num_classes,
                                        int log2_threads_per_tree,
                                        bool predict_proba)
  {
    size_t phase1 = (FIL_TPB - FIL_TPB % num_classes) * sizeof(vec<NITEMS, real_t>);
    size_t phase2 = predict_proba ? block_reduce_footprint_host<NITEMS, real_t>()
                                  : block_reduce_best_class_footprint_host<NITEMS, real_t>();
    return predict_proba ? phase1 + phase2 : std::max(phase1, phase2);
  }

  static size_t smem_accumulate_footprint(int num_classes) { return 0; }

  __device__ __forceinline__ tree_aggregator_t(predict_params params,
                                               void* accumulate_workspace,
                                               void* finalize_workspace,
                                               real_t* vector_leaf)
    : num_classes(params.num_classes),
      per_thread((vec<NITEMS, real_t>*)finalize_workspace),
      tmp_storage(params.predict_proba ? per_thread + num_classes : finalize_workspace)
  {
  }

  __device__ __forceinline__ void accumulate(vec<NITEMS, real_t> single_tree_prediction,
                                             int tree,
                                             int thread_num_rows)
  {
    acc += single_tree_prediction;
  }

  __device__ INLINE_CONFIG void finalize(real_t* out,
                                         int num_rows,
                                         int num_outputs,
                                         output_t transform,
                                         int num_trees,
                                         int log2_threads_per_tree)
  {
    __syncthreads();  // free up input row in case it was in shared memory
    // load margin into shared memory
    per_thread[threadIdx.x] = acc;
    __syncthreads();
    acc = multi_sum<6>(per_thread, num_classes, blockDim.x / num_classes);
    if (threadIdx.x < num_classes) per_thread[threadIdx.x] = acc;
    __syncthreads();  // per_thread needs to be fully populated

    class_margins_to_global_memory(per_thread,
                                   per_thread + num_classes,
                                   transform,
                                   num_trees / num_classes,
                                   tmp_storage,
                                   out,
                                   num_rows,
                                   num_outputs);
  }
};

template <int NITEMS, typename real_t>
struct tree_aggregator_t<NITEMS, real_t, GROVE_PER_CLASS_MANY_CLASSES> {
  vec<NITEMS, real_t> acc;
  /// at first, per class margin, then, possibly, different softmax partials
  vec<NITEMS, real_t>* per_class_margin;
  void* tmp_storage;
  int num_classes;

  static size_t smem_finalize_footprint(size_t data_row_size,
                                        int num_classes,
                                        int log2_threads_per_tree,
                                        bool predict_proba)
  {
    size_t phase1 = data_row_size + smem_accumulate_footprint(num_classes);
    size_t phase2 = predict_proba ? block_reduce_footprint_host<NITEMS, real_t>()
                                  : block_reduce_best_class_footprint_host<NITEMS, real_t>();
    return predict_proba ? phase1 + phase2 : std::max(phase1, phase2);
  }

  static __host__ __device__ size_t smem_accumulate_footprint(int num_classes)
  {
    return num_classes * sizeof(vec<NITEMS, real_t>);
  }

  __device__ __forceinline__ tree_aggregator_t(predict_params params,
                                               void* accumulate_workspace,
                                               void* finalize_workspace,
                                               real_t* vector_leaf)
    : per_class_margin((vec<NITEMS, real_t>*)accumulate_workspace),
      tmp_storage(params.predict_proba ? per_class_margin + num_classes : finalize_workspace),
      num_classes(params.num_classes)
  {
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x)
      per_class_margin[c] = vec<NITEMS, real_t>(0);
    // __syncthreads() is called in infer_k
  }

  __device__ __forceinline__ void accumulate(vec<NITEMS, real_t> single_tree_prediction,
                                             int tree,
                                             int thread_num_rows)
  {
    // since threads are assigned to consecutive classes, no need for atomics
    if (thread_num_rows > 0) { per_class_margin[tree % num_classes] += single_tree_prediction; }
    __syncthreads();
  }

  __device__ INLINE_CONFIG void finalize(real_t* out,
                                         int num_rows,
                                         int num_outputs,
                                         output_t transform,
                                         int num_trees,
                                         int log2_threads_per_tree)
  {
    class_margins_to_global_memory(per_class_margin,
                                   per_class_margin + num_classes,
                                   transform,
                                   num_trees / num_classes,
                                   tmp_storage,
                                   out,
                                   num_rows,
                                   num_outputs);
  }
};

template <int NITEMS, typename real_t>
struct tree_aggregator_t<NITEMS, real_t, VECTOR_LEAF> {
  // per_class_margin is a row-major matrix
  // of size num_threads_per_class * num_classes
  // used to acccumulate class values
  vec<NITEMS, real_t>* per_class_margin;
  vec<NITEMS, int>* vector_leaf_indices;
  int* thread_num_rows;
  int num_classes;
  int num_threads_per_class;
  real_t* vector_leaf;
  void* tmp_storage;

  static size_t smem_finalize_footprint(size_t data_row_size,
                                        int num_classes,
                                        int log2_threads_per_tree,
                                        bool predict_proba)
  {
    size_t phase1 = data_row_size + smem_accumulate_footprint(num_classes);
    size_t phase2 = predict_proba ? block_reduce_footprint_host<NITEMS, real_t>()
                                  : block_reduce_best_class_footprint_host<NITEMS, real_t>();
    return predict_proba ? phase1 + phase2 : std::max(phase1, phase2);
  }
  static size_t smem_accumulate_footprint(int num_classes)
  {
    return sizeof(vec<NITEMS, real_t>) * num_classes * max(1, FIL_TPB / num_classes) +
           sizeof(vec<NITEMS, int>) * FIL_TPB + sizeof(int) * FIL_TPB;
  }

  __device__ __forceinline__ tree_aggregator_t(predict_params params,
                                               void* accumulate_workspace,
                                               void* finalize_workspace,
                                               real_t* vector_leaf)
    : num_classes(params.num_classes),
      num_threads_per_class(max(1, blockDim.x / params.num_classes)),
      vector_leaf(vector_leaf),
      tmp_storage(finalize_workspace)
  {
    // Assign workspace
    char* ptr        = (char*)accumulate_workspace;
    per_class_margin = (vec<NITEMS, real_t>*)ptr;
    ptr += sizeof(vec<NITEMS, real_t>) * num_classes * num_threads_per_class;
    vector_leaf_indices = (vec<NITEMS, int>*)ptr;
    ptr += sizeof(vec<NITEMS, int>) * blockDim.x;
    thread_num_rows = (int*)ptr;

    // Initialise shared memory
    for (int i = threadIdx.x; i < num_classes * num_threads_per_class; i += blockDim.x) {
      per_class_margin[i] = vec<NITEMS, real_t>();
    }
    vector_leaf_indices[threadIdx.x] = vec<NITEMS, int>();
    thread_num_rows[threadIdx.x]     = 0;
    // __syncthreads() is called in infer_k
  }

  __device__ __forceinline__ void accumulate(vec<NITEMS, int> single_tree_prediction,
                                             int tree,
                                             int num_rows)
  {
    // Perform a transpose in shared memory
    // Assign each thread to a class, so they can accumulate without atomics
    __syncthreads();
    // Write indices to shared memory
    vector_leaf_indices[threadIdx.x] = single_tree_prediction;
    thread_num_rows[threadIdx.x]     = num_rows;
    __syncthreads();
    // i here refers to each element of the matrix per_class_margin
    for (int i = threadIdx.x; i < num_classes * num_threads_per_class; i += blockDim.x) {
      // if num_threads_per_class == 1, then c == i
      int c = i % num_classes;
      // iterate over original thread inputs with stride num_threads_per_class
      // j is the original thread input
      // we have num_classes threads for each j
      for (int j = i / num_classes; j < blockDim.x; j += num_threads_per_class) {
        for (int item = 0; item < thread_num_rows[j]; ++item) {
          real_t pred = vector_leaf[vector_leaf_indices[j][item] * num_classes + c];
          per_class_margin[i][item] += pred;
        }
      }
    }
  }
  __device__ INLINE_CONFIG void finalize(real_t* out,
                                         int num_rows,
                                         int num_outputs,
                                         output_t transform,
                                         int num_trees,
                                         int log2_threads_per_tree)
  {
    if (num_classes < blockDim.x) {
      __syncthreads();
      // Efficient implementation for small number of classes
      auto acc = multi_sum<6>(per_class_margin, num_classes, max(1, blockDim.x / num_classes));
      if (threadIdx.x < num_classes) per_class_margin[threadIdx.x] = acc;
      __syncthreads();
    }
    class_margins_to_global_memory(per_class_margin,
                                   per_class_margin + num_classes,
                                   transform,
                                   num_trees,
                                   tmp_storage,
                                   out,
                                   num_rows,
                                   num_outputs);
  }
};

template <int NITEMS, typename real_t>
struct tree_aggregator_t<NITEMS, real_t, CATEGORICAL_LEAF> {
  // could switch to uint16_t to save shared memory
  // provided raft::myAtomicAdd(short*) simulated with appropriate shifts
  int* votes;
  int num_classes;

  static size_t smem_finalize_footprint(size_t data_row_size,
                                        int num_classes,
                                        int log2_threads_per_tree,
                                        bool predict_proba)
  {
    // not accounting for lingering accumulate_footprint during finalize()
    return 0;
  }
  static size_t smem_accumulate_footprint(int num_classes)
  {
    return sizeof(int) * num_classes * NITEMS;
  }

  __device__ __forceinline__ tree_aggregator_t(predict_params params,
                                               void* accumulate_workspace,
                                               void* finalize_workspace,
                                               real_t* vector_leaf)
    : num_classes(params.num_classes), votes((int*)accumulate_workspace)
  {
    for (int c = threadIdx.x; c < num_classes; c += FIL_TPB * NITEMS)
      CUDA_PRAGMA_UNROLL
    for (int item = 0; item < NITEMS; ++item)
      votes[c * NITEMS + item] = 0;
    // __syncthreads() is called in infer_k
  }
  __device__ __forceinline__ void accumulate(vec<NITEMS, int> single_tree_prediction,
                                             int tree,
                                             int thread_num_rows)
  {
    if (thread_num_rows == 0) return;
    CUDA_PRAGMA_UNROLL
    for (int item = 0; item < NITEMS; ++item) {
      raft::myAtomicAdd(votes + single_tree_prediction[item] * NITEMS + item, 1);
    }
  }
  // class probabilities or regression. for regression, num_classes
  // is just the number of outputs for each data instance
  __device__ __forceinline__ void finalize_multiple_outputs(real_t* out, int num_rows)
  {
    __syncthreads();
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
      CUDA_PRAGMA_UNROLL
      for (int row = 0; row < num_rows; ++row)
        out[row * num_classes + c] = votes[c * NITEMS + row];
    }
  }
  // using this when predicting a single class label, as opposed to sparse class vector
  // or class probabilities or regression
  __device__ __forceinline__ void finalize_class_label(real_t* out, int num_rows)
  {
    __syncthreads();  // make sure all votes[] are final
    int item = threadIdx.x;
    int row  = item;
    if (item < NITEMS && row < num_rows) {
      int max_votes  = 0;
      int best_class = 0;
      for (int c = 0; c < num_classes; ++c) {
        if (votes[c * NITEMS + item] > max_votes) {
          max_votes  = votes[c * NITEMS + item];
          best_class = c;
        }
      }
      out[row] = best_class;
    }
  }
  __device__ INLINE_CONFIG void finalize(real_t* out,
                                         int num_rows,
                                         int num_outputs,
                                         output_t transform,
                                         int num_trees,
                                         int log2_threads_per_tree)
  {
    if (num_outputs > 1) {
      // only supporting num_outputs == num_classes
      finalize_multiple_outputs(out, num_rows);
    } else {
      finalize_class_label(out, num_rows);
    }
  }
};

template <typename real_t>
__device__ INLINE_CONFIG void load_data(real_t* sdata,
                                        const real_t* block_input,
                                        predict_params params,
                                        int rows_per_block,
                                        int block_num_rows)
{
  int num_cols     = params.num_cols;
  int sdata_stride = params.sdata_stride();
  // cache the row for all threads to reuse
  // 2021: latest SMs still do not have >256KiB of shared memory/block required to
  // exceed the uint16_t
  CUDA_PRAGMA_UNROLL
  for (uint16_t input_idx = threadIdx.x; input_idx < block_num_rows * num_cols;
       input_idx += blockDim.x) {
    // for even num_cols, we need to pad sdata_stride to reduce bank conflicts
    // assuming here that sdata_stride == num_cols + 1
    // then, idx / num_cols * sdata_stride + idx % num_cols == idx + idx / num_cols
    uint16_t sdata_idx =
      sdata_stride == num_cols ? input_idx : input_idx + input_idx / (uint16_t)num_cols;
    sdata[sdata_idx] = block_input[input_idx];
  }
  CUDA_PRAGMA_UNROLL
  for (int idx = block_num_rows * sdata_stride; idx < rows_per_block * sdata_stride;
       idx += blockDim.x)
    sdata[idx] = 0.0f;
}

template <int NITEMS,
          leaf_algo_t leaf_algo,
          bool cols_in_shmem,
          bool CATS_SUPPORTED,
          class storage_type>
__global__ void infer_k(storage_type forest, predict_params params)
{
  using real_t = typename storage_type::real_type;
  extern __shared__ char smem[];
  real_t* sdata      = reinterpret_cast<real_t*>(smem);
  int sdata_stride   = params.sdata_stride();
  int rows_per_block = NITEMS << params.log2_threads_per_tree;
  int num_cols       = params.num_cols;
  int thread_row0    = NITEMS * modpow2(threadIdx.x, params.log2_threads_per_tree);
  for (int64_t block_row0 = blockIdx.x * rows_per_block; block_row0 < params.num_rows;
       block_row0 += rows_per_block * gridDim.x) {
    int block_num_rows =
      max(0, (int)min((int64_t)rows_per_block, (int64_t)params.num_rows - block_row0));
    const real_t* block_input =
      reinterpret_cast<const real_t*>(params.data) + block_row0 * num_cols;
    if constexpr (cols_in_shmem)
      load_data(sdata, block_input, params, rows_per_block, block_num_rows);

    tree_aggregator_t<NITEMS, real_t, leaf_algo> acc(
      params, (char*)sdata + params.cols_shmem_size(), sdata, forest.vector_leaf_);

    __syncthreads();  // for both row cache init and acc init
    // one block works on NITEMS * threads_per_tree rows and the whole forest
    // one thread works on NITEMS rows

    int thread_tree0    = threadIdx.x >> params.log2_threads_per_tree;
    int tree_stride     = blockDim.x >> params.log2_threads_per_tree;
    int thread_num_rows = max(0, min(NITEMS, block_num_rows - thread_row0));
    for (int tree = thread_tree0; tree - thread_tree0 < forest.num_trees(); tree += tree_stride) {
      /* tree - thread_tree0 < forest.num_trees() is a necessary but block-uniform
         condition for "tree < forest.num_trees()". It lets use __syncthreads()
         and is made exact below.
         Same with thread_num_rows > 0
      */
      using pred_t = typename leaf_output_t<leaf_algo, real_t>::T;
      vec<NITEMS, pred_t> prediction;
      if (tree < forest.num_trees() && thread_num_rows != 0) {
        prediction = infer_one_tree<NITEMS, CATS_SUPPORTED, pred_t>(
          forest[tree],
          cols_in_shmem ? sdata + thread_row0 * sdata_stride : block_input + thread_row0 * num_cols,
          cols_in_shmem ? sdata_stride : num_cols,
          cols_in_shmem ? NITEMS : thread_num_rows);
      }
      // All threads must enter accumulate
      // Dummy threads can be marked as having 0 rows
      acc.accumulate(prediction, tree, tree < forest.num_trees() ? thread_num_rows : 0);
    }
    acc.finalize(reinterpret_cast<real_t*>(params.preds) + params.num_outputs * block_row0,
                 block_num_rows,
                 params.num_outputs,
                 params.transform,
                 forest.num_trees(),
                 params.log2_threads_per_tree);
    __syncthreads();  // free up acc's shared memory resources for next row set
  }
}

template <int NITEMS, typename real_t, leaf_algo_t leaf_algo>
size_t shmem_size_params::get_smem_footprint()
{
  size_t finalize_footprint = tree_aggregator_t<NITEMS, real_t, leaf_algo>::smem_finalize_footprint(
    cols_shmem_size(), num_classes, log2_threads_per_tree, predict_proba);
  size_t accumulate_footprint =
    tree_aggregator_t<NITEMS, real_t, leaf_algo>::smem_accumulate_footprint(num_classes) +
    cols_shmem_size();
  return std::max(accumulate_footprint, finalize_footprint);
}

template <class KernelParams>
int compute_smem_footprint::run(predict_params ssp)
{
  switch (ssp.sizeof_real) {
    case 4:
      return ssp
        .template get_smem_footprint<KernelParams::N_ITEMS, float, KernelParams::LEAF_ALGO>();
    case 8:
      return ssp
        .template get_smem_footprint<KernelParams::N_ITEMS, double, KernelParams::LEAF_ALGO>();
    default:
      ASSERT(false,
             "internal error: sizeof_real == %d, but must be 4 or 8",
             static_cast<int>(ssp.sizeof_real));
      // unreachable
      return 0;
  }
}

// make sure to instantiate all possible get_smem_footprint instantiations
template int dispatch_on_fil_template_params(compute_smem_footprint, predict_params);

template <typename storage_type>
struct infer_k_storage_template : dispatch_functor<void> {
  storage_type forest;
  cudaStream_t stream;
  infer_k_storage_template(storage_type forest_, cudaStream_t stream_)
    : forest(forest_), stream(stream_)
  {
  }

  template <class KernelParams = KernelTemplateParams<>>
  void run(predict_params params)
  {
    params.num_blocks = params.num_blocks != 0
                          ? params.num_blocks
                          : raft::ceildiv(int(params.num_rows), params.n_items);
    infer_k<KernelParams::N_ITEMS,
            KernelParams::LEAF_ALGO,
            KernelParams::COLS_IN_SHMEM,
            KernelParams::CATS_SUPPORTED>
      <<<params.num_blocks, params.block_dim_x, params.shm_sz, stream>>>(forest, params);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
};

template <typename storage_type>
void infer(storage_type forest, predict_params params, cudaStream_t stream)
{
  dispatch_on_fil_template_params(infer_k_storage_template<storage_type>(forest, stream), params);
}

template void infer<dense_storage_f32>(dense_storage_f32 forest,
                                       predict_params params,
                                       cudaStream_t stream);
template void infer<dense_storage_f64>(dense_storage_f64 forest,
                                       predict_params params,
                                       cudaStream_t stream);
template void infer<sparse_storage16_f32>(sparse_storage16_f32 forest,
                                          predict_params params,
                                          cudaStream_t stream);
template void infer<sparse_storage16_f64>(sparse_storage16_f64 forest,
                                          predict_params params,
                                          cudaStream_t stream);
template void infer<sparse_storage8>(sparse_storage8 forest,
                                     predict_params params,
                                     cudaStream_t stream);

}  // namespace fil
}  // namespace ML
