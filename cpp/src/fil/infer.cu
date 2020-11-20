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

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/transform_iterator.h>
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
  BinaryOp op;
  __device__ Vectorized(BinaryOp op_ = BinaryOp()) : op(op_) {}
  template <int NITEMS, typename T>
  constexpr __host__ __device__ __forceinline__ vec<NITEMS, T> operator()(
    vec<NITEMS, T> a, vec<NITEMS, T> b) const {
    vec<NITEMS, T> c;
#pragma unroll
    for (int i = 0; i < NITEMS; i++) c[i] = op(a[i], b[i]);
    return c;
  }
};

template <int N, typename T>
struct vec {
  T data[N];
  explicit __host__ __device__ vec(T t = T()) {
#pragma unroll
    for (int i = 0; i < N; ++i) data[i] = t;
  }
  __host__ __device__ T& operator[](int i) { return data[i]; }
  __host__ __device__ T operator[](int i) const { return data[i]; }
  template <typename Vec>
  friend __host__ __device__ vec<N, T> operator+(const vec<N, T>& a,
                                                 const Vec& b) {
    return Vectorized<cub::Sum>()(a, vec<N, T>(b));
  }
  template <typename Vec>
  friend __host__ __device__ void operator+=(vec<N, T>& a, const Vec& b) {
    a = a + b;
  }
  template <typename Vec>
  friend __host__ __device__ void operator/(vec<N, T>& a, const Vec& b) {
    return Vectorized<thrust::divides<T>>()(a, vec<N, T>(b));
  }
  template <typename Vec>
  friend __host__ __device__ void operator/=(vec<N, T>& a, const Vec& b) {
    a = a / b;
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
template <typename T, typename BinaryOp>
__device__ __forceinline__ T block_reduce(T value, BinaryOp op, void* storage) {
  typedef cub::BlockReduce<T, FIL_TPB> BlockReduceT;
  return BlockReduceT(*(typename BlockReduceT::TempStorage*)storage)
    .Reduce(value, op);
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
    acc = block_reduce(acc, Vectorized<cub::Sum>(), tmp_storage);
    if (threadIdx.x > 0) return;
    #pragma unroll
    for (int row = 0; row < num_rows; ++row)
      out[row * output_stride] = acc[row];
  }
};

template <typename T, typename BinaryOp>
__device__ __forceinline__ T block_allreduce(T value, BinaryOp op,
                                             void* storage) {
  auto result = block_reduce(value, op, storage);
  // broadcast sum to all threads
  __syncthreads();  // free up tmp_storage
  if (threadIdx.x == 0) *(T*)storage = result;
  __syncthreads();
  return *(T*)storage;
}

template <typename Iterator, typename BinaryOp>
__device__ __forceinline__ auto allreduce_shmem(Iterator begin, Iterator end,
                                                BinaryOp op, void* tmp_storage)
{
  Iterator::typename value_type thread_partial(0);
  for (Iterator it = begin; it < end; ++it)
    thread_sum = op(thread_partial, *it);
  return block_allreduce(thread_partial, op, tmp_storage);
}

template <typename Iterator>
__device__ __forceinline__ void write_best_class_in_block(
  Iterator begin, Iterator end, void* tmp_storage, float* out, int num_rows) {
  // find best class per block (for each of the NITEMS rows)
  auto best =
    allreduce_shmem(begin, end, Vectorized<cub::ArgMax>(), tmp_storage);
  // write it out to global memory
  if (threadIdx.x > 0) return;
#pragma unroll
  for (int row = 0; row < num_rows; ++row)
    if (row < num_rows) out[row] = best[row].key;
}

template <typename T>
class StridedIt
  : public thrust::iterator_adaptor<StridedIt<T>, T, thrust::use_default,
                                    thrust::forward_device_iterator_tag> {
 public:
  typedef thrust::iterator_adaptor<StridedIt<T>, T, thrust::use_default,
                                   thrust::forward_device_iterator_tag>
    super_t;

  explicit __device__ StridedIt(T p_, int stride_)
    : super_t(p_), begin(p_), stride(stride_) {}
  friend class thrust::iterator_core_access;

 private:
  const T begin;
  int stride;
  __device__ typename super_t::reference dereference() const {
    return *(begin + (this->base() - begin) * stride);
  }
};

template <typename T>
struct StridedItArray {
  T v;
  int size;
  __device__ StridedItArray(T v_, int size_) : v(v_), size(size_) {}

  typedef StridedIt<T> iterator;
  __device__ iterator begin() { return iterator(v + threadIdx.x, blockDim.x); }
  __device__ iterator end() { return iterator(v + size, blockDim.x); }
};

/// needed for softmax
__device__ float shifted_exp(float margin, float max) {
  return expf(margin - max);
}

template <typename Iterator>
__device__ __forceinline__ void block_softmax(Iterator begin, Iterator end,
                                              void* tmp_storage) {
  // subtract max before exponentiating for numerical stability
  typedef decltype(*begin) value_type;
  value_type max =
    allreduce_shmem(begin, end, Vectorized<cub::Max>(), tmp_storage);

  for (Iterator it = begin; it < end; ++it)
    *it = Vectorized<float(*)(float, float)>(shifted_exp)(*it, max);
  // sum of exponents
  value_type soe =
    allreduce_shmem(begin, end, Vectorized<cub::Sum>(), tmp_storage);
  // softmax phase 2: normalization
  for (Iterator it = begin; it < end; ++it) *it /= soe;
}

template <typename Iterator>
__device__ __forceinline__ void normalize_softmax_and_write(
  Iterator begin, Iterator end, output_t transform, int num_trees,
  void* tmp_storage, int num_classes, float* out, int num_rows) {
  if ((transform & output_t::AVG) != 0) {
    for (Iterator it = begin; it < end; ++it) *it /= num_trees;
  }
  if ((transform & output_t::SOFTMAX) != 0)
    block_softmax(begin, end, tmp_storage);
    // write result to global memory
#pragma unroll
  for (int row = 0; row < num_rows; ++row) {
    StridedItArray<float*> out_a(out + num_classes * row, num_classes);
    auto out = out_a.begin();
    for (Iterator in = begin; in < end;) *out++ = (*in++)[row];
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

    if (num_outputs == 1) {  // will output class
      __syncthreads();       // free up per_thread[] margin
      StridedItArray<vec<NITEMS, best_margin_label>*> to_reduce(tmp_storage,
                                                                num_classes);
      if (to_reduce.begin() != to_reduce.end())
        *to_reduce.begin() = to_vec(threadIdx.x, acc);
      write_best_class_in_block(to_reduce.begin(), to_reduce.end(), tmp_storage,
                                out, num_rows);
    } else {  // output softmax-ed margin
      StridedItArray<vec<NITEMS, float>*> to_reduce(tmp_storage, num_classes);
      if (to_reduce.begin() != to_reduce.end()) *to_reduce.begin() = acc;
      normalize_softmax_and_write(to_reduce.begin(), to_reduce.end(), transform,
                                  num_trees, tmp_storage, num_classes, out,
                                  num_rows);
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
  StridedItArray<vec<NITEMS, float>*> per_class_a;

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
    : per_class_value(
        (vec<NITEMS, float>*)((char*)shared_workspace + data_row_size)),
      tmp_storage(shared_workspace),
      num_classes(num_classes_),
      per_class_a(per_class_value, num_classes_) {
    for (StridedIt<vec<NITEMS, float>*> it = per_class_a.begin(); it < per_class_a.end(); ++it)
      *it = vec<NITEMS, float>(0);
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
      auto candidate = [this] __device__(auto it) {
        return to_vec(&it - this->per_class_value, it);
      };
      write_best_class_in_block(
        thrust::make_transform_iterator(per_class_a.begin(), candidate),
        thrust::make_transform_iterator(per_class_a.end(), candidate),
        tmp_storage, out, num_rows);
    } else {  // will output softmax-ed margins
      normalize_softmax_and_write(
        per_class_a.begin(), per_class_a.end(), transform, num_trees,
        tmp_storage, num_classes, out, num_rows);
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
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
#pragma unroll
      for (int row = 0; row < num_rows; ++row)
        out[row * num_classes + c] = votes[c * NITEMS + row];
    }
  }
  // using this when predicting a single class label, as opposed to sparse class vector
  // or class probabilities or regression
  __device__ __forceinline__ void finalize_class_label(float* out,
                                                       int num_rows) {
    __syncthreads();
    int item = threadIdx.x;
    int row = item;
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
  for (size_t block_row0 = blockIdx.x * NITEMS; block_row0 < params.num_rows;
       block_row0 += NITEMS * gridDim.x) {
    // cache the row for all threads to reuse
    for (size_t j = 0; j < NITEMS; ++j) {
         size_t row = block_row0 + j;
#pragma unroll
	    for (int col = threadIdx.x; col < params.num_cols; col += blockDim.x) {
        sdata[j * params.num_cols + col] =
	          row < params.num_rows ? params.data[row * params.num_cols + col]
	                                : 0.0f;
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
  acc.finalize(params.preds + params.num_outputs * block_row0,
               min((size_t)NITEMS, params.num_rows - block_row0),
               params.num_outputs, params.transform, forest.num_trees());
  __syncthreads();
  }
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
      break; /*
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
      break;*/
    default:
      ASSERT(false, "internal error: nitems > 4");
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename storage_type>
void infer(storage_type forest, predict_params params, cudaStream_t stream) {
  switch (params.leaf_algo) {
    case FLOAT_UNARY_BINARY:
      //infer_k_launcher<FLOAT_UNARY_BINARY>(forest, params, stream, FIL_TPB);
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
      //infer_k_launcher<CATEGORICAL_LEAF>(forest, params, stream, FIL_TPB);
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
