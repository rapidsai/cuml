
#pragma once

#ifndef IF_DEBUG
  #define IF_DEBUG 0
#endif
#define TEST_NNZ 12021

#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_utils.h>

/* Change to <sparse/coo.h> */
#include "./sparse/coo.h"

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
using namespace thrust::placeholders;

#include <random/rng.h>
#include <stats/sum.h>
#include <sys/time.h>

/** check for cuda runtime API errors and assert accordingly */
#define CHECK(call)                                        \
  do {                                                     \
    cudaError_t status = call;                             \
    if (status != cudaSuccess) {                           \
      cfree_all();                                         \
      ASSERT(false, "FAIL: call='%s'. Reason:%s\n", #call, \
             cudaGetErrorString(status));                  \
    }                                                      \
  } while (0)

// ###################### General global funcs ######################
#define thrust_t thrust::device_ptr
#define to_thrust thrust::device_pointer_cast
#define __STREAM__ thrust::cuda::par.on(stream)
#define cuda_create_stream(x) CHECK(cudaStreamCreate(x))
#define cuda_destroy_stream(x) CHECK(cudaStreamDestroy(x))
#define cuda_synchronize() CHECK(cudaDeviceSynchronize())

#define COO_t Sparse::COO
#define MAX(a, b) ((a > b) ? a : b)
#define MIN(a, b) ((a > b) ? b : a)

// ###################### Garbage Collection Systems ######################
#define MAX_GARBAGE 100
static size_t garbage_no = 0;
#define ADD_GARBAGE(x) GlobalGarbage[garbage_no++] = x
static void *GlobalGarbage[MAX_GARBAGE] = {NULL};

// ###################### Functions ######################
namespace ML {
using namespace ML;
using namespace MLCommon;

// ###################### Debugging prints ######################
#define DEBUG(fmt, ...)                                \
  do {                                                 \
    if (IF_DEBUG) fprintf(stderr, fmt, ##__VA_ARGS__); \
  } while (0);

// ###################### Free some space ######################
#define cuda_free(x) CUDA_CHECK(cudaFree(x))
#define cfree(x) CUDA_CHECK(cudaFree(x))

// ###################### Collect the malloced space ######################
void cuda_free_all(void) {
  DEBUG("[----]   Number of malloced items = %zu\n", garbage_no);
  for (size_t i = 0; i < garbage_no; i++) {
    if (GlobalGarbage[i] != NULL) cfree(GlobalGarbage[i]);
    GlobalGarbage[i] = NULL;
  }
  garbage_no = 0;
}
#define cfree_all cuda_free_all

// ###################### Set memory to some value ######################
template <typename math_t>
static void __cuda_memset(math_t *&ptr, const math_t val, const size_t n,
                          cudaStream_t stream) {
  if (val == 0) CHECK(cudaMemset(ptr, 0, sizeof(math_t) * n));
  // else {
  //     thrust_t<math_t> begin = to_thrust(ptr);
  //     thrust::fill(__STREAM__, begin, begin+n, val);
  // }
}
#define cuda_memset(ptr, val, n) __cuda_memset(ptr, val, n, stream)
#define cmemset cuda_memset

// ###################### Malloc some space ######################
static void *__cuda_malloc(const size_t n, bool garbage_collect = true) {
  void *data = NULL;
  CHECK(cudaMalloc((void **)&data, n));

  if (data == NULL) {
    DEBUG("[Error]   Malloc Failed. Terminating TSNE.\n");
    cfree_all();
    exit(-1);
  }
  if (garbage_collect) ADD_GARBAGE(data);
  return data;
}

// ###################### Malloc and memset some space ######################
template <typename math_t = float>
static math_t *__cuda_malloc_memset(const size_t n, cudaStream_t stream,
                                    bool garbage_collect, const math_t val) {
  math_t *data = (math_t *)__cuda_malloc(n, garbage_collect);
  __cuda_memset(data, val, n / sizeof(math_t), stream);
  return data;
}
#define __cuda_malloc1(n) __cuda_malloc(n, true)
#define __cuda_malloc2(n, gc) __cuda_malloc(n, gc)
#define __cuda_malloc3(n, gc, val) __cuda_malloc_memset(n, stream, gc, val)

#define GET_MACRO(_1, _2, _3, NAME, ...) NAME
#define cuda_malloc(...)                                                 \
  GET_MACRO(__VA_ARGS__, __cuda_malloc3, __cuda_malloc2, __cuda_malloc1) \
  (__VA_ARGS__)
#define cmalloc cuda_malloc

// ###################### Creates a random uniform vector ######################
void random_vector(float *vector, const float minimum, const float maximum,
                   const int size, cudaStream_t stream, long long seed = -1) {
  if (seed <= 0) {
    // Get random seed based on time of day
    struct timeval tp;
    gettimeofday(&tp, NULL);
    seed = tp.tv_sec * 1000 + tp.tv_usec;
  }
  Random::Rng random(seed);
  random.uniform<float>(vector, size, minimum, maximum, stream);
  CUDA_CHECK(cudaPeekAtLastError());
}

inline void array_multiply(float *array, const float mult, const int n, cudaStream_t stream) {
	thrust_t<float> begin = to_thrust(array);
	thrust::transform(__STREAM__, begin, begin + n, begin, mult * _1);
}

// end namespace ML
}  // namespace ML
