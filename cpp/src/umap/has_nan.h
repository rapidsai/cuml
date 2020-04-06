#pragma once

#include <cuda_runtime.h>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/common/utils.hpp>
#include "common/device_buffer.hpp"

#include "distance/distance.h"

#include "datasets/digits.h"

#include <cuml/manifold/umapparams.h>
#include <metrics/trustworthiness.h>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/cuml.hpp>
#include <cuml/neighbors/knn.hpp>

#include "linalg/reduce_rows_by_key.h"
#include "random/make_blobs.h"

#include "common/device_buffer.hpp"
#include "umap/runner.h"

#include <cuda_utils.h>

#include <iostream>
#include <vector>

using namespace MLCommon;

template <typename T>
__global__ void has_nan_kernel(T* data, size_t len, bool* answer) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= len) return;
  bool val = data[tid];
  if (val != val) {
    *answer = true;
  }
}

template <typename T>
bool has_nan(T* data, size_t len, std::shared_ptr<deviceAllocator> alloc,
             cudaStream_t stream) {
  dim3 blk(256);
  dim3 grid(MLCommon::ceildiv(len, (size_t)blk.x));
  bool h_answer = false;
  device_buffer<bool> d_answer(alloc, stream, 1);
  updateDevice(d_answer.data(), &h_answer, 1, stream);
  has_nan_kernel<<<grid, blk, 0, stream>>>(data, len, d_answer.data());
  updateHost(&h_answer, d_answer.data(), 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return h_answer;
}

template <typename T>
void test_has_nan(T* data, size_t len, std::shared_ptr<deviceAllocator> alloc,
                  cudaStream_t stream, int location) {
  CUDA_CHECK(cudaStreamSynchronize(stream));
  bool res = has_nan(data, len, alloc, stream);
  if (res) {
    std::cout << "Nans at location : " << location << std::endl;
  }
}
