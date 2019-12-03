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
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#ifdef HAVE_CUB
#include <cuml/common/cubAllocatorAdapter.hpp>
#endif  //HAVE_CUB

#ifdef HAVE_RMM
#include <rmm/rmm.h>
#include <cuml/common/rmmAllocatorAdapter.hpp>
#endif  //HAVE_RMM

#include <cuml/tree/algo_helper.h>
#include <cuml/cuml.hpp>
#include <cuml/tree/decisiontree.hpp>

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                    \
  {                                                                           \
    cudaError_t cudaStatus = call;                                            \
    if (cudaSuccess != cudaStatus)                                            \
      fprintf(stderr,                                                         \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
              "%s (%d).\n",                                                   \
              #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus),      \
              cudaStatus);                                                    \
  }
#endif  //CUDA_RT_CALL

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg,
             const T default_val) {
  T argval = default_val;
  char** itr = std::find(begin, end, arg);
  if (itr != end && ++itr != end) {
    std::istringstream inbuf(*itr);
    inbuf >> argval;
  }
  return argval;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
  char** itr = std::find(begin, end, arg);
  if (itr != end) {
    return true;
  }
  return false;
}

void printUsage() {
  std::cout << "To run default example use:" << std::endl
            << "  decisiontree_example [-dev_id <GPU id>] [-num_rows <nr>]"
            << std::endl
            << "    [-num_cols <nc>] [-regression] [-max_depth <md>]"
            << std::endl
            << "    -dataset /path/to/dataset.csv" << std::endl;
  return;
}

void loadClassificationDataset(const std::string& file, int nRows, int nCols,
                               float* d_data, int* d_labels, cudaStream_t s) {
  std::ifstream fp;
  fp.open(file.c_str());
  std::string line;
  std::vector<float> data;
  data.resize(nRows * nCols);
  std::vector<int> labels;
  labels.resize(nRows);
  int counter = 0;
  while (std::getline(fp, line) && (counter < nRows)) {
    std::stringstream str(line);
    std::vector<float> row;
    float i;
    while (str >> i) {
      row.push_back(i);
      if (str.peek() == ',') str.ignore();
    }
    for (int j = 0; j < nCols; ++j) {
      data[counter + j * nRows] = row[j];
    }
    labels[counter] = int(row[nCols]);
    ++counter;
  }
  fp.close();
  CUDA_RT_CALL(cudaMemcpyAsync(d_data, data.data(),
                               sizeof(float) * nRows * nCols,
                               cudaMemcpyHostToDevice, s));
  CUDA_RT_CALL(cudaMemcpyAsync(d_labels, labels.data(), sizeof(int) * nRows,
                               cudaMemcpyHostToDevice, s));
}

void loadRegressionDataset(const std::string& file, int nRows, int nCols,
                           float* d_data, float* d_labels, cudaStream_t s) {
  std::ifstream fp;
  fp.open(file.c_str());
  std::string line;
  std::vector<float> data;
  data.resize(nRows * nCols);
  std::vector<int> labels;
  labels.resize(nRows);
  int counter = 0;
  while (std::getline(fp, line) && (counter < nRows)) {
    std::stringstream str(line);
    std::vector<float> row;
    float i;
    while (str >> i) {
      row.push_back(i);
      if (str.peek() == ',') str.ignore();
    }
    for (int j = 0; j < nCols; ++j) {
      data[counter + j * nRows] = row[j];
    }
    labels[counter] = row[nCols];
    ++counter;
  }
  fp.close();
  CUDA_RT_CALL(cudaMemcpyAsync(d_data, data.data(),
                               sizeof(float) * nRows * nCols,
                               cudaMemcpyHostToDevice, s));
  CUDA_RT_CALL(cudaMemcpyAsync(d_labels, labels.data(), sizeof(int) * nRows,
                               cudaMemcpyHostToDevice, s));
}

void setDeviceAllocator(ML::cumlHandle& h) {
  std::cout << "Setting device allocator..." << std::endl;
#ifdef HAVE_RMM
  std::shared_ptr<ML::deviceAllocator> allocator(new ML::rmmAllocatorAdapter());
#elif defined(HAVE_CUB)
  std::shared_ptr<ML::deviceAllocator> allocator(
    new ML::cachingDeviceAllocator());
#else
  std::shared_ptr<ML::deviceAllocator> allocator(
    new ML::defaultDeviceAllocator());
#endif  // HAVE_RMM
  h.setDeviceAllocator(allocator);
}

int main(int argc, char* argv[]) {
  int devId = get_argval<int>(argv, argv + argc, "-dev_id", 0);
  auto dataset = get_argval<std::string>(argv, argv + argc, "-dataset", "");
  auto regression = get_arg(argv, argv + argc, "-regression");
  int nRows, nCols;
  if (regression) {
    // from reg dataset!
    nRows = get_argval<int>(argv, argv + argc, "-num_rows", 99);
    nCols = get_argval<int>(argv, argv + argc, "-num_cols", 4);
  } else {
    // from iris dataset!
    nRows = get_argval<int>(argv, argv + argc, "-num_rows", 150);
    nCols = get_argval<int>(argv, argv + argc, "-num_cols", 4);
  }
  int maxDepth = get_argval<int>(argv, argv + argc, "-max_depth", 2);
  int nBins = get_argval<int>(argv, argv + argc, "-num_bins", 8);
  if (get_arg(argv, argv + argc, "-h")) {
    printUsage();
    return 0;
  }
  if (dataset.empty()) {
    std::cerr << "'-dataset' option is mandatory!" << std::endl;
    printUsage();
    return -1;
  }
  std::cout << "Creating cumlHandle..." << std::endl;
  ML::cumlHandle cumlHandle;
#ifdef HAVE_RMM
  rmmOptions_t rmmOptions;
  rmmOptions.allocation_mode = PoolAllocation;
  rmmOptions.initial_pool_size = 0;
  rmmOptions.enable_logging = false;
  rmmError_t rmmStatus = rmmInitialize(&rmmOptions);
  if (RMM_SUCCESS != rmmStatus) {
    std::cerr << "WARN: Could not initialize RMM: "
              << rmmGetErrorString(rmmStatus) << std::endl;
    return -1;
  }
#endif  //HAVE_RMM
  setDeviceAllocator(cumlHandle);
  std::cout << "Setting up buffers..." << std::endl;
  float *d_data, *d_labels_reg;
  int* d_labels;
  CUDA_RT_CALL(cudaMalloc((void**)&d_data, sizeof(float) * nRows * nCols));
  if (regression) {
    CUDA_RT_CALL(cudaMalloc((void**)&d_labels_reg, sizeof(float) * nRows));
  } else {
    CUDA_RT_CALL(cudaMalloc((void**)&d_labels, sizeof(int) * nRows));
  }
  cudaStream_t stream;
  CUDA_RT_CALL(cudaStreamCreate(&stream));
  cumlHandle.setStream(stream);
  std::cout << "Loading dataset..." << std::endl;
  if (regression) {
    loadRegressionDataset(dataset, nRows, nCols, d_data, d_labels_reg, stream);
  } else {
    loadClassificationDataset(dataset, nRows, nCols, d_data, d_labels, stream);
  }
  CUDA_RT_CALL(cudaStreamSynchronize(stream));
  std::cout << "Setting up parameters..." << std::endl;
  using namespace ML::DecisionTree;
  DecisionTreeParams params;
  set_tree_params(params, maxDepth, 1 << maxDepth, 1.f, nBins,
                  ML::SPLIT_ALGO::GLOBAL_QUANTILE, 2, 0.f, false,
                  regression ? ML::CRITERION::MSE : ML::CRITERION::GINI, true,
                  false, 128, 10, 4, 0);
  print(params);
  std::cout << "Starting the training..." << std::endl;
  if (regression) {
    ///@todo!!
  } else {
    ///@todo!!
  }
  std::cout << "Cleaning up..." << std::endl;
  if (regression) {
    CUDA_RT_CALL(cudaFree(d_labels_reg));
  } else {
    CUDA_RT_CALL(cudaFree(d_labels));
  }
  CUDA_RT_CALL(cudaFree(d_data));
  CUDA_RT_CALL(cudaStreamDestroy(stream));
  CUDA_RT_CALL(cudaDeviceSynchronize());
  return 0;
}
