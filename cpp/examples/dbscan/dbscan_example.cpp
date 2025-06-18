/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
#include <cuml/cluster/dbscan.hpp>
#include <cuml/common/distance_type.hpp>

#include <raft/core/handle.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                    \
  {                                                                           \
    cudaError_t cudaStatus = call;                                            \
    if (cudaSuccess != cudaStatus)                                            \
      fprintf(stderr,                                                         \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
              "%s (%d).\n",                                                   \
              #call,                                                          \
              __LINE__,                                                       \
              __FILE__,                                                       \
              cudaGetErrorString(cudaStatus),                                 \
              cudaStatus);                                                    \
  }
#endif  // CUDA_RT_CALL

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T default_val)
{
  T argval   = default_val;
  char** itr = std::find(begin, end, arg);
  if (itr != end && ++itr != end) {
    std::istringstream inbuf(*itr);
    inbuf >> argval;
  }
  return argval;
}

bool get_arg(char** begin, char** end, const std::string& arg)
{
  char** itr = std::find(begin, end, arg);
  if (itr != end) { return true; }
  return false;
}

void printUsage()
{
  std::cout << "To run default example use:" << std::endl
            << "    dbscan_example [-dev_id <GPU id>]" << std::endl
            << "For other cases:" << std::endl
            << "    dbscan_example [-dev_id <GPU id>] -input <samples-file> "
            << "-num_samples <number of samples> -num_features <number of features> "
            << "[-min_pts <minimum number of samples in a cluster>] "
            << "[-eps <maximum distance between any two samples of a cluster>] "
            << "[-max_bytes_per_batch <maximum memory to use (in bytes) for batch size "
               "calculation>] "
            << std::endl;
  return;
}

void loadDefaultDataset(std::vector<float>& inputData,
                        size_t& nRows,
                        size_t& nCols,
                        int& minPts,
                        float& eps,
                        size_t& max_bytes_per_batch)
{
  constexpr size_t NUM_ROWS = 25;
  constexpr size_t NUM_COLS = 3;
  constexpr int MIN_PTS     = 2;
  constexpr float EPS       = 1.0f;

  constexpr float data[NUM_ROWS * NUM_COLS] = {
    -7.497668f, 9.218568f,  -4.924911f, 8.001691f,  -2.377415f, -3.496702f, -7.402899f, 9.162857f,
    -4.894407f, -7.590056f, 9.375731f,  -4.762814f, 7.822048f,  -2.388025f, -3.403690f, -7.376115f,
    9.441934f,  -4.801385f, -7.531280f, 9.230399f,  -4.763294f, 8.042177f,  -2.665680f, -3.316565f,
    7.944115f,  -2.557312f, -3.185993f, 7.922114f,  -2.423922f, -3.194180f, 7.897527f,  -2.466402f,
    -3.311819f, -7.569343f, 9.266988f,  -4.779115f, -7.528063f, 9.156666f,  -4.887371f, -7.296247f,
    9.187418f,  -4.754778f, 7.825963f,  -2.351993f, -3.419239f, -7.608446f, 9.386856f,  -4.750009f,
    8.087856f,  -2.330975f, -3.392595f, -7.503101f, 9.391059f,  -4.762857f, 7.936867f,  -2.410410f,
    -3.397487f, -7.565027f, 9.248172f,  -5.000937f, -7.339392f, 9.317035f,  -4.778559f, 7.803362f,
    -2.304214f, -3.173147f, -7.510096f, 9.441537f,  -4.718324f, 8.025255f,  -2.585647f, -3.019001f,
    7.957931f,  -2.547737f, -3.283212f};
  nRows               = NUM_ROWS;
  nCols               = NUM_COLS;
  minPts              = MIN_PTS;
  eps                 = EPS;
  max_bytes_per_batch = 0;  // allow algorithm to set this

  inputData.insert(inputData.begin(), data, data + nRows * nCols);
}

int main(int argc, char* argv[])
{
  int devId         = get_argval<int>(argv, argv + argc, "-dev_id", 0);
  size_t nRows      = get_argval<size_t>(argv, argv + argc, "-num_samples", 0);
  size_t nCols      = get_argval<size_t>(argv, argv + argc, "-num_features", 0);
  std::string input = get_argval<std::string>(argv, argv + argc, "-input", std::string(""));
  int minPts        = get_argval<int>(argv, argv + argc, "-min_pts", 3);
  float eps         = get_argval<float>(argv, argv + argc, "-eps", 1.0f);
  size_t max_bytes_per_batch =
    get_argval<size_t>(argv, argv + argc, "-max_bytes_per_batch", (size_t)13e9);

  {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus             = cudaSetDevice(devId);
    if (cudaSuccess != cudaStatus) {
      std::cerr << "ERROR: Could not select CUDA device with the id: " << devId << "("
                << cudaGetErrorString(cudaStatus) << ")" << std::endl;
      return 1;
    }
    cudaStatus = cudaFree(0);
    if (cudaSuccess != cudaStatus) {
      std::cerr << "ERROR: Could not initialize CUDA on device: " << devId << "("
                << cudaGetErrorString(cudaStatus) << ")" << std::endl;
      return 1;
    }
  }

  std::vector<float> h_inputData;

  if (input == "") {
    // Samples file not specified, run with defaults
    std::cout << "Samples file not specified. (-input option)" << std::endl;
    std::cout << "Running with default dataset:" << std::endl;
    loadDefaultDataset(h_inputData, nRows, nCols, minPts, eps, max_bytes_per_batch);
  } else if (nRows == 0 || nCols == 0) {
    // Samples file specified but nRows and nCols is not specified
    // Print usage and quit
    std::cerr << "Samples file: " << input << std::endl;
    std::cerr << "Incorrect value for (num_samples x num_features): (" << nRows << " x " << nCols
              << ")" << std::endl;
    printUsage();
    return 1;
  } else {
    // All options are correctly specified
    // Try to read input file now
    std::ifstream input_stream(input, std::ios::in);
    if (!input_stream.is_open()) {
      std::cerr << "ERROR: Could not open input file " << input << std::endl;
      return 1;
    }
    std::cout << "Trying to read samples from " << input << std::endl;
    h_inputData.reserve(nRows * nCols);
    float val = 0.0;
    while (input_stream >> val) {
      h_inputData.push_back(val);
    }
    if (h_inputData.size() != nRows * nCols) {
      std::cerr << "ERROR: Read " << h_inputData.size() << " from " << input
                << ", while expecting to read: " << nRows * nCols << " (num_samples*num_features)"
                << std::endl;
      return 1;
    }
  }

  cudaStream_t stream;
  CUDA_RT_CALL(cudaStreamCreate(&stream));
  raft::handle_t handle{stream};

  std::vector<int> h_labels(nRows);
  int* d_labels      = nullptr;
  float* d_inputData = nullptr;

  CUDA_RT_CALL(cudaMalloc(&d_labels, nRows * sizeof(int)));
  CUDA_RT_CALL(cudaMalloc(&d_inputData, nRows * nCols * sizeof(float)));
  CUDA_RT_CALL(cudaMemcpyAsync(d_inputData,
                               h_inputData.data(),
                               nRows * nCols * sizeof(float),
                               cudaMemcpyHostToDevice,
                               stream));

  std::cout << "Running DBSCAN with following parameters:" << std::endl
            << "Number of samples - " << nRows << std::endl
            << "Number of features - " << nCols << std::endl
            << "min_pts - " << minPts << std::endl
            << "eps - " << eps << std::endl
            << "max_bytes_per_batch - " << max_bytes_per_batch << std::endl;

  ML::Dbscan::fit(handle,
                  d_inputData,
                  nRows,
                  nCols,
                  eps,
                  minPts,
                  ML::distance::DistanceType::L2SqrtUnexpanded,
                  d_labels,
                  nullptr,
                  nullptr,
                  max_bytes_per_batch,
                  ML::Dbscan::EpsNnMethod::BRUTE_FORCE,
                  rapids_logger::level_enum::off);
  CUDA_RT_CALL(cudaMemcpyAsync(
    h_labels.data(), d_labels, nRows * sizeof(int), cudaMemcpyDeviceToHost, stream));
  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  std::map<long, size_t> histogram;
  for (int row = 0; row < nRows; row++) {
    if (histogram.find(h_labels[row]) == histogram.end()) {
      histogram[h_labels[row]] = 1;
    } else {
      histogram[h_labels[row]]++;
    }
  }

  size_t nClusters = 0;
  size_t noise     = 0;
  std::cout << "Histogram of samples" << std::endl;
  std::cout << "Cluster id, Number samples" << std::endl;
  for (auto it = histogram.begin(); it != histogram.end(); it++) {
    if (it->first != -1) {
      std::cout << std::setw(10) << it->first << ", " << it->second << std::endl;
      nClusters++;
    } else {
      noise += it->second;
    }
  }

  std::cout << "Total number of clusters: " << nClusters << std::endl;
  std::cout << "Noise samples: " << noise << std::endl;

  CUDA_RT_CALL(cudaFree(d_labels));
  CUDA_RT_CALL(cudaFree(d_inputData));
  CUDA_RT_CALL(cudaStreamDestroy(stream));
  CUDA_RT_CALL(cudaDeviceSynchronize());
  return 0;
}
