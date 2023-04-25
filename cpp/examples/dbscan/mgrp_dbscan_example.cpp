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
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <vector>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuml/cluster/dbscan.hpp>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>

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
            << "    mgrp_dbscan_example [-dev_id <GPU id>]" << std::endl
            << "For other cases:" << std::endl
            << "    mgrp_dbscan_example [-dev_id <GPU id>] -input <samples-file> "
            << "-num_groups <number of groups> "
            << "-num_samples <number of samples> "
            << "-num_features <number of features> "
            << "[-min_pts <minimum number of samples in a cluster>] "
            << "[-eps <maximum distance between any two samples of a cluster>] "
            << "[-metric <1 for CosineExpanded, else for L2sqrtUnexpanded>] "
            << "[-cust_wsp <0 for rmm managed buffer, else for custom workspace buffer] "
            << "[-verb <level of logger>] "
            << "[-max_bytes_per_batch <maximum memory to use (in bytes) for batch size "
               "calculation>] "
            << std::endl;
  return;
}

template <typename DataT>
std::ostream& operator<<(std::ostream& os, const thrust::host_vector<DataT>& data)
{
  for (auto it : data)
    os << it << " ";
  return os;
}

void loadDefaultDataset(std::vector<float>& inputData,
                        size_t& nGroups,
                        size_t& nRows,
                        size_t& nCols,
                        int& minPts,
                        float& eps,
                        size_t& max_bytes_per_batch)
{
  constexpr size_t NUM_GRPS = 3;
  constexpr size_t NUM_ROWS = 25;
  constexpr size_t NUM_COLS = 3;
  constexpr int MIN_PTS     = 2;
  constexpr float EPS       = 1.0f;

  constexpr float data[NUM_GRPS * NUM_ROWS * NUM_COLS] = {
    -7.497668f, 9.218568f,  -4.924911f, 8.001691f,  -2.377415f, -3.496702f, -7.402899f, 9.162857f,
    -4.894407f, -7.590056f, 9.375731f,  -4.762814f, 7.822048f,  -2.388025f, -3.403690f, -7.376115f,
    9.441934f,  -4.801385f, -7.531280f, 9.230399f,  -4.763294f, 8.042177f,  -2.665680f, -3.316565f,
    7.944115f,  -2.557312f, -3.185993f, 7.922114f,  -2.423922f, -3.194180f, 7.897527f,  -2.466402f,
    -3.311819f, -7.569343f, 9.266988f,  -4.779115f, -7.528063f, 9.156666f,  -4.887371f, -7.296247f,
    9.187418f,  -4.754778f, 7.825963f,  -2.351993f, -3.419239f, -7.608446f, 9.386856f,  -4.750009f,
    8.087856f,  -2.330975f, -3.392595f, -7.503101f, 9.391059f,  -4.762857f, 7.936867f,  -2.410410f,
    -3.397487f, -7.565027f, 9.248172f,  -5.000937f, -7.339392f, 9.317035f,  -4.778559f, 7.803362f,
    -2.304214f, -3.173147f, -7.510096f, 9.441537f,  -4.718324f, 8.025255f,  -2.585647f, -3.019001f,
    7.957931f,  -2.547737f, -3.283212f, 8.069495f,  -2.465998f, -3.246544f, 0.885940f,  -2.654126f,
    -1.018474f, -7.631181f, 9.242371f,  -4.794718f, -2.837777f, -5.395402f, 5.538133f,  -5.049445f,
    -3.171856f, 9.684905f,  5.011832f,  -1.756882f, -3.168749f, -5.297946f, -7.101916f, 1.724579f,
    -5.012668f, -3.019594f, 9.623699f,  8.270521f,  -9.426013f, 7.332846f,  1.639379f,  -8.760420f,
    1.830559f,  -7.291831f, 7.189862f,  6.366914f,  7.994307f,  -2.428711f, -3.120755f, 1.842419f,
    -7.103554f, 7.178880f,  -1.972615f, 2.198118f,  -7.696833f, 1.737913f,  -7.209247f, 7.047458f,
    -2.796221f, -5.521312f, 5.491510f,  1.446300f,  -8.715239f, 1.896955f,  -1.067057f, 6.711888f,
    -7.342354f, -7.497436f, 7.332180f,  6.558944f,  -5.348785f, -7.041458f, 1.728055f,  -1.085605f,
    6.796169f,  -7.504171f, 0.862283f,  -2.541425f, -1.097588f, -4.268553f, -0.597631f, 7.444093f,
    -6.580936f, -1.914008f, 2.827801f,  -7.531569f, 9.501262f,  -4.686692f, 7.830602f,  -4.767288f,
    -1.379315f, 2.072869f,  1.145346f,  3.840629f,  -0.229785f, 4.054196f,  -6.561411f, -9.751370f,
    -0.625692f, 2.580033f,  8.300755f,  9.995937f,  -4.128877f, 0.953095f,  -4.868012f, 5.830784f,
    -2.770719f, 3.010745f,  3.732297f,  0.230208f,  8.106454f,  -6.575977f, 6.787752f,  7.759267f,
    9.666306f,  -3.025806f, -8.178691f, 0.313673f,  8.173984f,  -4.595605f, -1.492277f, -0.340185f,
    9.202333f,  -7.853731f, 1.991260f,  3.752417f,  9.296068f,  2.109616f,  3.774991f,  9.226852f,
    -5.685891f, 5.654886f,  7.176744f,  0.354228f,  7.994700f,  -6.671954f, -9.734419f, -0.810781f,
    2.761060f,  -2.841335f, -8.061689f, 0.412604f,  -5.625323f, 5.588506f,  7.097187f,  0.659263f,
    -4.742221f, 5.800708f,  8.292029f,  2.196709f,  7.590376f,  2.030531f,  1.021889f,  3.648936f,
    8.328954f,  -0.093709f, 4.517338f,  8.187078f,  -0.073398f, 4.464756f,  8.535991f,  9.996729f,
    -3.943532f};
  nGroups             = NUM_GRPS;
  nRows               = NUM_ROWS;
  nCols               = NUM_COLS;
  minPts              = MIN_PTS;
  eps                 = EPS;
  max_bytes_per_batch = 0;  // allow algorithm to set this

  inputData.insert(inputData.begin(), data, data + nGroups * nRows * nCols);
}

int main(int argc, char* argv[])
{
  int devId         = get_argval<int>(argv, argv + argc, "-dev_id", 0);
  size_t nGroups    = get_argval<size_t>(argv, argv + argc, "-num_groups", 1);
  size_t nRows      = get_argval<size_t>(argv, argv + argc, "-num_samples", 0);
  size_t nCols      = get_argval<size_t>(argv, argv + argc, "-num_features", 0);
  std::string input = get_argval<std::string>(argv, argv + argc, "-input", std::string(""));
  int minPts        = get_argval<int>(argv, argv + argc, "-min_pts", 3);
  float eps         = get_argval<float>(argv, argv + argc, "-eps", 1.0f);
  bool bCustomWsp   = static_cast<bool>(get_argval<int>(argv, argv + argc, "-cust_wsp", 0));
  int bMetric       = get_argval<int>(argv, argv + argc, "-metric", 0);
  int verbosity     = get_argval<int>(argv, argv + argc, "-verb", CUML_LEVEL_INFO);
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

  thrust::host_vector<size_t> vRows;
  for (size_t i = 0; i < nGroups; ++i) {
    size_t nRowsPerGrp = nRows;
    nRowsPerGrp        = std::max(nRowsPerGrp, (size_t)1);
    vRows.push_back(nRowsPerGrp);
  }
  assert(nGroups <= vRows.size());
  size_t nTotalRows = thrust::reduce(thrust::host, vRows.begin(), vRows.end(), 0);

  std::vector<float> h_inputData;

  if (input == "") {
    // Samples file not specified, run with defaults
    std::cout << "Samples file not specified. (-input option)" << std::endl;
    std::cout << "Running with default dataset:" << std::endl;
    loadDefaultDataset(h_inputData, nGroups, nRows, nCols, minPts, eps, max_bytes_per_batch);
    vRows.clear();
    for (size_t i = 0; i < nGroups; ++i) {
      size_t nRowsPerGrp = nRows;
      nRowsPerGrp        = std::max(nRowsPerGrp, (size_t)1);
      vRows.push_back(nRowsPerGrp);
    }
    nTotalRows = thrust::reduce(thrust::host, vRows.begin(), vRows.end(), 0);
  } else if (nGroups == 0 || vRows.empty() || nCols == 0) {
    // Samples file specified but nRows and nCols is not specified
    // Print usage and quit
    std::cerr << "Samples file: " << input << std::endl;
    std::cerr << "Incorrect value for (num_groups, num_samples, num_features): (" << nGroups << ", "
              << nTotalRows << ", " << nCols << ")" << std::endl;
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
    h_inputData.reserve(nTotalRows * nCols);
    float val = 0.0;
    for (int g = 0; g < nGroups; ++g) {
      for (int n = 0; n < nRows * nCols; ++n) {
        if (!(input_stream >> val && h_inputData.size() <= nTotalRows * nCols)) break;
        if (n >= vRows[g] * nCols) continue;
        h_inputData.push_back(val);
      }
    }
    if (h_inputData.size() != nTotalRows * nCols) {
      std::cerr << "ERROR: Read " << h_inputData.size() << " from " << input
                << ", while expecting to read: " << nTotalRows * nCols
                << " (num_samples*num_features)" << std::endl;
      return 1;
    }
  }

  cudaStream_t stream;
  CUDA_RT_CALL(cudaStreamCreate(&stream));
  raft::handle_t handle{stream};

  auto metric = (bMetric == 1) ? raft::distance::CosineExpanded : raft::distance::L2SqrtUnexpanded;
  std::string metricMsg = (bMetric == 1) ? "CosineExpanded" : "L2SqrtUnexpanded";
  std::cout << "Running DBSCAN with following parameters:" << std::endl
            << "Number of groups - " << nGroups << std::endl
            << "Number of samples in all groups - [" << vRows << "] sum: " << nTotalRows
            << std::endl
            << "Number of features - " << nCols << std::endl
            << "min_pts - " << minPts << std::endl
            << "eps - " << eps << std::endl
            << "metric - " << metricMsg << std::endl
            << "max_bytes_per_batch - " << max_bytes_per_batch << std::endl;

  std::vector<int> h_labels(nTotalRows);
  int* d_labels      = nullptr;
  float* d_inputData = nullptr;
  void* workspace    = nullptr;

  std::vector<float> vEps(nGroups, eps);
  std::vector<int> vMinPts(nGroups, minPts);
  std::vector<int> vCastRows(nGroups, 0);
  for (int i = 0; i < nGroups; ++i)
    vCastRows[i] = static_cast<int>(vRows[i]);

  CUDA_RT_CALL(cudaMalloc(&d_labels, nTotalRows * sizeof(int)));
  CUDA_RT_CALL(cudaMalloc(&d_inputData, nTotalRows * nCols * sizeof(float)));
  CUDA_RT_CALL(cudaMemcpyAsync(d_inputData,
                               h_inputData.data(),
                               nTotalRows * nCols * sizeof(float),
                               cudaMemcpyHostToDevice,
                               stream));

  if (bCustomWsp) {
    size_t workspace_size = 0;
    ML::Dbscan::fit(handle,
                    d_inputData,
                    nGroups,
                    vCastRows.data(),
                    nCols,
                    vEps.data(),
                    vMinPts.data(),
                    metric,
                    nullptr,
                    nullptr,
                    max_bytes_per_batch,
                    verbosity,
                    nullptr,
                    &workspace_size,
                    false);
    CUDA_RT_CALL(cudaMalloc(&workspace, workspace_size));
    std::cout << "Alloc <<" << workspace_size << "bytes buffer: " << workspace;
  }
  std::cout << "=== Run Dbscan (multi groups) ===" << std::endl;
  ML::Dbscan::fit(handle,
                  d_inputData,
                  nGroups,
                  vCastRows.data(),
                  nCols,
                  vEps.data(),
                  vMinPts.data(),
                  metric,
                  d_labels,
                  nullptr,
                  max_bytes_per_batch,
                  verbosity,
                  workspace,
                  nullptr,
                  false);

  if (bCustomWsp) { CUDA_RT_CALL(cudaFree(workspace)); }

  CUDA_RT_CALL(cudaMemcpyAsync(
    h_labels.data(), d_labels, nTotalRows * sizeof(int), cudaMemcpyDeviceToHost, stream));
  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  for (int group = 0, offset = 0; group < nGroups; ++group) {
    int nRowsPerGrp = vCastRows[group];
    std::map<long, size_t> histogram;
    for (int row = 0; row < nRowsPerGrp; row++) {
      if (histogram.find(h_labels[offset + row]) == histogram.end()) {
        histogram[h_labels[offset + row]] = 1;
      } else {
        histogram[h_labels[offset + row]]++;
      }
    }
    offset += nRowsPerGrp;

    size_t nClusters = 0;
    size_t noise     = 0;
    std::cout << "Group: " << group << std::endl;
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
  }

  CUDA_RT_CALL(cudaFree(d_labels));
  CUDA_RT_CALL(cudaFree(d_inputData));
  CUDA_RT_CALL(cudaStreamDestroy(stream));
  CUDA_RT_CALL(cudaDeviceSynchronize());
  return 0;
}