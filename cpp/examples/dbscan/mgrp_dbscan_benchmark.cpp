/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <utility>
#include <vector>

#include <raft/core/nvtx.hpp>

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
            << "    dbscan_example [-dev_id <GPU id>]" << std::endl
            << "For other cases:" << std::endl
            << "    dbscan_example [-dev_id <GPU id>] -input <samples-file> "
            << "-num_samples <number of samples per group> -num_features <number of features> "
               "-num_groups <number of groups>"
            << "[-min_pts <minimum number of samples in a cluster>] "
            << "[-eps <maximum distance between any two samples of a cluster>] "
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

template <typename DataT>
double getHostDiff(const DataT* src1, const DataT* src2, int size)
{
  double diff = 0.f;
  for (int i = 0; i < size; ++i) {
    diff += std::abs(static_cast<double>(src1[i] - src2[i]));
  }
  return diff;
}

template <typename DataT>
double getDevDiff(const DataT* d_src1, const DataT* d_src2, int size)
{
  double diff = 0.f;
  DataT* src1 = new DataT[size];
  DataT* src2 = new DataT[size];
  CUDA_RT_CALL(cudaMemcpy(src1, d_src1, size * sizeof(DataT), cudaMemcpyDeviceToHost));
  CUDA_RT_CALL(cudaMemcpy(src2, d_src2, size * sizeof(DataT), cudaMemcpyDeviceToHost));
  for (int i = 0; i < size; ++i) {
    diff += std::abs(static_cast<double>(src1[i] - src2[i]));
  }
  delete[] src1;
  delete[] src2;
  return diff;
}

template <typename DataT>
void printVec(const DataT* src, int size)
{
  if (src == nullptr) return;
  std::cout << "[ ";
  for (int i = 0; i < size; ++i) {
    std::cout << src[i] << " ";
  }
  std::cout << "]" << std::endl;
}

void generateDefaultDataset(std::vector<float>& inputData,
                            size_t batchSize,
                            size_t* pNbRows,
                            size_t nCols,
                            int minPts,
                            float eps,
                            size_t& max_bytes_per_batch)
{
  constexpr size_t MAX_NUM_ROWS = 10000;
  constexpr size_t MAX_NUM_COLS = 1024;
  size_t nRows                  = 0;
  for (int b = 0; b < batchSize; ++b) {
    nRows += pNbRows[b];
  }
  assert(nRows <= MAX_NUM_ROWS);
  assert(nCols <= MAX_NUM_COLS);

  std::vector<float> vData(nRows * nCols, 0.f);
  std::srand(std::time(nullptr));
  for (auto& it : vData) {
    // it = static_cast<float>(std::rand()) / RAND_MAX * 1.0 - 0.5;
    it = static_cast<float>(std::rand()) / RAND_MAX * 3.0 - 1.5;
  }

  max_bytes_per_batch = 0;  // allow algorithm to set this
  inputData.assign(vData.begin(), vData.end());
}

template <typename Index_t = int>
void run_sg_dbscan(const raft::handle_t& handle,
                   float* d_inputData,
                   Index_t nGroups,
                   Index_t nTotalRows,
                   Index_t nCols,
                   Index_t* h_nRows,
                   const float* pEps,
                   const Index_t* pMinPts,
                   Index_t* d_labels,
                   Index_t* d_corepts_indices,
                   raft::distance::DistanceType metric,
                   size_t& max_bytes_per_batch,
                   void* workspace,
                   const Index_t nWarmup,
                   const Index_t nLoops,
                   double& cpuLatency,
                   double& gpuLatency,
                   int verbosity)
{
  cudaStream_t stream = handle.get_stream();
  thrust::cuda::par.on(stream);
  cudaEvent_t start_event, stop_event;
  CUDA_RT_CALL(cudaEventCreate(&start_event));
  CUDA_RT_CALL(cudaEventCreate(&stop_event));

  thrust::host_vector<Index_t> pfx_nRows(nGroups, 0);
  thrust::exclusive_scan(
    thrust::host, h_nRows, h_nRows + nGroups, thrust::raw_pointer_cast(pfx_nRows.data()));

  std::cout << "=== Run Dbscan (native) ===" << std::endl;
  for (Index_t g = 0; g < nGroups; ++g) {
    Index_t nRows   = h_nRows[g];
    float* input    = d_inputData + pfx_nRows[g] * nCols;
    float eps       = pEps[g];
    Index_t minPts  = pMinPts[g];
    Index_t* labels = d_labels + pfx_nRows[g];
    Index_t* core_pts_indices =
      (d_corepts_indices == nullptr) ? nullptr : d_corepts_indices + pfx_nRows[g];
    ML::Dbscan::fit(handle,
                    input,
                    nRows,
                    nCols,
                    eps,
                    minPts,
                    metric,
                    labels,
                    core_pts_indices,
                    max_bytes_per_batch,
                    verbosity,
                    false);
  }

  cpuLatency = 0;
  gpuLatency = 0;
  for (Index_t i = 0; i < nWarmup; ++i) {
    for (Index_t g = 0; g < nGroups; ++g) {
      Index_t nRows  = h_nRows[g];
      float* input   = d_inputData + pfx_nRows[g] * nCols;
      float eps      = pEps[g];
      Index_t minPts = pMinPts[g];
      ML::Dbscan::fit(handle,
                      input,
                      nRows,
                      nCols,
                      eps,
                      minPts,
                      metric,
                      reinterpret_cast<Index_t*>(workspace),
                      nullptr,
                      max_bytes_per_batch,
                      verbosity,
                      false);
    }
  }

  for (Index_t i = 0; i < nLoops; ++i) {
    float epoch_time  = 0.f;
    auto cpuStartTime = std::chrono::high_resolution_clock::now();
    CUDA_RT_CALL(cudaEventRecord(start_event, stream));
    for (Index_t g = 0; g < nGroups; ++g) {
      Index_t nRows  = h_nRows[g];
      float* input   = d_inputData + pfx_nRows[g] * nCols;
      float eps      = pEps[g];
      Index_t minPts = pMinPts[g];
      ML::Dbscan::fit(handle,
                      input,
                      nRows,
                      nCols,
                      eps,
                      minPts,
                      metric,
                      reinterpret_cast<Index_t*>(workspace),
                      nullptr,
                      max_bytes_per_batch,
                      verbosity,
                      false);
    }
    CUDA_RT_CALL(cudaEventRecord(stop_event, stream));
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    cpuLatency += (std::chrono::duration_cast<std::chrono::duration<double>>(
                     std::chrono::high_resolution_clock::now() - cpuStartTime)
                     .count()) *
                  1000;
    CUDA_RT_CALL(cudaEventElapsedTime(&epoch_time, start_event, stop_event));
    gpuLatency += epoch_time;
  }
  cpuLatency /= nLoops;
  gpuLatency /= nLoops;
  return;
}

template <typename Index_t = int>
void run_mg_dbscan(const raft::handle_t& handle,
                   float* d_inputData,
                   Index_t nGroups,
                   Index_t nTotalRows,
                   Index_t nCols,
                   Index_t* h_nRows,
                   float* pEps,
                   Index_t* pMinPts,
                   Index_t* d_labels,
                   Index_t* d_corepts_indices,
                   raft::distance::DistanceType metric,
                   size_t& max_bytes_per_batch,
                   void* workspace,
                   const Index_t nWarmup,
                   const Index_t nLoops,
                   double& cpuLatency,
                   double& gpuLatency,
                   int verbosity,
                   bool useCustomWorkspace = false)
{
  cudaStream_t stream = handle.get_stream();
  thrust::cuda::par.on(stream);
  cudaEvent_t start_event, stop_event;
  CUDA_RT_CALL(cudaEventCreate(&start_event));
  CUDA_RT_CALL(cudaEventCreate(&stop_event));

  void* workspace_buffer = nullptr;
  if (useCustomWorkspace) {
    size_t workspace_size = 0;
    ML::Dbscan::fit(handle,
                    d_inputData,
                    nGroups,
                    h_nRows,
                    nCols,
                    pEps,
                    pMinPts,
                    metric,
                    nullptr,
                    nullptr,
                    max_bytes_per_batch,
                    verbosity,
                    nullptr,
                    &workspace_size,
                    false);
    // std::printf("Alloc %lu bytes buffer: %p\n", workspace_size, workspace_buffer);
    CUDA_RT_CALL(cudaMalloc(&workspace_buffer, workspace_size));
  }
  CUDA_RT_CALL(cudaStreamSynchronize(stream));

  std::cout << "=== Run Dbscan (multi groups) ===" << std::endl;
  ML::Dbscan::fit(handle,
                  d_inputData,
                  nGroups,
                  h_nRows,
                  nCols,
                  pEps,
                  pMinPts,
                  metric,
                  d_labels,
                  d_corepts_indices,
                  max_bytes_per_batch,
                  verbosity,
                  workspace_buffer,
                  nullptr,
                  false);

  cpuLatency = 0;
  gpuLatency = 0;
  for (Index_t i = 0; i < nWarmup; ++i) {
    ML::Dbscan::fit(handle,
                    d_inputData,
                    nGroups,
                    h_nRows,
                    nCols,
                    pEps,
                    pMinPts,
                    metric,
                    reinterpret_cast<Index_t*>(workspace),
                    nullptr,
                    max_bytes_per_batch,
                    verbosity,
                    workspace_buffer,
                    nullptr,
                    false);
  }

  for (Index_t i = 0; i < nLoops; ++i) {
    float epoch_time  = 0.f;
    auto cpuStartTime = std::chrono::high_resolution_clock::now();
    CUDA_RT_CALL(cudaEventRecord(start_event, stream));
    ML::Dbscan::fit(handle,
                    d_inputData,
                    nGroups,
                    h_nRows,
                    nCols,
                    pEps,
                    pMinPts,
                    metric,
                    reinterpret_cast<Index_t*>(workspace),
                    nullptr,
                    max_bytes_per_batch,
                    verbosity,
                    workspace_buffer,
                    nullptr,
                    false);
    CUDA_RT_CALL(cudaEventRecord(stop_event, stream));
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
    cpuLatency += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::high_resolution_clock::now() - cpuStartTime)
                                        .count()) /
                  1000;
    CUDA_RT_CALL(cudaEventElapsedTime(&epoch_time, start_event, stop_event));
    gpuLatency += epoch_time;
  }
  cpuLatency /= nLoops;
  gpuLatency /= nLoops;
  if (useCustomWorkspace) { CUDA_RT_CALL(cudaFree(workspace_buffer)); }
  return;
}

int main(int argc, char* argv[])
{
  int devId         = get_argval<int>(argv, argv + argc, "-dev_id", 0);
  size_t nGroups    = get_argval<size_t>(argv, argv + argc, "-num_groups", 1);
  size_t maxRows    = get_argval<size_t>(argv, argv + argc, "-num_samples", 20);
  size_t nCols      = get_argval<size_t>(argv, argv + argc, "-num_features", 3);
  std::string input = get_argval<std::string>(argv, argv + argc, "-input", std::string(""));
  int minPts        = get_argval<int>(argv, argv + argc, "-min_pts", 3);
  float eps         = get_argval<float>(argv, argv + argc, "-eps", 1.0f);
  int nLoops        = get_argval<int>(argv, argv + argc, "-num_loops", 0);
  bool bStaticShape = static_cast<bool>(get_argval<int>(argv, argv + argc, "-static", 0));
  bool bNoWarmup    = static_cast<bool>(get_argval<int>(argv, argv + argc, "-nowarmup", 1));
  bool bPrint       = static_cast<bool>(get_argval<int>(argv, argv + argc, "-print", 0));
  bool bCustomWsp   = static_cast<bool>(get_argval<int>(argv, argv + argc, "-cust", 0));
  int bMetric       = get_argval<int>(argv, argv + argc, "-metric", 0);
  int verbosity     = get_argval<int>(argv, argv + argc, "-verb", CUML_LEVEL_INFO);
  size_t max_bytes_per_batch =
    get_argval<size_t>(argv, argv + argc, "-max_bytes_per_batch", (size_t)13e9);
  const int nWarmup = (bNoWarmup) ? 0 : 30;
  size_t nRows      = (bStaticShape) ? maxRows : maxRows * 2 / 3;

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

  std::srand(std::time(nullptr));
  thrust::host_vector<size_t> vRows;
  for (size_t i = 0; i < nGroups; ++i) {
    size_t nRowsPG = (bStaticShape) ? nRows : (rand() % nRows + rand() % maxRows) / 2;
    nRowsPG        = std::max(nRowsPG, (size_t)1);
    vRows.push_back(nRowsPG);
  }
  assert(nGroups <= vRows.size());
  size_t nTotalRows = thrust::reduce(thrust::host, vRows.begin(), vRows.end(), 0);

  std::vector<float> h_inputData;
  if (input == "") {
    // Samples file not specified, run with defaults
    std::cout << "Samples file not specified. (-input option)" << std::endl;
    std::cout << "Running with default dataset:" << std::endl;
    generateDefaultDataset(h_inputData,
                           nGroups,
                           thrust::raw_pointer_cast(vRows.data()),
                           nCols,
                           minPts,
                           eps,
                           max_bytes_per_batch);
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
      for (int n = 0; n < maxRows * nCols; ++n) {
        if (!(input_stream >> val && h_inputData.size() <= nTotalRows * nCols)) break;
        if (n >= vRows[g] * nCols) continue;
        h_inputData.push_back(val);
      }
    }

    // while (input_stream >> val && h_inputData.size() <= nTotalRows * nCols) {
    //   h_inputData.push_back(val);
    // }
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

  std::cout << "Running DBSCAN with following parameters:" << std::endl
            << "Number of loops - " << nLoops << std::endl
            << "Number of groups - " << nGroups << std::endl
            << "Number of samples in all groups - [" << vRows << "] sum: " << nTotalRows
            << std::endl
            << "Number of features - " << nCols << std::endl
            << "min_pts - " << minPts << std::endl
            << "eps - " << eps << std::endl
            << "max_bytes_per_batch - " << max_bytes_per_batch << std::endl;

  float* d_inputData_sg = nullptr;
  float* d_inputData_mg = nullptr;
  CUDA_RT_CALL(cudaMalloc(&d_inputData_sg, nTotalRows * nCols * sizeof(float)));
  CUDA_RT_CALL(cudaMalloc(&d_inputData_mg, nTotalRows * nCols * sizeof(float)));
  CUDA_RT_CALL(cudaMemcpyAsync(d_inputData_sg,
                               h_inputData.data(),
                               nTotalRows * nCols * sizeof(float),
                               cudaMemcpyHostToDevice,
                               stream));
  CUDA_RT_CALL(cudaMemcpyAsync(d_inputData_mg,
                               h_inputData.data(),
                               nTotalRows * nCols * sizeof(float),
                               cudaMemcpyHostToDevice,
                               stream));

  if (bPrint)
    std::printf("Diff in input: %lf\n",
                getDevDiff(d_inputData_sg, d_inputData_mg, nTotalRows * nCols));

  {
    using Index_t = int;
    // auto metric = raft::distance::L2SqrtUnexpanded;
    auto metric =
      (bMetric == 1) ? raft::distance::CosineExpanded : raft::distance::L2SqrtUnexpanded;
    std::printf("Running with metric %s\n", (bMetric == 1) ? "CosineExpanded" : "L2SqrtUnexpanded");
    std::vector<float> vEps(nGroups, eps);
    std::vector<Index_t> vMinPts(nGroups, minPts);

    Index_t* labels_sg             = nullptr;
    Index_t* corepoints_indices_sg = nullptr;
    Index_t* labels_mg             = nullptr;
    Index_t* corepoints_indices_mg = nullptr;
    void* workspace                = nullptr;
    CUDA_RT_CALL(cudaMalloc(&labels_sg, nTotalRows * sizeof(Index_t)));
    CUDA_RT_CALL(cudaMalloc(&corepoints_indices_sg, nTotalRows * sizeof(Index_t)));
    CUDA_RT_CALL(cudaMalloc(&labels_mg, nTotalRows * sizeof(Index_t)));
    CUDA_RT_CALL(cudaMalloc(&corepoints_indices_mg, nTotalRows * sizeof(Index_t)));
    CUDA_RT_CALL(cudaMalloc(&workspace, nTotalRows * sizeof(Index_t)));

    std::vector<Index_t> vTyRows(nGroups, 0);
    for (int i = 0; i < nGroups; ++i)
      vTyRows[i] = static_cast<Index_t>(vRows[i]);
    double cpuSgLatency, gpuSgLatency, cpuMgLatency, gpuMgLatency;
    raft::common::nvtx::push_range("Trace::Example::run_sg_dbscan");
    run_sg_dbscan<Index_t>(handle,
                           d_inputData_sg,
                           nGroups,
                           nTotalRows,
                           nCols,
                           vTyRows.data(),
                           vEps.data(),
                           vMinPts.data(),
                           labels_sg,
                           corepoints_indices_sg,
                           metric,
                           max_bytes_per_batch,
                           workspace,
                           nWarmup,
                           nLoops,
                           cpuSgLatency,
                           gpuSgLatency,
                           verbosity);
    raft::common::nvtx::pop_range();

    raft::common::nvtx::push_range("Trace::Example::run_mg_dbscan");
    run_mg_dbscan<Index_t>(handle,
                           d_inputData_mg,
                           nGroups,
                           nTotalRows,
                           nCols,
                           vTyRows.data(),
                           vEps.data(),
                           vMinPts.data(),
                           labels_mg,
                           corepoints_indices_mg,
                           metric,
                           max_bytes_per_batch,
                           workspace,
                           nWarmup,
                           nLoops,
                           cpuMgLatency,
                           gpuMgLatency,
                           verbosity,
                           bCustomWsp);
    raft::common::nvtx::pop_range();

    // Compare the results
    Index_t* h_labels_sg  = new Index_t[nTotalRows];
    Index_t* h_labels_mg  = new Index_t[nTotalRows];
    Index_t* h_corepts_sg = new Index_t[nTotalRows];
    Index_t* h_corepts_mg = new Index_t[nTotalRows];

    CUDA_RT_CALL(
      cudaMemcpy(h_labels_sg, labels_sg, nTotalRows * sizeof(Index_t), cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(
      cudaMemcpy(h_labels_mg, labels_mg, nTotalRows * sizeof(Index_t), cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaMemcpy(
      h_corepts_sg, corepoints_indices_sg, nTotalRows * sizeof(Index_t), cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaMemcpy(
      h_corepts_mg, corepoints_indices_mg, nTotalRows * sizeof(Index_t), cudaMemcpyDeviceToHost));

    std::vector<Index_t> vResDiff(nGroups, 0);
    size_t startRow = 0;
    for (int g = 0; g < nGroups; ++g) {
      thrust::host_vector<Index_t> h_diff(vRows[g]);
      for (int i = 0; i < vRows[g]; ++i) {
        h_diff[i] = std::abs(h_labels_sg[startRow + i] - h_labels_mg[startRow + i]);
      }
      vResDiff[g] = thrust::reduce(thrust::host, h_diff.begin(), h_diff.end());
      startRow += vRows[g];
    }
    Index_t sum_diff =
      thrust::reduce(thrust::host, vResDiff.data(), vResDiff.data() + vResDiff.size());
    std::cout << "diff: " << sum_diff << " ===> (";
    std::for_each(vResDiff.begin(), vResDiff.end(), [](Index_t x) { std::cout << x << " "; });
    std::cout << ")" << std::endl;

    std::printf(
      "Latency: on cpu %.4lf ms, on gpu %.4lf ms (single);\n"
      "Latency: on cpu %.4lf ms, on gpu %.4lf ms (multi);\n"
      "Speedup: on cpu %.4lf x, on gpu %.4lf x\n",
      cpuSgLatency,
      gpuSgLatency,
      cpuMgLatency,
      gpuMgLatency,
      cpuSgLatency / cpuMgLatency,
      gpuSgLatency / gpuMgLatency);

    CUDA_RT_CALL(cudaFree(labels_sg));
    CUDA_RT_CALL(cudaFree(corepoints_indices_sg));
    CUDA_RT_CALL(cudaFree(labels_mg));
    CUDA_RT_CALL(cudaFree(corepoints_indices_mg));
    CUDA_RT_CALL(cudaFree(workspace));

    delete[] h_labels_sg;
    delete[] h_labels_mg;
    delete[] h_corepts_sg;
    delete[] h_corepts_mg;
  }

  CUDA_RT_CALL(cudaFree(d_inputData_sg));
  CUDA_RT_CALL(cudaFree(d_inputData_mg));
  CUDA_RT_CALL(cudaStreamDestroy(stream));
  return 0;
}