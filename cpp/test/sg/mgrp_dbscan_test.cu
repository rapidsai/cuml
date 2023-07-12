/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <gtest/gtest.h>

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

#include <cuml/cluster/dbscan.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/datasets/make_blobs.hpp>
#include <cuml/metrics/metrics.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>

#include <test_utils.h>

template <typename T, typename Index_t = int>
void run_sg_dbscan(const raft::handle_t& handle,
                   T* d_inputData,
                   Index_t nGroups,
                   Index_t nTotalRows,
                   Index_t nCols,
                   Index_t* h_nRows,
                   const T* pEps,
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
  RAFT_CUDA_TRY(cudaEventCreate(&start_event));
  RAFT_CUDA_TRY(cudaEventCreate(&stop_event));

  thrust::host_vector<Index_t> pfx_nRows(nGroups, 0);
  thrust::exclusive_scan(
    thrust::host, h_nRows, h_nRows + nGroups, thrust::raw_pointer_cast(pfx_nRows.data()));

  std::cout << "=== Run Dbscan (native) ===" << std::endl;
  for (Index_t g = 0; g < nGroups; ++g) {
    Index_t nRows   = h_nRows[g];
    T* input        = d_inputData + pfx_nRows[g] * nCols;
    T eps           = pEps[g];
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
      T* input       = d_inputData + pfx_nRows[g] * nCols;
      T eps          = pEps[g];
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
    RAFT_CUDA_TRY(cudaEventRecord(start_event, stream));
    for (Index_t g = 0; g < nGroups; ++g) {
      Index_t nRows  = h_nRows[g];
      T* input       = d_inputData + pfx_nRows[g] * nCols;
      T eps          = pEps[g];
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
    RAFT_CUDA_TRY(cudaEventRecord(stop_event, stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    cpuLatency += (std::chrono::duration_cast<std::chrono::duration<double>>(
                     std::chrono::high_resolution_clock::now() - cpuStartTime)
                     .count()) *
                  1000;
    RAFT_CUDA_TRY(cudaEventElapsedTime(&epoch_time, start_event, stop_event));
    gpuLatency += epoch_time;
  }
  cpuLatency /= nLoops;
  gpuLatency /= nLoops;
  return;
}

template <typename T, typename Index_t = int>
void run_mg_dbscan(const raft::handle_t& handle,
                   T* d_inputData,
                   Index_t nGroups,
                   Index_t nTotalRows,
                   Index_t nCols,
                   Index_t* h_nRows,
                   T* pEps,
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
  RAFT_CUDA_TRY(cudaEventCreate(&start_event));
  RAFT_CUDA_TRY(cudaEventCreate(&stop_event));

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
    RAFT_CUDA_TRY(cudaMalloc(&workspace_buffer, workspace_size));
  }
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

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
    RAFT_CUDA_TRY(cudaEventRecord(start_event, stream));
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
    RAFT_CUDA_TRY(cudaEventRecord(stop_event, stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    cpuLatency += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                        std::chrono::high_resolution_clock::now() - cpuStartTime)
                                        .count()) /
                  1000;
    RAFT_CUDA_TRY(cudaEventElapsedTime(&epoch_time, start_event, stop_event));
    gpuLatency += epoch_time;
  }
  cpuLatency /= nLoops;
  gpuLatency /= nLoops;
  if (useCustomWorkspace) { RAFT_CUDA_TRY(cudaFree(workspace_buffer)); }
  return;
}

namespace ML {
template <typename T, typename IdxT = int>
struct MgrpDbscanInputs {
  const T* points;
  IdxT n_groups;
  const IdxT* n_rows_ptr;
  IdxT n_cols;
  const T* eps_ptr;
  const IdxT* min_pts_ptr;
  raft::distance::DistanceType metric;
  bool bCustomWsp;
};

typedef struct {
  double hostSgrpLatency;
  double devSgrpLatency;
  double hostMgrpLatency;
  double devMgrpLatency;
} clk;

template <typename T, typename IdxT>
class MgrpDbscanTest : public ::testing::TestWithParam<MgrpDbscanInputs<T, IdxT>> {
 protected:
  void basicTest()
  {
    raft::handle_t handle;
    auto stream = handle.get_stream();

    params = ::testing::TestWithParam<MgrpDbscanInputs<T, IdxT>>::GetParam();

    raft::distance::DistanceType metric = params.metric;
    size_t max_bytes_per_batch          = (size_t)13e9;
    int verbosity                       = CUML_LEVEL_INFO;
    const int nWarmup                   = 5;
    const int nLoops                    = 100;

    IdxT sum_rows =
      thrust::reduce(thrust::host, params.n_rows_ptr, params.n_rows_ptr + params.n_groups);
    std::vector<IdxT> pfxsum_rows(params.n_groups + 1, 0);
    thrust::exclusive_scan(thrust::host,
                           params.n_rows_ptr,
                           params.n_rows_ptr + (params.n_groups + 1),
                           pfxsum_rows.data());

    rmm::device_uvector<T> inputs_bsl(sum_rows * params.n_cols, stream);
    rmm::device_uvector<T> inputs_mgrp(sum_rows * params.n_cols, stream);
    rmm::device_uvector<IdxT> labels_bsl(sum_rows, stream);
    rmm::device_uvector<IdxT> labels_mgrp(sum_rows, stream);
    rmm::device_uvector<IdxT> coreinds_bsl(sum_rows, stream);
    rmm::device_uvector<IdxT> coreinds_mgrp(sum_rows, stream);
    rmm::device_uvector<IdxT> workspace(sum_rows, stream);

    raft::copy(inputs_bsl.data(), params.points, sum_rows * params.n_cols, stream);
    raft::copy(inputs_mgrp.data(), params.points, sum_rows * params.n_cols, stream);

    handle.sync_stream(stream);
    raft::common::nvtx::push_range("Trace::Example::run_sg_dbscan");
    run_sg_dbscan<T, IdxT>(handle,
                           inputs_bsl.data(),
                           params.n_groups,
                           sum_rows,
                           params.n_cols,
                           (IdxT*)params.n_rows_ptr,
                           (T*)params.eps_ptr,
                           (IdxT*)params.min_pts_ptr,
                           labels_bsl.data(),
                           coreinds_bsl.data(),
                           metric,
                           max_bytes_per_batch,
                           workspace.data(),
                           nWarmup,
                           nLoops,
                           latency.hostSgrpLatency,
                           latency.devSgrpLatency,
                           verbosity);
    raft::common::nvtx::pop_range();

    raft::common::nvtx::push_range("Trace::Example::run_mg_dbscan");
    run_mg_dbscan<T, IdxT>(handle,
                           inputs_mgrp.data(),
                           params.n_groups,
                           sum_rows,
                           params.n_cols,
                           (IdxT*)params.n_rows_ptr,
                           (T*)params.eps_ptr,
                           (IdxT*)params.min_pts_ptr,
                           labels_mgrp.data(),
                           coreinds_mgrp.data(),
                           metric,
                           max_bytes_per_batch,
                           workspace.data(),
                           nWarmup,
                           nLoops,
                           latency.hostMgrpLatency,
                           latency.devMgrpLatency,
                           verbosity,
                           params.bCustomWsp);
    raft::common::nvtx::pop_range();

    handle.sync_stream(stream);
    score =
      ML::Metrics::adjusted_rand_index(handle, labels_bsl.data(), labels_mgrp.data(), sum_rows);

    if (score < 1.0) {
      auto str = raft::arr2Str(labels_bsl.data(), sum_rows, "labels_bsl", handle.get_stream());
      CUML_LOG_DEBUG("y: %s", str.c_str());
      str = raft::arr2Str(labels_mgrp.data(), sum_rows, "labels_mgrp", handle.get_stream());
      CUML_LOG_DEBUG("y_hat: %s", str.c_str());
      CUML_LOG_DEBUG("Score = %lf", score);
    }

    printSpeedup();
  }

  void SetUp() override { basicTest(); }

  void printSpeedup()
  {
    std::printf(
      "Latency: on host %.4lf ms, on gpu %.4lf ms (single);\n"
      "Latency: on host %.4lf ms, on gpu %.4lf ms (multi);\n"
      "Speedup: on host %.4lf x, on gpu %.4lf x\n",
      latency.hostSgrpLatency,
      latency.devSgrpLatency,
      latency.hostMgrpLatency,
      latency.devMgrpLatency,
      latency.hostSgrpLatency / latency.hostMgrpLatency,
      latency.devSgrpLatency / latency.devMgrpLatency);
  }

 protected:
  MgrpDbscanInputs<T, IdxT> params;
  double score;
  clk latency;
};

const std::size_t test_ngroups = 3;
const std::size_t test_ncols   = 3;

// Random dataset
const std::vector<float> test_data_f = {
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

const std::vector<int> test_nrows_i(test_ngroups, 25);
const std::vector<int> test_minpts_i(test_ngroups, 2);
const std::vector<float> test_eps_f(test_ngroups, 1.0f);

const std::vector<double> test_data_d(test_data_f.begin(), test_data_f.end());
const std::vector<double> test_eps_d(test_eps_f.begin(), test_eps_f.end());

const std::vector<int64_t> test_nrows_l(test_nrows_i.begin(), test_nrows_i.end());
const std::vector<int64_t> test_minpts_l(test_minpts_i.begin(), test_minpts_i.end());

const std::vector<MgrpDbscanInputs<float, int>> inputs_fi = {
  {test_data_f.data(),
   (int)test_ngroups,
   test_nrows_i.data(),
   (int)test_ncols,
   test_eps_f.data(),
   test_minpts_i.data(),
   raft::distance::L2SqrtUnexpanded,
   false},
  {test_data_f.data(),
   (int)test_ngroups,
   test_nrows_i.data(),
   (int)test_ncols,
   test_eps_f.data(),
   test_minpts_i.data(),
   raft::distance::CosineExpanded,
   false},
  {test_data_f.data(),
   (int)test_ngroups,
   test_nrows_i.data(),
   (int)test_ncols,
   test_eps_f.data(),
   test_minpts_i.data(),
   raft::distance::L2SqrtUnexpanded,
   true},
  {test_data_f.data(),
   (int)test_ngroups,
   test_nrows_i.data(),
   (int)test_ncols,
   test_eps_f.data(),
   test_minpts_i.data(),
   raft::distance::CosineExpanded,
   true},
};

const std::vector<MgrpDbscanInputs<double, int>> inputs_di = {
  {test_data_d.data(),
   (int)test_ngroups,
   test_nrows_i.data(),
   (int)test_ncols,
   test_eps_d.data(),
   test_minpts_i.data(),
   raft::distance::L2SqrtUnexpanded,
   false},
  {test_data_d.data(),
   (int)test_ngroups,
   test_nrows_i.data(),
   (int)test_ncols,
   test_eps_d.data(),
   test_minpts_i.data(),
   raft::distance::CosineExpanded,
   false},
  {test_data_d.data(),
   (int)test_ngroups,
   test_nrows_i.data(),
   (int)test_ncols,
   test_eps_d.data(),
   test_minpts_i.data(),
   raft::distance::L2SqrtUnexpanded,
   true},
  {test_data_d.data(),
   (int)test_ngroups,
   test_nrows_i.data(),
   (int)test_ncols,
   test_eps_d.data(),
   test_minpts_i.data(),
   raft::distance::CosineExpanded,
   true},
};

const std::vector<MgrpDbscanInputs<float, int64_t>> inputs_fl = {
  {test_data_f.data(),
   (int64_t)test_ngroups,
   test_nrows_l.data(),
   (int64_t)test_ncols,
   test_eps_f.data(),
   test_minpts_l.data(),
   raft::distance::L2SqrtUnexpanded,
   false},
  {test_data_f.data(),
   (int64_t)test_ngroups,
   test_nrows_l.data(),
   (int64_t)test_ncols,
   test_eps_f.data(),
   test_minpts_l.data(),
   raft::distance::CosineExpanded,
   false},
  {test_data_f.data(),
   (int64_t)test_ngroups,
   test_nrows_l.data(),
   (int64_t)test_ncols,
   test_eps_f.data(),
   test_minpts_l.data(),
   raft::distance::L2SqrtUnexpanded,
   true},
  {test_data_f.data(),
   (int64_t)test_ngroups,
   test_nrows_l.data(),
   (int64_t)test_ncols,
   test_eps_f.data(),
   test_minpts_l.data(),
   raft::distance::CosineExpanded,
   true},
};

const std::vector<MgrpDbscanInputs<double, int64_t>> inputs_dl = {
  {test_data_d.data(),
   (int64_t)test_ngroups,
   test_nrows_l.data(),
   (int64_t)test_ncols,
   test_eps_d.data(),
   test_minpts_l.data(),
   raft::distance::L2SqrtUnexpanded,
   false},
  {test_data_d.data(),
   (int64_t)test_ngroups,
   test_nrows_l.data(),
   (int64_t)test_ncols,
   test_eps_d.data(),
   test_minpts_l.data(),
   raft::distance::CosineExpanded,
   false},
  {test_data_d.data(),
   (int64_t)test_ngroups,
   test_nrows_l.data(),
   (int64_t)test_ncols,
   test_eps_d.data(),
   test_minpts_l.data(),
   raft::distance::L2SqrtUnexpanded,
   true},
  {test_data_d.data(),
   (int64_t)test_ngroups,
   test_nrows_l.data(),
   (int64_t)test_ncols,
   test_eps_d.data(),
   test_minpts_l.data(),
   raft::distance::CosineExpanded,
   true},
};

typedef MgrpDbscanTest<float, int> MgrpDbscanTestF_Int;
TEST_P(MgrpDbscanTestF_Int, Result) { ASSERT_TRUE(score == 1.0); }

typedef MgrpDbscanTest<double, int> MgrpDbscanTestD_Int;
TEST_P(MgrpDbscanTestD_Int, Result) { ASSERT_TRUE(score == 1.0); }

typedef MgrpDbscanTest<float, int64_t> MgrpDbscanTestF_Long;
TEST_P(MgrpDbscanTestF_Long, Result) { ASSERT_TRUE(score == 1.0); }

typedef MgrpDbscanTest<double, int64_t> MgrpDbscanTestD_Long;
TEST_P(MgrpDbscanTestD_Long, Result) { ASSERT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(MgrpDbscanTest, MgrpDbscanTestF_Int, ::testing::ValuesIn(inputs_fi));
INSTANTIATE_TEST_CASE_P(MgrpDbscanTest, MgrpDbscanTestD_Int, ::testing::ValuesIn(inputs_di));
INSTANTIATE_TEST_CASE_P(MgrpDbscanTest, MgrpDbscanTestF_Long, ::testing::ValuesIn(inputs_fl));
INSTANTIATE_TEST_CASE_P(MgrpDbscanTest, MgrpDbscanTestD_Long, ::testing::ValuesIn(inputs_dl));

}  // namespace ML