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

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <utils.h>
#include <cstdio>
#include <iostream>
#include <matrix/batched_matrix.hpp>
#include "batched_kalman.hpp"
#include "cuda_utils.h"

#include <unistd.h>
#include <fstream>

#include <nvToolsExt.h>
#include <common/nvtx.hpp>

#include <cub/cub.cuh>

#include <chrono>
#include <ratio>

#include <timeSeries/jones_transform.h>

#include <cuML.hpp>

using std::vector;

using MLCommon::allocate;
using MLCommon::updateDevice;
using MLCommon::updateHost;
using MLCommon::Matrix::b_gemm;
using MLCommon::Matrix::b_kron;
using MLCommon::Matrix::b_solve;
using MLCommon::Matrix::BatchedMatrix;
using MLCommon::Matrix::BatchedMatrixMemoryPool;

////////////////////////////////////////////////////////////
#include <iostream>

namespace ML {

void nvtx_range_push(std::string msg) { ML::PUSH_RANGE(msg.c_str()); }

void nvtx_range_pop() { ML::POP_RANGE(); }

// Because the kalman filter is typically called many times within the ARIMA
// `fit()` method, allocations end up very costly. We avoid this by re-using the
// variables stored in this global `GPUContext` object.
class GPUContext {
 public:
  int m_p = 0;
  int m_q = 0;
  int m_num_batches = 0;
  double* d_ys = nullptr;
  double* d_vs = nullptr;
  double* d_Fs = nullptr;
  double* d_loglike = nullptr;
  double* d_sigma2 = nullptr;

  // memory allocators
  std::shared_ptr<BatchedMatrixMemoryPool> pool = nullptr;
  std::shared_ptr<ML::deviceAllocator> allocator;

  // TODO: This handle will probably be passed in externally later and we will only
  // permanently store the allocator shared_ptr.
  cumlHandle m_handle;

  // // batched_jones
  double* d_ar = nullptr;
  double* d_Tar = nullptr;
  double* d_ma = nullptr;
  double* d_Tma = nullptr;

  GPUContext(int p, int q, int num_batches)
    : m_p(p), m_q(q), m_num_batches(num_batches) {
    d_ys = nullptr;
    d_vs = nullptr;
    d_Fs = nullptr;
    d_loglike = nullptr;
    d_sigma2 = nullptr;
    pool = nullptr;
    d_ar = nullptr;
    d_Tar = nullptr;
    d_ma = nullptr;
    d_Tma = nullptr;

    m_handle.setStream(0);
    allocator = m_handle.getDeviceAllocator();
  }

  // Note: Tried to re-use these device vectors, but it caused segfaults, so we ignore them for now.
  // thrust::device_vector<double> d_Z_b;
  // thrust::device_vector<double> d_R_b;
  // thrust::device_vector<double> d_T_b;

  // Only allocates when the pointer is uninitialized.
  void allocate_if_zero(double*& ptr, size_t size) {
    if (ptr == nullptr) {
      ptr = (double*)allocator->allocate(sizeof(double) * size, 0);
    }
  }

  bool orderEquals(int p, int q, int num_batches) {
    return (m_p == p) && (m_q == q) && (m_num_batches == num_batches);
  }

  // static void resize_if_zero(thrust::device_vector<double> v, size_t size) {
  //   if (v.size() == 0) {
  //     v.resize(size);
  //   }
  // }

  ~GPUContext() noexcept(false) {
    ////////////////////////////////////////////////////////////
    // free memory
    if (d_ys != nullptr) allocator->deallocate(d_ys, 0, 0);
    if (d_vs != nullptr) allocator->deallocate(d_vs, 0, 0);
    if (d_Fs != nullptr) allocator->deallocate(d_Fs, 0, 0);
    if (d_sigma2 != nullptr) allocator->deallocate(d_sigma2, 0, 0);
    if (d_loglike != nullptr) allocator->deallocate(d_loglike, 0, 0);

    if (d_ar != nullptr) allocator->deallocate(d_ar, 0, 0);
    if (d_Tar != nullptr) allocator->deallocate(d_Tar, 0, 0);
    if (d_ma != nullptr) allocator->deallocate(d_ma, 0, 0);
    if (d_Tma != nullptr) allocator->deallocate(d_Tma, 0, 0);
    CUDA_CHECK(cudaPeekAtLastError());
  }
};

//! A global context variable which saves allocations between invocations.
GPUContext* GPU_CTX = nullptr;

void process_mem_usage(double& vm_usage, double& resident_set) {
  vm_usage = 0.0;
  resident_set = 0.0;

  // the two fields we want
  unsigned long vsize;
  long rss;
  {
    std::string ignore;
    std::ifstream ifs("/proc/self/stat", std::ios_base::in);
    ifs >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >>
      ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >>
      ignore >> ignore >> ignore >> ignore >> ignore >> ignore >> ignore >>
      ignore >> vsize >> rss;
  }

  long page_size_kb = sysconf(_SC_PAGE_SIZE) /
                      1024;  // in case x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;
}

//! Thread-local Matrix-Vector multiplication.
template <int r>
__device__ void Mv_l(double* A, double* v, double* out) {
  for (int i = 0; i < r; i++) {
    out[i] = 0.0;
    for (int j = 0; j < r; j++) {
      out[i] += A[i + j * r] * v[j];
    }
  }
}

//! Thread-local Matrix-Matrix multiplication.
template <int r>
__device__ void MM_l(double* A, double* B, double* out) {
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < r; j++) {
      out[i + j * r] = 0.0;
      for (int k = 0; k < r; k++) {
        out[i + j * r] += A[i + k * r] * B[k + j * r];
      }
    }
  }
}

//! Kalman loop. Each thread computes kalman filter for a single series and
//! stores relevant matrices in registers.
template <int N>
__global__ void batched_kalman_loop_kernel_v2(double* ys, int nobs,
                                              double** T,      // \in R^(r x r)
                                              double** Z,      // \in R^(1 x r)
                                              double** RRT,    // \in R^(r x r)
                                              double** P,      // \in R^(r x r)
                                              double** alpha,  // \in R^(r x 1)
                                              int num_batches, double* vs,
                                              double* Fs, double* sum_logFs) {
  double l_RRT[N * N];
  double l_T[N * N];
  double l_Z[N];
  double l_P[N * N];
  double l_alpha[N];
  double l_K[N];
  double l_tmpA[N * N];
  double l_tmpB[N * N];

  int bid = blockDim.x * blockIdx.x + threadIdx.x;

  if (bid < num_batches) {
    int r2 = N * N;

    // load GM into registers
    for (int i = 0; i < r2; i++) {
      l_RRT[i] = RRT[bid][i];
      l_T[i] = T[bid][i];
      l_P[i] = P[bid][i];
    }
    for (int i = 0; i < N; i++) {
      l_Z[i] = Z[bid][i];
      l_alpha[i] = alpha[bid][i];
    }

    double bid_sum_logFs = 0.0;

    for (int it = 0; it < nobs; it++) {
      // 1. & 2.
      vs[it + bid * nobs] = ys[it + bid * nobs] - l_alpha[0];
      Fs[it + bid * nobs] = l_P[0];
      bid_sum_logFs += log(l_P[0]);

      // 3.
      // MatrixT K = 1.0/Fs[it] * (T * P * Z.transpose());
      // tmpA = P*Z.T
      Mv_l<N>(l_P, l_Z, l_tmpA);
      // tmpB = T*tmpA
      Mv_l<N>(l_T, l_tmpA, l_tmpB);

      // K = 1/Fs[it] * tmpB
      double _1_Fs = 1.0 / Fs[it + bid * nobs];
      for (int i = 0; i < N; i++) {
        l_K[i] = _1_Fs * l_tmpB[i];
      }

      // 4.
      // alpha = T*alpha + K*vs[it];
      Mv_l<N>(l_T, l_alpha, l_tmpA);
      double vs_it = vs[it + bid * nobs];
      for (int i = 0; i < N; i++) {
        l_alpha[i] = l_tmpA[i] + l_K[i] * vs_it;
      }

      // 5.
      // MatrixT L = T - K*Z;
      // tmpA = KZ
      // tmpA[0] = K[0]*Z[0]
      // tmpA[1] = K[1]*Z[0]
      // tmpA[2] = K[0]*Z[1]
      // tmpA[3] = K[1]*Z[1]
      // pytest [i % 3 for i in range(9)] -> 0 1 2 0 1 2 0 1 2
      // pytest [i//3 % 3 for i in range(9)] -> 0 0 0 1 1 1 2 2 2
      for (int tid = 0; tid < N * N; tid++) {
        l_tmpA[tid] = l_K[tid % N] * l_Z[(tid / N) % N];
      }

      // tmpA = T-tmpA
      for (int tid = 0; tid < N * N; tid++) {
        l_tmpA[tid] = l_T[tid] - l_tmpA[tid];
      }
      // note: L = tmpA

      // 6.
      // tmpB = tmpA.transpose()
      // tmpB[0] = tmpA[0]
      // tmpB[1] = tmpA[2]
      // tmpB[2] = tmpA[1]
      // tmpB[3] = tmpA[3]
      for (int tid = 0; tid < N; tid++) {
        for (int i = 0; i < N; i++) {
          l_tmpB[tid + i * N] = l_tmpA[tid * N + i];
        }
      }
      // note: L.T = tmpB

      // P = T * P * L.transpose() + R * R.transpose();
      // tmpA = P*L.T
      MM_l<N>(l_P, l_tmpB, l_tmpA);
      // tmpB = T*tmpA;
      MM_l<N>(l_T, l_tmpA, l_tmpB);
      // P = tmpB + RRT
      for (int tid = 0; tid < N * N; tid++) {
        l_P[tid] = l_tmpB[tid] + l_RRT[tid];
      }
    }
    sum_logFs[bid] = bid_sum_logFs;
  }
}

void batched_kalman_loop(double* ys, int nobs, const BatchedMatrix& T,
                         const BatchedMatrix& Z, const BatchedMatrix& RRT,
                         const BatchedMatrix& P0, const BatchedMatrix& alpha,
                         int r, double* vs, double* Fs, double* sum_logFs) {
  const int num_batches = T.batches();

  // const int num_blocks = num_batches;
  // const int num_threads = r * r;
  // const size_t bytes_shared_memory = (5 * r * r + 3 * r) * sizeof(double);
  // batched_kalman_loop_kernel<<<num_blocks, num_threads, bytes_shared_memory>>>(
  //   ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(), r,
  //   num_batches, vs, Fs, sum_logFs);
  // CUDA_CHECK(cudaPeekAtLastError());
  // CUDA_CHECK(cudaDeviceSynchronize());

  dim3 numThreadsPerBlock(32, 1);
  dim3 numBlocks(MLCommon::ceildiv<int>(num_batches, numThreadsPerBlock.x), 1);
  if (r == 1) {
    batched_kalman_loop_kernel_v2<1><<<numBlocks, numThreadsPerBlock, 0>>>(
      ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(),
      num_batches, vs, Fs, sum_logFs);
  } else if (r == 2) {
    batched_kalman_loop_kernel_v2<2><<<numBlocks, numThreadsPerBlock, 0>>>(
      ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(),
      num_batches, vs, Fs, sum_logFs);
  } else if (r == 3) {
    batched_kalman_loop_kernel_v2<3><<<numBlocks, numThreadsPerBlock, 0>>>(
      ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(),
      num_batches, vs, Fs, sum_logFs);
  } else if (r == 4) {
    batched_kalman_loop_kernel_v2<4><<<numBlocks, numThreadsPerBlock, 0>>>(
      ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(),
      num_batches, vs, Fs, sum_logFs);
  } else if (r == 5) {
    batched_kalman_loop_kernel_v2<5><<<numBlocks, numThreadsPerBlock, 0>>>(
      ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(),
      num_batches, vs, Fs, sum_logFs);
  } else if (r == 6) {
    batched_kalman_loop_kernel_v2<6><<<numBlocks, numThreadsPerBlock, 0>>>(
      ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(),
      num_batches, vs, Fs, sum_logFs);
  } else if (r == 7) {
    batched_kalman_loop_kernel_v2<7><<<numBlocks, numThreadsPerBlock, 0>>>(
      ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(),
      num_batches, vs, Fs, sum_logFs);
  } else if (r == 8) {
    batched_kalman_loop_kernel_v2<8><<<numBlocks, numThreadsPerBlock, 0>>>(
      ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(),
      num_batches, vs, Fs, sum_logFs);
  } else {
    throw std::runtime_error(
      "ERROR: Currently unsupported number of parameters (r).");
  }

  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}  // namespace ML

__global__ void batched_kalman_loglike_kernel(double* d_vs, double* d_Fs,
                                              double* d_sumLogFs, int nobs,
                                              int num_batches, double* sigma2,
                                              double* loglike) {
  using BlockReduce = cub::BlockReduce<double, 128>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int num_threads = blockDim.x;
  double bid_sigma2 = 0.0;
  for (int it = 0; it < nobs; it += num_threads) {
    // vs and Fs are in time-major order (memory layout: column major)
    int idx = (it + tid) + bid * nobs;
    double d_vs2_Fs = 0.0;
    if (idx < nobs * num_batches) {
      d_vs2_Fs = d_vs[idx] * d_vs[idx] / d_Fs[idx];
    }
    __syncthreads();
    double partial_sum = BlockReduce(temp_storage).Sum(d_vs2_Fs, nobs - it);
    bid_sigma2 += partial_sum;
  }
  if (tid == 0) {
    bid_sigma2 /= nobs;
    sigma2[bid] = bid_sigma2;
    loglike[bid] = -.5 * (d_sumLogFs[bid] + nobs * log(bid_sigma2)) -
                   nobs / 2. * (log(2 * M_PI) + 1);
  }
}

void batched_kalman_loglike(double* d_vs, double* d_Fs, double* d_sumLogFs,
                            int nobs, int num_batches, double* sigma2,
                            double* loglike) {
  // BlockReduce uses 128 threads, so here also use 128 threads.
  const int num_threads = 128;
  batched_kalman_loglike_kernel<<<num_batches, num_threads>>>(
    d_vs, d_Fs, d_sumLogFs, nobs, num_batches, sigma2, loglike);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// Internal Kalman filter implementation that assumes data exists on GPU.
void _batched_kalman_filter(double* d_ys, int nobs, const BatchedMatrix& Zb,
                            const BatchedMatrix& Tb, const BatchedMatrix& Rb,
                            int r, double* d_vs, double* d_Fs,
                            double* d_loglike, double* d_sigma2,
                            bool initP_with_kalman_iterations = false) {
  const size_t num_batches = Zb.batches();

  BatchedMatrix RRT = b_gemm(Rb, Rb, false, true);

  BatchedMatrix P(r, r, num_batches, Zb.pool(), false);
  if (initP_with_kalman_iterations)
    // A single Kalman iteration
    P = b_gemm(Tb, Tb, false, true) -
        Tb * b_gemm(Zb, b_gemm(Zb, Tb, false, true), true, false) + RRT;
  else {
    // # (Durbin Koopman "Time Series Analysis" pg 138)
    // NumPy version
    //   invImTT = np.linalg.pinv(np.eye(r**2) - np.kron(T_bi, T_bi))
    //   P0 = np.reshape(invImTT @ (R_bi @ R_bi.T).ravel(), (r, r), order="F")
    BatchedMatrix I_m_TxT =
      BatchedMatrix::Identity(r * r, num_batches, Zb.pool()) - b_kron(Tb, Tb);
    BatchedMatrix invI_m_TxT_x_RRTvec = b_solve(I_m_TxT, RRT.vec());
    BatchedMatrix P0 = invI_m_TxT_x_RRTvec.mat(r, r);
    P = P0;
    // auto& stream = std::cout;
    // stream.precision(16);
    // MLCommon::myPrintDevVector("P0", P[0], 4*P0.batches(), stream);
  }

  // init alpha to zero
  BatchedMatrix alpha(r, 1, num_batches, Zb.pool(), true);

  // init vs, Fs
  // In batch-major format.
  double* d_sumlogFs;

  d_sumlogFs =
    (double*)GPU_CTX->allocator->allocate(sizeof(double) * num_batches, 0);

  CUDA_CHECK(cudaPeekAtLastError());

  // Reference implementation
  // For it = 1:nobs
  //  // 1.
  //   vs[it] = ys[it] - alpha(0,0);
  //  // 2.
  //   Fs[it] = P(0,0);

  //   if(Fs[it] < 0) {
  //     std::cout << "P=" << P << "\n";
  //     throw std::runtime_error("ERROR: F < 0");
  //   }
  //   3.
  //   MatrixT K = 1.0/Fs[it] * (T * P * Z.transpose());
  //   4.
  //   alpha = T*alpha + K*vs[it];
  //   5.
  //   MatrixT L = T - K*Z;
  //   6.
  //   P = T * P * L.transpose() + R * R.transpose();
  //   loglikelihood += std::log(Fs[it]);
  // }

  batched_kalman_loop(d_ys, nobs, Tb, Zb, RRT, P, alpha, r, d_vs, d_Fs,
                      d_sumlogFs);

  // Finalize loglikelihood
  // 7. & 8.
  // double sigma2 = ((vs.array().pow(2.0)).array() / Fs.array()).mean();
  // double loglike = -.5 * (loglikelihood + nobs * std::log(sigma2));
  // loglike -= nobs / 2. * (std::log(2 * M_PI) + 1);

  batched_kalman_loglike(d_vs, d_Fs, d_sumlogFs, nobs, num_batches, d_sigma2,
                         d_loglike);
  GPU_CTX->allocator->deallocate(d_sumlogFs, sizeof(double) * num_batches, 0);
}

void init_batched_kalman_matrices(const vector<double>& b_ar_params,
                                  const vector<double>& b_ma_params,
                                  const int num_batches, const int p,
                                  const int q, int& r,
                                  thrust::device_vector<double>& d_Z_b,
                                  thrust::device_vector<double>& d_R_b,
                                  thrust::device_vector<double>& d_T_b) {
  using thrust::device_vector;
  using thrust::fill;
  using thrust::host_vector;

  ML::PUSH_RANGE("init_batched_kalman_matrices");

  device_vector<double> d_b_ar_params = b_ar_params;
  device_vector<double> d_b_ma_params = b_ma_params;
  const int nb = num_batches;
  // see (3.18) in TSA by D&K
  r = std::max(p, q + 1);

  d_Z_b.resize(r * nb);
  d_R_b.resize(r * nb);
  d_T_b.resize(r * r * nb);

  thrust::fill(d_Z_b.begin(), d_Z_b.end(), 0.0);
  thrust::fill(d_R_b.begin(), d_R_b.end(), 0.0);
  thrust::fill(d_T_b.begin(), d_T_b.end(), 0.0);

  // wish we didn't have to do this casting dance...
  double* d_b_ar_params_data = thrust::raw_pointer_cast(d_b_ar_params.data());
  double* d_b_ma_params_data = thrust::raw_pointer_cast(d_b_ma_params.data());
  double* d_Z_b_data = thrust::raw_pointer_cast(d_Z_b.data());
  double* d_R_b_data = thrust::raw_pointer_cast(d_R_b.data());
  double* d_T_b_data = thrust::raw_pointer_cast(d_T_b.data());

  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(counting, counting + nb, [=] __device__(int bid) {
    // See TSA pg. 54 for Z,R,T matrices
    // Z = [1 0 0 0 ... 0]
    d_Z_b_data[bid * r] = 1.0;
    // for (int i = 1; i < r; i++) {
    //   d_Z_b_data[bid * r + i] = 0.0;
    // }

    /*
        |1.0   |
    R = |ma_1  |
        |ma_r-1|
     */
    d_R_b_data[bid * r] = 1.0;
    for (int i = 0; i < q; i++) {
      d_R_b_data[bid * r + i + 1] = d_b_ma_params_data[bid * q + i];
    }

    /*
           |ar_1  1.0  0.0  ...  0.0|
           | .         1.0          |
           | .             .        |
       T = | .               .   0.0|
           | .                 .    |
           | .                   1.0|
           |ar_r  0.0  0.0  ...  0.0|
    */

    for (int i = 0; i < r; i++) {
      // note: ar_i is zero if (i > p)
      if (i < p) {
        d_T_b_data[bid * r * r + i] = d_b_ar_params_data[bid * p + i];
      }

      // shifted identity
      if (i < r - 1) {
        d_T_b_data[bid * r * r + (i + 1) * r + i] = 1.0;
      }
    }
  });
  ML::POP_RANGE();
}

void batched_kalman_filter(double* h_ys, int nobs,
                           const vector<double>& b_ar_params,
                           const vector<double>& b_ma_params, int p, int q,
                           int num_batches, std::vector<double>& h_loglike_b,
                           std::vector<vector<double>>& h_vs_b,
                           bool initP_with_kalman_iterations) {
  ML::PUSH_RANGE("batched_akalman_filter");

  if (GPU_CTX == nullptr) {
    GPU_CTX = new GPUContext(p, q, num_batches);
  }
  if (!GPU_CTX->orderEquals(p, q, num_batches)) {
    delete GPU_CTX;
    GPU_CTX = new GPUContext(p, q, num_batches);
  }

  const size_t ys_len = nobs;
  ////////////////////////////////////////////////////////////
  // xfer batched series from host to device
  GPU_CTX->allocate_if_zero(GPU_CTX->d_ys, nobs * num_batches);
  double* d_ys = GPU_CTX->d_ys;
  updateDevice(d_ys, h_ys, nobs * num_batches, 0);

  int r;

  thrust::device_vector<double> d_Z_b;
  thrust::device_vector<double> d_R_b;
  thrust::device_vector<double> d_T_b;

  init_batched_kalman_matrices(b_ar_params, b_ma_params, num_batches, p, q, r,
                               d_Z_b, d_R_b, d_T_b);

  if (GPU_CTX->pool == nullptr) {
    GPU_CTX->pool = std::make_shared<BatchedMatrixMemoryPool>(
      num_batches, GPU_CTX->m_handle.getDeviceAllocator());
  }
  auto memory_pool = GPU_CTX->pool;

  BatchedMatrix Zb(1, r, num_batches, memory_pool);
  BatchedMatrix Tb(r, r, num_batches, memory_pool);
  BatchedMatrix Rb(r, 1, num_batches, memory_pool);

  ////////////////////////////////////////////////////////////
  // Copy matrix raw data into `BatchedMatrix` memory

  //Zb
  double* d_Z_data = thrust::raw_pointer_cast(d_Z_b.data());
  CUDA_CHECK(cudaMemcpy(Zb[0], d_Z_data, sizeof(double) * r * num_batches,
                        cudaMemcpyDeviceToDevice));

  // Rb
  double* d_R_data = thrust::raw_pointer_cast(d_R_b.data());
  CUDA_CHECK(cudaMemcpy(Rb[0], d_R_data, sizeof(double) * r * num_batches,
                        cudaMemcpyDeviceToDevice));

  //Tb
  double* d_T_data = thrust::raw_pointer_cast(d_T_b.data());
  CUDA_CHECK(cudaMemcpy(Tb[0], d_T_data, sizeof(double) * r * r * num_batches,
                        cudaMemcpyDeviceToDevice));

  ////////////////////////////////////////////////////////////
  // Computation
  GPU_CTX->allocate_if_zero(GPU_CTX->d_vs, ys_len * num_batches);
  GPU_CTX->allocate_if_zero(GPU_CTX->d_Fs, ys_len * num_batches);

  GPU_CTX->allocate_if_zero(GPU_CTX->d_sigma2, num_batches);
  GPU_CTX->allocate_if_zero(GPU_CTX->d_loglike, num_batches);

  _batched_kalman_filter(d_ys, nobs, Zb, Tb, Rb, r, GPU_CTX->d_vs,
                         GPU_CTX->d_Fs, GPU_CTX->d_loglike, GPU_CTX->d_sigma2,
                         initP_with_kalman_iterations);

  ////////////////////////////////////////////////////////////
  // xfer results from GPU
  h_loglike_b.resize(num_batches);
  updateHost(h_loglike_b.data(), GPU_CTX->d_loglike, num_batches, 0);

  vector<double> h_vs(ys_len * num_batches);
  updateHost(h_vs.data(), GPU_CTX->d_vs, ys_len * num_batches, 0);

  h_vs_b.resize(num_batches);
  for (int i = 0; i < num_batches; i++) {
    h_vs_b[i].resize(ys_len);
    for (int j = 0; j < ys_len; j++) {
      h_vs_b[i][j] =
        h_vs[j +
             i *
               ys_len];  // vs is in time-major order (column major, time first)
    }
  }
  ML::POP_RANGE();
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
  if (!v.empty()) {
    out << '[';
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}

std::vector<double> fix_ar_ma_invparams(const vector<double>& old_params,
                                        int pq, bool isAr = true) {
  std::vector<double> params = old_params;
  int num_batches = params.size() / pq;
  int n = pq;

  // The parameter must be within a "triangle" region. If not, we bring the parameter inside by 1%.
  double eps = 0.99;

  for (int ib = 0; ib < num_batches; ib++) {
    for (int i = 0; i < n; i++) {
      double sum = 0.0;
      for (int j = 0; j < i; j++) {
        sum += params[n - j - 1 + ib * n];
      }

      // AR is minus
      if (isAr) {
        // param < 1-sum(param)
        params[n - i - 1 + ib * n] =
          std::min((1 - sum) * eps, params[n - i - 1 + ib * n]);
        // param > -(1-sum(param))
        params[n - i - 1 + ib * n] =
          std::max(-(1 - sum) * eps, params[n - i - 1 + ib * n]);
      } else {
        // MA is plus
        // param < 1+sum(param)
        params[n - i - 1 + ib * n] =
          std::min((1 + sum) * eps, params[n - i - 1 + ib * n]);
        // param > -(1+sum(param))
        params[n - i - 1 + ib * n] =
          std::max(-(1 + sum) * eps, params[n - i - 1 + ib * n]);
      }
    }
  }
  return params;
}

void batched_jones_transform(int p, int q, int batchSize, bool isInv,
                             const vector<double>& ar, const vector<double>& ma,
                             vector<double>& Tar, vector<double>& Tma) {
  ML::PUSH_RANGE("batched_jones_transform");

  if (GPU_CTX == nullptr) GPU_CTX = new GPUContext(p, q, batchSize);
  if (!GPU_CTX->orderEquals(p, q, batchSize)) {
    delete GPU_CTX;
    GPU_CTX = new GPUContext(p, q, batchSize);
  }

  std::shared_ptr<MLCommon::deviceAllocator> allocator(
    new MLCommon::defaultDeviceAllocator());
  cudaStream_t stream = 0;

  if (p > 0) {
    // inverse transform will produce NaN if parameters are outside of a "triangle" region
    vector<double> ar_fixed;
    if (isInv) {
      ar_fixed = fix_ar_ma_invparams(ar, p, true);
    } else {
      ar_fixed = ar;
    }

    Tar.resize(p * batchSize);
    GPU_CTX->allocate_if_zero(GPU_CTX->d_ar, p * batchSize);
    GPU_CTX->allocate_if_zero(GPU_CTX->d_Tar, p * batchSize);

    MLCommon::updateDevice(GPU_CTX->d_ar, ar_fixed.data(), p * batchSize,
                           stream);

    MLCommon::TimeSeries::jones_transform(GPU_CTX->d_ar, batchSize, p,
                                          GPU_CTX->d_Tar, true, isInv,
                                          allocator, stream);

    MLCommon::updateHost(Tar.data(), GPU_CTX->d_Tar, p * batchSize, stream);
  }
  if (q > 0) {
    // inverse transform will produce NaN if parameters are outside of a "triangle" region
    vector<double> ma_fixed;
    if (isInv) {
      ma_fixed = fix_ar_ma_invparams(ma, q, false);
    } else {
      ma_fixed = ma;
    }

    Tma.resize(q * batchSize);

    GPU_CTX->allocate_if_zero(GPU_CTX->d_ma, q * batchSize);
    GPU_CTX->allocate_if_zero(GPU_CTX->d_Tma, q * batchSize);

    MLCommon::updateDevice(GPU_CTX->d_ma, ma_fixed.data(), q * batchSize,
                           stream);

    MLCommon::TimeSeries::jones_transform(GPU_CTX->d_ma, batchSize, q,
                                          GPU_CTX->d_Tma, false, isInv,
                                          allocator, stream);
    MLCommon::updateHost(Tma.data(), GPU_CTX->d_Tma, q * batchSize, stream);
  }
  ML::POP_RANGE();
}
}  // namespace ML
