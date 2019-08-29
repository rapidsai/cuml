#include <matrix/batched_matrix.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <utils.h>
#include <cstdio>
#include <iostream>
#include "batched_kalman.h"

#include <unistd.h>
#include <fstream>

#include <nvToolsExt.h>
#include <common/nvtx.hpp>

#include <cub/cub.cuh>

#include <chrono>
#include <ratio>

#include <timeSeries/jones_transform.h>

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

void nvtx_range_push(std::string msg) { ML::PUSH_RANGE(msg.c_str()); }

void nvtx_range_pop() { ML::POP_RANGE(); }

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

//! Matrix-Vector multiplication
__device__ void Mv(double* A, double* v, int r, int tid, double* out) {
  out[tid] = 0.0;
  if (tid < r) {
    for (int i = 0; i < r; i++) {
      out[tid] += A[tid + r * i] * v[i];
    }
  }
}

//! Matrix-Matrix multiplication
__device__ void MM(double* A, double* B, int r, int tid, double* out) {
  out[tid] = 0.0;
  for (int i = 0; i < r; i++) {
    // access pattern should be:
    // out[0] += A[0 + r*i] * B[i + 0*r];
    // out[1] += A[1 + r*i] * B[i + 0*r];
    // out[2] += A[0 + r*i] * B[i + 1*r];
    // out[3] += A[1 + r*i] * B[i + 1*r];

    out[tid] += A[tid % r + r * i] * B[i + (tid / r % r) * r];
  }
}

extern __shared__ double s_array[];  // size = r*r x 5 + r x 3
__global__ void batched_kalman_loop_kernel(double* ys, int nobs,
                                           double** T,      // \in R^(r x r)
                                           double** Z,      // \in R^(1 x r)
                                           double** RRT,    // \in R^(r x r)
                                           double** P,      // \in R^(r x r)
                                           double** alpha,  // \in R^(r x 1)
                                           int r, int num_batches, double* vs,
                                           double* Fs, double* sum_logFs) {
  // kalman matrices and temporary storage
  int r2 = r * r;
  double* s_RRT = &s_array[0];              // rxr
  double* s_T = &s_array[r2];               // rxr
  double* s_Z = &s_array[2 * r2];           // r
  double* s_P = &s_array[2 * r2 + r];       // rxr
  double* s_alpha = &s_array[3 * r2 + r];   // r
  double* s_K = &s_array[3 * r2 + 2 * r];   // r
  double* tmpA = &s_array[3 * r2 + 3 * r];  // rxr
  double* tmpB = &s_array[4 * r2 + 3 * r];  // rxr

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  // preload kalman matrices from GM.
  s_RRT[tid] = RRT[bid][tid];
  s_T[tid] = T[bid][tid];
  s_P[tid] = P[bid][tid];
  if (tid < r) {
    s_Z[tid] = Z[bid][tid];
    s_alpha[tid] = alpha[bid][tid];
  }
  __syncthreads();

  double bid_sum_logFs = 0.0;

  for (int it = 0; it < nobs; it++) {
    // 1. & 2.
    // vs[it] = ys[it] - alpha(0,0);
    // Fs[it] = P(0,0);
    if (tid == 0) {
      vs[it + bid * nobs] = ys[it + bid * nobs] - s_alpha[0];
      Fs[it + bid * nobs] = s_P[0];
      bid_sum_logFs += log(s_P[0]);
    }
    __syncthreads();

    // 3.
    // MatrixT K = 1.0/Fs[it] * (T * P * Z.transpose());
    // tmpA = P*Z.T
    Mv(s_P, s_Z, r, tid, tmpA);
    __syncthreads();
    // tmpB = T*tmpA
    Mv(s_T, tmpA, r, tid, tmpB);
    __syncthreads();
    // tmpB = 1/Fs[it] * tmpB
    if (tid < r) {
      s_K[tid] = 1 / Fs[it + bid * nobs] * tmpB[tid];
    }
    __syncthreads();

    // 4.
    // alpha = T*alpha + K*vs[it];
    Mv(s_T, s_alpha, r, tid, tmpA);
    if (tid < r) {
      s_alpha[tid] = tmpA[tid] + s_K[tid] * vs[it + bid * nobs];
    }
    __syncthreads();

    // 5.
    // MatrixT L = T - K*Z;
    // tmpA = KZ
    // tmpA[0] = K[0]*Z[0]
    // tmpA[1] = K[1]*Z[0]
    // tmpA[2] = K[0]*Z[1]
    // tmpA[3] = K[1]*Z[1]
    // pytest [i % 3 for i in range(9)] -> 0 1 2 0 1 2 0 1 2
    // pytest [i//3 % 3 for i in range(9)] -> 0 0 0 1 1 1 2 2 2

    tmpA[tid] = s_K[tid % r] * s_Z[(tid / r) % r];

    __syncthreads();
    // tmpA = T-tmpA
    tmpA[tid] = s_T[tid] - tmpA[tid];
    __syncthreads();
    // L = tmpA

    // 6.
    // tmpB = tmpA.transpose()
    // tmpB[0] = tmpA[0]
    // tmpB[1] = tmpA[2]
    // tmpB[2] = tmpA[1]
    // tmpB[3] = tmpA[3]
    if (tid < r) {
      // TODO: There is probably a clever way to make this work without a loop
      for (int i = 0; i < r; i++) {
        tmpB[tid + i * r] = tmpA[tid * r + i];
      }
    }
    // L.T = tmpB
    __syncthreads();

    // P = T * P * L.transpose() + R * R.transpose();
    // tmpA = P*L.T
    MM(s_P, tmpB, r, tid, tmpA);
    __syncthreads();
    // tmpB = T*tmpA;
    MM(s_T, tmpA, r, tid, tmpB);
    __syncthreads();
    // P = tmpB + RRT
    s_P[tid] = tmpB[tid] + s_RRT[tid];
    __syncthreads();
  }
  if (tid == 0) {
    sum_logFs[bid] = bid_sum_logFs;
  }
}

void batched_kalman_loop(double* ys, int nobs, const BatchedMatrix& T,
                         const BatchedMatrix& Z, const BatchedMatrix& RRT,
                         const BatchedMatrix& P0, const BatchedMatrix& alpha,
                         int r, double* vs, double* Fs, double* sum_logFs) {
  const int num_batches = T.batches();
  const int num_blocks = num_batches;
  const int num_threads = r * r;
  const size_t bytes_shared_memory = (5 * r * r + 3 * r) * sizeof(double);

  batched_kalman_loop_kernel<<<num_blocks, num_threads, bytes_shared_memory>>>(
    ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(), r,
    num_batches, vs, Fs, sum_logFs);

  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

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
    // vs and Fs are in time-major order
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

  allocate(d_sumlogFs, num_batches);

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
}

// Because the kalman filter is typically called many times within the ARIMA
// `fit()` method, allocations end up very costly. We avoid this by re-using the
// variables stored in this global `GPUContext` object.
class GPUContext {
 public:
  int m_p = 0;
  int m_q = 0;
  double* d_ys = 0;
  double* d_vs = 0;
  double* d_Fs = 0;
  double* d_loglike = 0;
  double* d_sigma2 = 0;
  std::shared_ptr<BatchedMatrixMemoryPool> pool = 0;

  // // batched_jones
  double* d_ar = 0;
  double* d_Tar = 0;
  double* d_ma = 0;
  double* d_Tma = 0;

  GPUContext(int p, int q) : m_p(p), m_q(q) {
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
  }

  // Note: Tried to re-use these device vectors, but it caused segfaults, so we ignore them for now.
  // thrust::device_vector<double> d_Z_b;
  // thrust::device_vector<double> d_R_b;
  // thrust::device_vector<double> d_T_b;

  // Only allocates when the pointer is uninitialized.
  static void allocate_if_zero(double*& ptr, size_t size) {
    if (ptr == 0) {
      MLCommon::allocate(ptr, size);
    }
  }

  bool orderEquals(int p, int q) { return (m_p == p) && (m_q = q); }

  // static void resize_if_zero(thrust::device_vector<double> v, size_t size) {
  //   if (v.size() == 0) {
  //     v.resize(size);
  //   }
  // }

  ~GPUContext() noexcept(false) {
    ////////////////////////////////////////////////////////////
    // free memory
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_vs));
    CUDA_CHECK(cudaFree(d_Fs));
    CUDA_CHECK(cudaFree(d_sigma2));
    CUDA_CHECK(cudaFree(d_loglike));

    CUDA_CHECK(cudaFree(d_ar));
    CUDA_CHECK(cudaFree(d_Tar));
    CUDA_CHECK(cudaFree(d_ma));
    CUDA_CHECK(cudaFree(d_Tma));
  }
};

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

GPUContext* GPU_CTX = nullptr;

//! The public batched kalman filter.
//! `h_ys`: (nobs, num_batches) in column-major (series-major) layout
void batched_kalman_filter(double* h_ys, int nobs,
                           const vector<double>& b_ar_params,
                           const vector<double>& b_ma_params, int p, int q,
                           int num_batches, std::vector<double>& h_loglike_b,
                           std::vector<vector<double>>& h_vs_b,
                           bool initP_with_kalman_iterations) {
  ML::PUSH_RANGE("batched_akalman_filter");

  if (GPU_CTX == nullptr) {
    GPU_CTX = new GPUContext(p, q);
  }
  if (!GPU_CTX->orderEquals(p, q)) {
    delete GPU_CTX;
    GPU_CTX = new GPUContext(p, q);
  }

  const size_t ys_len = nobs;
  ////////////////////////////////////////////////////////////
  // xfer batched series from host to device
  GPUContext::allocate_if_zero(GPU_CTX->d_ys, nobs * num_batches);
  double* d_ys = GPU_CTX->d_ys;
  updateDevice(d_ys, h_ys, nobs * num_batches, 0);

  int r;

  thrust::device_vector<double> d_Z_b;
  thrust::device_vector<double> d_R_b;
  thrust::device_vector<double> d_T_b;

  init_batched_kalman_matrices(b_ar_params, b_ma_params, num_batches, p, q, r,
                               d_Z_b, d_R_b, d_T_b);

  if (GPU_CTX->pool == nullptr) {
    GPU_CTX->pool = std::make_shared<BatchedMatrixMemoryPool>(num_batches);
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
  GPUContext::allocate_if_zero(GPU_CTX->d_vs, ys_len * num_batches);
  GPUContext::allocate_if_zero(GPU_CTX->d_Fs, ys_len * num_batches);

  GPUContext::allocate_if_zero(GPU_CTX->d_sigma2, num_batches);
  GPUContext::allocate_if_zero(GPU_CTX->d_loglike, num_batches);

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
      h_vs_b[i][j] = h_vs[j + i * ys_len];  // vs is in time-major order
    }
  }
  ML::POP_RANGE();
}

//! Public interface to batched "jones transform" used in ARIMA to ensure
//! certain properties of the AR and MA parameters.
void batched_jones_transform(int p, int q, int batchSize, bool isInv,
                             const vector<double>& ar, const vector<double>& ma,
                             vector<double>& Tar, vector<double>& Tma) {
  ML::PUSH_RANGE("batched_jones_transform");

  if (GPU_CTX == nullptr) GPU_CTX = new GPUContext(p, q);
  if (!GPU_CTX->orderEquals(p, q)) {
    delete GPU_CTX;
    GPU_CTX = new GPUContext(p, q);
  }

  std::shared_ptr<MLCommon::deviceAllocator> allocator(
    new MLCommon::defaultDeviceAllocator());
  cudaStream_t stream = 0;

  if (p > 0) {
    Tar.resize(p * batchSize);
    GPUContext::allocate_if_zero(GPU_CTX->d_ar, p * batchSize);
    GPUContext::allocate_if_zero(GPU_CTX->d_Tar, p * batchSize);

    MLCommon::updateDevice(GPU_CTX->d_ar, ar.data(), p * batchSize, stream);

    MLCommon::TimeSeries::jones_transform(GPU_CTX->d_ar, batchSize, p,
                                          GPU_CTX->d_Tar, true, isInv,
                                          allocator, stream);

    MLCommon::updateHost(Tar.data(), GPU_CTX->d_Tar, p * batchSize, stream);
  }
  if (q > 0) {
    Tma.resize(q * batchSize);

    GPUContext::allocate_if_zero(GPU_CTX->d_ma, q * batchSize);
    GPUContext::allocate_if_zero(GPU_CTX->d_Tma, q * batchSize);

    MLCommon::updateDevice(GPU_CTX->d_ma, ma.data(), q * batchSize, stream);

    MLCommon::TimeSeries::jones_transform(GPU_CTX->d_ma, batchSize, q,
                                          GPU_CTX->d_Tma, false, isInv,
                                          allocator, stream);
    MLCommon::updateHost(Tma.data(), GPU_CTX->d_Tma, q * batchSize, stream);
  }
  ML::POP_RANGE();
}
