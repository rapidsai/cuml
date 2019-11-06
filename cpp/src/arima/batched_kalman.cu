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
#include "batched_kalman.hpp"
#include "cuda_utils.h"

#include <unistd.h>
#include <fstream>

#include <nvToolsExt.h>
#include <common/nvtx.hpp>

#include <cub/cub.cuh>

#include <chrono>
#include <ratio>

#include <cuml/cuml.hpp>

#include <timeSeries/jones_transform.h>

using std::vector;

#include <matrix/batched_matrix.hpp>

using MLCommon::allocate;
using MLCommon::updateDevice;
using MLCommon::updateHost;
using MLCommon::Matrix::b_gemm;
using MLCommon::Matrix::b_kron;
using MLCommon::Matrix::b_solve;
using BatchedMatrix = MLCommon::Matrix::BatchedMatrix<double>;

////////////////////////////////////////////////////////////
#include <iostream>

namespace ML {

void nvtx_range_push(std::string msg) { ML::PUSH_RANGE(msg.c_str()); }

void nvtx_range_pop() { ML::POP_RANGE(); }

//! Thread-local Matrix-Vector multiplication.
template <int r>
__device__ void Mv_l(double* A, double* v, double* out) {
  for (int i = 0; i < r; i++) {
    double sum = 0.0;
    for (int j = 0; j < r; j++) {
      sum += A[i + j * r] * v[j];
    }
    out[i] = sum;
  }
}

//! Thread-local Matrix-Matrix multiplication.
template <int r>
__device__ void MM_l(double* A, double* B, double* out) {
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < r; j++) {
      double sum = 0.0;
      for (int k = 0; k < r; k++) {
        sum += A[i + k * r] * B[k + j * r];
      }
      out[i + j * r] = sum;
    }
  }
}

// TODO: N is super confusing, shouldn't we rename the template param to r?

//! Kalman loop. Each thread computes kalman filter for a single series and
//! stores relevant matrices in registers.
template <int N>
__global__ void batched_kalman_loop_kernel(double* ys, int nobs,
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
  auto stream = T.stream();
  dim3 numThreadsPerBlock(32, 1);
  dim3 numBlocks(MLCommon::ceildiv<int>(num_batches, numThreadsPerBlock.x), 1);
  if (r == 1) {
    batched_kalman_loop_kernel<1><<<numBlocks, numThreadsPerBlock, 0, stream>>>(
      ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(),
      num_batches, vs, Fs, sum_logFs);
  } else if (r == 2) {
    batched_kalman_loop_kernel<2><<<numBlocks, numThreadsPerBlock, 0, stream>>>(
      ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(),
      num_batches, vs, Fs, sum_logFs);
  } else if (r == 3) {
    batched_kalman_loop_kernel<3><<<numBlocks, numThreadsPerBlock, 0, stream>>>(
      ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(),
      num_batches, vs, Fs, sum_logFs);
  } else if (r == 4) {
    batched_kalman_loop_kernel<4><<<numBlocks, numThreadsPerBlock, 0, stream>>>(
      ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(),
      num_batches, vs, Fs, sum_logFs);
  } else if (r == 5) {
    batched_kalman_loop_kernel<5><<<numBlocks, numThreadsPerBlock, 0, stream>>>(
      ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(),
      num_batches, vs, Fs, sum_logFs);
  } else if (r == 6) {
    batched_kalman_loop_kernel<6><<<numBlocks, numThreadsPerBlock, 0, stream>>>(
      ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(),
      num_batches, vs, Fs, sum_logFs);
  } else if (r == 7) {
    batched_kalman_loop_kernel<7><<<numBlocks, numThreadsPerBlock, 0, stream>>>(
      ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(),
      num_batches, vs, Fs, sum_logFs);
  } else if (r == 8) {
    batched_kalman_loop_kernel<8><<<numBlocks, numThreadsPerBlock, 0, stream>>>(
      ys, nobs, T.data(), Z.data(), RRT.data(), P0.data(), alpha.data(),
      num_batches, vs, Fs, sum_logFs);
  } else {
    throw std::runtime_error(
      "ERROR: Currently unsupported number of parameters (r).");
  }
  CUDA_CHECK(cudaGetLastError());
}  // namespace ML

template <int NUM_THREADS>
__global__ void batched_kalman_loglike_kernel(double* d_vs, double* d_Fs,
                                              double* d_sumLogFs, int nobs,
                                              int num_batches, double* sigma2,
                                              double* loglike) {
  using BlockReduce = cub::BlockReduce<double, NUM_THREADS>;
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
                            double* loglike, cudaStream_t stream) {
  const int NUM_THREADS = 128;
  batched_kalman_loglike_kernel<NUM_THREADS>
    <<<num_batches, NUM_THREADS, 0, stream>>>(d_vs, d_Fs, d_sumLogFs, nobs,
                                              num_batches, sigma2, loglike);
  CUDA_CHECK(cudaGetLastError());
}

// Internal Kalman filter implementation that assumes data exists on GPU.
void _batched_kalman_filter(cumlHandle& handle, double* d_ys, int nobs,
                            const BatchedMatrix& Zb, const BatchedMatrix& Tb,
                            const BatchedMatrix& Rb, int r, double* d_vs,
                            double* d_Fs, double* d_loglike, double* d_sigma2,
                            bool initP_with_kalman_iterations = false) {
  const size_t num_batches = Zb.batches();
  auto stream = handle.getStream();

  BatchedMatrix RRT = b_gemm(Rb, Rb, false, true);

  BatchedMatrix P(r, r, num_batches, handle.getImpl().getCublasHandle(),
                  handle.getDeviceAllocator(), stream, false);
  if (initP_with_kalman_iterations)
    // A single Kalman iteration
    P = b_gemm(Tb, Tb, false, true) -
        Tb * b_gemm(Zb, b_gemm(Zb, Tb, false, true), true, false) + RRT;
  else {
    // # (Durbin Koopman "Time Series Analysis" pg 138)
    // NumPy version
    //   invImTT = np.linalg.pinv(np.eye(r**2) - np.kron(T_bi, T_bi))
    //   P0 = np.reshape(invImTT @ (R_bi @ R_bi.T).ravel(), (r, r), order="F")
    ML::PUSH_RANGE("P0: (I-TxT)");
    BatchedMatrix I_m_TxT =
      BatchedMatrix::Identity(r * r, num_batches,
                              handle.getImpl().getCublasHandle(),
                              handle.getDeviceAllocator(), stream) -
      b_kron(Tb, Tb);
    ML::POP_RANGE();
    ML::PUSH_RANGE("(I-TxT)\\(R.R^T)");
    BatchedMatrix invI_m_TxT_x_RRTvec = b_solve(I_m_TxT, RRT.vec());
    ML::POP_RANGE();
    BatchedMatrix P0 = invI_m_TxT_x_RRTvec.mat(r, r);
    P = P0;
  }

  // init alpha to zero
  BatchedMatrix alpha(r, 1, num_batches, handle.getImpl().getCublasHandle(),
                      handle.getDeviceAllocator(), stream, true);

  // init vs, Fs
  // In batch-major format.
  double* d_sumlogFs;

  d_sumlogFs = (double*)handle.getDeviceAllocator()->allocate(
    sizeof(double) * num_batches, stream);

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
                         d_loglike, stream);
  handle.getDeviceAllocator()->deallocate(d_sumlogFs,
                                          sizeof(double) * num_batches, stream);
}

void init_batched_kalman_matrices(cumlHandle& handle,
                                  const double* d_b_ar_params,
                                  const double* d_b_ma_params,
                                  const int num_batches, const int p,
                                  const int q, int r, double* d_Z_b,
                                  double* d_R_b, double* d_T_b) {
  ML::PUSH_RANGE("init_batched_kalman_matrices");

  const int nb = num_batches;

  auto stream = handle.getStream();

  cudaMemsetAsync(d_Z_b, 0.0, r * nb * sizeof(double), stream);
  cudaMemsetAsync(d_R_b, 0.0, r * nb * sizeof(double), stream);
  cudaMemsetAsync(d_T_b, 0.0, r * r * nb * sizeof(double), stream);

  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(thrust::cuda::par.on(stream), counting, counting + nb,
                   [=] __device__(int bid) {
                     // See TSA pg. 54 for Z,R,T matrices
                     // Z = [1 0 0 0 ... 0]
                     d_Z_b[bid * r] = 1.0;

                     /*
                       |1.0   |
                   R = |ma_1  |
                       |ma_r-1|
                     */
                     d_R_b[bid * r] = 1.0;
                     for (int i = 0; i < q; i++) {
                       d_R_b[bid * r + i + 1] = d_b_ma_params[bid * q + i];
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
                         d_T_b[bid * r * r + i] = d_b_ar_params[bid * p + i];
                       }

                       // shifted identity
                       if (i < r - 1) {
                         d_T_b[bid * r * r + (i + 1) * r + i] = 1.0;
                       }
                     }
                   });
  ML::POP_RANGE();
}

void batched_kalman_filter(cumlHandle& handle, double* d_ys, int nobs,
                           const double* d_b_ar_params,
                           const double* d_b_ma_params, int p, int q,
                           int num_batches, std::vector<double>& h_loglike_b,
                           double* d_vs, bool initP_with_kalman_iterations) {
  ML::PUSH_RANGE("batched_kalman_filter");

  const size_t ys_len = nobs;

  auto cublasHandle = handle.getImpl().getCublasHandle();
  auto stream = handle.getStream();
  auto allocator = handle.getDeviceAllocator();

  // see (3.18) in TSA by D&K
  int r = std::max(p, q + 1);

  double* d_Z_b =
    (double*)allocator->allocate(r * num_batches * sizeof(double), stream);
  double* d_R_b =
    (double*)allocator->allocate(r * num_batches * sizeof(double), stream);
  double* d_T_b =
    (double*)allocator->allocate(r * r * num_batches * sizeof(double), stream);

  init_batched_kalman_matrices(handle, d_b_ar_params, d_b_ma_params,
                               num_batches, p, q, r, d_Z_b, d_R_b, d_T_b);

  BatchedMatrix Zb(1, r, num_batches, cublasHandle, allocator, stream);
  BatchedMatrix Tb(r, r, num_batches, cublasHandle, allocator, stream);
  BatchedMatrix Rb(r, 1, num_batches, cublasHandle, allocator, stream);

  ////////////////////////////////////////////////////////////
  // Copy matrix raw data into `BatchedMatrix` memory

  //Zb
  CUDA_CHECK(cudaMemcpyAsync(Zb[0], d_Z_b, sizeof(double) * r * num_batches,
                             cudaMemcpyDeviceToDevice, stream));

  // Rb
  CUDA_CHECK(cudaMemcpyAsync(Rb[0], d_R_b, sizeof(double) * r * num_batches,
                             cudaMemcpyDeviceToDevice, stream));

  //Tb
  CUDA_CHECK(cudaMemcpyAsync(Tb[0], d_T_b, sizeof(double) * r * r * num_batches,
                             cudaMemcpyDeviceToDevice, stream));

  /// TODO: avoid copy by simply creating matrices before and passing
  /// their pointer

  ////////////////////////////////////////////////////////////
  // Computation

  double* d_Fs =
    (double*)allocator->allocate(ys_len * num_batches * sizeof(double), stream);
  double* d_sigma2 =
    (double*)allocator->allocate(num_batches * sizeof(double), stream);
  double* d_loglike =
    (double*)allocator->allocate(num_batches * sizeof(double), stream);

  _batched_kalman_filter(handle, d_ys, nobs, Zb, Tb, Rb, r, d_vs, d_Fs,
                         d_loglike, d_sigma2, initP_with_kalman_iterations);

  ////////////////////////////////////////////////////////////
  // xfer loglikelihood from GPU
  h_loglike_b.resize(num_batches);
  updateHost(h_loglike_b.data(), d_loglike, num_batches, stream);

  allocator->deallocate(d_Fs, ys_len * num_batches * sizeof(double), stream);

  allocator->deallocate(d_sigma2, num_batches * sizeof(double), stream);

  allocator->deallocate(d_loglike, num_batches * sizeof(double), stream);

  allocator->deallocate(d_Z_b, r * num_batches * sizeof(double), stream);
  allocator->deallocate(d_R_b, r * num_batches * sizeof(double), stream);
  allocator->deallocate(d_T_b, r * r * num_batches * sizeof(double), stream);

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

/* AR and MA parameters have to be within a "triangle" region (i.e., subject to
 * an inequality) for the inverse transform to not return 'NaN' due to the
 * logarithm within the inverse. This function ensures that inequality is
 * satisfied for all parameters.
 */
void fix_ar_ma_invparams(const double* d_old_params, double* d_new_params,
                         int num_batches, int pq, cudaStream_t stream,
                         bool isAr = true) {
  CUDA_CHECK(cudaMemcpyAsync(d_new_params, d_old_params,
                             num_batches * pq * sizeof(double),
                             cudaMemcpyDeviceToDevice, stream));
  int n = pq;

  // The parameter must be within a "triangle" region. If not, we bring the parameter inside by 1%.
  double eps = 0.99;
  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(
    thrust::cuda::par.on(stream), counting, counting + num_batches,
    [=] __device__(int ib) {
      for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
          sum += d_new_params[n - j - 1 + ib * n];
        }
        // AR is minus
        if (isAr) {
          // param < 1-sum(param)
          d_new_params[n - i - 1 + ib * n] =
            fmin((1 - sum) * eps, d_new_params[n - i - 1 + ib * n]);
          // param > -(1-sum(param))
          d_new_params[n - i - 1 + ib * n] =
            fmax(-(1 - sum) * eps, d_new_params[n - i - 1 + ib * n]);
        } else {
          // MA is plus
          // param < 1+sum(param)
          d_new_params[n - i - 1 + ib * n] =
            fmin((1 + sum) * eps, d_new_params[n - i - 1 + ib * n]);
          // param > -(1+sum(param))
          d_new_params[n - i - 1 + ib * n] =
            fmax(-(1 + sum) * eps, d_new_params[n - i - 1 + ib * n]);
        }
      }
    });
}

void unpack(const double* d_params, double* d_mu, double* d_ar, double* d_ma,
            int batchSize, int p, int d, int q, cudaStream_t stream) {
  int N = (p + d + q);
  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(thrust::cuda::par.on(stream), counting, counting + batchSize,
                   [=] __device__(int bid) {
                     if (d > 0) d_mu[bid] = d_params[bid * N];
                     for (int ip = 0; ip < p; ip++) {
                       d_ar[p * bid + ip] = d_params[bid * N + d + ip];
                     }
                     for (int iq = 0; iq < q; iq++) {
                       d_ma[q * bid + iq] = d_params[bid * N + d + p + iq];
                     }
                   });
}

void pack(int batchSize, int p, int d, int q, const double* d_mu,
          const double* d_ar, const double* d_ma, double* d_params,
          cudaStream_t stream) {
  int N = (p + d + q);
  auto counting = thrust::make_counting_iterator(0);
  thrust::for_each(thrust::cuda::par.on(stream), counting, counting + batchSize,
                   [=] __device__(int bid) {
                     if (d > 0) d_params[bid * N] = d_mu[bid];
                     for (int ip = 0; ip < p; ip++) {
                       d_params[bid * N + d + ip] = d_ar[p * bid + ip];
                     }
                     for (int iq = 0; iq < q; iq++) {
                       d_params[bid * N + d + p + iq] = d_ma[q * bid + iq];
                     }
                   });
}

void batched_jones_transform(cumlHandle& handle, int p, int d, int q,
                             int batchSize, bool isInv, const double* h_params,
                             double* h_Tparams) {
  int N = p + d + q;
  auto alloc = handle.getDeviceAllocator();
  auto stream = handle.getStream();
  double* d_params =
    (double*)alloc->allocate(N * batchSize * sizeof(double), stream);
  double* d_Tparams =
    (double*)alloc->allocate(N * batchSize * sizeof(double), stream);
  double* d_mu =
    (double*)alloc->allocate(d * batchSize * sizeof(double), stream);
  double* d_ar =
    (double*)alloc->allocate(p * batchSize * sizeof(double), stream);
  double* d_ma =
    (double*)alloc->allocate(q * batchSize * sizeof(double), stream);
  double* d_Tar =
    (double*)alloc->allocate(p * batchSize * sizeof(double), stream);
  double* d_Tma =
    (double*)alloc->allocate(q * batchSize * sizeof(double), stream);

  MLCommon::updateDevice(d_params, h_params, N * batchSize, stream);

  unpack(d_params, d_mu, d_ar, d_ma, batchSize, p, d, q, stream);

  batched_jones_transform(handle, p, q, batchSize, isInv, d_ar, d_ma, d_Tar,
                          d_Tma);

  pack(batchSize, p, d, q, d_mu, d_Tar, d_Tma, d_Tparams, stream);

  MLCommon::updateHost(h_Tparams, d_Tparams, N * batchSize, stream);

  alloc->deallocate(d_params, N * batchSize * sizeof(double), stream);
  alloc->deallocate(d_Tparams, N * batchSize * sizeof(double), stream);
  alloc->deallocate(d_mu, d * batchSize * sizeof(double), stream);
  alloc->deallocate(d_ar, p * batchSize * sizeof(double), stream);
  alloc->deallocate(d_ma, q * batchSize * sizeof(double), stream);
  alloc->deallocate(d_Tar, p * batchSize * sizeof(double), stream);
  alloc->deallocate(d_Tma, q * batchSize * sizeof(double), stream);
}

void batched_jones_transform(cumlHandle& handle, int p, int q, int batchSize,
                             bool isInv, const double* d_ar, const double* d_ma,
                             double* d_Tar, double* d_Tma) {
  ML::PUSH_RANGE("batched_jones_transform");

  auto allocator = handle.getDeviceAllocator();
  auto stream = handle.getStream();

  if (p > 0) {
    // inverse transform will produce NaN if parameters are outside of a "triangle" region
    double* d_ar_fixed =
      (double*)allocator->allocate(sizeof(double) * batchSize * p, stream);
    if (isInv) {
      fix_ar_ma_invparams(d_ar, d_ar_fixed, batchSize, p, stream, true);
    } else {
      CUDA_CHECK(cudaMemcpyAsync(d_ar_fixed, d_ar,
                                 sizeof(double) * batchSize * p,
                                 cudaMemcpyDeviceToDevice, stream));
    }
    MLCommon::TimeSeries::jones_transform(d_ar_fixed, batchSize, p, d_Tar, true,
                                          isInv, allocator, stream);

    allocator->deallocate(d_ar_fixed, sizeof(double) * batchSize * p, stream);
  }
  if (q > 0) {
    // inverse transform will produce NaN if parameters are outside of a "triangle" region
    double* d_ma_fixed;
    d_ma_fixed =
      (double*)allocator->allocate(sizeof(double) * batchSize * q, stream);
    if (isInv) {
      fix_ar_ma_invparams(d_ma, d_ma_fixed, batchSize, q, stream, false);
    } else {
      CUDA_CHECK(cudaMemcpyAsync(d_ma_fixed, d_ma,
                                 sizeof(double) * batchSize * q,
                                 cudaMemcpyDeviceToDevice, stream));
    }

    MLCommon::TimeSeries::jones_transform(d_ma_fixed, batchSize, q, d_Tma,
                                          false, isInv, allocator, stream);

    allocator->deallocate(d_ma_fixed, sizeof(double) * batchSize * q, stream);
  }
  ML::POP_RANGE();
}

}  // namespace ML
