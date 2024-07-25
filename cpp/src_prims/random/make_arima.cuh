/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#pragma once

#include <cuml/tsa/arima_common.h>

#include <raft/random/rng.cuh>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <timeSeries/arima_helpers.cuh>

#include <random>

namespace MLCommon {
namespace Random {

/**
 * Main kernel to generate time series by simulating an ARIMA process
 *
 * @tparam     DataT      Scalar type
 * @param[out] d_diff     Generated series (before un-differencing)
 * @param[in]  d_res      Residuals (normal noise)
 * @param[in]  d_mu       Parameters mu
 * @param[in]  d_ar       Parameters ar
 * @param[in]  d_ma       Parameters ma
 * @param[in]  d_sar      Parameters sar
 * @param[in]  d_sma      Parameters sma
 * @param[in]  n_obs_diff Number of observations per series in d_diff
 * @param[in]  p          Parameter p
 * @param[in]  q          Parameter q
 * @param[in]  P          Parameter P
 * @param[in]  Q          Parameter Q
 * @param[in]  s          Parameter s
 * @param[in]  k          Parameter k
 */
template <typename DataT>
CUML_KERNEL void make_arima_kernel(DataT* d_diff,
                                   const DataT* d_res,
                                   const DataT* d_mu,
                                   const DataT* d_ar,
                                   const DataT* d_ma,
                                   const DataT* d_sar,
                                   const DataT* d_sma,
                                   int n_obs_diff,
                                   int p,
                                   int q,
                                   int P,
                                   int Q,
                                   int s,
                                   int k)
{
  int n_phi   = p + s * P;
  int n_theta = q + s * Q;

  // Load phi, theta and mu to registers
  DataT phi = 0, theta = 0;
  if (threadIdx.x < n_phi) {
    phi = TimeSeries::reduced_polynomial<true>(blockIdx.x, d_ar, p, d_sar, P, s, threadIdx.x + 1);
  }
  if (threadIdx.x < n_theta) {
    theta =
      TimeSeries::reduced_polynomial<false>(blockIdx.x, d_ma, q, d_sma, Q, s, threadIdx.x + 1);
  }
  DataT mu = (k && threadIdx.x == 0) ? d_mu[blockIdx.x] : (DataT)0;

  // Shared memory: set pointers and load the residuals
  // Note: neutral type to avoid a float/double definition conflict
  extern __shared__ char make_arima_shared_mem[];
  DataT* b_diff = (DataT*)make_arima_shared_mem;
  DataT* b_res  = (DataT*)make_arima_shared_mem + n_obs_diff;
  for (int i = threadIdx.x; i < n_obs_diff; i += blockDim.x) {
    b_res[i] = d_res[n_obs_diff * blockIdx.x + i];
  }

  // Main loop
  char* temp_smem = (char*)(make_arima_shared_mem + 2 * n_obs_diff * sizeof(DataT));
  DataT obs;
  for (int i = 0; i < n_obs_diff; i++) {
    __syncthreads();

    obs = 0;
    // AR component
    obs += phi * ((threadIdx.x < min(i, n_phi)) ? b_diff[i - threadIdx.x - 1] : mu);
    // MA component
    obs += (threadIdx.x < min(i, n_theta)) ? theta * b_res[i - threadIdx.x - 1] : 0;

    obs = raft::blockReduce(obs, temp_smem);

    if (threadIdx.x == 0) {
      // Intercept and residual
      obs += mu + b_res[i];

      // Write a data point in shared memory
      b_diff[i] = obs;
    }
  }
  __syncthreads();

  // Copy the generated data to global memory
  for (int i = threadIdx.x; i < n_obs_diff; i += blockDim.x) {
    d_diff[n_obs_diff * blockIdx.x + i] = b_diff[i];
  }
}

/**
 * Generates a dataset of time series by simulating an ARIMA process
 * of a given order.
 *
 * @tparam     DataT          Scalar type
 * @param[out] out            Generated time series
 * @param[in]  batch_size     Batch size
 * @param[in]  n_obs          Number of observations per series
 * @param[in]  order          ARIMA order
 * @param[in]  stream         CUDA stream
 * @param[in]  scale          Scale used to draw the starting values
 * @param[in]  noise_scale    Scale used to draw the residuals
 * @param[in]  intercept_sale Scale used to draw the intercept
 * @param[in]  seed           Seed for the random number generator
 * @param[in]  type           Type of random number generator
 */
template <typename DataT>
void make_arima(DataT* out,
                int batch_size,
                int n_obs,
                ML::ARIMAOrder order,
                cudaStream_t stream,
                DataT scale                      = (DataT)1.0,
                DataT noise_scale                = (DataT)0.2,
                DataT intercept_scale            = (DataT)1.0,
                uint64_t seed                    = 0ULL,
                raft::random::GeneratorType type = raft::random::GenPhilox)
{
  int d_sD      = order.d + order.s * order.D;
  int n_phi     = order.p + order.s * order.P;
  int n_theta   = order.q + order.s * order.Q;
  auto counting = thrust::make_counting_iterator(0);

  // Create CPU/GPU random generators and distributions
  raft::random::Rng gpu_gen(seed, type);

  // Generate parameters. We draw temporary random parameters and transform
  // them to create the final parameters.
  ML::ARIMAParams<DataT> params_temp, params;
  params_temp.allocate(order, batch_size, stream, false);
  params.allocate(order, batch_size, stream, true);
  if (order.k) {
    gpu_gen.uniform(params_temp.mu, batch_size, -intercept_scale, intercept_scale, stream);
  }
  if (order.p) {
    gpu_gen.uniform(params_temp.ar, batch_size * order.p, (DataT)-1.0, (DataT)1.0, stream);
  }
  if (order.q) {
    gpu_gen.uniform(params_temp.ma, batch_size * order.q, (DataT)-1.0, (DataT)1.0, stream);
  }
  if (order.P) {
    gpu_gen.uniform(params_temp.sar, batch_size * order.P, (DataT)-1.0, (DataT)1.0, stream);
  }
  if (order.Q) {
    gpu_gen.uniform(params_temp.sma, batch_size * order.Q, (DataT)-1.0, (DataT)1.0, stream);
  }
  // Note: sigma2 is unused, we just memset it to zero
  RAFT_CUDA_TRY(cudaMemsetAsync(params_temp.sigma2, 0, batch_size * sizeof(DataT), stream));
  // No need to copy, just reuse the pointer
  params.mu = params_temp.mu;
  TimeSeries::batched_jones_transform(order, batch_size, false, params_temp, params, stream);

  // Generate d+s*D starting values per series with a random walk
  // We first generate random values between -1 and 1 and then use a kernel to
  // create the random walk
  rmm::device_uvector<DataT> starting_values(0, stream);
  if (d_sD) {
    starting_values.resize(batch_size * d_sD, stream);
    DataT* d_start_val = starting_values.data();

    // First generate random values between - 1 and 1
    gpu_gen.uniform(starting_values.data(), batch_size * d_sD, (DataT)-1, (DataT)1, stream);

    // Then use a kernel to create the random walk
    DataT walk_scale = 0.5 * scale;
    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int ib) {
        DataT* b_start_val = d_start_val + d_sD * ib;
        b_start_val[0] *= scale;
        for (int i = 1; i < d_sD; i++) {
          b_start_val[i] = b_start_val[i - 1] + walk_scale * b_start_val[i];
        }
      });
  }

  // Create a buffer for the differenced series
  DataT* d_diff;
  rmm::device_uvector<DataT> diff_data(0, stream);
  if (d_sD) {
    diff_data.resize(batch_size * (n_obs - d_sD), stream);
    d_diff = diff_data.data();
  } else {
    d_diff = out;
  }

  // Generate noise/residuals
  rmm::device_uvector<DataT> residuals(batch_size * (n_obs - d_sD), stream);
  gpu_gen.normal(residuals.data(), batch_size * (n_obs - d_sD), (DataT)0.0, noise_scale, stream);

  // Call the main kernel to generate the differenced series
  int n_warps            = std::max(raft::ceildiv<int>(std::max(n_phi, n_theta), 32), 1);
  size_t shared_mem_size = (2 * (n_obs - d_sD) + n_warps) * sizeof(double);
  make_arima_kernel<<<batch_size, 32 * n_warps, shared_mem_size, stream>>>(d_diff,
                                                                           residuals.data(),
                                                                           params.mu,
                                                                           params.ar,
                                                                           params.ma,
                                                                           params.sar,
                                                                           params.sma,
                                                                           n_obs - d_sD,
                                                                           order.p,
                                                                           order.q,
                                                                           order.P,
                                                                           order.Q,
                                                                           order.s,
                                                                           order.k);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Final time series
  if (d_sD) {
    TimeSeries::finalize_forecast(d_diff,
                                  starting_values.data(),
                                  n_obs - d_sD,
                                  batch_size,
                                  d_sD,
                                  d_sD,
                                  order.d,
                                  order.D,
                                  order.s,
                                  stream);
  }

  // Copy to output if we didn't write directly to the output vector
  if (d_sD) {
    DataT* d_starting_values = starting_values.data();
    thrust::for_each(
      thrust::cuda::par.on(stream), counting, counting + batch_size, [=] __device__(int ib) {
        for (int i = 0; i < d_sD; i++) {
          out[ib * n_obs + i] = d_starting_values[d_sD * ib + i];
        }
        for (int i = 0; i < n_obs - d_sD; i++) {
          out[ib * n_obs + d_sD + i] = d_diff[(n_obs - d_sD) * ib + i];
        }
      });
  }
}

}  // namespace Random
}  // namespace MLCommon
