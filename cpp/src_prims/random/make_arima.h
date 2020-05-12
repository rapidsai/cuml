/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <random>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "cuml/common/cuml_allocator.hpp"
#include "cuml/tsa/arima_common.h"
#include "rng.h"
#include "timeSeries/arima_helpers.h"

namespace MLCommon {
namespace Random {

/**
 * @brief Time series generator for a given ARIMA order
 *
 * @tparam  DataT  Scalar type
 * @todo: docs (+ modify interface?) + way to return params
 *                                     (as coef in make_regression)
 */
template <typename DataT>
void make_arima(DataT* out, int batch_size, int n_obs, ML::ARIMAOrder order,
                std::shared_ptr<deviceAllocator> allocator, cudaStream_t stream,
                DataT scale = (DataT)1.0, DataT noise_scale = (DataT)0.2,
                DataT intercept_scale = (DataT)1.0, uint64_t seed = 0ULL,
                GeneratorType type = GenPhilox) {
  int d_sD = order.d + order.s * order.D;
  int p_sP = order.p + order.s * order.P;
  int q_sQ = order.q + order.s * order.Q;
  auto counting = thrust::make_counting_iterator(0);

  // Create CPU/GPU random generators and distributions
  std::default_random_engine cpu_gen(seed);
  Rng gpu_gen(seed, type);
  std::uniform_real_distribution<DataT> udis((DataT)0.0, (DataT)1.0);

  // Generate parameters. We draw temporary random parameters and transform
  // them to create the final parameters.
  // Note: sigma2 is unused so we don't even initialize it
  ML::ARIMAParams<DataT> params_temp, params;
  params_temp.allocate(order, batch_size, allocator, stream, false);
  params.allocate(order, batch_size, allocator, stream, true);
  if (order.k) {
    gpu_gen.uniform(params_temp.mu, batch_size, -intercept_scale,
                    intercept_scale, stream);
  }
  if (order.p) {
    gpu_gen.uniform(params_temp.ar, batch_size * order.p, (DataT)-1.0,
                    (DataT)1.0, stream);
  }
  if (order.q) {
    gpu_gen.uniform(params_temp.ma, batch_size * order.q, (DataT)-1.0,
                    (DataT)1.0, stream);
  }
  if (order.P) {
    gpu_gen.uniform(params_temp.sar, batch_size * order.P, (DataT)-1.0,
                    (DataT)1.0, stream);
  }
  if (order.Q) {
    gpu_gen.uniform(params_temp.sma, batch_size * order.Q, (DataT)-1.0,
                    (DataT)1.0, stream);
  }
  params.mu = params_temp.mu;  // No need to copy, just reuse the pointer
  TimeSeries::batched_jones_transform(order, batch_size, false, params_temp,
                                      params, allocator, stream);

  // Create lag coefficient vectors for the AR+SAR and MA+SMA components
  /// TODO: fuse all in a single kernel
  device_buffer<DataT> ar_vec(allocator, stream);
  device_buffer<DataT> ma_vec(allocator, stream);
  ar_vec.resize(batch_size * p_sP, stream);
  ma_vec.resize(batch_size * q_sQ, stream);
  DataT* d_ar_vec = ar_vec.data();
  DataT* d_ma_vec = ma_vec.data();
  DataT *d_ar = params.ar, *d_ma = params.ma, *d_sar = params.sar,
        *d_sma = params.sma;
  if (p_sP) {
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int ib) {
                       DataT* b_ar_vec = d_ar_vec + ib * p_sP;
                       for (int ip = 0; ip < p_sP; ip++) {
                         b_ar_vec[ip] = TimeSeries::reduced_polynomial<true>(
                           ib, d_ar, order.p, d_sar, order.P, order.s, ip + 1);
                       }
                     });
  }
  if (q_sQ) {
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int ib) {
                       DataT* b_ma_vec = d_ma_vec + ib * q_sQ;
                       for (int iq = 0; iq < q_sQ; iq++) {
                         b_ma_vec[iq] = TimeSeries::reduced_polynomial<false>(
                           ib, d_ma, order.q, d_sma, order.Q, order.s, iq + 1);
                       }
                     });
  }

  // Generate d+s*D starting values per series
  /// TODO: generate with a random walk
  device_buffer<DataT> starting_values(allocator, stream);
  if (d_sD) {
    starting_values.resize(batch_size * d_sD, stream);
    DataT mean = udis(cpu_gen);
    gpu_gen.uniform(starting_values.data(), batch_size * d_sD, mean - scale,
                    mean + scale, stream);
  }

  // Create buffer for the differenced series
  DataT* d_diff;
  device_buffer<DataT> diff_data(allocator, stream);
  if (d_sD) {
    diff_data.resize(batch_size * (n_obs - d_sD), stream);
    d_diff = diff_data.data();
  } else {
    d_diff = out;
  }

  // Generate noise/residuals
  device_buffer<DataT> residuals(allocator, stream);
  residuals.resize(batch_size * (n_obs - d_sD), stream);
  gpu_gen.normal(residuals.data(), batch_size * (n_obs - d_sD), (DataT)0.0,
                 noise_scale, stream);
  const DataT* d_res = residuals.data();

  // Iterate to generate the differenced series
  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + batch_size, [=] __device__(int ib) {
                     const DataT* b_ar_vec = d_ar_vec + ib * p_sP;
                     const DataT* b_ma_vec = d_ma_vec + ib * q_sQ;
                     const DataT* b_res = d_res + ib * (n_obs - d_sD);
                     DataT* b_diff = d_diff + ib * (n_obs - d_sD);
                     DataT b_mu = order.k ? params.mu[ib] : 0;
                     for (int i = 0; i < n_obs - d_sD; i++) {
                       // Observation noise
                       DataT yi = b_mu + b_res[i];
                       // AR component
                       for (int ip = 0; ip < p_sP; ip++) {
                         if (i - 1 - ip >= 0)
                           yi += b_ar_vec[ip] * b_diff[i - 1 - ip];
                       }
                       // MA component
                       for (int iq = 0; iq < q_sQ; iq++) {
                         if (i - 1 - iq >= 0)
                           yi += b_ma_vec[iq] * b_res[i - 1 - iq];
                       }

                       b_diff[i] = yi;
                     }
                   });

  // Final time series
  if (d_sD || order.k) {
    TimeSeries::finalize_forecast(d_diff, starting_values.data(), n_obs - d_sD,
                                  batch_size, d_sD, d_sD, order.d, order.D,
                                  order.s, stream);
  }

  // Copy to output if we didn't write directly to the output vector
  if (d_sD) {
    DataT* d_starting_values = starting_values.data();
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + batch_size, [=] __device__(int ib) {
                       for (int i = 0; i < d_sD; i++) {
                         out[ib * n_obs + i] = d_starting_values[d_sD * ib + i];
                       }
                       for (int i = 0; i < n_obs - d_sD; i++) {
                         out[ib * n_obs + d_sD + i] =
                           d_diff[(n_obs - d_sD) * ib + i];
                       }
                     });
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace Random
}  // namespace MLCommon