/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cufft_utils.h>
#include <linalg/init.h>
#include <cmath>
#include <common/device_buffer.hpp>
#include <common/device_utils.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/stats/sum.cuh>
#include "fft_kernels.cuh"
#include "utils.cuh"

namespace ML {
namespace TSNE {

/**
 * @brief Fast Dimensionality reduction via TSNE using the Barnes Hut O(NlogN) approximation.
 * @param[in] VAL: The values in the attractive forces COO matrix.
 * @param[in] COL: The column indices in the attractive forces COO matrix.
 * @param[in] ROW: The row indices in the attractive forces COO matrix.
 * @param[in] NNZ: The number of non zeros in the attractive forces COO matrix.
 * @param[in] handle: The GPU handle.
 * @param[out] Y: The final embedding (col-major).
 * @param[in] n: Number of rows in data X.
 * @param[in] early_exaggeration: How much pressure to apply to clusters to spread out during the exaggeration phase.
 * @param[in] late_exaggeration: How much pressure to apply to clusters to spread out after the exaggeration phase.
 * @param[in] exaggeration_iter: How many iterations you want the early pressure to run for.
 * @param[in] pre_learning_rate: The learning rate during the exaggeration phase.
 * @param[in] post_learning_rate: The learning rate after the exaggeration phase.
 * @param[in] max_iter: The maximum number of iterations TSNE should run for.
 * @param[in] min_grad_norm: The smallest gradient norm TSNE should terminate on.
 * @param[in] pre_momentum: The momentum used during the exaggeration phase.
 * @param[in] post_momentum: The momentum used after the exaggeration phase.
 * @param[in] random_state: Set this to -1 for random intializations or >= 0 to see the PRNG.
 * @param[in] initialize_embeddings: Whether to overwrite the current Y vector with random noise.
 */
void FFT_TSNE(float *VAL, const int *COL, const int *ROW, const int NNZ,
              const raft::handle_t &handle, float *Y, const int n,
              const float early_exaggeration, const float late_exaggeration,
              const int exaggeration_iter, const float pre_learning_rate,
              const float post_learning_rate, const int max_iter,
              const float min_grad_norm, const float pre_momentum,
              const float post_momentum, const long long random_state,
              const bool initialize_embeddings) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  // Get device properites
  //---------------------------------------------------
  const int mp_count = raft::getMultiProcessorCount();
  const int dev_major_version = MLCommon::getDeviceCapability().first;
  // These came from the CannyLab implementation, but I don't know how they were
  // determined. TODO check/optimize.
  const int integration_kernel_factor =
    dev_major_version >= 6
      ? 2
      : dev_major_version == 5 ? 1 : dev_major_version == 3 ? 2 : 3;

  constexpr int n_interpolation_points = 3;
  constexpr int min_num_intervals = 50;
  // The number of "charges" or s+2 sums i.e. number of kernel sums
  constexpr int n_terms = 4;
  int n_boxes_per_dim = min_num_intervals;

  // FFTW is faster on numbers that can be written as 2^a 3^b 5^c 7^d 11^e 13^f
  // where e+f is either 0 or 1, and the other exponents are arbitrary
  int allowed_n_boxes_per_dim[20] = {25,  36,  50,  55,  60,  65,  70,
                                     75,  80,  85,  90,  96,  100, 110,
                                     120, 130, 140, 150, 175, 200};
  if (n_boxes_per_dim < allowed_n_boxes_per_dim[19]) {
    // Round up to nearest grid point
    int chosen_i = 0;
    while (allowed_n_boxes_per_dim[chosen_i] < n_boxes_per_dim) chosen_i++;
    n_boxes_per_dim = allowed_n_boxes_per_dim[chosen_i];
  }

  int n_total_boxes = n_boxes_per_dim * n_boxes_per_dim;
  int total_interpolation_points =
    n_total_boxes * n_interpolation_points * n_interpolation_points;
  int n_fft_coeffs_half = n_interpolation_points * n_boxes_per_dim;
  int n_fft_coeffs = 2 * n_interpolation_points * n_boxes_per_dim;
  int n_interpolation_points_1d = n_interpolation_points * n_boxes_per_dim;

#define DB(type, name, size) \
  MLCommon::device_buffer<type> name(d_alloc, stream, size)

  DB(float, repulsive_forces_device, n * 2);
  MLCommon::LinAlg::zero(repulsive_forces_device.data(),
                         repulsive_forces_device.size(), stream);
  DB(float, attractive_forces_device, n * 2);
  DB(float, gains_device, n * 2);
  auto gains_device_thrust = thrust::device_pointer_cast(gains_device.data());
  thrust::fill(thrust::cuda::par.on(stream), gains_device_thrust,
               gains_device_thrust + n * 2, 1.0f);
  DB(float, old_forces_device, n * 2);
  MLCommon::LinAlg::zero(old_forces_device.data(), old_forces_device.size(),
                         stream);
  DB(float, normalization_vec_device, n);
  DB(int, point_box_idx_device, n);
  DB(float, x_in_box_device, n);
  DB(float, y_in_box_device, n);
  DB(float, y_tilde_values, total_interpolation_points *n_terms);
  DB(float, x_interpolated_values_device, n *n_interpolation_points);
  DB(float, y_interpolated_values_device, n *n_interpolation_points);
  DB(float, potentialsQij_device, n *n_terms);
  DB(float, w_coefficients_device, total_interpolation_points *n_terms);
  DB(float, all_interpolated_values_device,
     n_terms *n_interpolation_points *n_interpolation_points *n);
  DB(float, output_values,
     n_terms *n_interpolation_points *n_interpolation_points *n);
  DB(int, all_interpolated_indices,
     n_terms *n_interpolation_points *n_interpolation_points *n);
  DB(int, output_indices,
     n_terms *n_interpolation_points *n_interpolation_points *n);
  DB(float, chargesQij_device, n *n_terms);
  DB(float, box_lower_bounds_device, 2 * n_total_boxes);
  DB(float, kernel_tilde_device, n_fft_coeffs *n_fft_coeffs);
  DB(cufftComplex, fft_kernel_tilde_device,
     2 * n_interpolation_points_1d * 2 * n_interpolation_points_1d);
  DB(float, fft_input, n_terms *n_fft_coeffs *n_fft_coeffs);
  DB(cufftComplex, fft_w_coefficients,
     n_terms * n_fft_coeffs * (n_fft_coeffs / 2 + 1));
  DB(float, fft_output, n_terms *n_fft_coeffs *n_fft_coeffs);
  DB(float, sum_d, 1);

  float h = 1.0f / n_interpolation_points;
  float y_tilde_spacings[n_interpolation_points];
  y_tilde_spacings[0] = h / 2;
  for (int i = 1; i < n_interpolation_points; i++) {
    y_tilde_spacings[i] = y_tilde_spacings[i - 1] + h;
  }
  float denominator[n_interpolation_points];
  for (int i = 0; i < n_interpolation_points; i++) {
    denominator[i] = 1;
    for (int j = 0; j < n_interpolation_points; j++) {
      if (i != j) {
        denominator[i] *= y_tilde_spacings[i] - y_tilde_spacings[j];
      }
    }
  }

  DB(float, y_tilde_spacings_device, n_interpolation_points);
  CUDA_CHECK(cudaMemcpyAsync(y_tilde_spacings_device.data(), y_tilde_spacings,
                             n_interpolation_points * sizeof(*y_tilde_spacings),
                             cudaMemcpyHostToDevice, stream));
  DB(float, denominator_device, n_interpolation_points);
  CUDA_CHECK(cudaMemcpyAsync(denominator_device.data(), denominator,
                             n_interpolation_points * sizeof(*denominator),
                             cudaMemcpyHostToDevice, stream));
#undef DB

  raft::CuFFTHandle plan_kernel_tilde;
  raft::CuFFTHandle plan_dft;
  raft::CuFFTHandle plan_idft;

  size_t work_size, work_size_dft, work_size_idft;
  int fft_dimensions[2] = {n_fft_coeffs, n_fft_coeffs};
  CUFFT_TRY(cufftMakePlan2d(plan_kernel_tilde, fft_dimensions[0],
                            fft_dimensions[1], CUFFT_R2C, &work_size));
  CUFFT_TRY(cufftMakePlanMany(
    plan_dft, 2, fft_dimensions, NULL, 1, n_fft_coeffs * n_fft_coeffs, NULL, 1,
    n_fft_coeffs * (n_fft_coeffs / 2 + 1), CUFFT_R2C, n_terms, &work_size_dft));
  CUFFT_TRY(cufftMakePlanMany(plan_idft, 2, fft_dimensions, NULL, 1,
                              n_fft_coeffs * (n_fft_coeffs / 2 + 1), NULL, 1,
                              n_fft_coeffs * n_fft_coeffs, CUFFT_C2R, n_terms,
                              &work_size_idft));

  if (initialize_embeddings) {
    random_vector(Y, -0.0001f, 0.0001f, n * 2, stream, random_state);
  }

  float momentum = pre_momentum;
  float learning_rate = pre_learning_rate;
  float exaggeration = early_exaggeration;

  for (int iter = 0; iter < max_iter; iter++) {
    MLCommon::LinAlg::zero(w_coefficients_device.data(),
                           w_coefficients_device.size(), stream);
    MLCommon::LinAlg::zero(potentialsQij_device.data(),
                           potentialsQij_device.size(), stream);
    // TODO is this necessary inside the loop? IntegrationKernel zeros it.
    MLCommon::LinAlg::zero(attractive_forces_device.data(),
                           attractive_forces_device.size(), stream);

    if (iter == exaggeration_iter) {
      momentum = post_momentum;
      // learning_rate = post_learning_rate;  // TODO CannyLab doesn't switch (line 309)
      exaggeration = late_exaggeration;
      const float div = 1.0f / early_exaggeration;
      raft::linalg::scalarMultiply(VAL, VAL, div, NNZ, stream);
    }

    {  // Compute charges Q_ij
      const int num_threads = 1024;
      const int num_blocks = raft::ceildiv(n, num_threads);
      FFT::compute_chargesQij<<<num_blocks, num_threads, 0, stream>>>(
        chargesQij_device.data(), Y, Y + n, n, n_terms);
      CUDA_CHECK(cudaPeekAtLastError());
    }

    auto y_thrust = thrust::device_pointer_cast(Y);
    auto minimax_iter = thrust::minmax_element(thrust::cuda::par.on(stream),
                                               y_thrust, y_thrust + n * 2);
    float min_coord = *minimax_iter.first;
    float max_coord = *minimax_iter.second;
    float box_width =
      (max_coord - min_coord) / static_cast<float>(n_boxes_per_dim);

    //// Precompute FFT

    {  // Left and right bounds of each box, first the lower bounds in the x
      // direction, then in the y direction
      const int num_threads = 32;
      const int num_blocks = raft::ceildiv(n_total_boxes, num_threads);
      FFT::compute_bounds<<<num_blocks, num_threads, 0, stream>>>(
        box_lower_bounds_device.data(), box_width, min_coord, min_coord,
        n_boxes_per_dim, n_total_boxes);
      CUDA_CHECK(cudaPeekAtLastError());
    }

    {  // Evaluate the kernel at the interpolation nodes and form the embedded
      // generating kernel vector for a circulant matrix.
      // Coordinates of all the equispaced interpolation points
      float h = box_width / n_interpolation_points;
      const int num_threads = 32;
      const int num_blocks = raft::ceildiv(
        n_interpolation_points_1d * n_interpolation_points_1d, num_threads);
      FFT::compute_kernel_tilde<<<num_blocks, num_threads, 0, stream>>>(
        kernel_tilde_device.data(), min_coord, min_coord, h,
        n_interpolation_points_1d, n_fft_coeffs);
      CUDA_CHECK(cudaPeekAtLastError());
    }

    {  // Precompute the FFT of the kernel generating matrix
      CUFFT_TRY(cufftExecR2C(plan_kernel_tilde, kernel_tilde_device.data(),
                             fft_kernel_tilde_device.data()));
    }

    //// Run N-body FFT

    {
      const int num_threads = 128;

      int num_blocks = raft::ceildiv(n, num_threads);
      FFT::compute_point_box_idx<<<num_blocks, num_threads, 0, stream>>>(
        point_box_idx_device.data(), x_in_box_device.data(),
        y_in_box_device.data(), Y, Y + n, box_lower_bounds_device.data(),
        min_coord, box_width, n_boxes_per_dim, n_total_boxes, n);
      CUDA_CHECK(cudaPeekAtLastError());

      // Step 1: Interpolate kernel using Lagrange polynomials and compute the w
      // coefficients.

      // Compute the interpolated values at each real point with each Lagrange
      // polynomial in the `x` direction
      num_blocks = raft::ceildiv(n * n_interpolation_points, num_threads);
      FFT::interpolate_device<<<num_blocks, num_threads, 0, stream>>>(
        x_interpolated_values_device.data(), x_in_box_device.data(),
        y_tilde_spacings_device.data(), denominator_device.data(),
        n_interpolation_points, n);
      CUDA_CHECK(cudaPeekAtLastError());

      // ...and in the `y` direction
      FFT::interpolate_device<<<num_blocks, num_threads, 0, stream>>>(
        y_interpolated_values_device.data(), y_in_box_device.data(),
        y_tilde_spacings_device.data(), denominator_device.data(),
        n_interpolation_points, n);
      CUDA_CHECK(cudaPeekAtLastError());

      num_blocks = raft::ceildiv(
        n_terms * n_interpolation_points * n_interpolation_points * n,
        num_threads);
      FFT::compute_interpolated_indices<<<num_blocks, num_threads, 0, stream>>>(
        w_coefficients_device.data(), point_box_idx_device.data(),
        chargesQij_device.data(), x_interpolated_values_device.data(),
        y_interpolated_values_device.data(), n, n_interpolation_points,
        n_boxes_per_dim, n_terms);
      CUDA_CHECK(cudaPeekAtLastError());

      // Step 2: Compute the values v_{m, n} at the equispaced nodes, multiply
      // the kernel matrix with the coefficients w
      num_blocks = raft::ceildiv(
        n_terms * n_fft_coeffs_half * n_fft_coeffs_half, num_threads);
      FFT::copy_to_fft_input<<<num_blocks, num_threads, 0, stream>>>(
        fft_input.data(), w_coefficients_device.data(), n_fft_coeffs,
        n_fft_coeffs_half, n_terms);
      CUDA_CHECK(cudaPeekAtLastError());

      // Compute fft values at interpolated nodes
      CUFFT_TRY(
        cufftExecR2C(plan_dft, fft_input.data(), fft_w_coefficients.data()));
      CUDA_CHECK(cudaPeekAtLastError());

      // Take the broadcasted Hadamard product of a complex matrix and a complex
      // vector.
      {
        const int nn = n_fft_coeffs * (n_fft_coeffs / 2 + 1);
        const int num_threads = 32;
        const int num_blocks = raft::ceildiv(nn * n_terms, num_threads);
        FFT::broadcast_column_vector<<<num_blocks, num_threads, 0, stream>>>(
          fft_w_coefficients.data(), fft_kernel_tilde_device.data(), nn,
          n_terms);
        CUDA_CHECK(cudaPeekAtLastError());
      }

      // Invert the computed values at the interpolated nodes.
      CUFFT_TRY(
        cufftExecC2R(plan_idft, fft_w_coefficients.data(), fft_output.data()));
      FFT::copy_from_fft_output<<<num_blocks, num_threads, 0, stream>>>(
        y_tilde_values.data(), fft_output.data(), n_fft_coeffs,
        n_fft_coeffs_half, n_terms);
      CUDA_CHECK(cudaPeekAtLastError());

      // Step 3: Compute the potentials \tilde{\phi}
      num_blocks = raft::ceildiv(
        n_terms * n_interpolation_points * n_interpolation_points * n,
        num_threads);
      FFT::compute_potential_indices<n_terms, n_interpolation_points>
        <<<num_blocks, num_threads, 0, stream>>>(
          potentialsQij_device.data(), point_box_idx_device.data(),
          y_tilde_values.data(), x_interpolated_values_device.data(),
          y_interpolated_values_device.data(), n, n_boxes_per_dim);
      CUDA_CHECK(cudaPeekAtLastError());
    }

    float normalization;
    {  // Compute repulsive forces
      // Make the negative term, or F_rep in the equation 3 of the paper.
      const int num_threads = 1024;
      const int num_blocks = raft::ceildiv(n, num_threads);
      FFT::
        compute_repulsive_forces_kernel<<<num_blocks, num_threads, 0, stream>>>(
          repulsive_forces_device.data(), normalization_vec_device.data(), Y,
          Y + n, potentialsQij_device.data(), n, n_terms);
      CUDA_CHECK(cudaPeekAtLastError());

      // TODO is it faster to atomicAdd in compute_repulsive_forces_kernel?
      raft::stats::sum(sum_d.data(), normalization_vec_device.data(), 1, n,
                       true, stream);
      float sumQ;
      CUDA_CHECK(cudaMemcpyAsync(&sumQ, sum_d.data(), sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
      normalization = sumQ - n;
    }

    {  // Compute attractive forces
      const int num_threads = 1024;
      const int num_blocks = raft::ceildiv(NNZ, num_threads);
      FFT::compute_Pij_x_Qij_kernel<<<num_blocks, num_threads, 0, stream>>>(
        attractive_forces_device.data(), VAL, ROW, COL, Y, n, NNZ);
      CUDA_CHECK(cudaPeekAtLastError());
    }

    {  // Apply Forces
      const int num_threads = 1024;
      const int num_blocks = mp_count * integration_kernel_factor;
      FFT::IntegrationKernel<<<num_blocks, num_threads, 0, stream>>>(
        Y, attractive_forces_device.data(), repulsive_forces_device.data(),
        gains_device.data(), old_forces_device.data(), learning_rate,
        normalization, momentum, exaggeration, n);
      CUDA_CHECK(cudaPeekAtLastError());
    }

    // TODO if (iter > exaggeration_iter && grad_norm < min_grad_norm) break
  }
}

}  // namespace TSNE
}  // namespace ML
