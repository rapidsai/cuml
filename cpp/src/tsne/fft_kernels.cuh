/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

/*
 * This code is based on https://github.com/CannyLab/tsne-cuda (licensed under
 * the BSD 3-clause license at cannylabs_tsne_license.txt), which is in turn a
 * CUDA implementation of Linderman et al.'s FIt-SNE (MIT license)
 * (https://github.com/KlugerLab/FIt-SNE).
 */

#pragma once

#include <cuComplex.h>

namespace ML {
namespace TSNE {
namespace FFT {

template <typename value_idx, typename value_t>
CUML_KERNEL void compute_chargesQij(volatile value_t* __restrict__ chargesQij,
                                    const value_t* __restrict__ xs,
                                    const value_t* __restrict__ ys,
                                    const value_idx num_points,
                                    const value_idx n_terms)
{
  int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= num_points) return;

  value_t x_pt = xs[TID];
  value_t y_pt = ys[TID];

  chargesQij[TID * n_terms + 0] = 1;
  chargesQij[TID * n_terms + 1] = x_pt;
  chargesQij[TID * n_terms + 2] = y_pt;
  chargesQij[TID * n_terms + 3] = x_pt * x_pt + y_pt * y_pt;
}

template <typename value_idx, typename value_t>
CUML_KERNEL void compute_bounds(volatile value_t* __restrict__ box_lower_bounds,
                                const value_t box_width,
                                const value_t x_min,
                                const value_t y_min,
                                const value_idx n_boxes,
                                const value_idx n_total_boxes)
{
  const int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_boxes * n_boxes) return;

  const int i = TID / n_boxes;
  const int j = TID % n_boxes;

  box_lower_bounds[i * n_boxes + j]                 = j * box_width + x_min;
  box_lower_bounds[n_total_boxes + i * n_boxes + j] = i * box_width + y_min;
}

template <typename value_t>
HDI value_t squared_cauchy_2d(value_t x1, value_t x2, value_t y1, value_t y2)
{
  value_t x1_m_y1 = x1 - y1;
  value_t x2_m_y2 = x2 - y2;
  value_t t       = 1.0f + x1_m_y1 * x1_m_y1 + x2_m_y2 * x2_m_y2;
  return 1.0f / (t * t);
}

template <typename value_idx, typename value_t>
CUML_KERNEL void compute_kernel_tilde(volatile value_t* __restrict__ kernel_tilde,
                                      const value_t x_min,
                                      const value_t y_min,
                                      const value_t h,
                                      const value_idx n_interpolation_points_1d,
                                      const value_idx n_fft_coeffs)
{
  const int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_interpolation_points_1d * n_interpolation_points_1d) return;

  const value_idx i = TID / n_interpolation_points_1d;
  const value_idx j = TID % n_interpolation_points_1d;

  value_t tmp =
    squared_cauchy_2d(y_min + h / 2, x_min + h / 2, y_min + h / 2 + i * h, x_min + h / 2 + j * h);
  const value_idx n_interpolation_points_1d_p_i = n_interpolation_points_1d + i;
  const value_idx n_interpolation_points_1d_m_i = n_interpolation_points_1d - i;
  const value_idx n_interpolation_points_1d_p_j = n_interpolation_points_1d + j;
  const value_idx n_interpolation_points_1d_m_j = n_interpolation_points_1d - j;
  const value_idx p_i_n                         = n_interpolation_points_1d_p_i * n_fft_coeffs;
  const value_idx m_i_n                         = n_interpolation_points_1d_m_i * n_fft_coeffs;
  kernel_tilde[p_i_n + n_interpolation_points_1d_p_j] = tmp;
  kernel_tilde[m_i_n + n_interpolation_points_1d_p_j] = tmp;
  kernel_tilde[p_i_n + n_interpolation_points_1d_m_j] = tmp;
  kernel_tilde[m_i_n + n_interpolation_points_1d_m_j] = tmp;
}

template <typename value_idx, typename value_t>
CUML_KERNEL void compute_point_box_idx(volatile value_idx* __restrict__ point_box_idx,
                                       volatile value_t* __restrict__ x_in_box,
                                       volatile value_t* __restrict__ y_in_box,
                                       const value_t* const xs,
                                       const value_t* const ys,
                                       const value_t* const box_lower_bounds,
                                       const value_t min_coord,
                                       const value_t box_width,
                                       const value_idx n_boxes,
                                       const value_idx n_total_boxes,
                                       const value_idx N)
{
  const value_idx TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= N) return;

  value_idx x_idx = static_cast<value_idx>((xs[TID] - min_coord) / box_width);
  value_idx y_idx = static_cast<value_idx>((ys[TID] - min_coord) / box_width);

  x_idx = max((value_idx)0, x_idx);
  x_idx = min(n_boxes - 1, x_idx);

  y_idx = max((value_idx)0, y_idx);
  y_idx = min(n_boxes - 1, y_idx);

  value_idx box_idx  = y_idx * n_boxes + x_idx;
  point_box_idx[TID] = box_idx;

  x_in_box[TID] = (xs[TID] - box_lower_bounds[box_idx]) / box_width;
  y_in_box[TID] = (ys[TID] - box_lower_bounds[n_total_boxes + box_idx]) / box_width;
}

template <typename value_idx, typename value_t>
CUML_KERNEL void interpolate_device(volatile value_t* __restrict__ interpolated_values,
                                    const value_t* const y_in_box,
                                    const value_t* const y_tilde_spacings,
                                    const value_t* const denominator,
                                    const value_idx n_interpolation_points,
                                    const value_idx N)
{
  const value_idx TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= N * n_interpolation_points) return;

  value_idx i = TID % N;
  value_idx j = TID / N;

  value_t value  = 1.0f;
  value_t ybox_i = y_in_box[i];

  for (value_idx k = 0; k < n_interpolation_points; k++) {
    if (j != k) { value *= ybox_i - y_tilde_spacings[k]; }
  }

  interpolated_values[j * N + i] = value / denominator[j];
}

template <typename value_idx, typename value_t>
CUML_KERNEL void compute_interpolated_indices(value_t* __restrict__ w_coefficients_device,
                                              const value_idx* const point_box_indices,
                                              const value_t* const chargesQij,
                                              const value_t* const x_interpolated_values,
                                              const value_t* const y_interpolated_values,
                                              const value_idx N,
                                              const value_idx n_interpolation_points,
                                              const value_idx n_boxes,
                                              const value_idx n_terms)
{
  value_idx TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_terms * n_interpolation_points * n_interpolation_points * N) return;

  value_idx current_term = TID % n_terms;
  value_idx i            = (TID / n_terms) % N;
  value_idx interp_j     = ((TID / n_terms) / N) % n_interpolation_points;
  value_idx interp_i     = ((TID / n_terms) / N) / n_interpolation_points;

  value_idx box_idx = point_box_indices[i];
  value_idx box_i   = box_idx % n_boxes;
  value_idx box_j   = box_idx / n_boxes;

  value_idx idx = (box_i * n_interpolation_points + interp_i) * (n_boxes * n_interpolation_points) +
                  (box_j * n_interpolation_points) + interp_j;
  atomicAdd(w_coefficients_device + idx * n_terms + current_term,
            x_interpolated_values[i + interp_i * N] * y_interpolated_values[i + interp_j * N] *
              chargesQij[i * n_terms + current_term]);
}

template <typename value_idx, typename value_t>
CUML_KERNEL void copy_to_fft_input(volatile value_t* __restrict__ fft_input,
                                   const value_t* w_coefficients_device,
                                   const value_idx n_fft_coeffs,
                                   const value_idx n_fft_coeffs_half,
                                   const value_idx n_terms)
{
  const value_idx TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_terms * n_fft_coeffs_half * n_fft_coeffs_half) return;

  value_idx current_term = TID / (n_fft_coeffs_half * n_fft_coeffs_half);
  value_idx current_loc  = TID % (n_fft_coeffs_half * n_fft_coeffs_half);

  value_idx i = current_loc / n_fft_coeffs_half;
  value_idx j = current_loc % n_fft_coeffs_half;

  fft_input[current_term * (n_fft_coeffs * n_fft_coeffs) + i * n_fft_coeffs + j] =
    w_coefficients_device[current_term + current_loc * n_terms];
}

template <typename value_idx, typename value_t>
CUML_KERNEL void copy_from_fft_output(volatile value_t* __restrict__ y_tilde_values,
                                      const value_t* fft_output,
                                      const value_idx n_fft_coeffs,
                                      const value_idx n_fft_coeffs_half,
                                      const value_idx n_terms)
{
  const value_idx TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_terms * n_fft_coeffs_half * n_fft_coeffs_half) return;

  value_idx current_term = TID / (n_fft_coeffs_half * n_fft_coeffs_half);
  value_idx current_loc  = TID % (n_fft_coeffs_half * n_fft_coeffs_half);

  value_idx i = current_loc / n_fft_coeffs_half + n_fft_coeffs_half;
  value_idx j = current_loc % n_fft_coeffs_half + n_fft_coeffs_half;

  y_tilde_values[current_term + n_terms * current_loc] =
    fft_output[current_term * (n_fft_coeffs * n_fft_coeffs) + i * n_fft_coeffs + j] /
    (value_t)(n_fft_coeffs * n_fft_coeffs);
}

// Template so that division is by compile-time divisors.
template <typename value_idx, typename value_t, int n_terms, int n_interpolation_points>
CUML_KERNEL void compute_potential_indices(value_t* __restrict__ potentialsQij,
                                           const value_idx* const point_box_indices,
                                           const value_t* const y_tilde_values,
                                           const value_t* const x_interpolated_values,
                                           const value_t* const y_interpolated_values,
                                           const value_idx N,
                                           const value_idx n_boxes)
{
  const value_idx TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_terms * n_interpolation_points * n_interpolation_points * N) return;

  value_idx current_term = TID % n_terms;
  value_idx i            = (TID / n_terms) % N;
  value_idx interp_j     = ((TID / n_terms) / N) % n_interpolation_points;
  value_idx interp_i     = ((TID / n_terms) / N) / n_interpolation_points;

  value_idx box_idx = point_box_indices[i];
  value_idx box_i   = box_idx % n_boxes;
  value_idx box_j   = box_idx / n_boxes;

  value_idx idx = (box_i * n_interpolation_points + interp_i) * (n_boxes * n_interpolation_points) +
                  (box_j * n_interpolation_points) + interp_j;
  // interpolated_values[TID] = x_interpolated_values[i + interp_i * N] *
  // y_interpolated_values[i + interp_j * N] * y_tilde_values[idx * n_terms + current_term];
  // interpolated_indices[TID] = i * n_terms + current_term;
  atomicAdd(potentialsQij + i * n_terms + current_term,
            x_interpolated_values[i + interp_i * N] * y_interpolated_values[i + interp_j * N] *
              y_tilde_values[idx * n_terms + current_term]);
}

template <typename value_idx>
CUML_KERNEL void broadcast_column_vector(cuComplex* __restrict__ mat,
                                         cuComplex* __restrict__ vec,
                                         value_idx n,
                                         value_idx m)
{
  const value_idx TID = threadIdx.x + blockIdx.x * blockDim.x;
  const value_idx i   = TID % n;
  const value_idx j   = TID / n;
  if (j < m) {
    value_idx idx = j * n + i;
    mat[idx]      = cuCmulf(mat[idx], vec[i]);
  }
}

template <typename value_idx, typename value_t>
CUML_KERNEL void compute_repulsive_forces_kernel(
  volatile value_t* __restrict__ repulsive_forces_device,
  volatile value_t* __restrict__ normalization_vec_device,
  const value_t* const xs,
  const value_t* const ys,
  const value_t* const potentialsQij,
  const value_idx num_points,
  const value_idx n_terms)
{
  value_idx TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= num_points) return;

  value_t phi1 = potentialsQij[TID * n_terms + 0];
  value_t phi2 = potentialsQij[TID * n_terms + 1];
  value_t phi3 = potentialsQij[TID * n_terms + 2];
  value_t phi4 = potentialsQij[TID * n_terms + 3];

  value_t x_pt = xs[TID];
  value_t y_pt = ys[TID];

  normalization_vec_device[TID] =
    (1 + x_pt * x_pt + y_pt * y_pt) * phi1 - 2 * (x_pt * phi2 + y_pt * phi3) + phi4;

  repulsive_forces_device[TID]              = x_pt * phi1 - phi2;
  repulsive_forces_device[TID + num_points] = y_pt * phi1 - phi3;
}

template <typename value_idx, typename value_t>
CUML_KERNEL void compute_Pij_x_Qij_kernel(value_t* __restrict__ attr_forces,
                                          value_t* __restrict__ Qs,
                                          const value_t* __restrict__ pij,
                                          const value_idx* __restrict__ coo_rows,
                                          const value_idx* __restrict__ coo_cols,
                                          const value_t* __restrict__ points,
                                          const value_idx num_points,
                                          const value_idx num_nonzero,
                                          const value_t dof)
{
  const value_idx TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= num_nonzero) return;
  const value_idx i = coo_rows[TID];
  const value_idx j = coo_cols[TID];

  value_t ix = points[i];
  value_t iy = points[num_points + i];
  value_t jx = points[j];
  value_t jy = points[num_points + j];

  value_t dx = ix - jx;
  value_t dy = iy - jy;

  const value_t dist = (dx * dx) + (dy * dy);

  const value_t P  = pij[TID];
  const value_t Q  = compute_q(dist, dof);
  const value_t PQ = P * Q;

  atomicAdd(attr_forces + i, PQ * dx);
  atomicAdd(attr_forces + num_points + i, PQ * dy);

  if (Qs) {  // when computing KL div
    Qs[TID] = Q;
  }
}

template <typename value_idx, typename value_t>
CUML_KERNEL void IntegrationKernel(volatile value_t* __restrict__ points,
                                   volatile value_t* __restrict__ attr_forces,
                                   volatile value_t* __restrict__ rep_forces,
                                   volatile value_t* __restrict__ gains,
                                   volatile value_t* __restrict__ old_forces,
                                   const value_t eta,
                                   const value_t normalization,
                                   const value_t momentum,
                                   const value_t exaggeration,
                                   const value_idx num_points)
{
  // iterate over all bodies assigned to thread
  const value_idx inc = blockDim.x * gridDim.x;
  for (value_idx i = threadIdx.x + blockIdx.x * blockDim.x; i < num_points; i += inc) {
    value_t ux = old_forces[i];
    value_t uy = old_forces[num_points + i];
    value_t gx = gains[i];
    value_t gy = gains[num_points + i];
    value_t dx = exaggeration * attr_forces[i] - (rep_forces[i] / normalization);
    value_t dy =
      exaggeration * attr_forces[i + num_points] - (rep_forces[i + num_points] / normalization);

    gx = signbit(dx) != signbit(ux) ? gx + 0.2 : gx * 0.8;
    gy = signbit(dy) != signbit(uy) ? gy + 0.2 : gy * 0.8;
    gx = gx < 0.01 ? 0.01 : gx;
    gy = gy < 0.01 ? 0.01 : gy;

    ux = momentum * ux - eta * gx * dx;
    uy = momentum * uy - eta * gy * dy;

    points[i] += ux;
    points[i + num_points] += uy;

    attr_forces[i]              = 0.0f;
    attr_forces[num_points + i] = 0.0f;
    rep_forces[i]               = 0.0f;
    rep_forces[num_points + i]  = 0.0f;
    old_forces[i]               = ux;
    old_forces[num_points + i]  = uy;
    gains[i]                    = gx;
    gains[num_points + i]       = gy;
  }
}

}  // namespace FFT
}  // namespace TSNE
}  // namespace ML
