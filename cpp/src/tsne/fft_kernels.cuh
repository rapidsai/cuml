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

#include <cuComplex.h>

namespace ML {
namespace TSNE {
namespace FFT {

__global__ void compute_chargesQij(volatile float* __restrict__ chargesQij,
                                   const float* __restrict__ xs,
                                   const float* __restrict__ ys,
                                   const int num_points, const int n_terms) {
  int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= num_points) return;

  float x_pt = xs[TID];
  float y_pt = ys[TID];

  chargesQij[TID * n_terms + 0] = 1;
  chargesQij[TID * n_terms + 1] = x_pt;
  chargesQij[TID * n_terms + 2] = y_pt;
  chargesQij[TID * n_terms + 3] = x_pt * x_pt + y_pt * y_pt;
}

__global__ void compute_bounds(volatile float* __restrict__ box_lower_bounds,
                               const float box_width, const float x_min,
                               const float y_min, const int n_boxes,
                               const int n_total_boxes) {
  const int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_boxes * n_boxes) return;

  const int i = TID / n_boxes;
  const int j = TID % n_boxes;

  box_lower_bounds[i * n_boxes + j] = j * box_width + x_min;
  box_lower_bounds[n_total_boxes + i * n_boxes + j] = i * box_width + y_min;
}

HDI float squared_cauchy_2d(float x1, float x2, float y1, float y2) {
  float x1_m_y1 = x1 - y1;
  float x2_m_y2 = x2 - y2;
  float t = 1.0f + x1_m_y1 * x1_m_y1 + x2_m_y2 * x2_m_y2;
  return 1.0f / (t * t);
}

__global__ void compute_kernel_tilde(volatile float* __restrict__ kernel_tilde,
                                     const float x_min, const float y_min,
                                     const float h,
                                     const int n_interpolation_points_1d,
                                     const int n_fft_coeffs) {
  const int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_interpolation_points_1d * n_interpolation_points_1d) return;

  const int i = TID / n_interpolation_points_1d;
  const int j = TID % n_interpolation_points_1d;

  float tmp = squared_cauchy_2d(y_min + h / 2, x_min + h / 2,
                                y_min + h / 2 + i * h, x_min + h / 2 + j * h);
  kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs +
               (n_interpolation_points_1d + j)] = tmp;
  kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs +
               (n_interpolation_points_1d + j)] = tmp;
  kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs +
               (n_interpolation_points_1d - j)] = tmp;
  kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs +
               (n_interpolation_points_1d - j)] = tmp;
}

__global__ void compute_point_box_idx(volatile int* __restrict__ point_box_idx,
                                      volatile float* __restrict__ x_in_box,
                                      volatile float* __restrict__ y_in_box,
                                      const float* const xs,
                                      const float* const ys,
                                      const float* const box_lower_bounds,
                                      const float min_coord,
                                      const float box_width, const int n_boxes,
                                      const int n_total_boxes, const int N) {
  const int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= N) return;

  int x_idx = static_cast<int>((xs[TID] - min_coord) / box_width);
  int y_idx = static_cast<int>((ys[TID] - min_coord) / box_width);

  x_idx = max(0, x_idx);
  x_idx = min(n_boxes - 1, x_idx);

  y_idx = max(0, y_idx);
  y_idx = min(n_boxes - 1, y_idx);

  int box_idx = y_idx * n_boxes + x_idx;
  point_box_idx[TID] = box_idx;

  x_in_box[TID] = (xs[TID] - box_lower_bounds[box_idx]) / box_width;
  y_in_box[TID] =
    (ys[TID] - box_lower_bounds[n_total_boxes + box_idx]) / box_width;
}

__global__ void interpolate_device(
  volatile float* __restrict__ interpolated_values, const float* const y_in_box,
  const float* const y_tilde_spacings, const float* const denominator,
  const int n_interpolation_points, const int N) {
  const int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= N * n_interpolation_points) return;

  int i = TID % N;
  int j = TID / N;

  float value = 1.0f;
  float ybox_i = y_in_box[i];

  for (int k = 0; k < n_interpolation_points; k++) {
    if (j != k) {
      value *= ybox_i - y_tilde_spacings[k];
    }
  }

  interpolated_values[j * N + i] = value / denominator[j];
}

__global__ void compute_interpolated_indices(
  float* __restrict__ w_coefficients_device, const int* const point_box_indices,
  const float* const chargesQij, const float* const x_interpolated_values,
  const float* const y_interpolated_values, const int N,
  const int n_interpolation_points, const int n_boxes, const int n_terms) {
  int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_terms * n_interpolation_points * n_interpolation_points * N)
    return;

  int current_term = TID % n_terms;
  int i = (TID / n_terms) % N;
  int interp_j = ((TID / n_terms) / N) % n_interpolation_points;
  int interp_i = ((TID / n_terms) / N) / n_interpolation_points;

  int box_idx = point_box_indices[i];
  int box_i = box_idx % n_boxes;
  int box_j = box_idx / n_boxes;

  // interpolated_values[TID] = x_interpolated_values[i + interp_i * N] * y_interpolated_values[i + interp_j * N] * chargesQij[i * n_terms + current_term];
  int idx = (box_i * n_interpolation_points + interp_i) *
              (n_boxes * n_interpolation_points) +
            (box_j * n_interpolation_points) + interp_j;
  // interpolated_indices[TID] = idx * n_terms + current_term;
  atomicAdd(w_coefficients_device + idx * n_terms + current_term,
            x_interpolated_values[i + interp_i * N] *
              y_interpolated_values[i + interp_j * N] *
              chargesQij[i * n_terms + current_term]);
}

__global__ void copy_to_fft_input(volatile float* __restrict__ fft_input,
                                  const float* w_coefficients_device,
                                  const int n_fft_coeffs,
                                  const int n_fft_coeffs_half,
                                  const int n_terms) {
  const int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_terms * n_fft_coeffs_half * n_fft_coeffs_half) return;

  int current_term = TID / (n_fft_coeffs_half * n_fft_coeffs_half);
  int current_loc = TID % (n_fft_coeffs_half * n_fft_coeffs_half);

  int i = current_loc / n_fft_coeffs_half;
  int j = current_loc % n_fft_coeffs_half;

  fft_input[current_term * (n_fft_coeffs * n_fft_coeffs) + i * n_fft_coeffs +
            j] = w_coefficients_device[current_term + current_loc * n_terms];
}

__global__ void copy_from_fft_output(
  volatile float* __restrict__ y_tilde_values, const float* fft_output,
  const int n_fft_coeffs, const int n_fft_coeffs_half, const int n_terms) {
  const int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_terms * n_fft_coeffs_half * n_fft_coeffs_half) return;

  int current_term = TID / (n_fft_coeffs_half * n_fft_coeffs_half);
  int current_loc = TID % (n_fft_coeffs_half * n_fft_coeffs_half);

  int i = current_loc / n_fft_coeffs_half + n_fft_coeffs_half;
  int j = current_loc % n_fft_coeffs_half + n_fft_coeffs_half;

  y_tilde_values[current_term + n_terms * current_loc] =
    fft_output[current_term * (n_fft_coeffs * n_fft_coeffs) + i * n_fft_coeffs +
               j] /
    (float)(n_fft_coeffs * n_fft_coeffs);
}

// Template so that division is by compile-time divisors.
template <int n_terms, int n_interpolation_points>
__global__ void compute_potential_indices(
  float* __restrict__ potentialsQij, const int* const point_box_indices,
  const float* const y_tilde_values, const float* const x_interpolated_values,
  const float* const y_interpolated_values, const int N, const int n_boxes) {
  const int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_terms * n_interpolation_points * n_interpolation_points * N)
    return;

  int current_term = TID % n_terms;
  int i = (TID / n_terms) % N;
  int interp_j = ((TID / n_terms) / N) % n_interpolation_points;
  int interp_i = ((TID / n_terms) / N) / n_interpolation_points;

  int box_idx = point_box_indices[i];
  int box_i = box_idx % n_boxes;
  int box_j = box_idx / n_boxes;

  int idx = (box_i * n_interpolation_points + interp_i) *
              (n_boxes * n_interpolation_points) +
            (box_j * n_interpolation_points) + interp_j;
  // interpolated_values[TID] = x_interpolated_values[i + interp_i * N] * y_interpolated_values[i + interp_j * N] * y_tilde_values[idx * n_terms + current_term];
  // interpolated_indices[TID] = i * n_terms + current_term;
  atomicAdd(potentialsQij + i * n_terms + current_term,
            x_interpolated_values[i + interp_i * N] *
              y_interpolated_values[i + interp_j * N] *
              y_tilde_values[idx * n_terms + current_term]);
}

__global__ void broadcast_column_vector(cuComplex* __restrict__ mat,
                                        cuComplex* __restrict__ vec, int n,
                                        int m) {
  const int TID = threadIdx.x + blockIdx.x * blockDim.x;
  const int i = TID % n;
  const int j = TID / n;
  if (j < m) {
    int idx = j * n + i;
    mat[idx] = cuCmulf(mat[idx], vec[i]);
  }
}

__global__ void compute_repulsive_forces_kernel(
  volatile float* __restrict__ repulsive_forces_device,
  volatile float* __restrict__ normalization_vec_device, const float* const xs,
  const float* const ys, const float* const potentialsQij, const int num_points,
  const int n_terms) {
  int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= num_points) return;

  float phi1 = potentialsQij[TID * n_terms + 0];
  float phi2 = potentialsQij[TID * n_terms + 1];
  float phi3 = potentialsQij[TID * n_terms + 2];
  float phi4 = potentialsQij[TID * n_terms + 3];

  float x_pt = xs[TID];
  float y_pt = ys[TID];

  normalization_vec_device[TID] = (1 + x_pt * x_pt + y_pt * y_pt) * phi1 -
                                  2 * (x_pt * phi2 + y_pt * phi3) + phi4;

  repulsive_forces_device[TID] = x_pt * phi1 - phi2;
  repulsive_forces_device[TID + num_points] = y_pt * phi1 - phi3;
}

__global__ void compute_Pij_x_Qij_kernel(float* __restrict__ attr_forces,
                                         const float* __restrict__ pij,
                                         const int* __restrict__ coo_rows,
                                         const int* __restrict__ coo_cols,
                                         const float* __restrict__ points,
                                         const int num_points,
                                         const int num_nonzero) {
  const int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= num_nonzero) return;
  const int i = coo_rows[TID];
  const int j = coo_cols[TID];

  float ix = points[i];
  float iy = points[num_points + i];
  float jx = points[j];
  float jy = points[num_points + j];
  float dx = ix - jx;
  float dy = iy - jy;
  float pijqij = pij[TID] / (1 + dx * dx + dy * dy);
  atomicAdd(attr_forces + i, pijqij * dx);
  atomicAdd(attr_forces + num_points + i, pijqij * dy);
}

__global__ void IntegrationKernel(
  volatile float* __restrict__ points, volatile float* __restrict__ attr_forces,
  volatile float* __restrict__ rep_forces, volatile float* __restrict__ gains,
  volatile float* __restrict__ old_forces, const float eta,
  const float normalization, const float momentum, const float exaggeration,
  const int num_points) {
  // iterate over all bodies assigned to thread
  const int inc = blockDim.x * gridDim.x;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_points;
       i += inc) {
    float ux = old_forces[i];
    float uy = old_forces[num_points + i];
    float gx = gains[i];
    float gy = gains[num_points + i];
    float dx = exaggeration * attr_forces[i] - (rep_forces[i] / normalization);
    float dy = exaggeration * attr_forces[i + num_points] -
               (rep_forces[i + num_points] / normalization);

    gx = signbit(dx) != signbit(ux) ? gx + 0.2 : gx * 0.8;
    gy = signbit(dy) != signbit(uy) ? gy + 0.2 : gy * 0.8;
    gx = gx < 0.01 ? 0.01 : gx;
    gy = gy < 0.01 ? 0.01 : gy;

    ux = momentum * ux - eta * gx * dx;
    uy = momentum * uy - eta * gy * dy;

    points[i] += ux;
    points[i + num_points] += uy;

    attr_forces[i] = 0.0f;
    attr_forces[num_points + i] = 0.0f;
    rep_forces[i] = 0.0f;
    rep_forces[num_points + i] = 0.0f;
    old_forces[i] = ux;
    old_forces[num_points + i] = uy;
    gains[i] = gx;
    gains[num_points + i] = gy;
  }
}

}  // namespace FFT
}  // namespace TSNE
}  // namespace ML
