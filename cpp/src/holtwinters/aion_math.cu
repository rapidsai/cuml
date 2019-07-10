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

#include "aion_utils.hpp"

template <typename Dtype>
__global__ void div_kernel(int n, const Dtype* a, const Dtype* b, Dtype* y) {
  GPU_LOOP(i, n) { y[i] = a[i] / b[i]; }
}

template <typename Dtype>
void aion::math::div_gpu(int n, const Dtype* a, const Dtype* b, Dtype* y) {
  div_kernel<Dtype>
    <<<GET_NUM_BLOCKS(n), GET_THREADS_PER_BLOCK(n)>>>(n, a, b, y);
}

template <typename Dtype>
__global__ void subtract_kernel(int n, const Dtype* a, const Dtype* b,
                                Dtype* y) {
  GPU_LOOP(i, n) { y[i] = a[i] - b[i]; }
}

template <typename Dtype>
void aion::math::subtract_gpu(int n, const Dtype* a, const Dtype* b, Dtype* y) {
  subtract_kernel<Dtype>
    <<<GET_NUM_BLOCKS(n), GET_THREADS_PER_BLOCK(n)>>>(n, a, b, y);
}

template void aion::math::div_gpu<float>(int n, const float* a, const float* b,
                                         float* y);
template void aion::math::div_gpu<double>(int n, const double* a,
                                          const double* b, double* y);

template void aion::math::subtract_gpu<float>(int n, const float* a,
                                              const float* b, float* y);
template void aion::math::subtract_gpu<double>(int n, const double* a,
                                               const double* b, double* y);
