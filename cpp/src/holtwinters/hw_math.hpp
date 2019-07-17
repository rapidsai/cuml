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

#pragma once

namespace ML {
namespace HoltWinters {
namespace Math {

template <typename Dtype>
void div_gpu(int n, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void subtract_gpu(int n, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void subtract_gpu(int n, const Dtype* a, const Dtype* b, Dtype* y);

__device__ __forceinline__ float log_device(float x) { return logf(x); }
__device__ __forceinline__ double log_device(double x) { return log(x); }

__device__ __forceinline__ float exp_device(float x) { return expf(x); }
__device__ __forceinline__ double exp_device(double x) { return exp(x); }

__device__ __forceinline__ float expm1_device(float x) { return expm1f(x); }
__device__ __forceinline__ double expm1_device(double x) { return expm1(x); }

__device__ __forceinline__ float pow_device(float base, float exponent) {
  return powf(base, exponent);
}
__device__ __forceinline__ double pow_device(double base, double exponent) {
  return pow(base, exponent);
}

__device__ __forceinline__ float log1p_device(float x) { return log1pf(x); }
__device__ __forceinline__ double log1p_device(double x) { return log1p(x); }

__device__ __forceinline__ float abs_device(float x) { return fabsf(x); }
__device__ __forceinline__ double abs_device(double x) { return fabs(x); }

__device__ __forceinline__ float sqrt_device(float x) { return sqrtf(x); }
__device__ __forceinline__ double sqrt_device(double x) { return sqrt(x); }

}  // namespace Math
}  // namespace HoltWinters
}  // namespace ML
