/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#include <utils.h>
#include <thrust/extrema.h>

float minimum(float *d_samples, int num_samples)
{
  float minval;
  float *min_ptr = thrust::min_element(thrust::device, d_samples, d_samples + num_samples);
  CUDA_CHECK(cudaMemcpy(&minval, min_ptr, sizeof(float), cudaMemcpyDeviceToHost));
  return minval;
}

float maximum(float *d_samples, int num_samples)
{
  float maxval;
  float *max_ptr = thrust::max_element(thrust::device, d_samples, d_samples + num_samples);
  CUDA_CHECK(cudaMemcpy(&maxval, max_ptr, sizeof(float), cudaMemcpyDeviceToHost));
  return maxval;
}

// According to https://thrust.github.io/doc/group__extrema.html#gaa9dcee5e36206a3ef7215a4b3984e002
// minmax_element "[..] function is potentially more efficient than separate cols to min_element and max_element."

void min_and_max(float * d_samples, int num_samples, float & min_val, float & max_val, const cudaStream_t stream=0) {
	thrust::pair<float *, float *> result;
	//result = thrust::minmax_element(thrust::cuda::par.on(stream), thrust::device, d_samples, d_samples + num_samples);
	result = thrust::minmax_element(thrust::cuda::par.on(stream), d_samples, d_samples + num_samples);
	CUDA_CHECK(cudaMemcpyAsync(&min_val, result.first, sizeof(float), cudaMemcpyDeviceToHost, stream));
 	CUDA_CHECK(cudaMemcpyAsync(&max_val, result.second, sizeof(float), cudaMemcpyDeviceToHost, stream));
}
