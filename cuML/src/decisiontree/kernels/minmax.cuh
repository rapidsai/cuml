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

void min_and_max(float * d_samples, int num_samples, TemporaryMemory* tempmem) {

	tempmem->d_min_max_thrust = thrust::minmax_element(thrust::cuda::par.on(tempmem->stream), d_samples, d_samples + num_samples);
	
/*	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
	float thrust_min_max[2];
	float our_min_max[2];
	CUDA_CHECK(cudaMemcpyAsync(&thrust_min_max[0], tempmem->d_min_max_thrust.first, sizeof(float), cudaMemcpyDeviceToHost, tempmem->stream));
	CUDA_CHECK(cudaMemcpyAsync(&thrust_min_max[1], tempmem->d_min_max_thrust.second, sizeof(float), cudaMemcpyDeviceToHost, tempmem->stream));
	CUDA_CHECK(cudaMemcpyAsync(&our_min_max[0], &tempmem->d_min_max[0], sizeof(float), cudaMemcpyDeviceToHost, tempmem->stream));
	CUDA_CHECK(cudaMemcpyAsync(&our_min_max[1], &tempmem->d_min_max[1], sizeof(float), cudaMemcpyDeviceToHost, tempmem->stream));
	CUDA_CHECK(cudaStreamSynchronize(tempmem->stream));
	
	//std::cout << "Thrust min max " << thrust_min_max[0] << " " << thrust_min_max[1] << std::endl;
	//std::cout << "Our min max " << our_min_max[0] << " " << our_min_max[1] << std::endl;
	ASSERT(thrust_min_max[0] == our_min_max[0], "Min mismatch thrust %f ours %f\n", thrust_min_max[0], our_min_max[0]);
	ASSERT(thrust_min_max[1] == our_min_max[1], "Max mismatch thrust %f ours %f\n", thrust_min_max[1], our_min_max[1]);
*/
}
