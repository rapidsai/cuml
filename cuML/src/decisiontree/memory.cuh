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
#include <utils.h>
#include "cub/cub.cuh"
#include <thrust/extrema.h>
#include "common/cumlHandle.hpp"
#include <common/device_buffer.hpp>
#include <common/host_buffer.hpp>

template<class T, class L>
struct TemporaryMemory
{
	// Labels after boostrapping
	MLCommon::device_buffer<L> *sampledlabels;

	// Used for gini histograms (root tree node)
	MLCommon::device_buffer<int> *d_hist;
	MLCommon::host_buffer<int> *h_hist;
	
	//Host/Device histograms and device minmaxs
	MLCommon::device_buffer<T> *d_globalminmax;
	MLCommon::device_buffer<int> *d_histout, *d_colids;
	MLCommon::host_buffer<int> *h_histout;
	MLCommon::device_buffer<T> *d_mseout, *d_predout;
	MLCommon::host_buffer<T> *h_mseout, *h_predout;
	
	//Below pointers are shared for split functions
	MLCommon::device_buffer<char> *d_flags_left, *d_flags_right;
	MLCommon::host_buffer<int> *nrowsleftright;
	MLCommon::device_buffer<char> *d_split_temp_storage = nullptr;
	size_t split_temp_storage_bytes = 0;

	MLCommon::device_buffer<int> *d_num_selected_out, *temprowids;
	MLCommon::device_buffer<T> *question_value, *temp_data;

	//Total temp mem
	size_t totalmem = 0;

	//CUDA stream
	cudaStream_t stream;

	//For quantiles
	MLCommon::device_buffer<T> *d_quantile = nullptr;
	MLCommon::device_buffer<T> *d_temp_sampledcolumn = nullptr;

	const ML::cumlHandle_impl& ml_handle;

	TemporaryMemory(const ML::cumlHandle_impl& handle, int N, int Ncols, int maxstr, int n_unique, int n_bins, const int split_algo):ml_handle(handle)
	{
		
		//Assign Stream from cumlHandle
		stream = ml_handle.getStream();
		
		int n_hist_elements = n_unique * n_bins;

		h_hist = new MLCommon::host_buffer<int>(handle.getHostAllocator(), stream, n_hist_elements);
		d_hist = new MLCommon::device_buffer<int>(handle.getDeviceAllocator(), stream, n_hist_elements);
		nrowsleftright = new MLCommon::host_buffer<int>(handle.getHostAllocator(), stream, 2);
		
		int extra_elements = Ncols;
		int quantile_elements = (split_algo == ML::SPLIT_ALGO::GLOBAL_QUANTILE) ? extra_elements : 1;

		temp_data = new MLCommon::device_buffer<T>(handle.getDeviceAllocator(), stream, N * extra_elements);
		totalmem += n_hist_elements * sizeof(int) + N * extra_elements * sizeof(T);

		if (split_algo == ML::SPLIT_ALGO::GLOBAL_QUANTILE) {
			d_quantile = new MLCommon::device_buffer<T>(handle.getDeviceAllocator(), stream, n_bins * quantile_elements);
			d_temp_sampledcolumn = new MLCommon::device_buffer<T>(handle.getDeviceAllocator(), stream, N * extra_elements);
			totalmem += (n_bins + N) * extra_elements * sizeof(T);
		}

		sampledlabels = new MLCommon::device_buffer<L>(handle.getDeviceAllocator(), stream, N);
		totalmem += N*sizeof(L);

		//Allocate Temporary for split functions
		d_num_selected_out = new MLCommon::device_buffer<int>(handle.getDeviceAllocator(), stream, 1);
		d_flags_left = new MLCommon::device_buffer<char>(handle.getDeviceAllocator(), stream, N);
		d_flags_right = new MLCommon::device_buffer<char>(handle.getDeviceAllocator(), stream, N);
		temprowids = new MLCommon::device_buffer<int>(handle.getDeviceAllocator(), stream, N);
		question_value = new MLCommon::device_buffer<T>(handle.getDeviceAllocator(), stream, 1);

		cub::DeviceSelect::Flagged(d_split_temp_storage, split_temp_storage_bytes, temprowids->data(), d_flags_left->data(), temprowids->data(), d_num_selected_out->data(), N);
		d_split_temp_storage = new MLCommon::device_buffer<char>(handle.getDeviceAllocator(), stream, split_temp_storage_bytes);

		totalmem += split_temp_storage_bytes + (N + 1)*sizeof(int) + 2*N*sizeof(char) + sizeof(T);

		h_histout = new MLCommon::host_buffer<int>(handle.getHostAllocator(), stream, n_hist_elements * Ncols);
		int mse_elements = Ncols * n_bins;
		h_mseout = new MLCommon::host_buffer<T>(handle.getHostAllocator(), stream, 2*mse_elements);
		h_predout = new MLCommon::host_buffer<T>(handle.getHostAllocator(), stream, mse_elements);
		
		d_globalminmax = new MLCommon::device_buffer<T>(handle.getDeviceAllocator(), stream, Ncols * 2);
		d_histout = new MLCommon::device_buffer<int>(handle.getDeviceAllocator(), stream, n_hist_elements * Ncols);
		d_mseout = new MLCommon::device_buffer<T>(handle.getDeviceAllocator(), stream, 2*mse_elements);
		d_predout = new MLCommon::device_buffer<T>(handle.getDeviceAllocator(), stream, mse_elements);
		
		d_colids = new MLCommon::device_buffer<int>(handle.getDeviceAllocator(), stream, Ncols);
		totalmem += (n_hist_elements * sizeof(int) + sizeof(int) + 2*sizeof(T) + n_bins * sizeof(T))* Ncols;

	}

	void print_info()
	{
		std::cout << " Total temporary memory usage--> "<< ((double)totalmem/ (1024*1024)) << "  MB" << std::endl;
		return;
	}

	~TemporaryMemory()
	{

		h_hist->release(stream);
		d_hist->release(stream);
		nrowsleftright->release(stream);
		temp_data->release(stream);

		delete h_hist;
		delete d_hist;
		delete temp_data;

		if (d_quantile != nullptr) {
			d_quantile->release(stream);
			delete d_quantile;
		}
		if (d_temp_sampledcolumn != nullptr) {
			d_temp_sampledcolumn->release(stream);
			delete d_temp_sampledcolumn;
		}

		sampledlabels->release(stream);
		d_split_temp_storage->release(stream);
		d_num_selected_out->release(stream);
		d_flags_left->release(stream);
		d_flags_right->release(stream);
		temprowids->release(stream);
		question_value->release(stream);
		h_histout->release(stream);
		h_mseout->release(stream);
		h_predout->release(stream);
		
		delete sampledlabels;
		delete d_split_temp_storage;
		delete d_num_selected_out;
		delete d_flags_left;
		delete d_flags_right;
		delete temprowids;
		delete question_value;
		delete h_histout;
		delete h_mseout;
		delete h_predout;
		
		d_globalminmax->release(stream);
		d_histout->release(stream);
		d_mseout->release(stream);
		d_predout->release(stream);
		d_colids->release(stream);

		delete d_globalminmax;
		delete d_histout;
		delete d_mseout;
		delete d_predout;
		delete d_colids;

	}

};
