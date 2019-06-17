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
#include "memory.h"
#include <fstream>

template<class T>
TemporaryMemory<T>::TemporaryMemory(const ML::cumlHandle_impl& handle, int N, int Ncols, int maxstr, int n_unique, int n_bins, const int split_algo):ml_handle(handle)
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

		sampledlabels = new MLCommon::device_buffer<int>(handle.getDeviceAllocator(), stream, N);
		totalmem += N*sizeof(int);

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

		d_globalminmax = new MLCommon::device_buffer<T>(handle.getDeviceAllocator(), stream, Ncols * 2);
		d_histout = new MLCommon::device_buffer<int>(handle.getDeviceAllocator(), stream, n_hist_elements * Ncols);
		d_colids = new MLCommon::device_buffer<int>(handle.getDeviceAllocator(), stream, Ncols);
		totalmem += (n_hist_elements * sizeof(int) + sizeof(int) + 2*sizeof(T))* Ncols;

	}

template<class T>
void TemporaryMemory<T>::print_info()
	{
		std::cout <<" Inside the print_info function  \n" << std::flush;
		std::cout << " Total temporary memory usage--> "<< ((double)totalmem/ (1024*1024)) << "  MB" << std::endl;
		return;
	}

template<class T>
TemporaryMemory<T>::~TemporaryMemory()
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

		delete sampledlabels;
		delete d_split_temp_storage;
		delete d_num_selected_out;
		delete d_flags_left;
		delete d_flags_right;
		delete temprowids;
		delete question_value;
		delete h_histout;

		d_globalminmax->release(stream);
		d_histout->release(stream);
		d_colids->release(stream);

		delete d_globalminmax;
		delete d_histout;
		delete d_colids;

	}

