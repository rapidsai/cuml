/*!
 * Modifications Copyright 2017-2018 H2O.ai, Inc.
 */
// original code from https://github.com/NVIDIA/kmeans (Apache V2.0 License)
#pragma once
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "kmeans_labels.h"

inline __device__ double my_atomic_add(double *address, double val) {
	unsigned long long int *address_as_ull = (unsigned long long int *) address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

inline __device__ float my_atomic_add(float *address, float val) {
	return (atomicAdd(address, val));
}

namespace kmeans {
namespace detail {

template<typename T>
__device__ __forceinline__
void update_centroid(int label, int dimension, int d, T accumulator,
		T *centroids, int count, int *counts) {
	int index = label * d + dimension;
	T *target = centroids + index;
	my_atomic_add(target, accumulator);
	if (dimension == 0) {
		atomicAdd(counts + label, count);
	}
}

template<typename T>
__global__ void calculate_centroids(int n, int d, int k, T *data,
		int *ordered_labels, int *ordered_indices, T *centroids, int *counts) {
	int in_flight = blockDim.y * gridDim.y;
	int labels_per_row = (n - 1) / in_flight + 1;
	for (int dimension = threadIdx.x; dimension < d; dimension += blockDim.x) {
		T accumulator = 0;
		int count = 0;
		int global_id = threadIdx.y + blockIdx.y * blockDim.y;
		int start = global_id * labels_per_row;
		int end = (global_id + 1) * labels_per_row;
		end = (end > n) ? n : end;
		int prior_label;
		if (start < n) {
			prior_label = ordered_labels[start];

			for (int label_number = start; label_number < end; label_number++) {
				int label = ordered_labels[label_number];
				if (label != prior_label) {
					update_centroid(prior_label, dimension, d, accumulator,
							centroids, count, counts);
					accumulator = 0;
					count = 0;
				}

				T value = data[dimension + ordered_indices[label_number] * d];
				accumulator += value;
				prior_label = label;
				count++;
			}
			update_centroid(prior_label, dimension, d, accumulator, centroids,
					count, counts);
		}
	}
}

template<typename T>
__global__ void revert_zeroed_centroids(int d, int k, T *tmp_centroids,
		T *centroids, int *counts) {
	int global_id_x = threadIdx.x + blockIdx.x * blockDim.x;
	int global_id_y = threadIdx.y + blockIdx.y * blockDim.y;
	if ((global_id_x < d) && (global_id_y < k)) {
		if (counts[global_id_y] < 1) {
			centroids[global_id_x + d * global_id_y] = tmp_centroids[global_id_x
					+ d * global_id_y];
		}
	}
}

template<typename T>
__global__ void scale_centroids(int d, int k, int *counts, T *centroids) {
	int global_id_x = threadIdx.x + blockIdx.x * blockDim.x;
	int global_id_y = threadIdx.y + blockIdx.y * blockDim.y;
	if ((global_id_x < d) && (global_id_y < k)) {
		int count = counts[global_id_y];
		//To avoid introducing divide by zero errors
		//If a centroid has no weight, we'll do no normalization
		//This will keep its coordinates defined.
		if (count < 1) {
			count = 1;
		}
		T scale = 1.0 / T(count);
		centroids[global_id_x + d * global_id_y] *= scale;
	}
}

// Scale - should be true when running on a single GPU
template<typename T>
void find_centroids(int q, int n, int d, int k, int k_max,
		thrust::device_vector<T> &data, thrust::device_vector<int> &labels,
		thrust::device_vector<T> &centroids, thrust::device_vector<int> &range,
		thrust::device_vector<int> &indices, thrust::device_vector<int> &counts,
		bool scale) {
	int dev_num;
	cudaGetDevice(&dev_num);

	// If no scaling then this step will be handled on the host
	// when aggregating centroids from all GPUs
	// Cache original centroids in case some centroids are not present in labels
	// and would get zeroed
	thrust::device_vector<T> tmp_centroids;
	if (scale) {
		tmp_centroids = thrust::device_vector<T>(k_max * d);
		memcpy(tmp_centroids, centroids);
	}

	memcpy(indices, range);
	// TODO the rest of the algorithm doesn't necessarily require labels/data to be sorted
	// but *might* make if faster due to less atomic updates
	thrust::sort_by_key(labels.begin(), labels.end(), indices.begin());
	// TODO cub is faster but sort_by_key_int isn't sorting, possibly a bug
//    mycub::sort_by_key_int(labels, indices);

#if(CHECK)
	gpuErrchk(cudaGetLastError());
#endif

	// Need to zero this - the algo uses this array to accumulate values for each centroid
	// which are then averaged to get a new centroid
	memzero(centroids);
	memzero(counts);

	//Calculate centroids
	int n_threads_x = 64; // TODO FIXME
	int n_threads_y = 16; // TODO FIXME
	//XXX Number of blocks here is hard coded at 30
	//This should be taken care of more thoughtfully.
	calculate_centroids<<<dim3(1, 30), dim3(n_threads_x, n_threads_y), // TODO FIXME
	0, cuda_stream[dev_num]>>>(n, d, k, thrust::raw_pointer_cast(data.data()),
			thrust::raw_pointer_cast(labels.data()),
			thrust::raw_pointer_cast(indices.data()),
			thrust::raw_pointer_cast(centroids.data()),
			thrust::raw_pointer_cast(counts.data()));

#if(CHECK)
	gpuErrchk(cudaGetLastError());
#endif

	// Scaling should take place on the GPU if n_gpus=1 so we don't
	// move centroids and counts between host and device all the time for nothing
	if (scale) {
		// Revert only if `scale`, otherwise this will be taken care of in the host
		// Revert reverts centroids for which count is equal 0
		revert_zeroed_centroids<<<dim3((d - 1) / 32 + 1, (k - 1) / 32 + 1),
				dim3(32, 32), // TODO FIXME
				0, cuda_stream[dev_num]>>>(d, k,
				thrust::raw_pointer_cast(tmp_centroids.data()),
				thrust::raw_pointer_cast(centroids.data()),
				thrust::raw_pointer_cast(counts.data()));

#if(CHECK)
		gpuErrchk(cudaGetLastError());
#endif

		//Averages the centroids
		scale_centroids<<<dim3((d - 1) / 32 + 1, (k - 1) / 32 + 1),
				dim3(32, 32), // TODO FIXME
				0, cuda_stream[dev_num]>>>(d, k,
				thrust::raw_pointer_cast(counts.data()),
				thrust::raw_pointer_cast(centroids.data()));
#if(CHECK)
		gpuErrchk(cudaGetLastError());
#endif
	}
}

}
}
