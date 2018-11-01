/*!
 * Modifications Copyright 2017-2018 H2O.ai, Inc.
 */
// original code from https://github.com/NVIDIA/kmeans (Apache V2.0 License)
#pragma once
#include <thrust/device_vector.h>
#include "cub/cub.cuh"
#include <iostream>
#include <sstream>
#include <cublas_v2.h>
#include <cfloat>
#include <unistd.h>
#include "kmeans_general.h"
#include <thrust/fill.h>
#include <thrust/gather.h>

inline void gpu_assert(cudaError_t code, const char *file, int line,
		bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
				line);
		std::stringstream ss;
		ss << file << "(" << line << ")";
		std::string file_and_line;
		ss >> file_and_line;
		thrust::system_error(code, thrust::cuda_category(), file_and_line);
	}
}

inline cudaError_t throw_on_cuda_error(cudaError_t code, const char *file,
		int line) {
	if (code != cudaSuccess) {
		std::stringstream ss;
		ss << file << "(" << line << ")";
		std::string file_and_line;
		ss >> file_and_line;
		thrust::system_error(code, thrust::cuda_category(), file_and_line);
	}

	return code;
}

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *cudaGetErrorEnum(cublasStatus_t error) {
	switch (error) {
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";

	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";

	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";

	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";

	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";

	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";

	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";

	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}
#endif

inline cublasStatus_t throw_on_cublas_error(cublasStatus_t code,
		const char *file, int line) {

	if (code != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "cublas error: %s %s %d\n", cudaGetErrorEnum(code),
				file, line);
		std::stringstream ss;
		ss << file << "(" << line << ")";
		std::string file_and_line;
		ss >> file_and_line;
		thrust::system_error(code, thrust::cuda_category(), file_and_line);
	}

	return code;
}

extern cudaStream_t cuda_stream[MAX_NGPUS];

template<unsigned int i>
extern __global__ void debugMark() {
}
;

namespace kmeans {
namespace detail {

void labels_init();
void labels_close();

extern cublasHandle_t cublas_handle[MAX_NGPUS];
template<typename T>
void memcpy(thrust::host_vector<T, std::allocator<T> > &H,
		thrust::device_vector<T, thrust::device_malloc_allocator<T> > &D) {
	int dev_num;
	safe_cuda(cudaGetDevice(&dev_num));
	safe_cuda(
			cudaMemcpyAsync(thrust::raw_pointer_cast(H.data()),
					thrust::raw_pointer_cast(D.data()), sizeof(T) * D.size(),
					cudaMemcpyDeviceToHost, cuda_stream[dev_num]));
}

template<typename T>
void memcpy(thrust::device_vector<T, thrust::device_malloc_allocator<T> > &D,
		thrust::host_vector<T, std::allocator<T> > &H) {
	int dev_num;
	safe_cuda(cudaGetDevice(&dev_num));
	safe_cuda(
			cudaMemcpyAsync(thrust::raw_pointer_cast(D.data()),
					thrust::raw_pointer_cast(H.data()), sizeof(T) * H.size(),
					cudaMemcpyHostToDevice, cuda_stream[dev_num]));
}
template<typename T>
void memcpy(thrust::device_vector<T, thrust::device_malloc_allocator<T> > &Do,
		thrust::device_vector<T, thrust::device_malloc_allocator<T> > &Di) {
	int dev_num;
	safe_cuda(cudaGetDevice(&dev_num));
	safe_cuda(
			cudaMemcpyAsync(thrust::raw_pointer_cast(Do.data()),
					thrust::raw_pointer_cast(Di.data()), sizeof(T) * Di.size(),
					cudaMemcpyDeviceToDevice, cuda_stream[dev_num]));
}
template<typename T>
void memzero(thrust::device_vector<T, thrust::device_malloc_allocator<T> >& D) {
	int dev_num;
	safe_cuda(cudaGetDevice(&dev_num));
	safe_cuda(
			cudaMemsetAsync(thrust::raw_pointer_cast(D.data()), 0,
					sizeof(T) * D.size(), cuda_stream[dev_num]));
}
void streamsync(int dev_num);

//n: number of points
//d: dimensionality of points
//data: points, laid out in row-major order (n rows, d cols)
//dots: result vector (n rows)
// NOTE:
//Memory accesses in this function are uncoalesced!!
//This is because data is in row major order
//However, in k-means, it's called outside the optimization loop
//on the large data array, and inside the optimization loop it's
//called only on a small array, so it doesn't really matter.
//If this becomes a performance limiter, transpose the data somewhere
template<typename T>
__global__ void self_dots(int n, int d, T* data, T* dots) {
	T accumulator = 0;
	int global_id = blockDim.x * blockIdx.x + threadIdx.x;

	if (global_id < n) {
		for (int i = 0; i < d; i++) {
			T value = data[i + global_id * d];
			accumulator += value * value;
		}
		dots[global_id] = accumulator;
	}
}

template<typename T>
void make_self_dots(int n, int d, thrust::device_vector<T>& data,
		thrust::device_vector<T>& dots) {
	int dev_num;
#define MAX_BLOCK_THREADS0 256
	const int GRID_SIZE = (n - 1) / MAX_BLOCK_THREADS0 + 1;
	safe_cuda(cudaGetDevice(&dev_num));
	self_dots<<<GRID_SIZE, MAX_BLOCK_THREADS0, 0, cuda_stream[dev_num]>>>(n, d,
			thrust::raw_pointer_cast(data.data()),
			thrust::raw_pointer_cast(dots.data()));
#if(CHECK)
	gpuErrchk(cudaGetLastError());
#endif

}

#define MAX_BLOCK_THREADS 32
template<typename T>
__global__ void all_dots(int n, int k, T* data_dots, T* centroid_dots,
		T* dots) {
	__shared__ T local_data_dots[MAX_BLOCK_THREADS];
	__shared__ T local_centroid_dots[MAX_BLOCK_THREADS];
	//        if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0) printf("inside %d %d %d\n",threadIdx.x,blockIdx.x,blockDim.x);

	int data_index = threadIdx.x + blockIdx.x * blockDim.x;
	if ((data_index < n) && (threadIdx.y == 0)) {
		local_data_dots[threadIdx.x] = data_dots[data_index];
	}

	int centroid_index = threadIdx.x + blockIdx.y * blockDim.y;
	if ((centroid_index < k) && (threadIdx.y == 1)) {
		local_centroid_dots[threadIdx.x] = centroid_dots[centroid_index];
	}

	__syncthreads();

	centroid_index = threadIdx.y + blockIdx.y * blockDim.y;
	//        printf("data_index=%d centroid_index=%d\n",data_index,centroid_index);
	if ((data_index < n) && (centroid_index < k)) {
		dots[data_index + centroid_index * n] = local_data_dots[threadIdx.x]
				+ local_centroid_dots[threadIdx.y];
	}
}

template<typename T>
void make_all_dots(int n, int k, size_t offset,
		thrust::device_vector<T>& data_dots,
		thrust::device_vector<T>& centroid_dots,
		thrust::device_vector<T>& dots) {
	int dev_num;
	safe_cuda(cudaGetDevice(&dev_num));
	const int BLOCK_THREADSX = MAX_BLOCK_THREADS; // BLOCK_THREADSX*BLOCK_THREADSY<=1024 on modern arch's (sm_61)
	const int BLOCK_THREADSY = MAX_BLOCK_THREADS;
	const int GRID_SIZEX = (n - 1) / BLOCK_THREADSX + 1; // on old arch's this has to be less than 2^16=65536
	const int GRID_SIZEY = (k - 1) / BLOCK_THREADSY + 1; // this has to be less than 2^16=65536
	//        printf("pre all_dots: %d %d %d %d\n",GRID_SIZEX,GRID_SIZEY,BLOCK_THREADSX,BLOCK_THREADSY); fflush(stdout);
	all_dots<<<dim3(GRID_SIZEX, GRID_SIZEY),
			dim3(BLOCK_THREADSX, BLOCK_THREADSY), 0, cuda_stream[dev_num]>>>(n,
			k, thrust::raw_pointer_cast(data_dots.data() + offset),
			thrust::raw_pointer_cast(centroid_dots.data()),
			thrust::raw_pointer_cast(dots.data()));
#if(CHECK)
	gpuErrchk(cudaGetLastError());
#endif
}
;

#define WARP_SIZE 32
#define BLOCK_SIZE 1024
#define BSIZE_DIV_WSIZE (BLOCK_SIZE/WARP_SIZE)
#define IDX(i,j,lda) ((i)+(j)*(lda))
template<typename T>
__global__ void rejectByCosines(int k, int *accept, T *dists,
		T *centroid_dots) {

	// Global indices
	int gidx, gidy;

	// Lengths from cosine_dots
	float lenA, lenB;
	// Threshold
	float thresh = 0.9;

	// Observation vector is determined by global y-index
	gidy = threadIdx.y + blockIdx.y * blockDim.y;
	while (gidy < k) {
		// Get lengths from global memory, stored in centroid_dots
		lenA = centroid_dots[gidy];

		gidx = threadIdx.x + blockIdx.x * blockDim.x;
		while (gidx < gidy) {
			lenB = centroid_dots[gidx];
			if (lenA > 1e-8 && lenB > 1e-8)
				dists[IDX(gidx, gidy, k)] /= lenA * lenB;
			if (dists[IDX(gidx, gidy, k)] > thresh
					&& ((lenA < 2.0 * lenB) && (lenB < 2.0 * lenA)))
				accept[gidy] = 0;
			gidx += blockDim.x * gridDim.x;
		}
		// Move to another centroid
		gidy += blockDim.y * gridDim.y;
	}

}

template<typename T>
void checkCosine(int d, int k, int *numChosen, thrust::device_vector<T> &dists,
		thrust::device_vector<T> &centroids,
		thrust::device_vector<T> &centroid_dots) {

	dim3 blockDim, gridDim;

	int h_accept[k];

	thrust::device_vector<int> accept(k);
	thrust::fill(accept.begin(), accept.begin() + k, 1);
	//printf("after fill accept\n");
	// Divide dists by centroid lengths to get cosine matrix
	blockDim.x = WARP_SIZE;
	blockDim.y = BLOCK_SIZE / WARP_SIZE;
	blockDim.z = 1;
	gridDim.x = min((k + WARP_SIZE - 1) / WARP_SIZE, 65535);
	gridDim.y = min((k + BSIZE_DIV_WSIZE - 1) / BSIZE_DIV_WSIZE, 65535);
	gridDim.z = 1;

	rejectByCosines<<<gridDim, blockDim>>>(k,
			thrust::raw_pointer_cast(accept.data()),
			thrust::raw_pointer_cast(dists.data()),
			thrust::raw_pointer_cast(centroid_dots.data()));

#if(CHECK)
	gpuErrchk(cudaGetLastError());
#endif

	*numChosen = thrust::reduce(accept.begin(), accept.begin() + k);

	CUDACHECK(
			cudaMemcpy(h_accept, thrust::raw_pointer_cast(accept.data()),
					k * sizeof(int), cudaMemcpyDeviceToHost));

	int skipcopy = 1;
	for (int z = 0; z < *numChosen; ++z) {
		if (h_accept[z] == 0)
			skipcopy = 0;
	}

	if (!skipcopy && (*numChosen > 1 && *numChosen < k)) {
		int i, j;
		int candidate_map[d * (*numChosen)];
		j = 0;
		for (i = 0; i < k; ++i) {
			if (h_accept[i]) {
				for (int m = 0; m < d; ++m)
					candidate_map[j * d + m] = i * d + m;
				j += 1;
			}
		}

		thrust::device_vector<int> d_candidate_map(d * (*numChosen));
		CUDACHECK(cudaMemcpy(thrust::raw_pointer_cast(d_candidate_map.data()), candidate_map, d*(*numChosen)*sizeof(int), cudaMemcpyHostToDevice))
		;

		thrust::device_vector<T> cent_copy(dists);

		thrust::copy_n(centroids.begin(), d * k, cent_copy.begin());

#if(CHECK)
		gpuErrchk(cudaGetLastError());
#endif

		// Gather accepted centroid candidates into centroid memory
		thrust::gather(d_candidate_map.begin(),
				d_candidate_map.begin() + d * (*numChosen), cent_copy.begin(),
				centroids.begin());
	}
}

template<typename T>
void calculate_distances(int verbose, int q, size_t n, int d, int k,
		thrust::device_vector<T>& data, size_t data_offset,
		thrust::device_vector<T>& centroids,
		thrust::device_vector<T>& data_dots,
		thrust::device_vector<T>& centroid_dots,
		thrust::device_vector<T>& pairwise_distances);

template<typename T, typename F>
void batch_calculate_distances(int verbose, int q, size_t n, int d, int k,
		thrust::device_vector<T> &data, thrust::device_vector<T> &centroids,
		thrust::device_vector<T> &data_dots,
		thrust::device_vector<T> &centroid_dots, F functor) {
	int fudges_size = 4;
	double fudges[] = { 1.0, 0.75, 0.5, 0.25 };
	for (const double fudge : fudges) {
		try {
			// Get info about available memory
			// This part of the algo can be very memory consuming
			// We might need to batch it
			size_t free_byte;
			size_t total_byte;
			CUDACHECK(cudaMemGetInfo(&free_byte, &total_byte));
			free_byte = free_byte * fudge;

			size_t required_byte = n * k * sizeof(T);

			size_t runs = std::ceil(required_byte / (double) free_byte);

			log_verbose(verbose,
					"Batch calculate distance - Rows %ld | K %ld | Data size %d",
					n, k, sizeof(T));

			log_verbose(verbose,
					"Batch calculate distance - Free memory %zu | Required memory %zu | Runs %d",
					free_byte, required_byte, runs);

			size_t offset = 0;
			size_t rows_per_run = n / runs;
			thrust::device_vector<T> pairwise_distances(rows_per_run * k);

			for (int run = 0; run < runs; run++) {
				if (run + 1 == runs && n % rows_per_run != 0) {
					rows_per_run = n % rows_per_run;
				}

				thrust::fill_n(pairwise_distances.begin(),
						pairwise_distances.size(), (T) 0.0);

				log_verbose(verbose, "Batch calculate distance - Allocated");

				kmeans::detail::calculate_distances(verbose, 0, rows_per_run, d,
						k, data, offset, centroids, data_dots, centroid_dots,
						pairwise_distances);

				log_verbose(verbose,
						"Batch calculate distance - Distances calculated");

				functor(rows_per_run, offset, pairwise_distances);

				log_verbose(verbose, "Batch calculate distance - Functor ran");

				offset += rows_per_run;
			}
		} catch (const std::bad_alloc& e) {
			cudaGetLastError();
			if (fudges[fudges_size - 1] != fudge) {
				log_warn(verbose,
						"Batch calculate distance - Failed to allocate memory for pairwise distances - retrying.");
				continue;
			} else {
				log_error(verbose,
						"Batch calculate distance - Failed to allocate memory for pairwise distances - exiting.");
				throw e;
			}
		}

		return;
	}
}

template<typename T>
__global__ void make_new_labels(int n, int k, T* pairwise_distances,
		int* labels) {
	T min_distance = FLT_MAX; //std::numeric_limits<T>::max(); // might be ok TODO FIXME
	T min_idx = -1;
	int global_id = threadIdx.x + blockIdx.x * blockDim.x;
	if (global_id < n) {
		for (int c = 0; c < k; c++) {
			T distance = pairwise_distances[c * n + global_id];
			if (distance < min_distance) {
				min_distance = distance;
				min_idx = c;
			}
		}
		labels[global_id] = min_idx;
	}
}

template<typename T>
void relabel(int n, int k, thrust::device_vector<T>& pairwise_distances,
		thrust::device_vector<int>& labels, size_t offset) {
	int dev_num;
	safe_cuda(cudaGetDevice(&dev_num));
#define MAX_BLOCK_THREADS2 256
	const int GRID_SIZE = (n - 1) / MAX_BLOCK_THREADS2 + 1;
	make_new_labels<<<GRID_SIZE, MAX_BLOCK_THREADS2, 0, cuda_stream[dev_num]>>>(
			n, k, thrust::raw_pointer_cast(pairwise_distances.data()),
			thrust::raw_pointer_cast(labels.data() + offset));
#if(CHECK)
	gpuErrchk(cudaGetLastError());
#endif
}

}
}
namespace mycub {

extern void *d_key_alt_buf[MAX_NGPUS];
extern unsigned int key_alt_buf_bytes[MAX_NGPUS];
extern void *d_value_alt_buf[MAX_NGPUS];
extern unsigned int value_alt_buf_bytes[MAX_NGPUS];
extern void *d_temp_storage[MAX_NGPUS];
extern size_t temp_storage_bytes[MAX_NGPUS];
extern void *d_temp_storage2[MAX_NGPUS];
extern size_t temp_storage_bytes2[MAX_NGPUS];
extern bool cub_initted;

void sort_by_key_int(thrust::device_vector<int>& keys,
		thrust::device_vector<int>& values);

template<typename T, typename U>
void sort_by_key(thrust::device_vector<T>& keys,
		thrust::device_vector<U>& values) {
	int dev_num;
	safe_cuda(cudaGetDevice(&dev_num));
	cudaStream_t this_stream = cuda_stream[dev_num];
	int SIZE = keys.size();
	if (key_alt_buf_bytes[dev_num] < sizeof(T) * SIZE) {
		if (d_key_alt_buf[dev_num])
			safe_cuda(cudaFree(d_key_alt_buf[dev_num]));
		safe_cuda(cudaMalloc(&d_key_alt_buf[dev_num], sizeof(T) * SIZE));
		key_alt_buf_bytes[dev_num] = sizeof(T) * SIZE;
		std::cout << "Malloc key_alt_buf" << std::endl;
	}
	if (value_alt_buf_bytes[dev_num] < sizeof(U) * SIZE) {
		if (d_value_alt_buf[dev_num])
			safe_cuda(cudaFree(d_value_alt_buf[dev_num]));
		safe_cuda(cudaMalloc(&d_value_alt_buf[dev_num], sizeof(U) * SIZE));
		value_alt_buf_bytes[dev_num] = sizeof(U) * SIZE;
		std::cout << "Malloc value_alt_buf" << std::endl;
	}
	cub::DoubleBuffer<T> d_keys(thrust::raw_pointer_cast(keys.data()),
			(T*) d_key_alt_buf[dev_num]);
	cub::DoubleBuffer<U> d_values(thrust::raw_pointer_cast(values.data()),
			(U*) d_value_alt_buf[dev_num]);
	cudaError_t err;

	// Determine temporary device storage requirements for sorting operation
	//if (temp_storage_bytes[dev_num] == 0) {
	void *d_temp;
	size_t temp_bytes;
	err = cub::DeviceRadixSort::SortPairs(d_temp_storage[dev_num], temp_bytes,
			d_keys, d_values, SIZE, 0, sizeof(T) * 8, this_stream);
	// Allocate temporary storage for sorting operation
	safe_cuda(cudaMalloc(&d_temp, temp_bytes));
	d_temp_storage[dev_num] = d_temp;
	temp_storage_bytes[dev_num] = temp_bytes;
	std::cout << "Malloc temp_storage. " << temp_storage_bytes[dev_num]
			<< " bytes" << std::endl;
	std::cout << "d_temp_storage[" << dev_num << "] = "
			<< d_temp_storage[dev_num] << std::endl;
	if (err) {
		std::cout << "Error " << err << " in SortPairs 1" << std::endl;
		std::cout << cudaGetErrorString(err) << std::endl;
	}
	//}
	// Run sorting operation
	err = cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys, d_values,
			SIZE, 0, sizeof(T) * 8, this_stream);
	if (err)
		std::cout << "Error in SortPairs 2" << std::endl;
	//cub::DeviceRadixSort::SortPairs(d_temp_storage[dev_num], temp_storage_bytes[dev_num], d_keys,
	//                                d_values, SIZE, 0, sizeof(T)*8, this_stream);

}
template<typename T>
void sum_reduce(thrust::device_vector<T>& values, T* sum) {
	int dev_num;
	safe_cuda(cudaGetDevice(&dev_num));
	if (!d_temp_storage2[dev_num]) {
		cub::DeviceReduce::Sum(d_temp_storage2[dev_num],
				temp_storage_bytes2[dev_num],
				thrust::raw_pointer_cast(values.data()), sum, values.size(),
				cuda_stream[dev_num]);
		// Allocate temporary storage for sorting operation
		safe_cuda(
				cudaMalloc(&d_temp_storage2[dev_num],
						temp_storage_bytes2[dev_num]));
	}
	cub::DeviceReduce::Sum(d_temp_storage2[dev_num],
			temp_storage_bytes2[dev_num],
			thrust::raw_pointer_cast(values.data()), sum, values.size(),
			cuda_stream[dev_num]);
}
void cub_init();
void cub_close();

void cub_init(int dev);
void cub_close(int dev);
}

namespace kmeans {
namespace detail {

template<typename T>
struct absolute_value {
	__host__ __device__
	void operator()(T &x) const {
		x = (x > 0 ? x : -x);
	}
};

cublasHandle_t cublas_handle[MAX_NGPUS];

void labels_init() {
	cublasStatus_t stat;
	cudaError_t err;
	int dev_num;
	safe_cuda(cudaGetDevice(&dev_num));
	stat = cublasCreate(&detail::cublas_handle[dev_num]);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		std::cout << "CUBLAS initialization failed" << std::endl;
		exit(1);
	}
	err = safe_cuda(cudaStreamCreate(&cuda_stream[dev_num]))
	;
	if (err != cudaSuccess) {
		std::cout << "Stream creation failed" << std::endl;

	}
	cublasSetStream(cublas_handle[dev_num], cuda_stream[dev_num]);
	mycub::cub_init(dev_num);
}

void labels_close() {
	int dev_num;
	safe_cuda(cudaGetDevice(&dev_num));
	safe_cublas(cublasDestroy(cublas_handle[dev_num]));
	safe_cuda(cudaStreamDestroy(cuda_stream[dev_num]));
	mycub::cub_close(dev_num);
}

void streamsync(int dev_num) {
	cudaStreamSynchronize(cuda_stream[dev_num]);
}

/**
 * Matrix multiplication: alpha * A^T * B + beta * C
 * Optimized for tall and skinny matrices
 *
 * @tparam float_t
 * @param A
 * @param B
 * @param C
 * @param alpha
 * @param beta
 * @param n
 * @param d
 * @param k
 * @param max_block_rows
 * @return
 */
template<typename float_t>
__global__ void matmul(const float_t *A, const float_t *B, float_t *C,
		const float_t alpha, const float_t beta, int n, int d, int k,
		int max_block_rows) {

	extern __shared__ __align__(sizeof(float_t)) unsigned char my_smem[];
	float_t *shared = reinterpret_cast<float_t *>(my_smem);

	float_t *s_A = shared;
	float_t *s_B = shared + max_block_rows * d;

	for (int i = threadIdx.x; i < d * k; i += blockDim.x) {
		s_B[i] = B[i];
	}

	size_t block_start_row_index = blockIdx.x * max_block_rows;
	size_t block_rows = max_block_rows;

	if (blockIdx.x == gridDim.x - 1 && n % max_block_rows != 0) {
		block_rows = n % max_block_rows;
	}

	for (size_t i = threadIdx.x; i < d * block_rows; i += blockDim.x) {
		s_A[i] = alpha * A[d * block_start_row_index + i];
	}

	__syncthreads();

	float_t elem_c = 0;

	int col_c = threadIdx.x % k;
	size_t abs_row_c = block_start_row_index + threadIdx.x / k;
	int row_c = threadIdx.x / k;

	// Thread/Block combination either too far for data array
	// Or is calculating for index that should be calculated in a different blocks - in some edge cases
	// "col_c * n + abs_row_c" can yield same result in different thread/block combinations
	if (abs_row_c >= n || threadIdx.x >= block_rows * k) {
		return;
	}

	for (size_t i = 0; i < d; i++) {
		elem_c += s_B[d * col_c + i] * s_A[d * row_c + i];
	}

	C[col_c * n + abs_row_c] = beta * C[col_c * n + abs_row_c] + elem_c;

}

template<>
void calculate_distances<double>(int verbose, int q, size_t n, int d, int k,
		thrust::device_vector<double> &data, size_t data_offset,
		thrust::device_vector<double> &centroids,
		thrust::device_vector<double> &data_dots,
		thrust::device_vector<double> &centroid_dots,
		thrust::device_vector<double> &pairwise_distances) {
	detail::make_self_dots(k, d, centroids, centroid_dots);
	detail::make_all_dots(n, k, data_offset, data_dots, centroid_dots,
			pairwise_distances);

	//||x-y||^2 = ||x||^2 + ||y||^2 - 2 x . y
	//pairwise_distances has ||x||^2 + ||y||^2, so beta = 1
	//The dgemm calculates x.y for all x and y, so alpha = -2.0
	double alpha = -2.0;
	double beta = 1.0;
	//If the data were in standard column major order, we'd do a
	//centroids * data ^ T
	//But the data is in row major order, so we have to permute
	//the arguments a little
	int dev_num;
	safe_cuda(cudaGetDevice(&dev_num));

	bool do_cublas = true;
	if (k <= 16 && d <= 64) {
		const int BLOCK_SIZE_MUL = 128;
		int block_rows = std::min((size_t) BLOCK_SIZE_MUL / k, n);
		int grid_size = std::ceil(static_cast<double>(n) / block_rows);

		int shared_size_B = d * k * sizeof(double);
		size_t shared_size_A = block_rows * d * sizeof(double);
		if (shared_size_B + shared_size_A < (1 << 15)) {

			matmul<<<grid_size, BLOCK_SIZE_MUL, shared_size_B + shared_size_A>>>(
					thrust::raw_pointer_cast(data.data() + data_offset * d),
					thrust::raw_pointer_cast(centroids.data()),
					thrust::raw_pointer_cast(pairwise_distances.data()), alpha,
					beta, n, d, k, block_rows);
			do_cublas = false;
		}
	}

	if (do_cublas) {
		cublasStatus_t stat =
				safe_cublas(
						cublasDgemm(detail::cublas_handle[dev_num], CUBLAS_OP_T, CUBLAS_OP_N, n, k, d, &alpha, thrust::raw_pointer_cast(data.data() + data_offset * d), d, //Has to be n or d
						thrust::raw_pointer_cast(centroids.data()), d,//Has to be k or d
						&beta, thrust::raw_pointer_cast(pairwise_distances.data()), n))
		; //Has to be n or k

		if (stat != CUBLAS_STATUS_SUCCESS) {
			std::cout << "Invalid Dgemm" << std::endl;
			exit(1);
		}
	}

	thrust::for_each(pairwise_distances.begin(), pairwise_distances.end(),
			absolute_value<double>()); // in-place transformation to ensure all distances are positive indefinite

#if(CHECK)
	gpuErrchk(cudaGetLastError());
#endif
}

template<>
void calculate_distances<float>(int verbose, int q, size_t n, int d, int k,
		thrust::device_vector<float> &data, size_t data_offset,
		thrust::device_vector<float> &centroids,
		thrust::device_vector<float> &data_dots,
		thrust::device_vector<float> &centroid_dots,
		thrust::device_vector<float> &pairwise_distances) {
	detail::make_self_dots(k, d, centroids, centroid_dots);
	detail::make_all_dots(n, k, data_offset, data_dots, centroid_dots,
			pairwise_distances);

	//||x-y||^2 = ||x||^2 + ||y||^2 - 2 x . y
	//pairwise_distances has ||x||^2 + ||y||^2, so beta = 1
	//The dgemm calculates x.y for all x and y, so alpha = -2.0
	float alpha = -2.0;
	float beta = 1.0;
	//If the data were in standard column major order, we'd do a
	//centroids * data ^ T
	//But the data is in row major order, so we have to permute
	//the arguments a little
	int dev_num;
	safe_cuda(cudaGetDevice(&dev_num));

	if (k <= 16 && d <= 64) {
		const int BLOCK_SIZE_MUL = 128;
		int block_rows = std::min((size_t) BLOCK_SIZE_MUL / k, n);
		int grid_size = std::ceil(static_cast<float>(n) / block_rows);

		int shared_size_B = d * k * sizeof(float);
		int shared_size_A = block_rows * d * sizeof(float);

		matmul<<<grid_size, BLOCK_SIZE_MUL, shared_size_B + shared_size_A>>>(
				thrust::raw_pointer_cast(data.data() + data_offset * d),
				thrust::raw_pointer_cast(centroids.data()),
				thrust::raw_pointer_cast(pairwise_distances.data()), alpha,
				beta, n, d, k, block_rows);
	} else {
		cublasStatus_t stat =
				safe_cublas(
						cublasSgemm(detail::cublas_handle[dev_num], CUBLAS_OP_T, CUBLAS_OP_N, n, k, d, &alpha, thrust::raw_pointer_cast(data.data() + data_offset * d), d, //Has to be n or d
						thrust::raw_pointer_cast(centroids.data()), d,//Has to be k or d
						&beta, thrust::raw_pointer_cast(pairwise_distances.data()), n))
		; //Has to be n or k

		if (stat != CUBLAS_STATUS_SUCCESS) {
			std::cout << "Invalid Sgemm" << std::endl;
			exit(1);
		}
	}

	thrust::for_each(pairwise_distances.begin(), pairwise_distances.end(),
			absolute_value<float>()); // in-place transformation to ensure all distances are positive indefinite

#if(CHECK)
	gpuErrchk(cudaGetLastError());
#endif
}

}
}

namespace mycub {

void *d_key_alt_buf[MAX_NGPUS];
unsigned int key_alt_buf_bytes[MAX_NGPUS];
void *d_value_alt_buf[MAX_NGPUS];
unsigned int value_alt_buf_bytes[MAX_NGPUS];
void *d_temp_storage[MAX_NGPUS];
size_t temp_storage_bytes[MAX_NGPUS];
void *d_temp_storage2[MAX_NGPUS];
size_t temp_storage_bytes2[MAX_NGPUS];
bool cub_initted;
void cub_init() {
	// std::cout <<"CUB init" << std::endl;
	for (int q = 0; q < MAX_NGPUS; q++) {
		d_key_alt_buf[q] = NULL;
		key_alt_buf_bytes[q] = 0;
		d_value_alt_buf[q] = NULL;
		value_alt_buf_bytes[q] = 0;
		d_temp_storage[q] = NULL;
		temp_storage_bytes[q] = 0;
		d_temp_storage2[q] = NULL;
		temp_storage_bytes2[q] = 0;
	}
	cub_initted = true;
}

void cub_init(int dev) {
	d_key_alt_buf[dev] = NULL;
	key_alt_buf_bytes[dev] = 0;
	d_value_alt_buf[dev] = NULL;
	value_alt_buf_bytes[dev] = 0;
	d_temp_storage[dev] = NULL;
	temp_storage_bytes[dev] = 0;
	d_temp_storage2[dev] = NULL;
	temp_storage_bytes2[dev] = 0;
}

void cub_close() {
	for (int q = 0; q < MAX_NGPUS; q++) {
		if (d_key_alt_buf[q])
			safe_cuda(cudaFree(d_key_alt_buf[q]));
		if (d_value_alt_buf[q])
			safe_cuda(cudaFree(d_value_alt_buf[q]));
		if (d_temp_storage[q])
			safe_cuda(cudaFree(d_temp_storage[q]));
		if (d_temp_storage2[q])
			safe_cuda(cudaFree(d_temp_storage2[q]));
		d_temp_storage[q] = NULL;
		d_temp_storage2[q] = NULL;
	}
	cub_initted = false;
}

void cub_close(int dev) {
	if (d_key_alt_buf[dev])
		safe_cuda(cudaFree(d_key_alt_buf[dev]));
	if (d_value_alt_buf[dev])
		safe_cuda(cudaFree(d_value_alt_buf[dev]));
	if (d_temp_storage[dev])
		safe_cuda(cudaFree(d_temp_storage[dev]));
	if (d_temp_storage2[dev])
		safe_cuda(cudaFree(d_temp_storage2[dev]));
	d_temp_storage[dev] = NULL;
	d_temp_storage2[dev] = NULL;
}

void sort_by_key_int(thrust::device_vector<int> &keys,
		thrust::device_vector<int> &values) {
	int dev_num;
	safe_cuda(cudaGetDevice(&dev_num));
	cudaStream_t this_stream = cuda_stream[dev_num];
	int SIZE = keys.size();
	//int *d_key_alt_buf, *d_value_alt_buf;
	if (key_alt_buf_bytes[dev_num] < sizeof(int) * SIZE) {
		if (d_key_alt_buf[dev_num])
			safe_cuda(cudaFree(d_key_alt_buf[dev_num]));
		safe_cuda(cudaMalloc(&d_key_alt_buf[dev_num], sizeof(int) * SIZE));
		key_alt_buf_bytes[dev_num] = sizeof(int) * SIZE;
	}
	if (value_alt_buf_bytes[dev_num] < sizeof(int) * SIZE) {
		if (d_value_alt_buf[dev_num])
			safe_cuda(cudaFree(d_value_alt_buf[dev_num]));
		safe_cuda(cudaMalloc(&d_value_alt_buf[dev_num], sizeof(int) * SIZE));
		value_alt_buf_bytes[dev_num] = sizeof(int) * SIZE;
	}
	cub::DoubleBuffer<int> d_keys(thrust::raw_pointer_cast(keys.data()),
			(int *) d_key_alt_buf[dev_num]);
	cub::DoubleBuffer<int> d_values(thrust::raw_pointer_cast(values.data()),
			(int *) d_value_alt_buf[dev_num]);

	// Determine temporary device storage requirements for sorting operation
	if (!d_temp_storage[dev_num]) {
		cub::DeviceRadixSort::SortPairs(d_temp_storage[dev_num],
				temp_storage_bytes[dev_num], d_keys, d_values, SIZE, 0,
				sizeof(int) * 8, this_stream);
		// Allocate temporary storage for sorting operation
		safe_cuda(
				cudaMalloc(&d_temp_storage[dev_num],
						temp_storage_bytes[dev_num]));
	}
	// Run sorting operation
	cub::DeviceRadixSort::SortPairs(d_temp_storage[dev_num],
			temp_storage_bytes[dev_num], d_keys, d_values, SIZE, 0,
			sizeof(int) * 8, this_stream);
	// Sorted keys and values are referenced by d_keys.Current() and d_values.Current()

	keys.data() = thrust::device_pointer_cast(d_keys.Current());
	values.data() = thrust::device_pointer_cast(d_values.Current());
}

}
