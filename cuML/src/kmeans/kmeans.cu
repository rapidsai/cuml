/*!
 * Copyright 2017-2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <iostream>
#include "cuda.h"
#include <cstdlib>
#include <unistd.h>
#include "kmeans_c.h"
#include "kmeans_impl.h"
#include "kmeans_general.h"
#include "kmeans.h"
#include <random>
#include <algorithm>
#include <vector>
#include <set>
#include <csignal>
#include "utils.h"
#include <math.h>

cudaStream_t cuda_stream[MAX_NGPUS];

/**
 * METHODS FOR DATA COPYING AND GENERATION
 */

template<typename T>
void random_data(int verbose, thrust::device_vector<T> &array, int m, int n) {
	thrust::host_vector<T> host_array(m * n);
	for (int i = 0; i < m * n; i++) {
		host_array[i] = (T) rand() / (T) RAND_MAX;
	}
	array = host_array;
}

/**
 * Copies data from srcdata to array
 * @tparam T
 * @param verbose Logging level
 * @param ord Column on row order of data
 * @param array Destination array
 * @param srcdata Source data
 * @param q Shard number (from 0 to n_gpu)
 * @param n
 * @param npergpu
 * @param d
 */
template<typename T>
void copy_data(int verbose, const char ord, thrust::device_vector<T> &array,
		const T *srcdata, int q, int n, size_t npergpu, int d) {
	if (ord == 'c') {
		thrust::host_vector<T> host_array(npergpu * d);
		log_debug(verbose, "Copy data COL ORDER -> ROW ORDER");

		for (size_t i = 0; i < npergpu * d; i++) {
			size_t indexi = i % d; // col
			size_t indexj = i / d + q * npergpu; // row (shifted by which gpu)
			host_array[i] = srcdata[indexi * n + indexj];
		}
		array = host_array;
	} else {
		log_debug(verbose, "Copy data ROW ORDER not changed");
		thrust::host_vector<T> host_array(srcdata + q * npergpu * d,
				srcdata + q * npergpu * d + npergpu * d);
		array = host_array;
	}
}

/**
 * Like copy_data but shuffles the data according to mapping from v
 * @tparam T
 * @param verbose
 * @param v
 * @param ord
 * @param array
 * @param srcdata
 * @param q
 * @param n
 * @param npergpu
 * @param d
 */
template<typename T>
void copy_data_shuffled(int verbose, std::vector<int> v, const char ord,
		thrust::device_vector<T> &array, const T *srcdata, int q, int n,
		int npergpu, int d) {
	thrust::host_vector<T> host_array(npergpu * d);
	if (ord == 'c') {
		log_debug(verbose, "Copy data shuffle COL ORDER -> ROW ORDER");

		for (int i = 0; i < npergpu; i++) {
			for (size_t j = 0; j < d; j++) {
				host_array[i * d + j] = srcdata[v[q * npergpu + i] + j * n]; // shift by which gpu
			}
		}
	} else {
		log_debug(verbose, "Copy data shuffle ROW ORDER not changed");

		for (int i = 0; i < npergpu; i++) {
			for (size_t j = 0; j < d; j++) {
				host_array[i * d + j] = srcdata[v[q * npergpu + i] * d + j]; // shift by which gpu
			}
		}
	}
	array = host_array;
}

template<typename T>
void copy_centroids_shuffled(int verbose, std::vector<int> v, const char ord,
		thrust::device_vector<T> &array, const T *srcdata, int n, int k,
		int d) {
	copy_data_shuffled(verbose, v, ord, array, srcdata, 0, n, k, d);
}

/**
 * Copies centroids from initial training set randomly.
 * @tparam T
 * @param verbose
 * @param seed
 * @param ord
 * @param array
 * @param srcdata
 * @param q
 * @param n
 * @param npergpu
 * @param d
 * @param k
 */
template<typename T>
void random_centroids(int verbose, int seed, const char ord,
		thrust::device_vector<T> &array, const T *srcdata, int q, int n,
		int npergpu, int d, int k) {
	thrust::host_vector<T> host_array(k * d);
	if (seed < 0) {
		std::random_device rd; //Will be used to obtain a seed for the random number engine
		seed = rd();
	}
	std::mt19937 gen(seed);
	std::uniform_int_distribution<> dis(0, n - 1); // random i in range from 0..n-1 (i.e. only 1 gpu gets centroids)

	if (ord == 'c') {
		log_debug(verbose, "Random centroids COL ORDER -> ROW ORDER");
		for (int i = 0; i < k; i++) { // clusters
			size_t reali = dis(gen); // + q*npergpu; // row sampled (called indexj above)
			for (size_t j = 0; j < d; j++) { // cols
				host_array[i * d + j] = srcdata[reali + j * n];
			}
		}
	} else {
		log_debug(verbose, "Random centroids ROW ORDER not changed");
		for (int i = 0; i < k; i++) { // rows
			size_t reali = dis(gen); // + q*npergpu ; // row sampled
			for (size_t j = 0; j < d; j++) { // cols
				host_array[i * d + j] = srcdata[reali * d + j];
			}
		}
	}
	array = host_array;
}

/**
 * KMEANS METHODS FIT, PREDICT, TRANSFORM
 */

#define __HBAR__                                                        \
  "----------------------------------------------------------------------------\n"

namespace h2o4gpukmeans {

template<typename T>
int kmeans_find_clusters(int verbose, const char ord, int seed,
		thrust::device_vector<T> **data, thrust::device_vector<int> **labels,
		thrust::device_vector<T> **d_centroids,
		thrust::device_vector<T> **data_dots, size_t rows, size_t cols,
		int init_from_data, int k, int k_max, T threshold, const T *srcdata,
		int n_gpu, std::vector<int> dList, T &residual, std::vector<int> v,
		int max_iterations);

template<typename T>
int kmeans_fit(int verbose, int seed, int gpu_idtry, int n_gputry, size_t rows,
		size_t cols, const char ord, int k, int k_max, int max_iterations,
		int init_from_data, T threshold, const T *srcdata, T **pred_centroids,
		int **pred_labels);

template<typename T>
int pick_point_idx_weighted(int seed, std::vector<T> *data,
		thrust::host_vector<T> weights) {
	T weighted_sum = 0;

	for (int i = 0; i < weights.size(); i++) {
		if (data) {
			weighted_sum += (data->data()[i] * weights.data()[i]);
		} else {
			weighted_sum += weights.data()[i];
		}
	}

	T best_prob = 0.0;
	int best_prob_idx = 0;

	std::mt19937 mt(seed);
	std::uniform_real_distribution<> dist(0.0, 1.0);

	int i = 0;
	for (i = 0; i <= weights.size(); i++) {
		if (weights.size() == i) {
			break;
		}

		T prob_threshold = (T) dist(mt);

		T data_val = weights.data()[i];
		if (data) {
			data_val *= data->data()[i];
		}

		T prob_x = (data_val / weighted_sum);

		if (prob_x > prob_threshold) {
			break;
		}

		if (prob_x >= best_prob) {
			best_prob = prob_x;
			best_prob_idx = i;
		}
	}

	return weights.size() == i ? best_prob_idx : i;
}

/**
 * Copies cols records, starting at position idx*cols from data to centroids. Removes them afterwards from data.
 * Removes record from weights at position idx.
 * @tparam T
 * @param idx
 * @param cols
 * @param data
 * @param weights
 * @param centroids
 */
template<typename T>
void add_centroid(int idx, int cols, thrust::host_vector<T> &data,
		thrust::host_vector<T> &weights, std::vector<T> &centroids) {
	for (int i = 0; i < cols; i++) {
		centroids.push_back(data[idx * cols + i]);
	}
	weights[idx] = 0;
}

struct square_root: public thrust::unary_function<float, float> {
	__host__ __device__
	float operator()(float x) const {
		return sqrtf(x);
	}
};

template<typename T>
void filterByDot(int d, int k, int *numChosen, thrust::device_vector<T> &dists,
		thrust::device_vector<T> &centroids,
		thrust::device_vector<T> &centroid_dots) {

	float alpha = 1.0f;
	float beta = 0.0f;

	CUDACHECK(cudaSetDevice(0));
	kmeans::detail::make_self_dots(k, d, centroids, centroid_dots);

	thrust::transform(centroid_dots.begin(), centroid_dots.begin() + k,
			centroid_dots.begin(), square_root());

	cublasStatus_t stat =
			safe_cublas(
					cublasSgemm(kmeans::detail::cublas_handle[0], CUBLAS_OP_T, CUBLAS_OP_N, k, k, d, &alpha, thrust::raw_pointer_cast(centroids.data()), d, thrust::raw_pointer_cast(centroids.data()), d, &beta, thrust::raw_pointer_cast(dists.data()), k))
	; //Has to be k or k

	//Check cosine angle between two vectors, must be < .9
	kmeans::detail::checkCosine(d, k, numChosen, dists, centroids,
			centroid_dots);
}

template<typename T>
struct min_calc_functor {
	T* all_costs_ptr;
	T* min_costs_ptr;
	T max = std::numeric_limits<T>::max();
	int potential_k_rows;
	int rows_per_run;

	min_calc_functor(T* _all_costs_ptr, T* _min_costs_ptr,
			int _potential_k_rows, int _rows_per_run) {
		all_costs_ptr = _all_costs_ptr;
		min_costs_ptr = _min_costs_ptr;
		potential_k_rows = _potential_k_rows;
		rows_per_run = _rows_per_run;
	}

	__host__ __device__
	void operator()(int idx) const {
		T best = max;
		for (int j = 0; j < potential_k_rows; j++) {
			best = min(best, std::abs(all_costs_ptr[j * rows_per_run + idx]));
		}
		min_costs_ptr[idx] = min(min_costs_ptr[idx], best);
	}
};

/**
 * K-Means|| initialization method implementation as described in "Scalable K-Means++".
 *
 * This is a probabilistic method, which tries to choose points as much spread out as possible as centroids.
 *
 * In case it finds more than k centroids a K-Means++ algorithm is ran on potential centroids to pick k best suited ones.
 *
 * http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf
 *
 * @tparam T
 * @param verbose
 * @param seed
 * @param ord
 * @param data
 * @param data_dots
 * @param centroids
 * @param rows
 * @param cols
 * @param k
 * @param num_gpu
 * @param threshold
 */
template<typename T>
thrust::host_vector<T> kmeans_parallel(int verbose, int seed, const char ord,
		thrust::device_vector<T> **data, thrust::device_vector<T> **data_dots,
		size_t rows, int cols, int k, int num_gpu, T threshold) {
	if (seed < 0) {
		std::random_device rd;
		int seed = rd();
	}

	size_t rows_per_gpu = rows / num_gpu;

	std::mt19937 gen(seed);
	std::uniform_int_distribution<> dis(0, rows - 1);

	// Find the position (GPU idx and idx on that GPU) of the initial centroid
	int first_center = dis(gen);
	int first_center_idx = first_center % rows_per_gpu;
	int first_center_gpu = first_center / rows_per_gpu;

	log_verbose(verbose, "KMeans|| - Initial centroid %d on GPU %d.",
			first_center_idx, first_center_gpu);

	// Copies the initial centroid to potential centroids vector. That vector will store all potential centroids found
	// in the previous iteration.
	thrust::host_vector<T> h_potential_centroids(cols);
	std::vector<thrust::host_vector<T>> h_potential_centroids_per_gpu(num_gpu);

	CUDACHECK(cudaSetDevice(first_center_gpu));

	thrust::copy((*data[first_center_gpu]).begin() + first_center_idx * cols,
			(*data[first_center_gpu]).begin() + (first_center_idx + 1) * cols,
			h_potential_centroids.begin());

	thrust::host_vector<T> h_all_potential_centroids = h_potential_centroids;

	// Initial the cost-to-potential-centroids and cost-to-closest-potential-centroid matrices. Initial cost is +infinity
	std::vector<thrust::device_vector<T>> d_min_costs(num_gpu);
	for (int q = 0; q < num_gpu; q++) {
		CUDACHECK(cudaSetDevice(q));
		d_min_costs[q].resize(rows_per_gpu);
		thrust::fill(d_min_costs[q].begin(), d_min_costs[q].end(),
				std::numeric_limits<T>::max());
	}

	double t0 = timer<double>();

	int curr_k = h_potential_centroids.size() / cols;
	int max_k = k;

	while (curr_k < max_k) {
		T total_min_cost = 0.0;

		int new_potential_centroids = 0;
#pragma omp parallel for
		for (int i = 0; i < num_gpu; i++) {
			CUDACHECK(cudaSetDevice(i));

			thrust::device_vector<T> d_potential_centroids =
					h_potential_centroids;

			int potential_k_rows = d_potential_centroids.size() / cols;

			// Compute all the costs to each potential centroid from previous iteration
			thrust::device_vector<T> centroid_dots(potential_k_rows);

			kmeans::detail::batch_calculate_distances(verbose, 0, rows_per_gpu,
					cols, potential_k_rows, *data[i], d_potential_centroids,
					*data_dots[i], centroid_dots,
					[&](int rows_per_run, size_t offset, thrust::device_vector<T> &pairwise_distances) {
						// Find the closest potential center cost for each row
						auto min_cost_counter = thrust::make_counting_iterator(0);
						auto all_costs_ptr = thrust::raw_pointer_cast(pairwise_distances.data());
						auto min_costs_ptr = thrust::raw_pointer_cast(d_min_costs[i].data() + offset);
						thrust::for_each(min_cost_counter,
								min_cost_counter + rows_per_run,
								// Functor instead of a lambda b/c nvcc is complaining about
								// nesting a __device__ lambda inside a regular lambda
								min_calc_functor<T>(all_costs_ptr, min_costs_ptr, potential_k_rows, rows_per_run));
					});
		}

		for (int i = 0; i < num_gpu; i++) {
			CUDACHECK(cudaSetDevice(i));
			total_min_cost += thrust::reduce(d_min_costs[i].begin(),
					d_min_costs[i].end());
		}

		log_verbose(verbose, "KMeans|| - Total min cost from centers %g.",
				total_min_cost);

		if (total_min_cost == (T) 0.0) {
			thrust::host_vector<T> final_centroids(0);
			if (verbose) {
				fprintf(stderr,
						"Too few points and centriods being found is getting 0 cost from centers\n");
				fflush(stderr);
			}

			return final_centroids;
		}

		std::set<int> copy_from_gpus;
#pragma omp parallel for
		for (int i = 0; i < num_gpu; i++) {
			CUDACHECK(cudaSetDevice(i));

			// Count how many potential centroids there are using probabilities
			// The further the row is from the closest cluster center the higher the probability
			auto pot_cent_filter_counter = thrust::make_counting_iterator(0);
			auto min_costs_ptr = thrust::raw_pointer_cast(
					d_min_costs[i].data());
int pot_cent_num = thrust::count_if(
                                            pot_cent_filter_counter,
                                            pot_cent_filter_counter + rows_per_gpu, [=]__device__(int idx){
                                              thrust::default_random_engine rng(seed);
                                              thrust::uniform_real_distribution<> dist(0.0, 1.0);
                                              int device;
                                              cudaGetDevice(&device);
                                              rng.discard(idx + device * rows_per_gpu);
                                              T prob_threshold = (T) dist(rng);

                                              T prob_x = (( 2.0 * k * min_costs_ptr[idx]) / total_min_cost);

                                              return prob_x > prob_threshold;
                                            }
                                            );

        			log_debug(verbose, "KMeans|| - Potential centroids on GPU %d = %d.",
					i, pot_cent_num);

			if (pot_cent_num > 0) {
				copy_from_gpus.insert(i);

				// Copy all potential cluster centers
				thrust::device_vector<T> d_new_potential_centroids(
						pot_cent_num * cols);

				auto range = thrust::make_counting_iterator(0);
				thrust::copy_if(
						(*data[i]).begin(), (*data[i]).end(), range,
						d_new_potential_centroids.begin(), [=] __device__(int idx) {
							int row = idx / cols;
							thrust::default_random_engine rng(seed);
							thrust::uniform_real_distribution<> dist(0.0, 1.0);
							int device;
							cudaGetDevice(&device);
							rng.discard(row + device * rows_per_gpu);
							T prob_threshold = (T) dist(rng);

							T prob_x = (( 2.0 * k * min_costs_ptr[row]) / total_min_cost);

							return prob_x > prob_threshold;
						});

				h_potential_centroids_per_gpu[i].clear();
				h_potential_centroids_per_gpu[i].resize(
						d_new_potential_centroids.size());

				new_potential_centroids += d_new_potential_centroids.size();

				thrust::copy(d_new_potential_centroids.begin(),
						d_new_potential_centroids.end(),
						h_potential_centroids_per_gpu[i].begin());

			}

		}

		log_verbose(verbose, "KMeans|| - New potential centroids %d.",
				new_potential_centroids);

		// Gather potential cluster centers from all GPUs
		if (new_potential_centroids > 0) {
			h_potential_centroids.clear();
			h_potential_centroids.resize(new_potential_centroids);

			int old_pot_centroids_size = h_all_potential_centroids.size();
			h_all_potential_centroids.resize(
					old_pot_centroids_size + new_potential_centroids);

			int offset = 0;
			for (int i = 0; i < num_gpu; i++) {
				if (copy_from_gpus.find(i) != copy_from_gpus.end()) {
					thrust::copy(h_potential_centroids_per_gpu[i].begin(),
							h_potential_centroids_per_gpu[i].end(),
							h_potential_centroids.begin() + offset);
					offset += h_potential_centroids_per_gpu[i].size();
				}
			}

			CUDACHECK(cudaSetDevice(0));
			thrust::device_vector<float> new_centroids = h_potential_centroids;

			thrust::device_vector<float> new_centroids_dist(
					(new_potential_centroids / cols)
							* (new_potential_centroids / cols));
			thrust::device_vector<float> new_centroids_dot(
					new_potential_centroids / cols);

			int numChosen = new_potential_centroids / cols;
			filterByDot(cols, numChosen, &numChosen, new_centroids_dist,
					new_centroids, new_centroids_dot);

			thrust::host_vector<T> h_new_centroids = new_centroids;
			h_all_potential_centroids.resize(
					old_pot_centroids_size + (numChosen * cols));
			thrust::copy(h_new_centroids.begin(),
					h_new_centroids.begin() + (numChosen * cols),
					h_all_potential_centroids.begin() + old_pot_centroids_size);
			curr_k = curr_k + numChosen;

		} else {
			thrust::host_vector<T> final_centroids(0);
			if (verbose) {
				fprintf(stderr,
						"Too few points , not able to find centroid candidate \n");
				fflush(stderr);
			}

			return final_centroids;
		}
	}

	thrust::host_vector<T> final_centroids(0);
	int potential_centroids_num = h_all_potential_centroids.size() / cols;

	final_centroids.resize(k * cols);
	thrust::copy(h_all_potential_centroids.begin(),
			h_all_potential_centroids.begin() + (max_k * cols),
			final_centroids.begin());

	return final_centroids;
}

volatile std::atomic_int flaggpu(0);

inline void my_function_gpu(int sig) { // can be called asynchronously
	fprintf(stderr, "Caught signal %d. Terminating shortly.\n", sig);
	flaggpu = 1;
}

std::vector<int> kmeans_init(int verbose, int *final_n_gpu, int n_gputry,
		int gpu_idtry, int rows) {
	if (rows > std::numeric_limits<int>::max()) {
		fprintf(stderr, "rows > %d not implemented\n",
				std::numeric_limits<int>::max());
		fflush(stderr);
		exit(0);
	}

	std::signal(SIGINT, my_function_gpu);
	std::signal(SIGTERM, my_function_gpu);

	// no more gpus than visible gpus
	int n_gpuvis;
	cudaGetDeviceCount(&n_gpuvis);
	int n_gpu = std::min(n_gpuvis, n_gputry);

	// no more than rows
	n_gpu = std::min(n_gpu, rows);

	if (verbose) {
		std::cout << n_gpu << " gpus." << std::endl;
	}

	int gpu_id = gpu_idtry % n_gpuvis;

	// setup GPU list to use
	std::vector<int> dList(n_gpu);
	for (int idx = 0; idx < n_gpu; idx++) {
		int device_idx = (gpu_id + idx) % n_gpuvis;
		dList[idx] = device_idx;
	}

	*final_n_gpu = n_gpu;
	return dList;
}

template<typename T>
H2O4GPUKMeans<T>::H2O4GPUKMeans(const T *A, int k, int n, int d) {
	_A = A;
	_k = k;
	_n = n;
	_d = d;
}

template<typename T>
int kmeans_find_clusters(int verbose, const char ord, int seed,
		thrust::device_vector<T> **data, thrust::device_vector<int> **labels,
		thrust::device_vector<T> **d_centroids,
		thrust::device_vector<T> **data_dots, size_t rows, size_t cols,
		int init_from_data, int k, int k_max, T threshold, const T *srcdata,
		int n_gpu, std::vector<int> dList, T &residual, std::vector<int> v,
		int max_iterations) {
	int bytecount = cols * k * sizeof(T);
	if (0 == init_from_data) {

		log_debug(verbose, "KMeans - Using random initialization.");

		int masterq = 0;
		CUDACHECK(cudaSetDevice(dList[masterq]));
		// DM: simply copies first k rows data into GPU_0
		copy_centroids_shuffled(verbose, v, ord, *d_centroids[masterq],
				&srcdata[0], rows, k, cols);

		// DM: can remove all of this
		// Copy centroids to all devices
		std::vector<cudaStream_t *> streams;
		streams.resize(n_gpu);
#pragma omp parallel for
		for (int q = 0; q < n_gpu; q++) {
			if (q == masterq)
				continue;

			CUDACHECK(cudaSetDevice(dList[q]));
			if (verbose > 0) {
				std::cout << "Copying centroid data to device: " << dList[q]
						<< std::endl;
			}

			streams[q] = reinterpret_cast<cudaStream_t *>(malloc(
					sizeof(cudaStream_t)));
			cudaStreamCreate(streams[q]);
			cudaMemcpyPeerAsync(thrust::raw_pointer_cast(&(*d_centroids[q])[0]),
					dList[q],
					thrust::raw_pointer_cast(&(*d_centroids[masterq])[0]),
					dList[masterq], bytecount, *(streams[q]));
		}
//#pragma omp parallel for
		for (int q = 0; q < n_gpu; q++) {
			if (q == masterq)
				continue;
			cudaSetDevice(dList[q]);
			cudaStreamDestroy(*(streams[q]));
#if(DEBUGKMEANS)
			thrust::host_vector<T> h_centroidq=*d_centroids[q];
			for(int ii=0;ii<k*d;ii++) {
				fprintf(stderr,"q=%d initcent[%d]=%g\n",q,ii,h_centroidq[ii]); fflush(stderr);
			}
#endif
		}
	} else if (1 == init_from_data) { // kmeans||
		log_debug(verbose, "KMeans - Using K-Means|| initialization.");

		thrust::host_vector<T> final_centroids = kmeans_parallel(verbose, seed,
				ord, data, data_dots, rows, cols, k, n_gpu, threshold);
		if (final_centroids.size() == 0) {
			if (verbose) {
				fprintf(stderr,
						"kmeans || failed to find %d number of cluster points \n",
						k);
				fflush(stderr);
			}

			residual = 0.0;
			return 0;
		}

#pragma omp parallel for
		for (int q = 0; q < n_gpu; q++) {
			CUDACHECK(cudaSetDevice(dList[q]));
			cudaMemcpy(thrust::raw_pointer_cast(&(*d_centroids[q])[0]),
					thrust::raw_pointer_cast(&final_centroids[0]), bytecount,
					cudaMemcpyHostToDevice);
		}

	}

#pragma omp parallel for
	for (int q = 0; q < n_gpu; q++) {
		CUDACHECK(cudaSetDevice(dList[q]));
		labels[q] = new thrust::device_vector<int>(rows / n_gpu);
	}

	double t0 = timer<double>();

	int iter = kmeans::kmeans<T>(verbose, &flaggpu, rows, cols, k, k_max, data,
			labels, d_centroids, data_dots, dList, n_gpu, max_iterations,
			threshold, true);

	if (iter < 0) {
		log_error(verbose, "KMeans algorithm failed.");
		return iter;
	}

	// Calculate the residual
	size_t rows_per_gpu = rows / n_gpu;
	std::vector<thrust::device_vector<T>> d_min_costs(n_gpu);
	for (int q = 0; q < n_gpu; q++) {
		CUDACHECK(cudaSetDevice(q));
		d_min_costs[q].resize(rows_per_gpu);
		thrust::fill(d_min_costs[q].begin(), d_min_costs[q].end(),
				std::numeric_limits<T>::max());
	}
#pragma omp parallel for
	for (int i = 0; i < n_gpu; i++) {
		CUDACHECK(cudaSetDevice(i));

		int potential_k_rows = k;
		// Compute all the costs to each potential centroid from previous iteration
		thrust::device_vector<T> centroid_dots(potential_k_rows);

		kmeans::detail::batch_calculate_distances(verbose, 0, rows_per_gpu,
				cols, k, *data[i], *d_centroids[i], *data_dots[i],
				centroid_dots,
				[&](int rows_per_run, size_t offset, thrust::device_vector<T> &pairwise_distances) {
					// Find the closest potential center cost for each row
					auto min_cost_counter = thrust::make_counting_iterator(0);
					auto all_costs_ptr = thrust::raw_pointer_cast(pairwise_distances.data());
					auto min_costs_ptr = thrust::raw_pointer_cast(d_min_costs[i].data() + offset);
					thrust::for_each(min_cost_counter,
							min_cost_counter + rows_per_run,
							// Functor instead of a lambda b/c nvcc is complaining about
							// nesting a __device__ lambda inside a regular lambda
							min_calc_functor<T>(all_costs_ptr, min_costs_ptr, potential_k_rows, rows_per_run));
				});
	}

	residual = 0.0;
	for (int i = 0; i < n_gpu; i++) {
		CUDACHECK(cudaSetDevice(i));
		residual += thrust::reduce(d_min_costs[i].begin(),
				d_min_costs[i].end());
	}

	double timefit = static_cast<double>(timer<double>() - t0);

	if (verbose) {
		std::cout << "  Time fit: " << timefit << " s" << std::endl;
		fprintf(stderr, "Time fir: %g \n", timefit);
		fflush(stderr);
	}

	return iter;
}

template<typename T>
int kmeans_fit(int verbose, int seed, int gpu_idtry, int n_gputry, size_t rows,
		size_t cols, const char ord, int k, int k_max, int max_iterations,
		int init_from_data, T threshold, const T *srcdata, T **pred_centroids,
		int **pred_labels) {
	// init random seed if use the C function rand()
	if (seed >= 0) {
		srand(seed);
	} else {
		srand(unsigned(time(NULL)));
	}

	// no more clusters than rows
	if (k_max > rows) {
		k_max = static_cast<int>(rows);
		fprintf(stderr,
				"Number of clusters adjusted to be equal to number of rows.\n");
		fflush(stderr);
	}

	int n_gpu;
	// only creates a list of GPUs to use. can be removed for single GPU
	std::vector<int> dList = kmeans_init(verbose, &n_gpu, n_gputry, gpu_idtry,
			rows);

	double t0t = timer<double>();
	thrust::device_vector<T> *data[n_gpu];
	thrust::device_vector<int> *labels[n_gpu];
	thrust::device_vector<T> *d_centroids[n_gpu];
	thrust::device_vector<T> *data_dots[n_gpu];
#pragma omp parallel for
	for (int q = 0; q < n_gpu; q++) {
		CUDACHECK(cudaSetDevice(dList[q]));
		data[q] = new thrust::device_vector<T>(rows / n_gpu * cols);
		d_centroids[q] = new thrust::device_vector<T>(k_max * cols);
		data_dots[q] = new thrust::device_vector<T>(rows / n_gpu);

		kmeans::detail::labels_init();
	}

	log_debug(verbose, "Number of points: %d", rows);
	log_debug(verbose, "Number of dimensions: %d", cols);
	log_debug(verbose, "Number of clusters: %d", k);
	log_debug(verbose, "Max number of clusters: %d", k_max);
	log_debug(verbose, "Max. number of iterations: %d", max_iterations);
	log_debug(verbose, "Stopping threshold: %d", threshold);

	std::vector<int> v(rows);
	std::iota(std::begin(v), std::end(v), 0); // Fill with 0, 1, ..., rows.

	if (seed >= 0) {
		std::shuffle(v.begin(), v.end(), std::default_random_engine(seed));
	} else {
		std::random_shuffle(v.begin(), v.end());
	}

	// Copy the data to devices
#pragma omp parallel for
	for (int q = 0; q < n_gpu; q++) {
		CUDACHECK(cudaSetDevice(dList[q]));
		if (verbose) {
			std::cout << "Copying data to device: " << dList[q] << std::endl;
		}

		copy_data(verbose, ord, *data[q], &srcdata[0], q, rows, rows / n_gpu,
				cols);

		// Pre-compute the data matrix norms
		kmeans::detail::make_self_dots(rows / n_gpu, cols, *data[q],
				*data_dots[q]);
	}

	// Host memory
	thrust::host_vector<T> results(k_max + 1, (T) (1e20));

	// Loop to find *best* k
	// Perform k-means in binary search
	int left = k;    //must be at least 2
	int right = k_max; //int(floor(len(data)/2)) #assumption of clusters of size 2 at least
	int mid = int(floor((right + left) / 2));
	int oldmid = mid;
	int tests;
	int iter = 0;
	T objective[2]; // 0= left of mid, 1= right of mid
	T minres = 0;
	T residual = 0.0;

	if (left == 1)
		left = 2; // at least do 2 clusters
	// eval left edge

	iter = kmeans_find_clusters(verbose, ord, seed, data, labels, d_centroids,
			data_dots, rows, cols, init_from_data, left, k_max, threshold,
			srcdata, n_gpu, dList, residual, v, max_iterations);
	results[left] = residual;

	if (left != right) {
		//eval right edge
		residual = 0.0;
		iter = kmeans_find_clusters(verbose, ord, seed, data, labels,
				d_centroids, data_dots, rows, cols, init_from_data, right,
				k_max, threshold, srcdata, n_gpu, dList, residual, v,
				max_iterations);
		int tmp_left = left;
		int tmp_right = right;
		T tmp_residual = 0.0;

		while ((residual == 0.0) && (right > 0)) {
			right = (tmp_left + tmp_right) / 2;
			// This k is already explored and need not be explored again
			if (right == tmp_left) {
				residual = tmp_residual;
				right = tmp_left;
				break;
			}
			iter = kmeans_find_clusters(verbose, ord, seed, data, labels,
					d_centroids, data_dots, rows, cols, init_from_data, right,
					k_max, threshold, srcdata, n_gpu, dList, residual, v,
					max_iterations);
			results[right] = residual;

			if (residual == 0.0) {
				tmp_right = right;
			} else {
				tmp_left = right;
				tmp_residual = residual;

				if (abs(tmp_left - tmp_right) == 1) {
					break;
				}
			}
			// Escape from an infinite loop if we come across
			if (tmp_left == tmp_right) {
				residual = tmp_residual;
				right = tmp_left;
				break;
			}

			residual = 0.0;
		}
		results[right] = residual;
		minres = residual * 0.9;
		mid = int(floor((right + left) / 2));
		oldmid = mid;
	}

	// binary search
	while (left < right - 1) {
		tests = 0;
		while (results[mid] > results[left] && tests < 3) {

			iter = kmeans_find_clusters(verbose, ord, seed, data, labels,
					d_centroids, data_dots, rows, cols, init_from_data, mid,
					k_max, threshold, srcdata, n_gpu, dList, residual, v,
					max_iterations);
			results[mid] = residual;
			if (results[mid] > results[left] && (mid + 1) < right) {
				mid += 1;
				results[mid] = 1e20;
			} else if (results[mid] > results[left] && (mid - 1) > left) {
				mid -= 1;
				results[mid] = 1e20;
			}
			tests += 1;
		}
		objective[0] = abs(results[left] - results[mid])
				/ (results[left] - minres);
		objective[0] /= mid - left;
		objective[1] = abs(results[mid] - results[right])
				/ (results[mid] - minres);
		objective[1] /= right - mid;
		if (objective[0] > 1.2 * objective[1]) { //abs(resid_reduction[left]-resid_reduction[mid])/(mid-left)) {
		// our point is in the left-of-mid side
			right = mid;
		} else {
			left = mid;
		}
		oldmid = mid;
		mid = int(floor((right + left) / 2));
	}

	int k_final = 0;
	k_final = right;
	if (results[left] < results[oldmid])
		k_final = left;

	// if k_star isn't what we just ran, re-run to get correct centroids and dist data on return-> this saves memory
	if (k_final != oldmid) {
		iter = kmeans_find_clusters(verbose, ord, seed, data, labels,
				d_centroids, data_dots, rows, cols, init_from_data, k_final,
				k_max, threshold, srcdata, n_gpu, dList, residual, v,
				max_iterations);
	}

	double timetransfer = static_cast<double>(timer<double>() - t0t);

	double t1 = timer<double>();

	// copy result of centroids (sitting entirely on each device) back to host
	// TODO FIXME: When do delete ctr and h_labels memory???
	thrust::host_vector<T> *ctr = new thrust::host_vector<T>(*d_centroids[0]);
	*pred_centroids = ctr->data();

	// copy assigned labels
	thrust::host_vector<int> *h_labels = new thrust::host_vector<int>(rows);
//#pragma omp parallel for
	for (int q = 0; q < n_gpu; q++) {
		int offset = labels[q]->size() * q;
		h_labels->insert(h_labels->begin() + offset, labels[q]->begin(),
				labels[q]->end());
	}

	*pred_labels = h_labels->data();

	// debug
	if (verbose >= H2O4GPU_LOG_VERBOSE) {
		for (unsigned int ii = 0; ii < k; ii++) {
			fprintf(stderr, "ii=%d of k=%d ", ii, k);
			for (unsigned int jj = 0; jj < cols; jj++) {
				fprintf(stderr, "%g ", (*pred_centroids)[cols * ii + jj]);
			}
			fprintf(stderr, "\n");
			fflush(stderr);
		}

		printf("Number of iteration: %d\n", iter);
	}

#pragma omp parallel for
	for (int q = 0; q < n_gpu; q++) {
		CUDACHECK(cudaSetDevice(dList[q]));
		delete (data[q]);
		delete (labels[q]);
		delete (d_centroids[q]);
		delete (data_dots[q]);
		kmeans::detail::labels_close();
	}

	double timecleanup = static_cast<double>(timer<double>() - t1);

	if (verbose) {
		fprintf(stderr, "Timetransfer: %g Timecleanup: %g\n", timetransfer,
				timecleanup);
		fflush(stderr);
	}

	return k_final;
}

template<typename T>
int kmeans_predict(int verbose, int gpu_idtry, int n_gputry, size_t rows,
		size_t cols, const char ord, int k, const T *srcdata,
		const T *centroids, int **pred_labels) {
	// Print centroids
	if (verbose >= H2O4GPU_LOG_VERBOSE) {
		std::cout << std::endl;
		for (int i = 0; i < cols * k; i++) {
			std::cout << centroids[i] << " ";
			if (i % cols == 1) {
				std::cout << std::endl;
			}
		}
	}

	int n_gpu;
	std::vector<int> dList = kmeans_init(verbose, &n_gpu, n_gputry, gpu_idtry,
			rows);

	thrust::device_vector<T> *d_data[n_gpu];
	thrust::device_vector<T> *d_centroids[n_gpu];
	thrust::device_vector<T> *data_dots[n_gpu];
	thrust::device_vector<T> *centroid_dots[n_gpu];
	thrust::host_vector<int> *h_labels = new thrust::host_vector<int>(0);
	std::vector<thrust::device_vector<int>> d_labels(n_gpu);

#pragma omp parallel for
	for (int q = 0; q < n_gpu; q++) {
		CUDACHECK(cudaSetDevice(dList[q]));
		kmeans::detail::labels_init();

		data_dots[q] = new thrust::device_vector<T>(rows / n_gpu);
		centroid_dots[q] = new thrust::device_vector<T>(k);

		d_centroids[q] = new thrust::device_vector<T>(k * cols);
		d_data[q] = new thrust::device_vector<T>(rows / n_gpu * cols);

		copy_data(verbose, 'r', *d_centroids[q], &centroids[0], 0, k, k, cols);

		copy_data(verbose, ord, *d_data[q], &srcdata[0], q, rows, rows / n_gpu,
				cols);

		kmeans::detail::make_self_dots(rows / n_gpu, cols, *d_data[q],
				*data_dots[q]);

		d_labels[q].resize(rows / n_gpu);

		kmeans::detail::batch_calculate_distances(verbose, q, rows / n_gpu,
				cols, k, *d_data[q], *d_centroids[q], *data_dots[q],
				*centroid_dots[q],
				[&](int n, size_t offset, thrust::device_vector<T> &pairwise_distances) {
					kmeans::detail::relabel(n, k, pairwise_distances, d_labels[q], offset);
				});

	}

	for (int q = 0; q < n_gpu; q++) {
		h_labels->insert(h_labels->end(), d_labels[q].begin(),
				d_labels[q].end());
	}

	*pred_labels = h_labels->data();

#pragma omp parallel for
	for (int q = 0; q < n_gpu; q++) {
		safe_cuda(cudaSetDevice(dList[q]));
		kmeans::detail::labels_close();
		delete (data_dots[q]);
		delete (centroid_dots[q]);
		delete (d_centroids[q]);
		delete (d_data[q]);
	}

	return 0;
}

template<typename T>
int kmeans_transform(int verbose, int gpu_idtry, int n_gputry, size_t rows,
		size_t cols, const char ord, int k, const T *srcdata,
		const T *centroids, T **preds) {
	// Print centroids
	if (verbose >= H2O4GPU_LOG_VERBOSE) {
		std::cout << std::endl;
		for (int i = 0; i < cols * k; i++) {
			std::cout << centroids[i] << " ";
			if (i % cols == 1) {
				std::cout << std::endl;
			}
		}
	}

	int n_gpu;
	std::vector<int> dList = kmeans_init(verbose, &n_gpu, n_gputry, gpu_idtry,
			rows);

	thrust::device_vector<T> *d_data[n_gpu];
	thrust::device_vector<T> *d_centroids[n_gpu];
	thrust::device_vector<T> *d_pairwise_distances[n_gpu];
	thrust::device_vector<T> *data_dots[n_gpu];
	thrust::device_vector<T> *centroid_dots[n_gpu];
#pragma omp parallel for
	for (int q = 0; q < n_gpu; q++) {
		CUDACHECK(cudaSetDevice(dList[q]));
		kmeans::detail::labels_init();

		data_dots[q] = new thrust::device_vector<T>(rows / n_gpu);
		centroid_dots[q] = new thrust::device_vector<T>(k);
		d_pairwise_distances[q] = new thrust::device_vector<T>(
				rows / n_gpu * k);

		d_centroids[q] = new thrust::device_vector<T>(k * cols);
		d_data[q] = new thrust::device_vector<T>(rows / n_gpu * cols);

		copy_data(verbose, 'r', *d_centroids[q], &centroids[0], 0, k, k, cols);

		copy_data(verbose, ord, *d_data[q], &srcdata[0], q, rows, rows / n_gpu,
				cols);

		kmeans::detail::make_self_dots(rows / n_gpu, cols, *d_data[q],
				*data_dots[q]);

		// TODO batch this
		kmeans::detail::calculate_distances(verbose, q, rows / n_gpu, cols, k,
				*d_data[q], 0, *d_centroids[q], *data_dots[q],
				*centroid_dots[q], *d_pairwise_distances[q]);
	}

#pragma omp parallel for
	for (int q = 0; q < n_gpu; q++) {
		CUDACHECK(cudaSetDevice(dList[q]));
		thrust::transform((*d_pairwise_distances[q]).begin(),
				(*d_pairwise_distances[q]).end(),
				(*d_pairwise_distances[q]).begin(), square_root());
	}

	// Move the resulting labels into host memory from all devices
	thrust::host_vector<T> *h_pairwise_distances = new thrust::host_vector<T>(
			0);
#pragma omp parallel for
	for (int q = 0; q < n_gpu; q++) {
		h_pairwise_distances->insert(h_pairwise_distances->end(),
				d_pairwise_distances[q]->begin(),
				d_pairwise_distances[q]->end());
	}
	*preds = h_pairwise_distances->data();

	// Print centroids
	if (verbose >= H2O4GPU_LOG_VERBOSE) {
		std::cout << std::endl;
		for (int i = 0; i < rows * cols; i++) {
			std::cout << h_pairwise_distances->data()[i] << " ";
			if (i % cols == 1) {
				std::cout << std::endl;
			}
		}
	}

#pragma omp parallel for
	for (int q = 0; q < n_gpu; q++) {
		safe_cuda(cudaSetDevice(dList[q]));
		kmeans::detail::labels_close();
		delete (d_pairwise_distances[q]);
		delete (data_dots[q]);
		delete (centroid_dots[q]);
		delete (d_centroids[q]);
		delete (d_data[q]);
	}

	return 0;
}

template<typename T>
int makePtr_dense(int dopredict, int verbose, int seed, int gpu_idtry,
		int n_gputry, size_t rows, size_t cols, const char ord, int k,
		int k_max, int max_iterations, int init_from_data, T threshold,
		const T *srcdata, const T *centroids, T **pred_centroids,
		int **pred_labels) {
	if (dopredict == 0) {
		return kmeans_fit(verbose, seed, gpu_idtry, n_gputry, rows, cols, ord,
				k, k_max, max_iterations, init_from_data, threshold, srcdata,
				pred_centroids, pred_labels);
	} else {
		return kmeans_predict(verbose, gpu_idtry, n_gputry, rows, cols, ord, k,
				srcdata, centroids, pred_labels);
	}
}

template int
makePtr_dense<float>(int dopredict, int verbose, int seed, int gpu_id,
		int n_gpu, size_t rows, size_t cols, const char ord, int k, int k_max,
		int max_iterations, int init_from_data, float threshold,
		const float *srcdata, const float *centroids, float **pred_centroids,
		int **pred_labels);

template int
makePtr_dense<double>(int dopredict, int verbose, int seed, int gpu_id,
		int n_gpu, size_t rows, size_t cols, const char ord, int k, int k_max,
		int max_iterations, int init_from_data, double threshold,
		const double *srcdata, const double *centroids, double **pred_centroids,
		int **pred_labels);

template int kmeans_fit<float>(int verbose, int seed, int gpu_idtry,
		int n_gputry, size_t rows, size_t cols, const char ord, int k,
		int k_max, int max_iterations, int init_from_data, float threshold,
		const float *srcdata, float **pred_centroids, int **pred_labels);

template int kmeans_fit<double>(int verbose, int seed, int gpu_idtry,
		int n_gputry, size_t rows, size_t cols, const char ord, int k,
		int k_max, int max_iterations, int init_from_data, double threshold,
		const double *srcdata, double **pred_centroids, int **pred_labels);

template int kmeans_find_clusters<float>(int verbose, const char ord, int seed,
		thrust::device_vector<float> **data,
		thrust::device_vector<int> **labels,
		thrust::device_vector<float> **d_centroids,
		thrust::device_vector<float> **data_dots, size_t rows, size_t cols,
		int init_from_data, int k, int k_max, float threshold,
		const float *srcdata, int n_gpu, std::vector<int> dList,
		float &residual, std::vector<int> v, int max_iterations);

template int kmeans_find_clusters<double>(int verbose, const char ord, int seed,
		thrust::device_vector<double> **data,
		thrust::device_vector<int> **labels,
		thrust::device_vector<double> **d_centroids,
		thrust::device_vector<double> **data_dots, size_t rows, size_t cols,
		int init_from_data, int k, int k_max, double threshold,
		const double *srcdata, int n_gpu, std::vector<int> dList,
		double &residual, std::vector<int> v, int max_iterations);

template int kmeans_predict<float>(int verbose, int gpu_idtry, int n_gputry,
		size_t rows, size_t cols, const char ord, int k, const float *srcdata,
		const float *centroids, int **pred_labels);

template int kmeans_predict<double>(int verbose, int gpu_idtry, int n_gputry,
		size_t rows, size_t cols, const char ord, int k, const double *srcdata,
		const double *centroids, int **pred_labels);

template int kmeans_transform<float>(int verbose, int gpu_id, int n_gpu,
		size_t m, size_t n, const char ord, int k, const float *src_data,
		const float *centroids, float **preds);

template int kmeans_transform<double>(int verbose, int gpu_id, int n_gpu,
		size_t m, size_t n, const char ord, int k, const double *src_data,
		const double *centroids, double **preds);

// Explicit template instantiation.
#if !defined(H2O4GPU_DOUBLE) || H2O4GPU_DOUBLE == 1

template
class H2O4GPUKMeans<double> ;

#endif

#if !defined(H2O4GPU_SINGLE) || H2O4GPU_SINGLE == 1

template
class H2O4GPUKMeans<float> ;

#endif

int get_n_gpus(int n_gputry) {
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	if (n_gputry < 0) { // get all available GPUs
		return nDevices;
	} else if (n_gputry > nDevices) {
		return nDevices;
	} else {
		return n_gputry;
	}
}

}  // namespace h2o4gpukmeans

namespace ML {
/*
 * Interface for other languages
 */

// Fit and Predict
void make_ptr_kmeans(int dopredict, int verbose, int seed, int gpu_id,
		int n_gpu, size_t mTrain, size_t n, const char ord, int k, int k_max,
		int max_iterations, int init_from_data, float threshold,
		const float *srcdata, const float *centroids, float *pred_centroids,
		int *pred_labels) {
	//float *h_srcdata = (float*) malloc(mTrain * n * sizeof(float));
	//cudaMemcpy((void*)h_srcdata, (void*)srcdata, mTrain*n * sizeof(float), cudaMemcpyDeviceToHost);

	const float *h_srcdata = srcdata;

	float *h_centroids = nullptr;
	if (dopredict) {
		h_centroids = (float*) malloc(k * n * sizeof(float));
		cudaMemcpy((void*) h_centroids, (void*) centroids,
				k * n * sizeof(float), cudaMemcpyDeviceToHost);
	}

	int *h_pred_labels = nullptr;
	float *h_pred_centroids = nullptr;
	int actual_n_gpu = h2o4gpukmeans::get_n_gpus(n_gpu);
	int actual_k = h2o4gpukmeans::makePtr_dense<float>(dopredict, verbose, seed,
			gpu_id, actual_n_gpu, mTrain, n, ord, k, k_max, max_iterations,
			init_from_data, threshold, h_srcdata, h_centroids,
			&h_pred_centroids, &h_pred_labels);

	cudaSetDevice(gpu_id);

	if (dopredict == 0) {
		cudaMemcpy(pred_centroids, h_pred_centroids, k * n * sizeof(float),
				cudaMemcpyHostToDevice);
	}

	cudaMemcpy(pred_labels, h_pred_labels, mTrain * sizeof(int),
			cudaMemcpyHostToDevice);

	//free(h_srcdata);
	if (dopredict) {
		free(h_centroids);
	}
}

void make_ptr_kmeans(int dopredict, int verbose, int seed, int gpu_id,
		int n_gpu, size_t mTrain, size_t n, const char ord, int k, int k_max,
		int max_iterations, int init_from_data, double threshold,
		const double *srcdata, const double *centroids, double *pred_centroids,
		int *pred_labels) {

	//double *h_srcdata = (double*) malloc(mTrain * n * sizeof(double));
	//cudaMemcpy((void*)h_srcdata, (void*)srcdata, mTrain*n * sizeof(double), cudaMemcpyDeviceToHost);

	const double *h_srcdata = srcdata;

	double *h_centroids = nullptr;
	if (dopredict) {
		h_centroids = (double*) malloc(k * n * sizeof(double));
		cudaMemcpy((void*) h_centroids, (void*) centroids,
				k * n * sizeof(double), cudaMemcpyDeviceToHost);
	}

	int *h_pred_labels = nullptr;
	double *h_pred_centroids = nullptr;
	int actual_n_gpu = h2o4gpukmeans::get_n_gpus(n_gpu);
	int actual_k = h2o4gpukmeans::makePtr_dense<double>(dopredict, verbose,
			seed, gpu_id, actual_n_gpu, mTrain, n, ord, k, k_max,
			max_iterations, init_from_data, threshold, h_srcdata, h_centroids,
			&h_pred_centroids, &h_pred_labels);

	cudaSetDevice(gpu_id);
	// int dev = -1;
	// cudaGetDevice(&dev);
	// printf("device: %d\n", dev);

	if (dopredict == 0) {
		cudaMemcpy(pred_centroids, h_pred_centroids, k * n * sizeof(double),
				cudaMemcpyHostToDevice);
	}

	cudaMemcpy(pred_labels, h_pred_labels, mTrain * sizeof(int),
			cudaMemcpyHostToDevice);

	//free(h_srcdata);
	if (dopredict) {
		free(h_centroids);
	}

}

// Transform
void kmeans_transform(int verbose, int gpu_id, int n_gpu, size_t m, size_t n,
		const char ord, int k, const float *src_data, const float *centroids,
		float *preds) {

	const float *h_srcdata = src_data;
	const float *h_centroids = centroids;

	float *h_preds = nullptr;
	int actual_n_gpu = h2o4gpukmeans::get_n_gpus(n_gpu);
	h2o4gpukmeans::kmeans_transform<float>(verbose, gpu_id, actual_n_gpu, m, n,
			ord, k, h_srcdata, h_centroids, &h_preds);

	cudaSetDevice(gpu_id);

	cudaMemcpy(preds, h_preds, m * k * sizeof(float), cudaMemcpyHostToDevice);
}

void kmeans_transform(int verbose, int gpu_id, int n_gpu, size_t m, size_t n,
		const char ord, int k, const double *src_data, const double *centroids,
		double *preds) {

	const double *h_srcdata = src_data;
	const double *h_centroids = centroids;

	double *h_preds = nullptr;
	int actual_n_gpu = h2o4gpukmeans::get_n_gpus(n_gpu);
	h2o4gpukmeans::kmeans_transform<double>(verbose, gpu_id, actual_n_gpu, m, n,
			ord, k, h_srcdata, h_centroids, &h_preds);

	cudaSetDevice(gpu_id);

	cudaMemcpy(preds, h_preds, m * k * sizeof(double), cudaMemcpyHostToDevice);
}

} // end namespace ML
