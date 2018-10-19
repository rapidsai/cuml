/*!
 * Modifications Copyright 2017-2018 H2O.ai, Inc.
 */
// original code from https://github.com/NVIDIA/kmeans (Apache V2.0 License)
#pragma once
#include <atomic>
#include <signal.h>
#include <string>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include "kmeans_centroids.h"
#include "kmeans_labels.h"
#include "kmeans_general.h"

namespace kmeans {

//! kmeans clusters data into k groups
/*!

 \param n Number of data points
 \param d Number of dimensions
 \param k Number of clusters
 \param data Data points, in row-major order. This vector must have
 size n * d, and since it's in row-major order, data point x occupies
 positions [x * d, (x + 1) * d) in the vector. The vector is passed
 by reference since it is shared with the caller and not copied.
 \param labels Cluster labels. This vector has size n.
 The vector is passed by reference since it is shared with the caller
 and not copied.
 \param centroids Centroid locations, in row-major order. This
 vector must have size k * d, and since it's in row-major order,
 centroid x occupies positions [x * d, (x + 1) * d) in the
 vector. The vector is passed by reference since it is shared
 with the caller and not copied.
 \param threshold This controls early termination of the kmeans
 iterations. If the ratio of points being reassigned to a different
 centroid is less than the threshold, than the iterations are
 terminated. Defaults to 1e-3.
 \param max_iterations Maximum number of iterations to run
 \return The number of iterations actually performed.
 */

template<typename T>
int kmeans(int verbose, volatile std::atomic_int *flag, int n, int d, int k,
		int k_max, thrust::device_vector<T> **data,
		thrust::device_vector<int> **labels,
		thrust::device_vector<T> **centroids,
		thrust::device_vector<T> **data_dots, std::vector<int> dList, int n_gpu,
		int max_iterations, double threshold = 1e-3, bool do_per_iter_check =
				true) {

	thrust::device_vector<T> *centroid_dots[n_gpu];
	thrust::device_vector<int> *labels_copy[n_gpu];
	thrust::device_vector<int> *range[n_gpu];
	thrust::device_vector<int> *indices[n_gpu];
	thrust::device_vector<int> *counts[n_gpu];
	thrust::device_vector<T> d_old_centroids;

	thrust::host_vector<int> h_counts(k);
	thrust::host_vector<int> h_counts_tmp(k);
	thrust::host_vector<T> h_centroids(k * d);
	h_centroids = *centroids[0]; // all should be equal
	thrust::host_vector<T> h_centroids_tmp(k_max * d);

	T *d_distance_sum[n_gpu];

	bool unable_alloc = false;
#pragma omp parallel for
	for (int q = 0; q < n_gpu; q++) {
		log_debug(verbose, "Before kmeans() Allocation: gpu: %d", q);

		safe_cuda(cudaSetDevice(dList[q]));
		safe_cuda(cudaMalloc(&d_distance_sum[q], sizeof(T)));

		try {
			centroid_dots[q] = new thrust::device_vector<T>(k);
			labels_copy[q] = new thrust::device_vector<int>(n / n_gpu);
			range[q] = new thrust::device_vector<int>(n / n_gpu);
			counts[q] = new thrust::device_vector<int>(k);
			indices[q] = new thrust::device_vector<int>(n / n_gpu);
		} catch (thrust::system_error &e) {
			log_error(verbose,
					"Unable to allocate memory for gpu: %d | n/n_gpu: %d | k: %d | d: %d | error: %s",
					q, n / n_gpu, k, d, e.what());
			unable_alloc = true;
			// throw std::runtime_error(ss.str());
		} catch (std::bad_alloc &e) {
			log_error(verbose,
					"Unable to allocate memory for gpu: %d | n/n_gpu: %d | k: %d | d: %d | error: %s",
					q, n / n_gpu, k, d, e.what());
			unable_alloc = true;
			//throw std::runtime_error(ss.str());
		}

		if (!unable_alloc) {
			//Create and save "range" for initializing labels
			thrust::copy(thrust::counting_iterator<int>(0),
					thrust::counting_iterator<int>(n / n_gpu),
					(*range[q]).begin());
		}
	}

	if (unable_alloc)
		return (-1);

	log_debug(verbose, "Before kmeans() Iterations");

	int i = 0;
	bool done = false;
	for (; i < max_iterations; i++) {
		log_verbose(verbose, "KMeans - Iteration %d/%d", i, max_iterations);

		if (*flag)
			continue;

		safe_cuda(cudaSetDevice(dList[0]));
		d_old_centroids = *centroids[dList[0]];

#pragma omp parallel for
		for (int q = 0; q < n_gpu; q++) {
			safe_cuda(cudaSetDevice(dList[q]));

			detail::batch_calculate_distances(verbose, q, n / n_gpu, d, k,
					*data[q], *centroids[q], *data_dots[q], *centroid_dots[q],
					[&](int n, size_t offset, thrust::device_vector<T> &pairwise_distances) {
						detail::relabel(n, k, pairwise_distances, *labels[q], offset);
					});

			log_verbose(verbose, "KMeans - Relabeled.");

			detail::memcpy(*labels_copy[q], *labels[q]);
			detail::find_centroids(q, n / n_gpu, d, k, k_max, *data[q],
					*labels_copy[q], *centroids[q], *range[q], *indices[q],
					*counts[q], n_gpu <= 1);
		}

		// Scale the centroids on host
		if (n_gpu > 1) {
			//Average the centroids from each device
			for (int p = 0; p < k; p++)
				h_counts[p] = 0.0;
			for (int q = 0; q < n_gpu; q++) {
				safe_cuda(cudaSetDevice(dList[q]));
				detail::memcpy(h_counts_tmp, *counts[q]);
				detail::streamsync(dList[q]);
				for (int p = 0; p < k; p++)
					h_counts[p] += h_counts_tmp[p];
			}

			// Zero the centroids only if any of the GPUs actually updated them
			for (int p = 0; p < k; p++) {
				for (int r = 0; r < d; r++) {
					if (h_counts[p] != 0) {
						h_centroids[p * d + r] = 0.0;
					}
				}
			}

			for (int q = 0; q < n_gpu; q++) {
				safe_cuda(cudaSetDevice(dList[q]));
				detail::memcpy(h_centroids_tmp, *centroids[q]);
				detail::streamsync(dList[q]);
				for (int p = 0; p < k; p++) {
					for (int r = 0; r < d; r++) {
						if (h_counts[p] != 0) {
							h_centroids[p * d + r] +=
									h_centroids_tmp[p * d + r];
						}
					}
				}
			}

			for (int p = 0; p < k; p++) {
				for (int r = 0; r < d; r++) {
					// If 0 counts that means we leave the original centroids
					if (h_counts[p] == 0) {
						h_counts[p] = 1;
					}
					h_centroids[p * d + r] /= h_counts[p];
				}
			}

			//Copy the averaged centroids to each device
#pragma omp parallel for
			for (int q = 0; q < n_gpu; q++) {
				safe_cuda(cudaSetDevice(dList[q]));
				detail::memcpy(*centroids[q], h_centroids);
			}
		}

		// whether to perform per iteration check
		if (do_per_iter_check) {
			safe_cuda(cudaSetDevice(dList[0]));

			T
			squared_norm = thrust::inner_product(
					d_old_centroids.begin(), d_old_centroids.end(),
					(*centroids[0]).begin(),
					(T) 0.0,
					thrust::plus<T>(),
					[=]__device__(T left, T right) {
						T diff = left - right;
						return diff * diff;
					}
			);

			if (squared_norm < threshold) {
				if (verbose) {
					std::cout << "Threshold triggered. Terminating early."
							<< std::endl;
				}
				done = true;
			}
		}

		if (*flag) {
			fprintf(stderr, "Signal caught. Terminated early.\n");
			fflush(stderr);
			*flag = 0; // set flag
			done = true;
		}

		if (done || i == max_iterations - 1) {
			// Final relabeling - uses final centroids
#pragma omp parallel for
			for (int q = 0; q < n_gpu; q++) {
				safe_cuda(cudaSetDevice(dList[q]));
				detail::batch_calculate_distances(verbose, q, n / n_gpu, d, k,
						*data[q], *centroids[q], *data_dots[q],
						*centroid_dots[q],
						[&](int n, size_t offset, thrust::device_vector<T> &pairwise_distances) {
							detail::relabel(n, k, pairwise_distances, *labels[q], offset);
						});
			}
			break;
		}
	}

//#pragma omp parallel for
	for (int q = 0; q < n_gpu; q++) {
		safe_cuda(cudaSetDevice(dList[q]));
		delete (centroid_dots[q]);
		delete (labels_copy[q]);
		delete (range[q]);
		delete (counts[q]);
		delete (indices[q]);
	}

	log_debug(verbose, "Iterations: %d", i);

	return i;
}

}
