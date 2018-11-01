/*!
 * Copyright 2017-2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */

#pragma once
#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#endif

#include <sstream>
#include <stdio.h>
#include "timer.h"

namespace h2o4gpukmeans {

template<typename M>
class H2O4GPUKMeans {
private:
	// Data
	const M *_A;
	int _k;
	int _n;
	int _d;
public:
	H2O4GPUKMeans(const M *A, int k, int n, int d);
};

template<typename M>
class H2O4GPUKMeansCPU {
private:
	// Data
	const M *_A;
	int _k;
	int _n;
	int _d;
public:
	H2O4GPUKMeansCPU(const M *A, int k, int n, int d);
};

template<typename T>
int makePtr_dense(int dopredict, int verbose, int seed, int gpu_id, int n_gpu,
		size_t rows, size_t cols, const char ord, int k, int max_iterations,
		int init_from_data, T threshold, const T *srcdata, const T *centroids,
		T **pred_centroids, int **pred_labels);

template<typename T>
int kmeans_transform(int verbose, int gpu_id, int n_gpu, size_t m, size_t n,
		const char ord, int k, const T *srcdata, const T *centroids, T **preds);

}  // namespace h2o4gpukmeans

namespace ML {

void make_ptr_kmeans(int dopredict, int verbose, int seed, int gpu_id,
		int n_gpu, size_t mTrain, size_t n, const char ord, int k, int k_max,
		int max_iterations, int init_from_data, float threshold,
		const float *srcdata, const float *centroids, float *pred_centroids,
		int *pred_labels);

void make_ptr_kmeans(int dopredict, int verbose, int seed, int gpu_id,
		int n_gpu, size_t mTrain, size_t n, const char ord, int k, int k_max,
		int max_iterations, int init_from_data, double threshold,
		const double *srcdata, const double *centroids, double *pred_centroids,
		int *pred_labels);

void kmeans_transform(int verbose, int gpu_id, int n_gpu, size_t m, size_t n,
		const char ord, int k, const float *srcdata, const float *centroids,
		float *preds);

void kmeans_transform(int verbose, int gpu_id, int n_gpu, size_t m, size_t n,
		const char ord, int k, const double *srcdata, const double *centroids,
		double *preds);

}
