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

#include <cuML.hpp>

namespace ML {

namespace kmeans{

enum InitMethod{KMeansPlusPlus, Random, Array};

/**
 * @brief Compute k-means clustering and optionally predicts cluster index for each sample in the input.
 *
 * @param[in]     cumlHandle  The handle to the cuML library context that manages the CUDA resources.
 * @param[in]     n_clusters  The number of clusters to form as well as the number of centroids to generate (default:8).
 * @param[in]     metric      Metric to use for distance computation. Any metric from MLCommon::Distance::DistanceType can be used
 * @param[in]     init        Method for initialization, defaults to k-means++:
 *                            - InitMethod::KMeansPlusPlus (k-means++): Use scalable k-means++ algorithm to select the initial cluster centers.
 *                            - InitMethod::Random (random): Choose 'n_clusters' observations (rows) at random from the input data for the initial centroids.
 *                            - InitMethod::Array (ndarray): Use 'centroids' as initial cluster centers. 
 * @param[in]     max_iter    Maximum number of iterations for the k-means algorithm.
 * @param[in]     tol         Relative tolerance with regards to inertia to declare convergence.
 * @param[in]     seed        Seed to the random number generator.
 * @param[in]     X           Training instances to cluster. It must be noted that the data must be in row-major format and stored in device accessible location.
 * @param[in]     n_samples   Number of samples in the input X.
 * @param[in]     n_features  Number of features or the dimensions of each sample.
 * @param[in|out] centroids   [in] When init is InitMethod::Array, use centroids as the initial cluster centers
 *                            [out] Otherwise, generated centroids from the kmeans algorithm is stored at the address pointed by 'centroids'.
 * @param[out]    labels      [optional] Index of the cluster each sample in X belongs to.
 */
void fit_predict(const ML::cumlHandle& handle, 
		 int n_clusters,
		 int metric,
		 InitMethod init,
		 int max_iter,
		 double tol,
		 int seed,
		 const float *X,
		 int n_samples,
		 int n_features,
		 float *centroids,
		 int *labels = 0,
		 int verbose = 0);

void fit_predict(const ML::cumlHandle& handle, 
		 int n_clusters,
		 int metric,
		 InitMethod init,
		 int max_iter,
		 double tol,
		 int seed,
		 const double *X,
		 int n_samples,
		 int n_features,
		 double *centroids,
		 int *labels = 0,
		 int verbose = 0);
  
  
/**
 * @brief Compute k-means clustering.
 *
 * @param[in]     cumlHandle  The handle to the cuML library context that manages the CUDA resources.
 * @param[in]     n_clusters  The number of clusters to form as well as the number of centroids to generate (default:8).
 * @param[in]     metric      Metric to use for distance computation. Any metric from MLCommon::Distance::DistanceType can be used
 * @param[in]     init        Method for initialization, defaults to k-means++:
 *                            - InitMethod::KMeansPlusPlus (k-means++): Use scalable k-means++ algorithm to select the initial cluster centers.
 *                            - InitMethod::Random (random): Choose 'n_clusters' observations (rows) at random from the input data for the initial centroids.
 *                            - InitMethod::Array (ndarray): Use 'centroids' as initial cluster centers. 
 * @param[in]     max_iter    Maximum number of iterations for the k-means algorithm.
 * @param[in]     tol         Relative tolerance with regards to inertia to declare convergence.
 * @param[in]     seed        Seed to the random number generator.
 * @param[in]     X           Training instances to cluster. It must be noted that the data must be in row-major format and stored in device accessible location.
 * @param[in]     n_samples   Number of samples in the input X.
 * @param[in]     n_features  Number of features or the dimensions of each sample.
 * @param[in|out] centroids   [in] When init is InitMethod::Array, use centroids as the initial cluster centers
 *                            [out] Otherwise, generated centroids from the kmeans algorithm is stored at the address pointed by 'centroids'.
 */
void fit(const ML::cumlHandle& handle, 
	 int n_clusters,
	 int metric,
	 InitMethod init,
	 int max_iter,
	 double tol,
	 int seed,
	 const float *X,
	 int n_samples,
	 int n_features,
	 float *centroids,
	 int verbose = 0);

void fit(const ML::cumlHandle& handle, 
	 int n_clusters,
	 int metric,
	 InitMethod init,
	 int max_iter,
	 double tol,
	 int seed,
	 const double *X,
	 int n_samples,
	 int n_features,
	 double *centroids,
	 int verbose = 0);
  
  


/**
 * @brief Predict the closest cluster each sample in X belongs to.
 *
 * @param[in]     cumlHandle  The handle to the cuML library context that manages the CUDA resources.
 * @param[in]     centroids   Cluster centroids. It must be noted that the data must be in row-major format and stored in device accessible location.
 * @param[in]     n_clusters  The number of clusters.
 * @param[in]     X           Training instances to cluster. It must be noted that the data must be in row-major format and stored in device accessible location.
 * @param[in]     n_samples   Number of samples in the input X.
 * @param[in]     n_features  Number of features or the dimensions of each sample in 'X' (value should be same as the dimension for each cluster centers in 'centroids').
 * @param[in]     metric      Metric to use for distance computation. Any metric from MLCommon::Distance::DistanceType can be used
 * @param[out]    labels      Index of the cluster each sample in X belongs to.
 */
void predict(const ML::cumlHandle& handle, 
	     float *centroids,
	     int n_clusters,
	     const float *X,
	     int n_samples,
	     int n_features,
	     int metric,
	     int *labels,
	     int verbose = 0);


void predict(const ML::cumlHandle& handle, 
	     double *centroids,
	     int n_clusters,
	     const double *X,
	     int n_samples,
	     int n_features,
	     int metric,
	     int *labels,
	     int verbose = 0);
  
  
  

/**
 * @brief Transform X to a cluster-distance space.
 *
 * @param[in]     cumlHandle  The handle to the cuML library context that manages the CUDA resources.
 * @param[in]     centroids   Cluster centroids. It must be noted that the data must be in row-major format and stored in device accessible location.
 * @param[in]     n_clusters  The number of clusters.
 * @param[in]     X           Training instances to cluster. It must be noted that the data must be in row-major format and stored in device accessible location.
 * @param[in]     n_samples   Number of samples in the input X.
 * @param[in]     n_features  Number of features or the dimensions of each sample in 'X' (it should be same as the dimension for each cluster centers in 'centroids').
 * @param[in]     metric      Metric to use for distance computation. Any metric from MLCommon::Distance::DistanceType can be used 
 * @param[out]    X_new       X transformed in the new space..
 */
void transform(const ML::cumlHandle& handle, 
	       const float *centroids,
	       int n_clusters,
	       const float *X,
	       int n_samples,
	       int n_features,
	       int metric,
	       float *X_new,
	       int verbose = 0);


void transform(const ML::cumlHandle& handle, 
	       const double *centroids,
	       int n_clusters,
	       const double *X,
	       int n_samples,
	       int n_features,
	       int metric,
	       double *X_new,
	       int verbose = 0);

  
};// end namespace kmeans
};// end namespace ML
