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

#include <distance/distance.h>
#include <linalg/binary_op.h>
#include <linalg/matrix_vector_op.h>
#include <linalg/mean_squared_error.h>
#include <linalg/reduce_rows_by_key.h>
#include <matrix/gather.h>
#include <random/permute.h>
#include <random/rng.h>
#include <random>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <numeric>

#include <ml_cuda_utils.h>
#include <common/allocatorAdapter.hpp>
#include <common/cumlHandle.hpp>
#include <common/device_buffer.hpp>
#include <common/host_buffer.hpp>
#include <common/tensor.hpp>
#include <cuML.hpp>

#include "kmeans.hpp"

namespace ML {
using namespace MLCommon;

/**
 * @tparam DataT 
 * @tparam IndexT
 */
template <typename DataT, typename IndexT = int>
class KMeans {
 public:
  struct Options {
    int seed;
    int oversampling_factor;
    Random::GeneratorType gtype;
    bool inertia_check;

   public:
    Options() {
      seed = -1;
      oversampling_factor = 0;
      gtype = Random::GeneratorType::GenPhilox;
      inertia_check = false;
    }
  };

  /**
     * @param[in] cumlHandle  The handle to the cuML library context that manages the CUDA resources.
     * @param[in] n_clusters  The number of clusters to form as well as the number of centroids to generate (default:8).
     * @param[in] metric      Metric to use for distance computation. Any metric from MLCommon::Distance::DistanceType can be used
     * @param[in] init        Method for initialization, defaults to k-means++:
     *                            - InitMethod::KMeansPlusPlus (k-means++): Use scalable k-means++ algorithm to select the initial cluster centers.
     *                            - InitMethod::Random (random): Choose 'n_clusters' observations (rows) at random from the input data for the initial centroids.
     *                            - InitMethod::Array (ndarray): Use the user provided input for initial centroids, it should be set using setCentroids method. 
     * @param[in] max_iter    Maximum number of iterations for the k-means algorithm.
     * @param[in] tol         Relative tolerance with regards to inertia to declare convergence.
     * @param[in] seed        Seed to the random number generator.
     */
  KMeans(const ML::cumlHandle_impl &cumlHandle, int n_clusters = 8,
         MLCommon::Distance::DistanceType metric =
           MLCommon::Distance::DistanceType::EucExpandedL2Sqrt,
         kmeans::InitMethod init = kmeans::InitMethod::KMeansPlusPlus,
         int max_iter = 300, double tol = 1e-4, int seed = -1, int verbose = 0);

  /**
     * @brief Compute k-means clustering.
     *
     * @param[in] X          Training instances to cluster. It must be noted that the data must be in row-major format and stored in device accessible location.
     * @param[in] n_samples  Number of samples in the input X.
     * @param[in] n_features Number of features or the dimensions of each sample.
     */
  void fit(const DataT *X, int n_samples, int n_features);

  /**
     * @brief Predict the closest cluster each sample in X belongs to.
     *
     * @param[in]  X          New data to predict.
     * @param[in]  n_samples  Number of samples in the input X.
     * @param[in]  n_features Number of features or the dimensions of each sample.
     * @param[out] labels     Index of the cluster each sample belongs to.
     */
  void predict(const DataT *X, int n_samples, int n_features, IndexT *labels);

  /**
     * @brief Transform X to a cluster-distance space. In the new space, each dimension is the distance to the cluster centers.
     *
     * @param[in]  X          New data to transform.
     * @param[in]  n_samples  Number of samples in the input X.
     * @param[in]  n_features Number of features or the dimensions of each sample.
     * @param[out] X_new      X transformed in the new space (output size is [n_samples x n_clusters].
     */
  void transform(const DataT *X, int n_samples, int n_features, DataT *X_new);

  /**
     * @brief Set centroids to be user provided value X
     *
     * @param[in]  X          New data to centroids.
     * @param[in]  n_samples  Number of samples in the input X.
     * @param[in]  n_features Number of features or the dimensions of each sample.
     */
  void setCentroids(const DataT *X, int n_samples, int n_features);

  // returns the raw pointer to the generated centroids
  DataT *centroids();

  // returns the inertia
  const double *getInertia();

  // release local resources
  ~KMeans();

  // internal functions (ideally must be private methods, but its not possible today because of the lambda limitations of CUDA compiler)
  void fit(Tensor<DataT, 2, IndexT> &);

  void predict(Tensor<DataT, 2, IndexT> &);

  void initRandom(Tensor<DataT, 2, IndexT> &);

  void initKMeansPlusPlus(Tensor<DataT, 2, IndexT> &);

 private:
  // Maximum number of iterations of the k-means algorithm for a single run.
  int max_iter;

  // Relative tolerance with regards to inertia to declare convergence.
  double tol;

  // Sum of squared distances of samples to their closest cluster center.
  double inertia;

  // Number of iterations run.
  int n_iter;

  // number of clusters to form as well as the number of centroids to generate.
  int n_clusters;

  // number of features or the dimensions of the input.
  int _n_features;

  // verbosity mode.
  int _verbose;

  // Device-accessible allocation of expandable storage used as temorary buffers
  MLCommon::device_buffer<char> _workspace;

  // underlying expandable storage that holds centroids data
  MLCommon::device_buffer<DataT> _centroidsRawData;

  // underlying expandable storage that holds labels
  MLCommon::device_buffer<IndexT> _labelsRawData;

  const ML::cumlHandle_impl &_handle;

  // method for initialization, defaults to 'k-means++':
  kmeans::InitMethod _init;

  MLCommon::Distance::DistanceType _metric;

  // optional
  Options options;
};
}  // namespace ML

#include "kmeans-inl.cuh"
