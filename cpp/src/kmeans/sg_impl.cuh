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

#include "common.cuh"

namespace ML {

namespace kmeans {

// Selects 'n_clusters' samples randomly from X
template <typename DataT, typename IndexT>
void initRandom(const ML::cumlHandle_impl &handle, const KMeansParams &params,
                Tensor<DataT, 2, IndexT> &X,
                MLCommon::device_buffer<DataT> &centroidsRawData) {
  cudaStream_t stream = handle.getStream();
  auto n_features = X.getSize(1);
  auto n_clusters = params.n_clusters;
  // allocate centroids buffer
  centroidsRawData.resize(n_clusters * n_features, stream);
  auto centroids = std::move(Tensor<DataT, 2, IndexT>(
    centroidsRawData.data(), {n_clusters, n_features}));

  kmeans::detail::shuffleAndGather(handle, X, centroids, n_clusters,
                                   params.seed, stream);
}

template <typename DataT, typename IndexT>
void fit(const ML::cumlHandle_impl &handle, const KMeansParams &params,
         Tensor<DataT, 2, IndexT> &X,
         MLCommon::device_buffer<DataT> &centroidsRawData, DataT &inertia,
         int &n_iter, MLCommon::device_buffer<char> &workspace) {
  cudaStream_t stream = handle.getStream();
  auto n_samples = X.getSize(0);
  auto n_features = X.getSize(1);
  auto n_clusters = params.n_clusters;

  MLCommon::Distance::DistanceType metric =
    static_cast<MLCommon::Distance::DistanceType>(params.metric);

  auto dataBatchSize = kmeans::detail::getDataBatchSize(params, n_samples);

  // stores (key, value) pair corresponding to each sample where
  //   - key is the index of nearest cluster
  //   - value is the distance to the nearest cluster
  Tensor<cub::KeyValuePair<IndexT, DataT>, 1, IndexT> minClusterAndDistance(
    {n_samples}, handle.getDeviceAllocator(), stream);

  // temporary buffer to store distance matrix, destructor releases the resource
  Tensor<DataT, 2, IndexT> pairwiseDistance(
    {dataBatchSize, n_clusters}, handle.getDeviceAllocator(), stream);

  // temporary buffer to store intermediate centroids, destructor releases the
  // resource
  Tensor<DataT, 2, IndexT> newCentroids({n_clusters, n_features},
                                        handle.getDeviceAllocator(), stream);

  // temporary buffer to store the sample count per cluster, destructor releases
  // the resource
  Tensor<int, 1, IndexT> sampleCountInCluster(
    {n_clusters}, handle.getDeviceAllocator(), stream);

  cub::KeyValuePair<IndexT, DataT> *clusterCostD =
    (cub::KeyValuePair<IndexT, DataT> *)handle.getDeviceAllocator()->allocate(
      sizeof(cub::KeyValuePair<IndexT, DataT>), stream);

  DataT priorClusteringCost = 0;
  for (n_iter = 0; n_iter < params.max_iter; ++n_iter) {
    LOG(params.verbose,
        "KMeans.fit: Iteration-%d: fitting the model using the initialize "
        "cluster centers\n",
        n_iter);

    auto centroids = std::move(Tensor<DataT, 2, IndexT>(
      centroidsRawData.data(), {n_clusters, n_features}));

    // computes minClusterAndDistance[0:n_samples) where
    // minClusterAndDistance[i] is a <key, value> pair where
    //   'key' is index to an sample in 'centroids' (index of the nearest
    //   centroid) and 'value' is the distance between the sample 'X[i]' and the
    //   'centroid[key]'
    kmeans::detail::minClusterAndDistance(
      handle, params, X, centroids, pairwiseDistance, minClusterAndDistance,
      workspace, metric, stream);

    // Using TransformInputIteratorT to dereference an array of
    // cub::KeyValuePair and converting them to just return the Key to be used
    // in reduce_rows_by_key prims
    kmeans::detail::KeyValueIndexOp<IndexT, DataT> conversion_op;
    cub::TransformInputIterator<IndexT,
                                kmeans::detail::KeyValueIndexOp<IndexT, DataT>,
                                cub::KeyValuePair<IndexT, DataT> *>
      itr(minClusterAndDistance.data(), conversion_op);

    workspace.resize(n_samples, stream);

    // Calculates sum of all the samples assigned to cluster-i and store the
    // result in newCentroids[i]
    MLCommon::LinAlg::reduce_rows_by_key(
      X.data(), X.getSize(1), itr, workspace.data(), X.getSize(0), X.getSize(1),
      n_clusters, newCentroids.data(), stream);

    // count # of samples in each cluster
    kmeans::detail::countLabels(handle, itr, sampleCountInCluster.data(),
                                n_samples, n_clusters, workspace, stream);

    // Computes newCentroids[i] = newCentroids[i]/sampleCountInCluster[i] where
    //   newCentroids[n_samples x n_features] - 2D array, newCentroids[i] has
    //   sum of all the samples assigned to cluster-i
    //   sampleCountInCluster[n_clusters] - 1D array, sampleCountInCluster[i]
    //   contains # of samples in cluster-i.
    // Note - when sampleCountInCluster[i] is 0, newCentroid[i] is reset to 0

    // transforms int values in sampleCountInCluster to its inverse and more
    // importantly to DataT because matrixVectorOp supports only when matrix and
    // vector are of same type
    workspace.resize(sampleCountInCluster.numElements() * sizeof(DataT),
                     stream);
    auto sampleCountInClusterInverse = std::move(
      Tensor<DataT, 1, IndexT>((DataT *)workspace.data(), {n_clusters}));

    ML::thrustAllocatorAdapter alloc(handle.getDeviceAllocator(), stream);
    auto execution_policy = thrust::cuda::par(alloc).on(stream);
    thrust::transform(
      execution_policy, sampleCountInCluster.begin(),
      sampleCountInCluster.end(), sampleCountInClusterInverse.begin(),
      [=] __device__(int count) {
        if (count == 0)
          return static_cast<DataT>(0);
        else
          return static_cast<DataT>(1.0) / static_cast<DataT>(count);
      });

    MLCommon::LinAlg::matrixVectorOp(
      newCentroids.data(), newCentroids.data(),
      sampleCountInClusterInverse.data(), newCentroids.getSize(1),
      newCentroids.getSize(0), true, false,
      [=] __device__(DataT mat, DataT vec) { return mat * vec; }, stream);

    // copy the centroids[i] to newCentroids[i] when sampleCountInCluster[i] is
    // 0
    cub::ArgIndexInputIterator<int *> itr_sc(sampleCountInCluster.data());
    MLCommon::Matrix::gather_if(
      centroids.data(), centroids.getSize(1), centroids.getSize(0), itr_sc,
      itr_sc, sampleCountInCluster.numElements(), newCentroids.data(),
      [=] __device__(cub::KeyValuePair<ptrdiff_t, int> map) {  // predicate
        // copy when the # of samples in the cluster is 0
        if (map.value == 0)
          return true;
        else
          return false;
      },
      [=] __device__(cub::KeyValuePair<ptrdiff_t, int> map) {  // map
        return map.key;
      },
      stream);

    // compute the squared norm between the newCentroids and the original
    // centroids, destructor releases the resource
    Tensor<DataT, 1> sqrdNorm({1}, handle.getDeviceAllocator(), stream);
    MLCommon::LinAlg::mapThenSumReduce(
      sqrdNorm.data(), newCentroids.numElements(),
      [=] __device__(const DataT a, const DataT b) {
        DataT diff = a - b;
        return diff * diff;
      },
      stream, centroids.data(), newCentroids.data());

    DataT sqrdNormError = 0;
    MLCommon::copy(&sqrdNormError, sqrdNorm.data(), sqrdNorm.numElements(),
                   stream);

    MLCommon::copy(centroidsRawData.data(), newCentroids.data(),
                   newCentroids.numElements(), stream);

    bool done = false;
    if (params.inertia_check) {
      // calculate cluster cost phi_x(C)
      kmeans::detail::computeClusterCost(
        handle, minClusterAndDistance, workspace, clusterCostD,
        [] __device__(const cub::KeyValuePair<IndexT, DataT> &a,
                      const cub::KeyValuePair<IndexT, DataT> &b) {
          cub::KeyValuePair<IndexT, DataT> res;
          res.key = 0;
          res.value = a.value + b.value;
          return res;
        },
        stream);

      DataT curClusteringCost = 0;
      MLCommon::copy(&curClusteringCost, &clusterCostD->value, 1, stream);

      CUDA_CHECK(cudaStreamSynchronize(stream));
      ASSERT(curClusteringCost != (DataT)0.0,
             "Too few points and centriods being found is getting 0 cost from "
             "centers\n");

      if (n_iter > 0) {
        DataT delta = curClusteringCost / priorClusteringCost;
        if (delta > 1 - params.tol) done = true;
      }
      priorClusteringCost = curClusteringCost;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (sqrdNormError < params.tol) done = true;

    if (done) {
      LOG(params.verbose,
          "Threshold triggered after %d iterations. Terminating early.\n",
          n_iter);
      break;
    }
  }

  // calculate cluster cost phi_x(C)
  kmeans::detail::computeClusterCost(
    handle, minClusterAndDistance, workspace, clusterCostD,
    [] __device__(const cub::KeyValuePair<IndexT, DataT> &a,
                  const cub::KeyValuePair<IndexT, DataT> &b) {
      cub::KeyValuePair<IndexT, DataT> res;
      res.key = 0;
      res.value = a.value + b.value;
      return res;
    },
    stream);

  MLCommon::copy(&inertia, &clusterCostD->value, 1, stream);

  handle.getDeviceAllocator()->deallocate(
    clusterCostD, sizeof(cub::KeyValuePair<IndexT, DataT>), stream);
}

/*
 * @brief Selects 'n_clusters' samples from X using scalable kmeans++ algorithm.

 * @note  This is the algorithm described in
 *        "Scalable K-Means++", 2012, Bahman Bahmani, Benjamin Moseley,
 *         Andrea Vattani, Ravi Kumar, Sergei Vassilvitskii,
 *         https://arxiv.org/abs/1203.6402

 * Scalable kmeans++ pseudocode
 * 1: C = sample a point uniformly at random from X
 * 2: psi = phi_X (C)
 * 3: for O( log(psi) ) times do
 * 4:   C' = sample each point x in X independently with probability
 *           p_x = l * (d^2(x, C) / phi_X (C) )
 * 5:   C = C U C'
 * 6: end for
 * 7: For x in C, set w_x to be the number of points in X closer to x than any
 * other point in C
 * 8: Recluster the weighted points in C into k clusters

 */
template <typename DataT, typename IndexT>
void initKMeansPlusPlus(const ML::cumlHandle_impl &handle,
                        const KMeansParams &params, Tensor<DataT, 2, IndexT> &X,
                        MLCommon::device_buffer<DataT> &centroidsRawData,
                        MLCommon::device_buffer<char> &workspace) {
  cudaStream_t stream = handle.getStream();
  auto n_samples = X.getSize(0);
  auto n_features = X.getSize(1);
  auto n_clusters = params.n_clusters;
  MLCommon::Distance::DistanceType metric =
    static_cast<MLCommon::Distance::DistanceType>(params.metric);

  MLCommon::Random::Rng rng(params.seed,
                            MLCommon::Random::GeneratorType::GenPhilox);

  // <<<< Step-1 >>> : C <- sample a point uniformly at random from X
  std::mt19937 gen(params.seed);
  std::uniform_int_distribution<> dis(0, n_samples - 1);

  int cIdx = dis(gen);
  auto initialCentroid = X.template view<2>({1, n_features}, {cIdx, 0});

  // flag the sample that is chosen as initial centroid
  MLCommon::host_buffer<int> h_isSampleCentroid(handle.getHostAllocator(),
                                                stream, n_samples);
  std::fill(h_isSampleCentroid.begin(), h_isSampleCentroid.end(), 0);
  h_isSampleCentroid[cIdx] = 1;

  // device buffer to flag the sample that is chosen as initial centroid
  Tensor<int, 1> isSampleCentroid({n_samples}, handle.getDeviceAllocator(),
                                  stream);

  MLCommon::copy(isSampleCentroid.data(), h_isSampleCentroid.data(),
                 isSampleCentroid.numElements(), stream);

  MLCommon::device_buffer<DataT> centroidsBuf(handle.getDeviceAllocator(),
                                              stream);

  // reset buffer to store the chosen centroid
  centroidsBuf.reserve(n_clusters * n_features, stream);
  centroidsBuf.resize(initialCentroid.numElements(), stream);
  MLCommon::copy(centroidsBuf.begin(), initialCentroid.data(),
                 initialCentroid.numElements(), stream);

  auto potentialCentroids = std::move(Tensor<DataT, 2, IndexT>(
    centroidsBuf.data(),
    {initialCentroid.getSize(0), initialCentroid.getSize(1)}));
  // <<< End of Step-1 >>>

  int dataBatchSize = kmeans::detail::getDataBatchSize(params, n_samples);

  MLCommon::device_buffer<DataT> pairwiseDistanceRaw(
    handle.getDeviceAllocator(), stream);
  pairwiseDistanceRaw.resize(dataBatchSize * n_clusters, stream);

  Tensor<DataT, 1, IndexT> minClusterDistance(
    {n_samples}, handle.getDeviceAllocator(), stream);
  Tensor<DataT, 1, IndexT> uniformRands({n_samples},
                                        handle.getDeviceAllocator(), stream);
  MLCommon::device_buffer<DataT> clusterCost(handle.getDeviceAllocator(),
                                             stream, 1);

  // <<< Step-2 >>>: psi <- phi_X (C)
  Tensor<DataT, 2, IndexT> pairwiseDistance(
    (DataT *)pairwiseDistanceRaw.data(),
    {dataBatchSize, potentialCentroids.getSize(0)});

  kmeans::detail::minClusterDistance(handle, params, X, potentialCentroids,
                                     pairwiseDistance, minClusterDistance,
                                     workspace, metric, stream);

  // compute partial cluster cost from the samples in rank
  kmeans::detail::computeClusterCost(
    handle, minClusterDistance, workspace, clusterCost.data(),
    [] __device__(const DataT &a, const DataT &b) { return a + b; }, stream);

  DataT psi = 0;
  MLCommon::copy(&psi, clusterCost.data(), clusterCost.size(), stream);

  // <<< End of Step-2 >>>

  // Scalable kmeans++ paper claims 8 rounds is sufficient
  CUDA_CHECK(cudaStreamSynchronize(stream));
  int niter = std::min(8, (int)ceil(log(psi)));
  // <<<< Step-3 >>> : for O( log(psi) ) times do
  for (int iter = 0; iter < niter; ++iter) {
    LOG(params.verbose,
        "KMeans|| - Iteration %d: # potential centroids sampled - %d\n", iter,
        potentialCentroids.getSize(0));

    pairwiseDistanceRaw.resize(dataBatchSize * potentialCentroids.getSize(0),
                               stream);
    Tensor<DataT, 2, IndexT> pairwiseDistance(
      (DataT *)pairwiseDistanceRaw.data(),
      {dataBatchSize, potentialCentroids.getSize(0)});

    kmeans::detail::minClusterDistance(handle, params, X, potentialCentroids,
                                       pairwiseDistance, minClusterDistance,
                                       workspace, metric, stream);

    kmeans::detail::computeClusterCost(
      handle, minClusterDistance, workspace, clusterCost.data(),
      [] __device__(const DataT &a, const DataT &b) { return a + b; }, stream);

    MLCommon::copy(&psi, clusterCost.data(), clusterCost.size(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // <<<< Step-4 >>> : Sample each point x in X independently and identify new
    // potentialCentroids
    rng.uniform(uniformRands.data(), uniformRands.getSize(0), (DataT)0,
                (DataT)1, stream);
    kmeans::detail::SamplingOp<DataT> select_op(
      psi, n_clusters * params.oversampling_factor, uniformRands.data(),
      isSampleCentroid.data());

    auto Cp = kmeans::detail::sampleCentroids(handle, X, minClusterDistance,
                                              isSampleCentroid, select_op,
                                              workspace, stream);
    /// <<<< End of Step-4 >>>>

    /// <<<< Step-5 >>> : C = C U C'
    // append the data in Cp to the buffer holding the potentialCentroids
    centroidsBuf.resize(centroidsBuf.size() + Cp.numElements(), stream);
    MLCommon::copy(centroidsBuf.end() - Cp.numElements(), Cp.data(),
                   Cp.numElements(), stream);

    int tot_centroids = potentialCentroids.getSize(0) + Cp.getSize(0);
    potentialCentroids = std::move(Tensor<DataT, 2, IndexT>(
      centroidsBuf.data(), {tot_centroids, n_features}));
    /// <<<< End of Step-5 >>>
  }  /// <<<< Step-6 >>>

  LOG(params.verbose, "KMeans||: # potential centroids sampled - %d\n",
      potentialCentroids.getSize(0));

  if (potentialCentroids.getSize(0) > n_clusters) {
    // <<< Step-7 >>>: For x in C, set w_x to be the number of pts closest to X
    // temporary buffer to store the sample count per cluster, destructor
    // releases the resource
    Tensor<int, 1, IndexT> weights({potentialCentroids.getSize(0)},
                                   handle.getDeviceAllocator(), stream);

    kmeans::detail::countSamplesInCluster(handle, params, X, potentialCentroids,
                                          workspace, metric, weights, stream);

    // <<< end of Step-7 >>>

    // Step-8: Recluster the weighted points in C into k clusters
    centroidsRawData.resize(n_clusters * n_features, stream);
    kmeans::detail::kmeansPlusPlus(handle, params, potentialCentroids, weights,
                                   metric, workspace, centroidsRawData, stream);

    DataT inertia = 0;
    int n_iter = 0;
    ML::kmeans::fit(handle, params, potentialCentroids, centroidsRawData,
                    inertia, n_iter, workspace);
  } else {
    ASSERT(potentialCentroids.getSize(0) == n_clusters,
           "failed to converge, restart with different seed");

    centroidsRawData.resize(n_clusters * n_features, stream);
    MLCommon::copy(centroidsRawData.data(), potentialCentroids.data(),
                   potentialCentroids.numElements(), stream);
  }
}

template <typename DataT, typename IndexT = int>
void fit(const ML::cumlHandle_impl &handle, const KMeansParams &params,
         const DataT *X, const int n_local_samples, const int n_features,
         DataT *centroids, DataT &inertia, int &n_iter) {
  cudaStream_t stream = handle.getStream();

  ASSERT(n_local_samples > 0, "# of samples must be > 0");

  ASSERT(params.oversampling_factor > 0,
         "oversampling factor must be > 0 (requested %d)",
         (int)params.oversampling_factor);

  ASSERT(memory_type(X) == cudaMemoryTypeDevice,
         "input data must be device accessible");

  Tensor<DataT, 2, IndexT> data((DataT *)X, {n_local_samples, n_features});

  // underlying expandable storage that holds centroids data
  MLCommon::device_buffer<DataT> centroidsRawData(handle.getDeviceAllocator(),
                                                  stream);

  // Device-accessible allocation of expandable storage used as temorary buffers
  MLCommon::device_buffer<char> workspace(handle.getDeviceAllocator(), stream);

  if (params.init == KMeansParams::InitMethod::Random) {
    // initializing with random samples from input dataset
    LOG(params.verbose,
        "KMeans.fit: initialize cluster centers by randomly choosing from the "
        "input data.\n");
    initRandom(handle, params, data, centroidsRawData);
  } else if (params.init == KMeansParams::InitMethod::KMeansPlusPlus) {
    // default method to initialize is kmeans++
    LOG(params.verbose,
        "KMeans.fit: initialize cluster centers using k-means++ algorithm.\n");
    initKMeansPlusPlus(handle, params, data, centroidsRawData, workspace);
  } else if (params.init == KMeansParams::InitMethod::Array) {
    LOG(params.verbose,
        "KMeans.fit: initialize cluster centers from the ndarray array input "
        "passed to init arguement.\n");

    ASSERT(centroids != nullptr,
           "centroids array is null (require a valid array of centroids for "
           "the requested initialization method)");

    centroidsRawData.resize(params.n_clusters * n_features, stream);
    MLCommon::copy(centroidsRawData.begin(), centroids,
                   params.n_clusters * n_features, stream);

  } else {
    THROW("unknown initialization method to select initial centers");
  }

  fit(handle, params, data, centroidsRawData, inertia, n_iter, workspace);

  MLCommon::copy(centroids, centroidsRawData.data(),
                 params.n_clusters * n_features, stream);
}

template <typename DataT, typename IndexT = int>
void predict(const ML::cumlHandle_impl &handle, const KMeansParams &params,
             const DataT *cptr, const DataT *Xptr, const int n_samples,
             const int n_features, IndexT *labelsRawPtr, DataT &inertia) {
  cudaStream_t stream = handle.getStream();
  auto n_clusters = params.n_clusters;

  ASSERT(n_clusters > 0 && cptr != nullptr, "no clusters exist");

  ASSERT(memory_type(Xptr) == cudaMemoryTypeDevice,
         "input data must be device accessible");

  ASSERT(memory_type(cptr) == cudaMemoryTypeDevice,
         "centroid data must be device accessible");

  MLCommon::Distance::DistanceType metric =
    static_cast<MLCommon::Distance::DistanceType>(params.metric);

  Tensor<DataT, 2, IndexT> X((DataT *)Xptr, {n_samples, n_features});
  Tensor<DataT, 2, IndexT> centroids((DataT *)cptr, {n_clusters, n_features});

  auto dataBatchSize = kmeans::detail::getDataBatchSize(params, n_samples);

  // underlying expandable storage that holds labels
  MLCommon::device_buffer<IndexT> labelsRawData(handle.getDeviceAllocator(),
                                                stream);

  // Device-accessible allocation of expandable storage used as temorary buffers
  MLCommon::device_buffer<char> workspace(handle.getDeviceAllocator(), stream);

  Tensor<cub::KeyValuePair<IndexT, DataT>, 1> minClusterAndDistance(
    {n_samples}, handle.getDeviceAllocator(), stream);
  Tensor<DataT, 2, IndexT> pairwiseDistance(
    {dataBatchSize, n_clusters}, handle.getDeviceAllocator(), stream);

  // computes minClusterAndDistance[0:n_samples) where  minClusterAndDistance[i]
  // is a <key, value> pair where
  //   'key' is index to an sample in 'centroids' (index of the nearest
  //   centroid) and 'value' is the distance between the sample 'X[i]' and the
  //   'centroid[key]'
  kmeans::detail::minClusterAndDistance(handle, params, X, centroids,
                                        pairwiseDistance, minClusterAndDistance,
                                        workspace, metric, stream);

  // calculate cluster cost phi_x(C)
  cub::KeyValuePair<IndexT, DataT> *clusterCostD =
    (cub::KeyValuePair<IndexT, DataT> *)handle.getDeviceAllocator()->allocate(
      sizeof(cub::KeyValuePair<IndexT, DataT>), stream);

  kmeans::detail::computeClusterCost(
    handle, minClusterAndDistance, workspace, clusterCostD,
    [] __device__(const cub::KeyValuePair<IndexT, DataT> &a,
                  const cub::KeyValuePair<IndexT, DataT> &b) {
      cub::KeyValuePair<IndexT, DataT> res;
      res.key = 0;
      res.value = a.value + b.value;
      return res;
    },
    stream);

  MLCommon::copy(&inertia, &clusterCostD->value, 1, stream);

  labelsRawData.resize(n_samples, stream);

  auto labels = std::move(Tensor<IndexT, 1>(labelsRawData.data(), {n_samples}));
  ML::thrustAllocatorAdapter alloc(handle.getDeviceAllocator(), stream);
  auto execution_policy = thrust::cuda::par(alloc).on(stream);
  thrust::transform(
    execution_policy, minClusterAndDistance.begin(),
    minClusterAndDistance.end(), labels.begin(),
    [=] __device__(cub::KeyValuePair<IndexT, DataT> pair) { return pair.key; });

  handle.getDeviceAllocator()->deallocate(
    clusterCostD, sizeof(cub::KeyValuePair<IndexT, DataT>), stream);

  MLCommon::copy(labelsRawPtr, labelsRawData.data(), n_samples, stream);
}

template <typename DataT, typename IndexT = int>
void transform(const ML::cumlHandle_impl &handle, const KMeansParams &params,
               const DataT *cptr, const DataT *Xptr, int n_samples,
               int n_features, int transform_metric, DataT *X_new) {
  cudaStream_t stream = handle.getStream();
  auto n_clusters = params.n_clusters;
  MLCommon::Distance::DistanceType metric =
    static_cast<MLCommon::Distance::DistanceType>(transform_metric);

  ASSERT(n_clusters > 0 && cptr != nullptr, "no clusters exist");

  ASSERT(memory_type(Xptr) == cudaMemoryTypeDevice,
         "input data must be device accessible");

  ASSERT(memory_type(cptr) == cudaMemoryTypeDevice,
         "centroid data must be device accessible");

  ASSERT(memory_type(X_new) == cudaMemoryTypeDevice,
         "output data storage must be device accessible");

  Tensor<DataT, 2, IndexT> dataset((DataT *)Xptr, {n_samples, n_features});
  Tensor<DataT, 2, IndexT> centroids((DataT *)cptr, {n_clusters, n_features});
  Tensor<DataT, 2, IndexT> pairwiseDistance((DataT *)X_new,
                                            {n_samples, n_clusters});

  // Device-accessible allocation of expandable storage used as temorary buffers
  MLCommon::device_buffer<char> workspace(handle.getDeviceAllocator(), stream);

  auto dataBatchSize = kmeans::detail::getDataBatchSize(params, n_samples);

  // tile over the input data and calculate distance matrix [n_samples x
  // n_clusters]
  for (int dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
    // # of samples for the current batch
    int ns = std::min(dataBatchSize, n_samples - dIdx);

    // datasetView [ns x n_features] - view representing the current batch of
    // input dataset
    auto datasetView = dataset.template view<2>({ns, n_features}, {dIdx, 0});

    // pairwiseDistanceView [ns x n_clusters]
    auto pairwiseDistanceView =
      pairwiseDistance.template view<2>({ns, n_clusters}, {dIdx, 0});

    // calculate pairwise distance between cluster centroids and current batch
    // of input dataset
    kmeans::detail::pairwiseDistance(handle, datasetView, centroids,
                                     pairwiseDistanceView, workspace, metric,
                                     stream);
  }
}

};  // namespace kmeans
};  // end namespace ML
