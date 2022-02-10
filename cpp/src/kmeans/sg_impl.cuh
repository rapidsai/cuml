/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include "common.cuh"
#include <cuml/cluster/kmeans.hpp>
#include <raft/cudart_utils.h>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace ML {

namespace kmeans {

namespace impl {

// Selects 'n_clusters' samples randomly from X
template <typename DataT, typename IndexT>
void initRandom(const raft::handle_t& handle,
                const KMeansParams& params,
                Tensor<DataT, 2, IndexT>& X,
                rmm::device_uvector<DataT>& centroidsRawData)
{
  cudaStream_t stream = handle.get_stream();
  auto n_features     = X.getSize(1);
  auto n_clusters     = params.n_clusters;
  // allocate centroids buffer
  centroidsRawData.resize(n_clusters * n_features, stream);
  auto centroids =
    std::move(Tensor<DataT, 2, IndexT>(centroidsRawData.data(), {n_clusters, n_features}));

  kmeans::detail::shuffleAndGather(handle, X, centroids, n_clusters, params.seed, stream);
}

template <typename DataT, typename IndexT>
void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         Tensor<DataT, 2, IndexT>& X,
         Tensor<DataT, 1, IndexT>& weight,
         rmm::device_uvector<DataT>& centroidsRawData,
         DataT& inertia,
         int& n_iter,
         rmm::device_uvector<char>& workspace)
{
  ML::Logger::get().setLevel(params.verbosity);
  cudaStream_t stream = handle.get_stream();
  auto n_samples      = X.getSize(0);
  auto n_features     = X.getSize(1);
  auto n_clusters     = params.n_clusters;

  raft::distance::DistanceType metric = static_cast<raft::distance::DistanceType>(params.metric);

  // stores (key, value) pair corresponding to each sample where
  //   - key is the index of nearest cluster
  //   - value is the distance to the nearest cluster
  Tensor<cub::KeyValuePair<IndexT, DataT>, 1, IndexT> minClusterAndDistance({n_samples}, stream);

  // temporary buffer to store L2 norm of centroids or distance matrix,
  // destructor releases the resource
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // temporary buffer to store intermediate centroids, destructor releases the
  // resource
  Tensor<DataT, 2, IndexT> newCentroids({n_clusters, n_features}, stream);

  // temporary buffer to store weights per cluster, destructor releases the
  // resource
  Tensor<DataT, 1, IndexT> wtInCluster({n_clusters}, stream);

  rmm::device_scalar<cub::KeyValuePair<IndexT, DataT>> clusterCostD(stream);

  // L2 norm of X: ||x||^2
  Tensor<DataT, 1> L2NormX({n_samples}, stream);
  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm(
      L2NormX.data(), X.data(), X.getSize(1), X.getSize(0), raft::linalg::L2Norm, true, stream);
  }

  LOG(handle,
      "Calling KMeans.fit with %d samples of input data and the initialized "
      "cluster centers",
      n_samples);

  DataT priorClusteringCost = 0;
  for (n_iter = 1; n_iter <= params.max_iter; ++n_iter) {
    LOG(handle,
        "KMeans.fit: Iteration-%d: fitting the model using the initialized "
        "cluster centers",
        n_iter);

    auto centroids =
      std::move(Tensor<DataT, 2, IndexT>(centroidsRawData.data(), {n_clusters, n_features}));

    // computes minClusterAndDistance[0:n_samples) where
    // minClusterAndDistance[i] is a <key, value> pair where
    //   'key' is index to an sample in 'centroids' (index of the nearest
    //   centroid) and 'value' is the distance between the sample 'X[i]' and the
    //   'centroid[key]'
    kmeans::detail::minClusterAndDistance(handle,
                                          params,
                                          X,
                                          centroids,
                                          minClusterAndDistance,
                                          L2NormX,
                                          L2NormBuf_OR_DistBuf,
                                          workspace,
                                          metric,
                                          stream);

    // Using TransformInputIteratorT to dereference an array of
    // cub::KeyValuePair and converting them to just return the Key to be used
    // in reduce_rows_by_key prims
    kmeans::detail::KeyValueIndexOp<IndexT, DataT> conversion_op;
    cub::TransformInputIterator<IndexT,
                                kmeans::detail::KeyValueIndexOp<IndexT, DataT>,
                                cub::KeyValuePair<IndexT, DataT>*>
      itr(minClusterAndDistance.data(), conversion_op);

    workspace.resize(n_samples, stream);

    // Calculates weighted sum of all the samples assigned to cluster-i and store the
    // result in newCentroids[i]
    raft::linalg::reduce_rows_by_key(X.data(),
                                     X.getSize(1),
                                     itr,
                                     weight.data(),
                                     workspace.data(),
                                     X.getSize(0),
                                     X.getSize(1),
                                     n_clusters,
                                     newCentroids.data(),
                                     stream);

    // Reduce weights by key to compute weight in each cluster
    raft::linalg::reduce_cols_by_key(
      weight.data(), itr, wtInCluster.data(), 1, weight.getSize(0), n_clusters, stream);

    // Computes newCentroids[i] = newCentroids[i]/wtInCluster[i] where
    //   newCentroids[n_clusters x n_features] - 2D array, newCentroids[i] has sum of all the
    //   samples assigned to cluster-i wtInCluster[n_clusters] - 1D array, wtInCluster[i] contains #
    //   of samples in cluster-i.
    // Note - when wtInCluster[i] is 0, newCentroid[i] is reset to 0
    raft::linalg::matrixVectorOp(
      newCentroids.data(),
      newCentroids.data(),
      wtInCluster.data(),
      newCentroids.getSize(1),
      newCentroids.getSize(0),
      true,
      false,
      [=] __device__(DataT mat, DataT vec) {
        if (vec == 0)
          return DataT(0);
        else
          return mat / vec;
      },
      stream);

    // copy centroids[i] to newCentroids[i] when wtInCluster[i] is 0
    cub::ArgIndexInputIterator<DataT*> itr_wt(wtInCluster.data());
    MLCommon::Matrix::gather_if(
      centroids.data(),
      centroids.getSize(1),
      centroids.getSize(0),
      itr_wt,
      itr_wt,
      wtInCluster.numElements(),
      newCentroids.data(),
      [=] __device__(cub::KeyValuePair<ptrdiff_t, DataT> map) {  // predicate
        // copy when the # of samples in the cluster is 0
        if (map.value == 0)
          return true;
        else
          return false;
      },
      [=] __device__(cub::KeyValuePair<ptrdiff_t, DataT> map) {  // map
        return map.key;
      },
      stream);

    // compute the squared norm between the newCentroids and the original
    // centroids, destructor releases the resource
    Tensor<DataT, 1> sqrdNorm({1}, stream);
    raft::linalg::mapThenSumReduce(
      sqrdNorm.data(),
      newCentroids.numElements(),
      [=] __device__(const DataT a, const DataT b) {
        DataT diff = a - b;
        return diff * diff;
      },
      stream,
      centroids.data(),
      newCentroids.data());

    DataT sqrdNormError = 0;
    raft::copy(&sqrdNormError, sqrdNorm.data(), sqrdNorm.numElements(), stream);

    raft::copy(centroidsRawData.data(), newCentroids.data(), newCentroids.numElements(), stream);

    bool done = false;
    if (params.inertia_check) {
      // calculate cluster cost phi_x(C)
      kmeans::detail::computeClusterCost(
        handle,
        minClusterAndDistance,
        workspace,
        clusterCostD.data(),
        [] __device__(const cub::KeyValuePair<IndexT, DataT>& a,
                      const cub::KeyValuePair<IndexT, DataT>& b) {
          cub::KeyValuePair<IndexT, DataT> res;
          res.key   = 0;
          res.value = a.value + b.value;
          return res;
        },
        stream);

      DataT curClusteringCost = 0;
      raft::copy(&curClusteringCost, &(clusterCostD.data()->value), 1, stream);

      handle.sync_stream(stream);
      ASSERT(curClusteringCost != (DataT)0.0,
             "Too few points and centriods being found is getting 0 cost from "
             "centers");

      if (n_iter > 1) {
        DataT delta = curClusteringCost / priorClusteringCost;
        if (delta > 1 - params.tol) done = true;
      }
      priorClusteringCost = curClusteringCost;
    }

    handle.sync_stream(stream);
    if (sqrdNormError < params.tol) done = true;

    if (done) {
      LOG(handle, "Threshold triggered after %d iterations. Terminating early.", n_iter);
      break;
    }
  }

  auto centroids =
    std::move(Tensor<DataT, 2, IndexT>(centroidsRawData.data(), {n_clusters, n_features}));

  kmeans::detail::minClusterAndDistance(handle,
                                        params,
                                        X,
                                        centroids,
                                        minClusterAndDistance,
                                        L2NormX,
                                        L2NormBuf_OR_DistBuf,
                                        workspace,
                                        metric,
                                        stream);

  thrust::transform(handle.get_thrust_policy(),
                    minClusterAndDistance.begin(),
                    minClusterAndDistance.end(),
                    weight.data(),
                    minClusterAndDistance.begin(),
                    [=] __device__(const cub::KeyValuePair<IndexT, DataT> kvp, DataT wt) {
                      cub::KeyValuePair<IndexT, DataT> res;
                      res.value = kvp.value * wt;
                      res.key   = kvp.key;
                      return res;
                    });

  // calculate cluster cost phi_x(C)
  kmeans::detail::computeClusterCost(
    handle,
    minClusterAndDistance,
    workspace,
    clusterCostD.data(),
    [] __device__(const cub::KeyValuePair<IndexT, DataT>& a,
                  const cub::KeyValuePair<IndexT, DataT>& b) {
      cub::KeyValuePair<IndexT, DataT> res;
      res.key   = 0;
      res.value = a.value + b.value;
      return res;
    },
    stream);

  raft::copy(&inertia, &(clusterCostD.data()->value), 1, stream);

  LOG(handle,
      "KMeans.fit: completed after %d iterations with %f inertia ",
      n_iter > params.max_iter ? n_iter - 1 : n_iter,
      inertia);
}

template <typename DataT, typename IndexT>
void initKMeansPlusPlus(const raft::handle_t& handle,
                        const KMeansParams& params,
                        Tensor<DataT, 2, IndexT>& X,
                        rmm::device_uvector<DataT>& centroidsRawData,
                        rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream                 = handle.get_stream();
  auto n_samples                      = X.getSize(0);
  auto n_features                     = X.getSize(1);
  auto n_clusters                     = params.n_clusters;
  raft::distance::DistanceType metric = static_cast<raft::distance::DistanceType>(params.metric);
  centroidsRawData.resize(n_clusters * n_features, stream);
  kmeans::detail::kmeansPlusPlus(handle, params, X, metric, workspace, centroidsRawData, stream);
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
void initScalableKMeansPlusPlus(const raft::handle_t& handle,
                                const KMeansParams& params,
                                Tensor<DataT, 2, IndexT>& X,
                                rmm::device_uvector<DataT>& centroidsRawData,
                                rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream                 = handle.get_stream();
  auto n_samples                      = X.getSize(0);
  auto n_features                     = X.getSize(1);
  auto n_clusters                     = params.n_clusters;
  raft::distance::DistanceType metric = static_cast<raft::distance::DistanceType>(params.metric);

  raft::random::Rng rng(params.seed, raft::random::GeneratorType::GenPhilox);

  // <<<< Step-1 >>> : C <- sample a point uniformly at random from X
  std::mt19937 gen(params.seed);
  std::uniform_int_distribution<> dis(0, n_samples - 1);

  int cIdx             = dis(gen);
  auto initialCentroid = X.template view<2>({1, n_features}, {cIdx, 0});

  // flag the sample that is chosen as initial centroid
  std::vector<int> h_isSampleCentroid(n_samples);
  std::fill(h_isSampleCentroid.begin(), h_isSampleCentroid.end(), 0);
  h_isSampleCentroid[cIdx] = 1;

  // device buffer to flag the sample that is chosen as initial centroid
  Tensor<int, 1> isSampleCentroid({n_samples}, stream);

  raft::copy(
    isSampleCentroid.data(), h_isSampleCentroid.data(), isSampleCentroid.numElements(), stream);

  rmm::device_uvector<DataT> centroidsBuf(0, stream);

  // reset buffer to store the chosen centroid
  centroidsBuf.resize(initialCentroid.numElements(), stream);
  raft::copy(centroidsBuf.begin(), initialCentroid.data(), initialCentroid.numElements(), stream);

  auto potentialCentroids = std::move(Tensor<DataT, 2, IndexT>(
    centroidsBuf.data(), {initialCentroid.getSize(0), initialCentroid.getSize(1)}));
  // <<< End of Step-1 >>>

  // temporary buffer to store L2 norm of centroids or distance matrix,
  // destructor releases the resource
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // L2 norm of X: ||x||^2
  Tensor<DataT, 1> L2NormX({n_samples}, stream);
  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm(
      L2NormX.data(), X.data(), X.getSize(1), X.getSize(0), raft::linalg::L2Norm, true, stream);
  }

  Tensor<DataT, 1, IndexT> minClusterDistance({n_samples}, stream);
  Tensor<DataT, 1, IndexT> uniformRands({n_samples}, stream);
  rmm::device_scalar<DataT> clusterCost(stream);

  // <<< Step-2 >>>: psi <- phi_X (C)
  kmeans::detail::minClusterDistance(handle,
                                     params,
                                     X,
                                     potentialCentroids,
                                     minClusterDistance,
                                     L2NormX,
                                     L2NormBuf_OR_DistBuf,
                                     workspace,
                                     metric,
                                     stream);

  // compute partial cluster cost from the samples in rank
  kmeans::detail::computeClusterCost(
    handle,
    minClusterDistance,
    workspace,
    clusterCost.data(),
    [] __device__(const DataT& a, const DataT& b) { return a + b; },
    stream);

  DataT psi = 0;
  psi       = clusterCost.value(stream);

  // <<< End of Step-2 >>>

  // Scalable kmeans++ paper claims 8 rounds is sufficient
  handle.sync_stream(stream);
  int niter = std::min(8, (int)ceil(log(psi)));
  LOG(handle, "KMeans||: psi = %g, log(psi) = %g, niter = %d ", psi, log(psi), niter);

  // <<<< Step-3 >>> : for O( log(psi) ) times do
  for (int iter = 0; iter < niter; ++iter) {
    LOG(handle,
        "KMeans|| - Iteration %d: # potential centroids sampled - %d",
        iter,
        potentialCentroids.getSize(0));

    kmeans::detail::minClusterDistance(handle,
                                       params,
                                       X,
                                       potentialCentroids,
                                       minClusterDistance,
                                       L2NormX,
                                       L2NormBuf_OR_DistBuf,
                                       workspace,
                                       metric,
                                       stream);

    kmeans::detail::computeClusterCost(
      handle,
      minClusterDistance,
      workspace,
      clusterCost.data(),
      [] __device__(const DataT& a, const DataT& b) { return a + b; },
      stream);

    psi = clusterCost.value(stream);

    // <<<< Step-4 >>> : Sample each point x in X independently and identify new
    // potentialCentroids
    rng.uniform(uniformRands.data(), uniformRands.getSize(0), (DataT)0, (DataT)1, stream);

    kmeans::detail::SamplingOp<DataT> select_op(
      psi, params.oversampling_factor, n_clusters, uniformRands.data(), isSampleCentroid.data());

    auto Cp = kmeans::detail::sampleCentroids(
      handle, X, minClusterDistance, isSampleCentroid, select_op, workspace, stream);
    /// <<<< End of Step-4 >>>>

    /// <<<< Step-5 >>> : C = C U C'
    // append the data in Cp to the buffer holding the potentialCentroids
    centroidsBuf.resize(centroidsBuf.size() + Cp.numElements(), stream);
    raft::copy(centroidsBuf.end() - Cp.numElements(), Cp.data(), Cp.numElements(), stream);

    int tot_centroids = potentialCentroids.getSize(0) + Cp.getSize(0);
    potentialCentroids =
      std::move(Tensor<DataT, 2, IndexT>(centroidsBuf.data(), {tot_centroids, n_features}));
    /// <<<< End of Step-5 >>>
  }  /// <<<< Step-6 >>>

  LOG(handle, "KMeans||: total # potential centroids sampled - %d", potentialCentroids.getSize(0));

  if (potentialCentroids.getSize(0) > n_clusters) {
    // <<< Step-7 >>>: For x in C, set w_x to be the number of pts closest to X
    // temporary buffer to store the sample count per cluster, destructor
    // releases the resource
    Tensor<DataT, 1, IndexT> weight({potentialCentroids.getSize(0)}, stream);

    kmeans::detail::countSamplesInCluster(
      handle, params, X, L2NormX, potentialCentroids, workspace, metric, weight, stream);

    // <<< end of Step-7 >>>

    // Step-8: Recluster the weighted points in C into k clusters
    centroidsRawData.resize(n_clusters * n_features, stream);
    kmeans::detail::kmeansPlusPlus(
      handle, params, potentialCentroids, metric, workspace, centroidsRawData, stream);

    DataT inertia = 0;
    int n_iter    = 0;
    KMeansParams default_params;
    default_params.n_clusters = params.n_clusters;

    ML::kmeans::impl::fit(handle,
                          default_params,
                          potentialCentroids,
                          weight,
                          centroidsRawData,
                          inertia,
                          n_iter,
                          workspace);

  } else if (potentialCentroids.getSize(0) < n_clusters) {
    // supplement with random
    auto n_random_clusters = n_clusters - potentialCentroids.getSize(0);

    LOG(handle,
        "[Warning!] KMeans||: found fewer than %d centroids during "
        "initialization (found %d centroids, remaining %d centroids will be "
        "chosen randomly from input samples)",
        n_clusters,
        potentialCentroids.getSize(0),
        n_random_clusters);

    // reset buffer to store the chosen centroid
    centroidsRawData.resize(n_clusters * n_features, stream);

    // generate `n_random_clusters` centroids
    KMeansParams rand_params;
    rand_params.init       = KMeansParams::InitMethod::Random;
    rand_params.n_clusters = n_random_clusters;
    initRandom(handle, rand_params, X, centroidsRawData);

    // copy centroids generated during kmeans|| iteration to the buffer
    raft::copy(centroidsRawData.data() + n_random_clusters * n_features,
               potentialCentroids.data(),
               potentialCentroids.numElements(),
               stream);
  } else {
    // found the required n_clusters
    centroidsRawData.resize(n_clusters * n_features, stream);
    raft::copy(
      centroidsRawData.data(), potentialCentroids.data(), potentialCentroids.numElements(), stream);
  }
}

template <typename DataT, typename IndexT = int>
void fit(const raft::handle_t& handle,
         const KMeansParams& km_params,
         const DataT* X,
         const int n_samples,
         const int n_features,
         const DataT* sample_weight,
         DataT* centroids,
         DataT& inertia,
         int& n_iter)
{
  ML::Logger::get().setLevel(km_params.verbosity);
  cudaStream_t stream = handle.get_stream();

  ASSERT(n_samples > 0, "# of samples must be > 0");

  ASSERT(km_params.oversampling_factor >= 0,
         "oversampling factor must be >= 0 (requested %f)",
         km_params.oversampling_factor);

  ASSERT(is_device_or_managed_type(X), "input data must be device accessible");

  Tensor<DataT, 2, IndexT> data((DataT*)X, {n_samples, n_features});

  Tensor<DataT, 1, IndexT> weight({n_samples}, stream);
  if (sample_weight != nullptr) {
    raft::copy(weight.data(), sample_weight, n_samples, stream);
  } else {
    thrust::fill(handle.get_thrust_policy(), weight.begin(), weight.end(), 1);
  }

  // underlying expandable storage that holds centroids data
  rmm::device_uvector<DataT> centroidsRawData(0, stream);

  // Device-accessible allocation of expandable storage used as temorary buffers
  rmm::device_uvector<char> workspace(0, stream);

  // check if weights sum up to n_samples
  kmeans::detail::checkWeights(handle, workspace, weight, stream);

  auto n_init = km_params.n_init;
  if (km_params.init == KMeansParams::InitMethod::Array && n_init != 1) {
    LOG(handle,
        "Explicit initial center position passed: performing only one init in "
        "k-means instead of n_init=%d",
        n_init);
    n_init = 1;
  }

  std::mt19937 gen(km_params.seed);
  inertia = std::numeric_limits<DataT>::max();

  // run k-means algorithm with different seeds
  for (auto seed_iter = 0; seed_iter < n_init; ++seed_iter) {
    // generate KMeansParams with different seed
    KMeansParams params = km_params;
    params.seed         = gen();

    DataT _inertia = std::numeric_limits<DataT>::max();
    int _n_iter    = 0;

    if (params.init == KMeansParams::InitMethod::Random) {
      // initializing with random samples from input dataset
      LOG(handle,
          "\n\nKMeans.fit (Iteration-%d/%d): initialize cluster centers by "
          "randomly choosing from the "
          "input data.",
          seed_iter + 1,
          n_init);
      initRandom(handle, params, data, centroidsRawData);
    } else if (params.init == KMeansParams::InitMethod::KMeansPlusPlus) {
      // default method to initialize is kmeans++
      LOG(handle,
          "\n\nKMeans.fit (Iteration-%d/%d): initialize cluster centers using "
          "k-means++ algorithm.",
          seed_iter + 1,
          n_init);
      if (params.oversampling_factor == 0)
        initKMeansPlusPlus(handle, params, data, centroidsRawData, workspace);
      else
        initScalableKMeansPlusPlus(handle, params, data, centroidsRawData, workspace);
    } else if (params.init == KMeansParams::InitMethod::Array) {
      LOG(handle,
          "\n\nKMeans.fit (Iteration-%d/%d): initialize cluster centers from "
          "the ndarray array input "
          "passed to init arguement.",
          seed_iter + 1,
          n_init);

      ASSERT(centroids != nullptr,
             "centroids array is null (require a valid array of centroids for "
             "the requested initialization method)");

      centroidsRawData.resize(params.n_clusters * n_features, stream);
      raft::copy(centroidsRawData.begin(), centroids, params.n_clusters * n_features, stream);

    } else {
      THROW("unknown initialization method to select initial centers");
    }

    fit(handle, params, data, weight, centroidsRawData, _inertia, _n_iter, workspace);

    if (_inertia < inertia) {
      inertia = _inertia;
      n_iter  = _n_iter;
      raft::copy(centroids, centroidsRawData.data(), params.n_clusters * n_features, stream);
    }

    LOG(handle,
        "KMeans.fit after iteration-%d/%d: inertia - %f, n_iter - %d",
        seed_iter + 1,
        n_init,
        inertia,
        n_iter);

    // auto centroidsT = std::move(Tensor<DataT, 2, IndexT>(
    //  centroids, {params.n_clusters, n_features}));
  }

  LOG(handle,
      "KMeans.fit: async call returned (fit could still be running on the "
      "device)");
}

template <typename DataT, typename IndexT = int>
void predict(const raft::handle_t& handle,
             const KMeansParams& params,
             const DataT* cptr,
             const DataT* Xptr,
             const int n_samples,
             const int n_features,
             const DataT* sample_weight,
             bool normalize_weights,
             IndexT* labelsRawPtr,
             DataT& inertia)
{
  ML::Logger::get().setLevel(params.verbosity);
  cudaStream_t stream = handle.get_stream();
  auto n_clusters     = params.n_clusters;

  ASSERT(n_clusters > 0 && cptr != nullptr, "no clusters exist");

  ASSERT(is_device_or_managed_type(Xptr), "input data must be device accessible");

  ASSERT(is_device_or_managed_type(cptr), "centroid data must be device accessible");

  raft::distance::DistanceType metric = static_cast<raft::distance::DistanceType>(params.metric);

  Tensor<DataT, 2, IndexT> X((DataT*)Xptr, {n_samples, n_features});
  Tensor<DataT, 2, IndexT> centroids((DataT*)cptr, {n_clusters, n_features});

  Tensor<DataT, 1, IndexT> weight({n_samples}, stream);
  if (sample_weight != nullptr) {
    raft::copy(weight.data(), sample_weight, n_samples, stream);
  } else {
    thrust::fill(handle.get_thrust_policy(), weight.begin(), weight.end(), 1);
  }

  // underlying expandable storage that holds labels
  rmm::device_uvector<IndexT> labelsRawData(0, stream);

  // Device-accessible allocation of expandable storage used as temorary buffers
  rmm::device_uvector<char> workspace(0, stream);

  // check if weights sum up to n_samples
  if (normalize_weights) kmeans::detail::checkWeights(handle, workspace, weight, stream);

  Tensor<cub::KeyValuePair<IndexT, DataT>, 1> minClusterAndDistance({n_samples}, stream);

  // temporary buffer to store L2 norm of centroids or distance matrix,
  // destructor releases the resource
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // L2 norm of X: ||x||^2
  Tensor<DataT, 1> L2NormX({n_samples}, stream);
  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm(
      L2NormX.data(), X.data(), X.getSize(1), X.getSize(0), raft::linalg::L2Norm, true, stream);
  }

  // computes minClusterAndDistance[0:n_samples) where  minClusterAndDistance[i]
  // is a <key, value> pair where
  //   'key' is index to an sample in 'centroids' (index of the nearest
  //   centroid) and 'value' is the distance between the sample 'X[i]' and the
  //   'centroid[key]'
  kmeans::detail::minClusterAndDistance(handle,
                                        params,
                                        X,
                                        centroids,
                                        minClusterAndDistance,
                                        L2NormX,
                                        L2NormBuf_OR_DistBuf,
                                        workspace,
                                        metric,
                                        stream);

  // calculate cluster cost phi_x(C)
  rmm::device_scalar<cub::KeyValuePair<IndexT, DataT>> clusterCostD(stream);

  thrust::transform(handle.get_thrust_policy(),
                    minClusterAndDistance.begin(),
                    minClusterAndDistance.end(),
                    weight.data(),
                    minClusterAndDistance.begin(),
                    [=] __device__(const cub::KeyValuePair<IndexT, DataT> kvp, DataT wt) {
                      cub::KeyValuePair<IndexT, DataT> res;
                      res.value = kvp.value * wt;
                      res.key   = kvp.key;
                      return res;
                    });

  kmeans::detail::computeClusterCost(
    handle,
    minClusterAndDistance,
    workspace,
    clusterCostD.data(),
    [] __device__(const cub::KeyValuePair<IndexT, DataT>& a,
                  const cub::KeyValuePair<IndexT, DataT>& b) {
      cub::KeyValuePair<IndexT, DataT> res;
      res.key   = 0;
      res.value = a.value + b.value;
      return res;
    },
    stream);

  raft::copy(&inertia, &(clusterCostD.data()->value), 1, stream);

  labelsRawData.resize(n_samples, stream);

  thrust::transform(handle.get_thrust_policy(),
                    minClusterAndDistance.begin(),
                    minClusterAndDistance.end(),
                    labelsRawData.data(),
                    [=] __device__(cub::KeyValuePair<IndexT, DataT> pair) { return pair.key; });

  raft::copy(labelsRawPtr, labelsRawData.data(), n_samples, stream);
}

template <typename DataT, typename IndexT = int>
void transform(const raft::handle_t& handle,
               const KMeansParams& params,
               const DataT* cptr,
               const DataT* Xptr,
               int n_samples,
               int n_features,
               int transform_metric,
               DataT* X_new)
{
  ML::Logger::get().setLevel(params.verbosity);
  cudaStream_t stream                 = handle.get_stream();
  auto n_clusters                     = params.n_clusters;
  raft::distance::DistanceType metric = static_cast<raft::distance::DistanceType>(transform_metric);

  ASSERT(n_clusters > 0 && cptr != nullptr, "no clusters exist");

  ASSERT(is_device_or_managed_type(Xptr), "input data must be device accessible");

  ASSERT(is_device_or_managed_type(cptr), "centroid data must be device accessible");

  ASSERT(is_device_or_managed_type(X_new), "output data storage must be device accessible");

  Tensor<DataT, 2, IndexT> dataset((DataT*)Xptr, {n_samples, n_features});
  Tensor<DataT, 2, IndexT> centroids((DataT*)cptr, {n_clusters, n_features});
  Tensor<DataT, 2, IndexT> pairwiseDistance((DataT*)X_new, {n_samples, n_clusters});

  // Device-accessible allocation of expandable storage used as temorary buffers
  rmm::device_uvector<char> workspace(0, stream);

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
    auto pairwiseDistanceView = pairwiseDistance.template view<2>({ns, n_clusters}, {dIdx, 0});

    // calculate pairwise distance between cluster centroids and current batch
    // of input dataset
    kmeans::detail::pairwise_distance(
      handle, datasetView, centroids, pairwiseDistanceView, workspace, metric, stream);
  }
}

};  // namespace impl
};  // namespace kmeans
};  // end namespace ML
