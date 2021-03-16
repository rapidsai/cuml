/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <raft/cudart_utils.h>

#include <cuml/cluster/kmeans.hpp>

#include "common.cuh"
#include "sg_impl.cuh"

namespace ML {
namespace kmeans {
namespace opg {
namespace impl {

#define KMEANS_COMM_ROOT 0

// Selects 'n_clusters' samples randomly from X
template <typename DataT, typename IndexT>
void initRandom(const raft::handle_t &handle, const KMeansParams &params,
                Tensor<DataT, 2, IndexT> &X,
                MLCommon::device_buffer<DataT> &centroidsRawData) {
  const auto &comm = handle.get_comms();
  cudaStream_t stream = handle.get_stream();
  auto n_local_samples = X.getSize(0);
  auto n_features = X.getSize(1);
  auto n_clusters = params.n_clusters;

  const int my_rank = comm.get_rank();
  const int n_ranks = comm.get_size();

  // allocate centroids buffer
  centroidsRawData.resize(n_clusters * n_features, stream);
  auto centroids = std::move(Tensor<DataT, 2, IndexT>(
    centroidsRawData.data(), {n_clusters, n_features}));

  std::vector<int> nCentroidsSampledByRank(n_ranks, 0);
  std::vector<size_t> nCentroidsElementsToReceiveFromRank(n_ranks, 0);

  const int nranks_reqd = std::min(n_ranks, n_clusters);
  ASSERT(KMEANS_COMM_ROOT < nranks_reqd,
         "KMEANS_COMM_ROOT must be in [0,  %d)\n", nranks_reqd);

  for (int rank = 0; rank < nranks_reqd; ++rank) {
    int nCentroidsSampledInRank = n_clusters / nranks_reqd;
    if (rank == KMEANS_COMM_ROOT) {
      nCentroidsSampledInRank +=
        n_clusters - nCentroidsSampledInRank * nranks_reqd;
    }
    nCentroidsSampledByRank[rank] = nCentroidsSampledInRank;
    nCentroidsElementsToReceiveFromRank[rank] =
      nCentroidsSampledInRank * n_features;
  }

  int nCentroidsSampledInRank = nCentroidsSampledByRank[my_rank];
  ASSERT(nCentroidsSampledInRank <= n_local_samples,
         "# random samples requested from rank-%d is larger than the available "
         "samples at the rank (requested is %d, available is %d)",
         my_rank, nCentroidsSampledInRank, n_local_samples);

  Tensor<DataT, 2, IndexT> centroidsSampledInRank(
    {nCentroidsSampledInRank, n_features}, handle.get_device_allocator(),
    stream);

  kmeans::detail::shuffleAndGather(handle, X, centroidsSampledInRank,
                                   nCentroidsSampledInRank, params.seed,
                                   stream);

  std::vector<size_t> displs(n_ranks);
  thrust::exclusive_scan(
    thrust::host, nCentroidsElementsToReceiveFromRank.begin(),
    nCentroidsElementsToReceiveFromRank.end(), displs.begin());

  // gather centroids from all ranks
  comm.allgatherv<DataT>(
    centroidsSampledInRank.data(),               // sendbuff
    centroids.data(),                            // recvbuff
    nCentroidsElementsToReceiveFromRank.data(),  // recvcount
    displs.data(), stream);
}

/*
* @brief Selects 'n_clusters' samples from X using scalable kmeans++ algorithm
* Scalable kmeans++ pseudocode
* 1: C = sample a point uniformly at random from X
* 2: psi = phi_X (C)
* 3: for O( log(psi) ) times do
* 4:   C' = sample each point x in X independently with probability
*           p_x = l * ( d^2(x, C) / phi_X (C) )
* 5:   C = C U C'
* 6: end for
* 7: For x in C, set w_x to be the number of points in X closer to x than any
*    other point in C
* 8: Recluster the weighted points in C into k clusters
*/
template <typename DataT, typename IndexT>
void initKMeansPlusPlus(const raft::handle_t &handle,
                        const KMeansParams &params, Tensor<DataT, 2, IndexT> &X,
                        MLCommon::device_buffer<DataT> &centroidsRawData,
                        MLCommon::device_buffer<char> &workspace) {
  const auto &comm = handle.get_comms();
  cudaStream_t stream = handle.get_stream();
  const int my_rank = comm.get_rank();
  const int n_rank = comm.get_size();

  auto n_samples = X.getSize(0);
  auto n_features = X.getSize(1);
  auto n_clusters = params.n_clusters;
  raft::distance::DistanceType metric =
    static_cast<raft::distance::DistanceType>(params.metric);

  raft::random::Rng rng(params.seed, raft::random::GeneratorType::GenPhilox);

  // <<<< Step-1 >>> : C <- sample a point uniformly at random from X
  //    1.1 - Select a rank r' at random from the available n_rank ranks with a
  //          probability of 1/n_rank [Note - with same seed all rank selects
  //          the same r' which avoids a call to comm]
  //    1.2 - Rank r' samples a point uniformly at random from the local dataset
  //          X which will be used as the initial centroid for kmeans++
  //    1.3 - Communicate the initial centroid chosen by rank-r' to all other
  //          ranks
  std::mt19937 gen(params.seed);
  std::uniform_int_distribution<> dis(0, n_rank - 1);
  int rp = dis(gen);

  // buffer to flag the sample that is chosen as initial centroids
  MLCommon::host_buffer<int> h_isSampleCentroid(handle.get_host_allocator(),
                                                stream, n_samples);
  std::fill(h_isSampleCentroid.begin(), h_isSampleCentroid.end(), 0);

  MLCommon::host_buffer<int> nPtsSampledByRank(handle.get_host_allocator(),
                                               stream, n_rank);

  Tensor<DataT, 2, IndexT> initialCentroid(
    {1, n_features}, handle.get_device_allocator(), stream);
  LOG(handle, "@Rank-%d : KMeans|| : initial centroid is sampled at rank-%d\n",
      my_rank, rp);

  //    1.2 - Rank r' samples a point uniformly at random from the local dataset
  //          X which will be used as the initial centroid for kmeans++
  if (my_rank == rp) {
    std::mt19937 gen(params.seed);
    std::uniform_int_distribution<> dis(0, n_samples - 1);

    int cIdx = dis(gen);
    auto centroidsView = X.template view<2>({1, n_features}, {cIdx, 0});

    raft::copy(initialCentroid.data(), centroidsView.data(),
               centroidsView.numElements(), stream);

    h_isSampleCentroid[cIdx] = 1;
  }

  // 1.3 - Communicate the initial centroid chosen by rank-r' to all other ranks
  comm.bcast<DataT>(initialCentroid.data(), initialCentroid.numElements(), rp,
                    stream);

  // device buffer to flag the sample that is chosen as initial centroid
  Tensor<int, 1> isSampleCentroid({n_samples}, handle.get_device_allocator(),
                                  stream);

  raft::copy(isSampleCentroid.data(), h_isSampleCentroid.data(),
             isSampleCentroid.numElements(), stream);

  MLCommon::device_buffer<DataT> centroidsBuf(handle.get_device_allocator(),
                                              stream);

  // reset buffer to store the chosen centroid
  centroidsBuf.reserve(n_clusters * n_features, stream);
  centroidsBuf.resize(initialCentroid.numElements(), stream);
  raft::copy(centroidsBuf.begin(), initialCentroid.data(),
             initialCentroid.numElements(), stream);

  auto potentialCentroids = std::move(Tensor<DataT, 2, IndexT>(
    centroidsBuf.data(),
    {initialCentroid.getSize(0), initialCentroid.getSize(1)}));
  // <<< End of Step-1 >>>

  MLCommon::device_buffer<DataT> L2NormBuf_OR_DistBuf(
    handle.get_device_allocator(), stream);

  // L2 norm of X: ||x||^2
  Tensor<DataT, 1> L2NormX({n_samples}, handle.get_device_allocator(), stream);
  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm(L2NormX.data(), X.data(), X.getSize(1), X.getSize(0),
                          raft::linalg::L2Norm, true, stream);
  }

  Tensor<DataT, 1, IndexT> minClusterDistance(
    {n_samples}, handle.get_device_allocator(), stream);
  Tensor<DataT, 1, IndexT> uniformRands({n_samples},
                                        handle.get_device_allocator(), stream);

  // <<< Step-2 >>>: psi <- phi_X (C)
  MLCommon::device_buffer<DataT> clusterCost(handle.get_device_allocator(),
                                             stream, 1);

  kmeans::detail::minClusterDistance(
    handle, params, X, potentialCentroids, minClusterDistance, L2NormX,
    L2NormBuf_OR_DistBuf, workspace, metric, stream);

  // compute partial cluster cost from the samples in rank
  kmeans::detail::computeClusterCost(
    handle, minClusterDistance, workspace, clusterCost.data(),
    [] __device__(const DataT &a, const DataT &b) { return a + b; }, stream);

  // compute total cluster cost by accumulating the partial cost from all the
  // ranks
  comm.allreduce(clusterCost.data(), clusterCost.data(), clusterCost.size(),
                 raft::comms::op_t::SUM, stream);

  DataT psi = 0;
  raft::copy(&psi, clusterCost.data(), clusterCost.size(), stream);

  // <<< End of Step-2 >>>

  ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
         "An error occurred in the distributed operation. This can result from "
         "a failed rank");

  // Scalable kmeans++ paper claims 8 rounds is sufficient
  int niter = std::min(8, (int)ceil(log(psi)));
  LOG(handle,
      "@Rank-%d:KMeans|| :phi - %f, max # of iterations for kmeans++ loop - "
      "%d\n",
      my_rank, psi, niter);

  // <<<< Step-3 >>> : for O( log(psi) ) times do
  for (int iter = 0; iter < niter; ++iter) {
    LOG(handle,
        "@Rank-%d:KMeans|| - Iteration %d: # potential centroids sampled - "
        "%d\n",
        my_rank, iter, potentialCentroids.getSize(0));

    kmeans::detail::minClusterDistance(
      handle, params, X, potentialCentroids, minClusterDistance, L2NormX,
      L2NormBuf_OR_DistBuf, workspace, metric, stream);

    kmeans::detail::computeClusterCost(
      handle, minClusterDistance, workspace, clusterCost.data(),
      [] __device__(const DataT &a, const DataT &b) { return a + b; }, stream);
    comm.allreduce(clusterCost.data(), clusterCost.data(), clusterCost.size(),
                   raft::comms::op_t::SUM, stream);
    raft::copy(&psi, clusterCost.data(), clusterCost.size(), stream);
    ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed operation. This can result "
           "from a failed rank");

    // <<<< Step-4 >>> : Sample each point x in X independently and identify new
    // potentialCentroids
    rng.uniform(uniformRands.data(), uniformRands.getSize(0), (DataT)0,
                (DataT)1, stream);
    kmeans::detail::SamplingOp<DataT> select_op(psi, params.oversampling_factor,
                                                n_clusters, uniformRands.data(),
                                                isSampleCentroid.data());

    auto inRankCp = kmeans::detail::sampleCentroids(
      handle, X, minClusterDistance, isSampleCentroid, select_op, workspace,
      stream);
    /// <<<< End of Step-4 >>>>

    /// <<<< Step-5 >>> : C = C U C'
    // append the data in Cp from all ranks to the buffer holding the
    // potentialCentroids
    std::fill(nPtsSampledByRank.begin(), nPtsSampledByRank.end(), 0);
    nPtsSampledByRank[my_rank] = inRankCp.getSize(0);
    comm.allgather(&nPtsSampledByRank[my_rank], nPtsSampledByRank.data(), 1,
                   stream);

    ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed operation. This can result "
           "from a failed rank");

    int nPtsSampled = thrust::reduce(thrust::host, nPtsSampledByRank.begin(),
                                     nPtsSampledByRank.end(), 0);

    // gather centroids from all ranks
    std::vector<size_t> sizes(n_rank);
    thrust::transform(thrust::host, nPtsSampledByRank.begin(),
                      nPtsSampledByRank.end(), sizes.begin(),
                      [&](int val) { return val * n_features; });

    std::vector<size_t> displs(n_rank);
    thrust::exclusive_scan(thrust::host, sizes.begin(), sizes.end(),
                           displs.begin());

    centroidsBuf.resize(centroidsBuf.size() + nPtsSampled * n_features, stream);
    comm.allgatherv<DataT>(inRankCp.data(),
                           centroidsBuf.end() - nPtsSampled * n_features,
                           sizes.data(), displs.data(), stream);

    int tot_centroids = potentialCentroids.getSize(0) + nPtsSampled;
    potentialCentroids = std::move(Tensor<DataT, 2, IndexT>(
      centroidsBuf.data(), {tot_centroids, n_features}));
    /// <<<< End of Step-5 >>>
  }  /// <<<< Step-6 >>>

  LOG(handle, "@Rank-%d:KMeans||: # potential centroids sampled - %d\n",
      my_rank, potentialCentroids.getSize(0));

  if (potentialCentroids.getSize(0) > n_clusters) {
    // <<< Step-7 >>>: For x in C, set w_x to be the number of pts closest to X
    // temporary buffer to store the sample count per cluster, destructor
    // releases the resource

    Tensor<DataT, 1, IndexT> weight({potentialCentroids.getSize(0)},
                                    handle.get_device_allocator(), stream);

    kmeans::detail::countSamplesInCluster(handle, params, X, L2NormX,
                                          potentialCentroids, workspace, metric,
                                          weight, stream);

    // merge the local histogram from all ranks
    comm.allreduce<DataT>(weight.data(),         // sendbuff
                          weight.data(),         // recvbuff
                          weight.numElements(),  // count
                          raft::comms::op_t::SUM, stream);

    // <<< end of Step-7 >>>

    // Step-8: Recluster the weighted points in C into k clusters
    // Note - reclustering step is duplicated across all ranks and with the same
    // seed they should generate the same potentialCentroids
    centroidsRawData.resize(n_clusters * n_features, stream);
    kmeans::detail::kmeansPlusPlus(handle, params, potentialCentroids, metric,
                                   workspace, centroidsRawData, stream);

    DataT inertia = 0;
    int n_iter = 0;
    KMeansParams default_params;
    default_params.n_clusters = params.n_clusters;

    ML::kmeans::impl::fit(handle, default_params, potentialCentroids, weight,
                          centroidsRawData, inertia, n_iter, workspace);

  } else if (potentialCentroids.getSize(0) < n_clusters) {
    // supplement with random
    auto n_random_clusters = n_clusters - potentialCentroids.getSize(0);
    LOG(handle,
        "[Warning!] KMeans||: found fewer than %d centroids during "
        "initialization (found %d centroids, remaining %d centroids will be "
        "chosen randomly from input samples)\n",
        n_clusters, potentialCentroids.getSize(0), n_random_clusters);

    // reset buffer to store the chosen centroid
    centroidsRawData.resize(n_clusters * n_features, stream);

    // generate `n_random_clusters` centroids
    KMeansParams rand_params;
    rand_params.init = KMeansParams::InitMethod::Random;
    rand_params.n_clusters = n_random_clusters;
    initRandom(handle, rand_params, X, centroidsRawData);

    // copy centroids generated during kmeans|| iteration to the buffer
    raft::copy(centroidsRawData.data() + n_random_clusters * n_features,
               potentialCentroids.data(), potentialCentroids.numElements(),
               stream);

  } else {
    // found the required n_clusters
    centroidsRawData.resize(n_clusters * n_features, stream);
    raft::copy(centroidsRawData.data(), potentialCentroids.data(),
               potentialCentroids.numElements(), stream);
  }
}

template <typename DataT, typename IndexT>
void fit(const raft::handle_t &handle, const KMeansParams &params,
         Tensor<DataT, 2, IndexT> &X,
         MLCommon::device_buffer<DataT> &centroidsRawData, DataT &inertia,
         int &n_iter, MLCommon::device_buffer<char> &workspace) {
  const auto &comm = handle.get_comms();
  cudaStream_t stream = handle.get_stream();
  auto n_samples = X.getSize(0);
  auto n_features = X.getSize(1);
  auto n_clusters = params.n_clusters;

  raft::distance::DistanceType metric =
    static_cast<raft::distance::DistanceType>(params.metric);

  // stores (key, value) pair corresponding to each sample where
  //   - key is the index of nearest cluster
  //   - value is the distance to the nearest cluster
  Tensor<cub::KeyValuePair<IndexT, DataT>, 1, IndexT> minClusterAndDistance(
    {n_samples}, handle.get_device_allocator(), stream);

  // temporary buffer to store L2 norm of centroids or distance matrix,
  // destructor releases the resource
  MLCommon::device_buffer<DataT> L2NormBuf_OR_DistBuf(
    handle.get_device_allocator(), stream);

  // temporary buffer to store intermediate centroids, destructor releases the
  // resource
  Tensor<DataT, 2, IndexT> newCentroids({n_clusters, n_features},
                                        handle.get_device_allocator(), stream);

  // temporary buffer to store the sample count per cluster, destructor releases
  // the resource
  Tensor<int, 1, IndexT> sampleCountInCluster(
    {n_clusters}, handle.get_device_allocator(), stream);

  // L2 norm of X: ||x||^2
  Tensor<DataT, 1> L2NormX({n_samples}, handle.get_device_allocator(), stream);
  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm(L2NormX.data(), X.data(), X.getSize(1), X.getSize(0),
                          raft::linalg::L2Norm, true, stream);
  }

  DataT priorClusteringCost = 0;
  for (n_iter = 0; n_iter < params.max_iter; ++n_iter) {
    LOG(handle,
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
      handle, params, X, centroids, minClusterAndDistance, L2NormX,
      L2NormBuf_OR_DistBuf, workspace, metric, stream);

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

    // merge the local histogram from all ranks
    comm.allreduce<int>(sampleCountInCluster.data(),         // sendbuff
                        sampleCountInCluster.data(),         // recvbuff
                        sampleCountInCluster.numElements(),  // count
                        raft::comms::op_t::SUM, stream);

    // reduces newCentroids from all ranks
    comm.allreduce<DataT>(newCentroids.data(),         // sendbuff
                          newCentroids.data(),         // recvbuff
                          newCentroids.numElements(),  // count
                          raft::comms::op_t::SUM, stream);

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

    ML::thrustAllocatorAdapter alloc(handle.get_device_allocator(), stream);
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

    raft::linalg::matrixVectorOp(
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
    Tensor<DataT, 1> sqrdNorm({1}, handle.get_device_allocator(), stream);
    raft::linalg::mapThenSumReduce(
      sqrdNorm.data(), newCentroids.numElements(),
      [=] __device__(const DataT a, const DataT b) {
        DataT diff = a - b;
        return diff * diff;
      },
      stream, centroids.data(), newCentroids.data());

    DataT sqrdNormError = 0;
    raft::copy(&sqrdNormError, sqrdNorm.data(), sqrdNorm.numElements(), stream);

    raft::copy(centroidsRawData.data(), newCentroids.data(),
               newCentroids.numElements(), stream);

    bool done = false;
    if (params.inertia_check) {
      cub::KeyValuePair<IndexT, DataT> *clusterCostD =
        (cub::KeyValuePair<IndexT, DataT> *)handle.get_device_allocator()
          ->allocate(sizeof(cub::KeyValuePair<IndexT, DataT>), stream);

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

      // Cluster cost phi_x(C) from all ranks
      comm.allreduce(&clusterCostD->value, &clusterCostD->value, 1,
                     raft::comms::op_t::SUM, stream);

      DataT curClusteringCost = 0;
      raft::copy(&curClusteringCost, &clusterCostD->value, 1, stream);

      ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
             "An error occurred in the distributed operation. This can result "
             "from a failed rank");
      ASSERT(curClusteringCost != (DataT)0.0,
             "Too few points and centriods being found is getting 0 cost from "
             "centers\n");

      if (n_iter > 0) {
        DataT delta = curClusteringCost / priorClusteringCost;
        if (delta > 1 - params.tol) done = true;
      }
      priorClusteringCost = curClusteringCost;

      handle.get_device_allocator()->deallocate(
        clusterCostD, sizeof(cub::KeyValuePair<IndexT, DataT>), stream);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (sqrdNormError < params.tol) done = true;

    if (done) {
      LOG(handle,
          "Threshold triggered after %d iterations. Terminating early.\n",
          n_iter);
      break;
    }
  }
}

template <typename DataT, typename IndexT = int>
void fit(const raft::handle_t &handle, const KMeansParams &params,
         const DataT *X, const int n_local_samples, const int n_features,
         DataT *centroids, DataT &inertia, int &n_iter) {
  cudaStream_t stream = handle.get_stream();

  ASSERT(n_local_samples > 0, "# of samples must be > 0");

  ASSERT(params.oversampling_factor > 0,
         "oversampling factor must be > 0 (requested %d)",
         (int)params.oversampling_factor);

  ASSERT(is_device_or_managed_type(X), "input data must be device accessible");

  Tensor<DataT, 2, IndexT> data((DataT *)X, {n_local_samples, n_features});

  // underlying expandable storage that holds centroids data
  MLCommon::device_buffer<DataT> centroidsRawData(handle.get_device_allocator(),
                                                  stream);

  // Device-accessible allocation of expandable storage used as temorary buffers
  MLCommon::device_buffer<char> workspace(handle.get_device_allocator(),
                                          stream);

  if (params.init == KMeansParams::InitMethod::Random) {
    // initializing with random samples from input dataset
    LOG(handle,
        "KMeans.fit: initialize cluster centers by randomly choosing from the "
        "input data.\n");
    initRandom(handle, params, data, centroidsRawData);
  } else if (params.init == KMeansParams::InitMethod::KMeansPlusPlus) {
    // default method to initialize is kmeans++
    LOG(handle,
        "KMeans.fit: initialize cluster centers using k-means++ algorithm.\n");
    initKMeansPlusPlus(handle, params, data, centroidsRawData, workspace);
  } else if (params.init == KMeansParams::InitMethod::Array) {
    LOG(handle,
        "KMeans.fit: initialize cluster centers from the ndarray array input "
        "passed to init arguement.\n");

    ASSERT(centroids != nullptr,
           "centroids array is null (require a valid array of centroids for "
           "the requested initialization method)");

    centroidsRawData.resize(params.n_clusters * n_features, stream);
    raft::copy(centroidsRawData.begin(), centroids,
               params.n_clusters * n_features, stream);

  } else {
    THROW("unknown initialization method to select initial centers");
  }

  fit(handle, params, data, centroidsRawData, inertia, n_iter, workspace);

  raft::copy(centroids, centroidsRawData.data(), params.n_clusters * n_features,
             stream);

  LOG(handle,
      "KMeans.fit: async call returned (fit could still be running on the "
      "device)\n");
}

};  // end namespace impl
};  // end namespace opg
};  // end namespace kmeans
};  // end namespace ML
