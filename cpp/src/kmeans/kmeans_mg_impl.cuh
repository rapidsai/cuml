/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <cuml/cluster/kmeans.hpp>
#include <cuml/common/logger.hpp>
#include <raft/cluster/kmeans.cuh>
#include <raft/cudart_utils.h>
#include <raft/matrix/gather.cuh>
#include <raft/mdarray.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <ml_cuda_utils.h>

namespace ML {

#define CUML_LOG_KMEANS(handle, fmt, ...)               \
  do {                                                  \
    bool isRoot = true;                                 \
    if (handle.comms_initialized()) {                   \
      const auto& comm  = handle.get_comms();           \
      const int my_rank = comm.get_rank();              \
      isRoot            = my_rank == 0;                 \
    }                                                   \
    if (isRoot) { CUML_LOG_DEBUG(fmt, ##__VA_ARGS__); } \
  } while (0)

namespace kmeans {
namespace opg {
namespace impl {

#define KMEANS_COMM_ROOT 0

// Selects 'n_clusters' samples randomly from X
template <typename DataT, typename IndexT>
void initRandom(const raft::handle_t& handle,
                const KMeansParams& params,
                const raft::device_matrix_view<const DataT>& X,
                const raft::device_matrix_view<DataT>& centroids)
{
  const auto& comm     = handle.get_comms();
  cudaStream_t stream  = handle.get_stream();
  auto n_local_samples = X.extent(0);
  auto n_features      = X.extent(1);
  auto n_clusters      = params.n_clusters;

  const int my_rank = comm.get_rank();
  const int n_ranks = comm.get_size();

  std::vector<int> nCentroidsSampledByRank(n_ranks, 0);
  std::vector<size_t> nCentroidsElementsToReceiveFromRank(n_ranks, 0);

  const int nranks_reqd = std::min(n_ranks, n_clusters);
  ASSERT(KMEANS_COMM_ROOT < nranks_reqd, "KMEANS_COMM_ROOT must be in [0,  %d)\n", nranks_reqd);

  for (int rank = 0; rank < nranks_reqd; ++rank) {
    int nCentroidsSampledInRank = n_clusters / nranks_reqd;
    if (rank == KMEANS_COMM_ROOT) {
      nCentroidsSampledInRank += n_clusters - nCentroidsSampledInRank * nranks_reqd;
    }
    nCentroidsSampledByRank[rank]             = nCentroidsSampledInRank;
    nCentroidsElementsToReceiveFromRank[rank] = nCentroidsSampledInRank * n_features;
  }

  auto nCentroidsSampledInRank = nCentroidsSampledByRank[my_rank];
  ASSERT((IndexT)nCentroidsSampledInRank <= (IndexT)n_local_samples,
         "# random samples requested from rank-%d is larger than the available "
         "samples at the rank (requested is %lu, available is %lu)",
         my_rank,
         (size_t)nCentroidsSampledInRank,
         (size_t)n_local_samples);

  auto centroidsSampledInRank =
    raft::make_device_matrix<DataT>(nCentroidsSampledInRank, n_features, stream);

  raft::cluster::shuffleAndGather<DataT, IndexT>(
    handle, X, centroidsSampledInRank.view(), nCentroidsSampledInRank, params.rng_state.seed);

  std::vector<size_t> displs(n_ranks);
  thrust::exclusive_scan(thrust::host,
                         nCentroidsElementsToReceiveFromRank.begin(),
                         nCentroidsElementsToReceiveFromRank.end(),
                         displs.begin());

  // gather centroids from all ranks
  comm.allgatherv<DataT>(centroidsSampledInRank.data(),               // sendbuff
                         centroids.data(),                            // recvbuff
                         nCentroidsElementsToReceiveFromRank.data(),  // recvcount
                         displs.data(),
                         stream);
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
void initKMeansPlusPlus(const raft::handle_t& handle,
                        const KMeansParams& params,
                        const raft::device_matrix_view<const DataT>& X,
                        const raft::device_matrix_view<DataT>& centroidsRawData,
                        rmm::device_uvector<char>& workspace)
{
  const auto& comm    = handle.get_comms();
  cudaStream_t stream = handle.get_stream();
  const int my_rank   = comm.get_rank();
  const int n_rank    = comm.get_size();

  auto n_samples  = X.extent(0);
  auto n_features = X.extent(1);
  auto n_clusters = params.n_clusters;
  auto metric     = params.metric;

  raft::random::RngState rng(params.rng_state.seed, raft::random::GeneratorType::GenPhilox);

  // <<<< Step-1 >>> : C <- sample a point uniformly at random from X
  //    1.1 - Select a rank r' at random from the available n_rank ranks with a
  //          probability of 1/n_rank [Note - with same seed all rank selects
  //          the same r' which avoids a call to comm]
  //    1.2 - Rank r' samples a point uniformly at random from the local dataset
  //          X which will be used as the initial centroid for kmeans++
  //    1.3 - Communicate the initial centroid chosen by rank-r' to all other
  //          ranks
  std::mt19937 gen(params.rng_state.seed);
  std::uniform_int_distribution<> dis(0, n_rank - 1);
  int rp = dis(gen);

  // buffer to flag the sample that is chosen as initial centroids
  std::vector<int> h_isSampleCentroid(n_samples);
  std::fill(h_isSampleCentroid.begin(), h_isSampleCentroid.end(), 0);

  auto initialCentroid = raft::make_device_matrix<DataT>(1, n_features, stream);
  CUML_LOG_KMEANS(
    handle, "@Rank-%d : KMeans|| : initial centroid is sampled at rank-%d\n", my_rank, rp);

  //    1.2 - Rank r' samples a point uniformly at random from the local dataset
  //          X which will be used as the initial centroid for kmeans++
  if (my_rank == rp) {
    std::mt19937 gen(params.rng_state.seed);
    std::uniform_int_distribution<> dis(0, n_samples - 1);

    int cIdx           = dis(gen);
    auto centroidsView = raft::make_device_matrix_view(X.data() + cIdx * n_features, 1, n_features);

    raft::copy(initialCentroid.data(), centroidsView.data(), centroidsView.size(), stream);

    h_isSampleCentroid[cIdx] = 1;
  }

  // 1.3 - Communicate the initial centroid chosen by rank-r' to all other ranks
  comm.bcast<DataT>(initialCentroid.data(), initialCentroid.size(), rp, stream);

  // device buffer to flag the sample that is chosen as initial centroid
  auto isSampleCentroid = raft::make_device_vector<int>(n_samples, stream);

  raft::copy(isSampleCentroid.data(), h_isSampleCentroid.data(), isSampleCentroid.size(), stream);

  rmm::device_uvector<DataT> centroidsBuf(0, stream);

  // reset buffer to store the chosen centroid
  centroidsBuf.resize(initialCentroid.size(), stream);
  raft::copy(centroidsBuf.begin(), initialCentroid.data(), initialCentroid.size(), stream);

  auto potentialCentroids = raft::make_device_matrix_view(
    centroidsBuf.data(), initialCentroid.extent(0), initialCentroid.extent(1));
  // <<< End of Step-1 >>>

  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // L2 norm of X: ||x||^2
  auto L2NormX = raft::make_device_vector<DataT>(n_samples, stream);
  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm(
      L2NormX.data(), X.data(), X.extent(1), X.extent(0), raft::linalg::L2Norm, true, stream);
  }

  auto minClusterDistance = raft::make_device_vector<DataT>(n_samples, stream);
  auto uniformRands       = raft::make_device_vector<DataT>(n_samples, stream);

  // <<< Step-2 >>>: psi <- phi_X (C)
  auto clusterCost = raft::make_device_scalar<DataT>(0, stream);

  raft::cluster::minClusterDistanceCompute<DataT, IndexT>(handle,
                                                          params,
                                                          X,
                                                          potentialCentroids,
                                                          minClusterDistance.view(),
                                                          L2NormX.view(),
                                                          L2NormBuf_OR_DistBuf,
                                                          workspace);

  // compute partial cluster cost from the samples in rank
  raft::cluster::computeClusterCost(
    handle,
    minClusterDistance.view(),
    workspace,
    clusterCost.view(),
    [] __device__(const DataT& a, const DataT& b) { return a + b; });

  // compute total cluster cost by accumulating the partial cost from all the
  // ranks
  comm.allreduce(clusterCost.data(), clusterCost.data(), 1, raft::comms::op_t::SUM, stream);

  DataT psi = 0;
  raft::copy(&psi, clusterCost.data(), 1, stream);

  // <<< End of Step-2 >>>

  ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
         "An error occurred in the distributed operation. This can result from "
         "a failed rank");

  // Scalable kmeans++ paper claims 8 rounds is sufficient
  int niter = std::min(8, (int)ceil(log(psi)));
  CUML_LOG_KMEANS(handle,
                  "@Rank-%d:KMeans|| :phi - %f, max # of iterations for kmeans++ loop - "
                  "%d\n",
                  my_rank,
                  psi,
                  niter);

  // <<<< Step-3 >>> : for O( log(psi) ) times do
  for (int iter = 0; iter < niter; ++iter) {
    CUML_LOG_KMEANS(handle,
                    "@Rank-%d:KMeans|| - Iteration %d: # potential centroids sampled - "
                    "%d\n",
                    my_rank,
                    iter,
                    potentialCentroids.extent(0));

    raft::cluster::minClusterDistanceCompute<DataT, IndexT>(handle,
                                                            params,
                                                            X,
                                                            potentialCentroids,
                                                            minClusterDistance.view(),
                                                            L2NormX.view(),
                                                            L2NormBuf_OR_DistBuf,
                                                            workspace);

    raft::cluster::computeClusterCost(
      handle,
      minClusterDistance.view(),
      workspace,
      clusterCost.view(),
      [] __device__(const DataT& a, const DataT& b) { return a + b; });
    comm.allreduce(clusterCost.data(), clusterCost.data(), 1, raft::comms::op_t::SUM, stream);
    raft::copy(&psi, clusterCost.data(), 1, stream);
    ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed operation. This can result "
           "from a failed rank");

    // <<<< Step-4 >>> : Sample each point x in X independently and identify new
    // potentialCentroids
    raft::random::uniform(
      handle, rng, uniformRands.data(), uniformRands.extent(0), (DataT)0, (DataT)1);
    raft::cluster::SamplingOp<DataT, IndexT> select_op(
      psi, params.oversampling_factor, n_clusters, uniformRands.data(), isSampleCentroid.data());

    rmm::device_uvector<DataT> inRankCp(0, stream);
    raft::cluster::sampleCentroids<DataT, IndexT>(handle,
                                                  X,
                                                  minClusterDistance.view(),
                                                  isSampleCentroid.view(),
                                                  select_op,
                                                  inRankCp,
                                                  workspace);
    /// <<<< End of Step-4 >>>>

    int* nPtsSampledByRank;
    RAFT_CUDA_TRY(cudaMallocHost(&nPtsSampledByRank, n_rank * sizeof(int)));

    /// <<<< Step-5 >>> : C = C U C'
    // append the data in Cp from all ranks to the buffer holding the
    // potentialCentroids
    // RAFT_CUDA_TRY(cudaMemsetAsync(nPtsSampledByRank, 0, n_rank * sizeof(int), stream));
    std::fill(nPtsSampledByRank, nPtsSampledByRank + n_rank, 0);
    nPtsSampledByRank[my_rank] = inRankCp.size();
    comm.allgather(&(nPtsSampledByRank[my_rank]), nPtsSampledByRank, 1, stream);
    ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed operation. This can result "
           "from a failed rank");

    auto nPtsSampled =
      thrust::reduce(thrust::host, nPtsSampledByRank, nPtsSampledByRank + n_rank, 0);

    // gather centroids from all ranks
    std::vector<size_t> sizes(n_rank);
    thrust::transform(
      thrust::host, nPtsSampledByRank, nPtsSampledByRank + n_rank, sizes.begin(), [&](int val) {
        return val * n_features;
      });

    RAFT_CUDA_TRY_NO_THROW(cudaFreeHost(nPtsSampledByRank));

    std::vector<size_t> displs(n_rank);
    thrust::exclusive_scan(thrust::host, sizes.begin(), sizes.end(), displs.begin());

    centroidsBuf.resize(centroidsBuf.size() + nPtsSampled * n_features, stream);
    comm.allgatherv<DataT>(inRankCp.data(),
                           centroidsBuf.end() - nPtsSampled * n_features,
                           sizes.data(),
                           displs.data(),
                           stream);

    auto tot_centroids = potentialCentroids.extent(0) + nPtsSampled;
    potentialCentroids =
      raft::make_device_matrix_view(centroidsBuf.data(), tot_centroids, n_features);
    /// <<<< End of Step-5 >>>
  }  /// <<<< Step-6 >>>

  CUML_LOG_KMEANS(handle,
                  "@Rank-%d:KMeans||: # potential centroids sampled - %d\n",
                  my_rank,
                  potentialCentroids.extent(0));

  if ((IndexT)potentialCentroids.extent(0) > (IndexT)n_clusters) {
    // <<< Step-7 >>>: For x in C, set w_x to be the number of pts closest to X
    // temporary buffer to store the sample count per cluster, destructor
    // releases the resource

    auto weight = raft::make_device_vector<DataT>(potentialCentroids.extent(0), stream);

    raft::cluster::countSamplesInCluster<DataT, IndexT>(
      handle, params, X, L2NormX.view(), potentialCentroids, workspace, weight.view());

    // merge the local histogram from all ranks
    comm.allreduce<DataT>(weight.data(),  // sendbuff
                          weight.data(),  // recvbuff
                          weight.size(),  // count
                          raft::comms::op_t::SUM,
                          stream);

    // <<< end of Step-7 >>>

    // Step-8: Recluster the weighted points in C into k clusters
    // Note - reclustering step is duplicated across all ranks and with the same
    // seed they should generate the same potentialCentroids
    raft::cluster::kmeansPlusPlus<DataT, IndexT>(
      handle, params, potentialCentroids, centroidsRawData, workspace);

    auto inertia = raft::make_host_scalar<DataT>(0);
    auto n_iter  = raft::make_host_scalar<IndexT>(0);
    KMeansParams default_params;
    default_params.n_clusters = params.n_clusters;

    raft::cluster::kmeans_fit_main<DataT, IndexT>(handle,
                                                  default_params,
                                                  potentialCentroids,
                                                  weight.view(),
                                                  centroidsRawData,
                                                  inertia.view(),
                                                  n_iter.view(),
                                                  workspace);

  } else if ((IndexT)potentialCentroids.extent(0) < (IndexT)n_clusters) {
    // supplement with random
    auto n_random_clusters = n_clusters - potentialCentroids.extent(0);
    CUML_LOG_KMEANS(handle,
                    "[Warning!] KMeans||: found fewer than %d centroids during "
                    "initialization (found %d centroids, remaining %d centroids will be "
                    "chosen randomly from input samples)\n",
                    n_clusters,
                    potentialCentroids.extent(0),
                    n_random_clusters);

    // generate `n_random_clusters` centroids
    KMeansParams rand_params;
    rand_params.init       = KMeansParams::InitMethod::Random;
    rand_params.n_clusters = n_random_clusters;
    initRandom<DataT, IndexT>(handle, rand_params, X, centroidsRawData);

    // copy centroids generated during kmeans|| iteration to the buffer
    raft::copy(centroidsRawData.data() + n_random_clusters * n_features,
               potentialCentroids.data(),
               potentialCentroids.size(),
               stream);

  } else {
    // found the required n_clusters
    raft::copy(
      centroidsRawData.data(), potentialCentroids.data(), potentialCentroids.size(), stream);
  }
}

template <typename DataT>
void checkWeights(const raft::handle_t& handle,
                  rmm::device_uvector<char>& workspace,
                  raft::device_vector_view<DataT> weight)
{
  cudaStream_t stream = handle.get_stream();
  rmm::device_scalar<DataT> wt_aggr(stream);

  const auto& comm = handle.get_comms();

  auto n_samples            = weight.extent(0);
  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceReduce::Sum(
    nullptr, temp_storage_bytes, weight.data(), wt_aggr.data(), n_samples, stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceReduce::Sum(
    workspace.data(), temp_storage_bytes, weight.data(), wt_aggr.data(), n_samples, stream));

  comm.allreduce<DataT>(wt_aggr.data(),  // sendbuff
                        wt_aggr.data(),  // recvbuff
                        1,               // count
                        raft::comms::op_t::SUM,
                        stream);
  DataT wt_sum = wt_aggr.value(stream);
  handle.sync_stream(stream);

  if (wt_sum != n_samples) {
    CUML_LOG_KMEANS(handle,
                    "[Warning!] KMeans: normalizing the user provided sample weights to "
                    "sum up to %d samples",
                    n_samples);

    DataT scale = n_samples / wt_sum;
    raft::linalg::unaryOp(
      weight.data(),
      weight.data(),
      weight.size(),
      [=] __device__(const DataT& wt) { return wt * scale; },
      stream);
  }
}

template <typename DataT, typename IndexT>
void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const raft::device_matrix_view<const DataT>& X,
         const raft::device_vector_view<DataT>& weight,
         const raft::device_matrix_view<DataT>& centroids,
         const raft::host_scalar_view<DataT>& inertia,
         const raft::host_scalar_view<IndexT>& n_iter,
         rmm::device_uvector<char>& workspace)
{
  const auto& comm    = handle.get_comms();
  cudaStream_t stream = handle.get_stream();
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = params.metric;

  // stores (key, value) pair corresponding to each sample where
  //   - key is the index of nearest cluster
  //   - value is the distance to the nearest cluster
  auto minClusterAndDistance =
    raft::make_device_vector<cub::KeyValuePair<IndexT, DataT>>(n_samples, stream);

  // temporary buffer to store L2 norm of centroids or distance matrix,
  // destructor releases the resource
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // temporary buffer to store intermediate centroids, destructor releases the
  // resource
  auto newCentroids = raft::make_device_matrix<DataT>(n_clusters, n_features, stream);

  // temporary buffer to store the weights per cluster, destructor releases
  // the resource
  auto wtInCluster = raft::make_device_vector<DataT>(n_clusters, stream);

  // L2 norm of X: ||x||^2
  auto L2NormX = raft::make_device_vector<DataT>(n_samples, stream);
  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm(
      L2NormX.data(), X.data(), X.extent(1), X.extent(0), raft::linalg::L2Norm, true, stream);
  }

  DataT priorClusteringCost = 0;
  for (n_iter[0] = 1; n_iter[0] <= params.max_iter; ++n_iter[0]) {
    CUML_LOG_KMEANS(handle,
                    "KMeans.fit: Iteration-%d: fitting the model using the initialize "
                    "cluster centers\n",
                    n_iter[0]);

    // computes minClusterAndDistance[0:n_samples) where
    // minClusterAndDistance[i] is a <key, value> pair where
    //   'key' is index to an sample in 'centroids' (index of the nearest
    //   centroid) and 'value' is the distance between the sample 'X[i]' and the
    //   'centroid[key]'
    raft::cluster::minClusterAndDistanceCompute<DataT, IndexT>(
      handle,
      params,
      X,
      (raft::device_matrix_view<const DataT>)centroids,
      minClusterAndDistance.view(),
      L2NormX.view(),
      L2NormBuf_OR_DistBuf,
      workspace);

    // Using TransformInputIteratorT to dereference an array of
    // cub::KeyValuePair and converting them to just return the Key to be used
    // in reduce_rows_by_key prims
    raft::cluster::KeyValueIndexOp<IndexT, DataT> conversion_op;
    cub::TransformInputIterator<IndexT,
                                raft::cluster::KeyValueIndexOp<IndexT, DataT>,
                                cub::KeyValuePair<IndexT, DataT>*>
      itr(minClusterAndDistance.data(), conversion_op);

    workspace.resize(n_samples, stream);

    // Calculates weighted sum of all the samples assigned to cluster-i and
    // store the result in newCentroids[i]
    raft::linalg::reduce_rows_by_key((DataT*)X.data(),
                                     X.extent(1),
                                     itr,
                                     weight.data(),
                                     workspace.data(),
                                     X.extent(0),
                                     X.extent(1),
                                     n_clusters,
                                     newCentroids.data(),
                                     stream);

    // Reduce weights by key to compute weight in each cluster
    raft::linalg::reduce_cols_by_key(weight.data(),
                                     itr,
                                     wtInCluster.data(),
                                     (IndexT)1,
                                     (IndexT)weight.extent(0),
                                     (IndexT)n_clusters,
                                     stream);

    // merge the local histogram from all ranks
    comm.allreduce<DataT>(wtInCluster.data(),  // sendbuff
                          wtInCluster.data(),  // recvbuff
                          wtInCluster.size(),  // count
                          raft::comms::op_t::SUM,
                          stream);

    // reduces newCentroids from all ranks
    comm.allreduce<DataT>(newCentroids.data(),  // sendbuff
                          newCentroids.data(),  // recvbuff
                          newCentroids.size(),  // count
                          raft::comms::op_t::SUM,
                          stream);

    // Computes newCentroids[i] = newCentroids[i]/wtInCluster[i] where
    //   newCentroids[n_clusters x n_features] - 2D array, newCentroids[i] has
    //   sum of all the samples assigned to cluster-i
    //   wtInCluster[n_clusters] - 1D array, wtInCluster[i] contains # of
    //   samples in cluster-i.
    // Note - when wtInCluster[i] is 0, newCentroid[i] is reset to 0

    raft::linalg::matrixVectorOp(
      newCentroids.data(),
      newCentroids.data(),
      wtInCluster.data(),
      newCentroids.extent(1),
      newCentroids.extent(0),
      true,
      false,
      [=] __device__(DataT mat, DataT vec) {
        if (vec == 0)
          return DataT(0);
        else
          return mat / vec;
      },
      stream);

    // copy the centroids[i] to newCentroids[i] when wtInCluster[i] is 0
    cub::ArgIndexInputIterator<DataT*> itr_wt(wtInCluster.data());
    raft::matrix::gather_if(
      centroids.data(),
      centroids.extent(1),
      centroids.extent(0),
      itr_wt,
      itr_wt,
      wtInCluster.size(),
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
    auto sqrdNorm = raft::make_device_scalar<DataT>(1, stream);
    raft::linalg::mapThenSumReduce(
      sqrdNorm.data(),
      newCentroids.size(),
      [=] __device__(const DataT a, const DataT b) {
        DataT diff = a - b;
        return diff * diff;
      },
      stream,
      centroids.data(),
      newCentroids.data());

    DataT sqrdNormError = 0;
    raft::copy(&sqrdNormError, sqrdNorm.data(), sqrdNorm.size(), stream);

    raft::copy(centroids.data(), newCentroids.data(), newCentroids.size(), stream);

    bool done = false;
    if (params.inertia_check) {
      rmm::device_scalar<cub::KeyValuePair<IndexT, DataT>> clusterCostD(stream);

      // calculate cluster cost phi_x(C)
      raft::cluster::computeClusterCost(handle,
                                        minClusterAndDistance.view(),
                                        workspace,
                                        raft::make_device_scalar_view(clusterCostD.data()),
                                        [] __device__(const cub::KeyValuePair<IndexT, DataT>& a,
                                                      const cub::KeyValuePair<IndexT, DataT>& b) {
                                          cub::KeyValuePair<IndexT, DataT> res;
                                          res.key   = 0;
                                          res.value = a.value + b.value;
                                          return res;
                                        });

      // Cluster cost phi_x(C) from all ranks
      comm.allreduce(&(clusterCostD.data()->value),
                     &(clusterCostD.data()->value),
                     1,
                     raft::comms::op_t::SUM,
                     stream);

      DataT curClusteringCost = 0;
      raft::copy(&curClusteringCost, &(clusterCostD.data()->value), 1, stream);

      ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
             "An error occurred in the distributed operation. This can result "
             "from a failed rank");
      ASSERT(curClusteringCost != (DataT)0.0,
             "Too few points and centriods being found is getting 0 cost from "
             "centers\n");

      if (n_iter[0] > 0) {
        DataT delta = curClusteringCost / priorClusteringCost;
        if (delta > 1 - params.tol) done = true;
      }
      priorClusteringCost = curClusteringCost;
    }

    handle.sync_stream(stream);
    if (sqrdNormError < params.tol) done = true;

    if (done) {
      CUML_LOG_KMEANS(
        handle, "Threshold triggered after %d iterations. Terminating early.\n", n_iter[0]);
      break;
    }
  }
}

template <typename DataT, typename IndexT = int>
void fit(const raft::handle_t& handle,
         const KMeansParams& params,
         const DataT* X,
         const IndexT n_local_samples,
         const IndexT n_features,
         const DataT* sample_weight,
         DataT* centroids,
         DataT& inertia,
         IndexT& n_iter)
{
  cudaStream_t stream = handle.get_stream();

  ASSERT(n_local_samples > 0, "# of samples must be > 0");
  ASSERT(params.oversampling_factor >= 0,
         "oversampling factor must be >= 0 (requested %d)",
         (int)params.oversampling_factor);
  ASSERT(is_device_or_managed_type(X), "input data must be device accessible");

  auto n_clusters = params.n_clusters;
  auto data       = raft::make_device_matrix_view(X, n_local_samples, n_features);
  auto weight     = raft::make_device_vector<DataT>(n_local_samples, stream);
  if (sample_weight != nullptr) {
    raft::copy(weight.data(), sample_weight, n_local_samples, stream);
  } else {
    thrust::fill(handle.get_thrust_policy(), weight.data(), weight.data() + weight.size(), 1);
  }

  // underlying expandable storage that holds centroids data
  auto centroidsRawData = raft::make_device_matrix<DataT>(n_clusters, n_features, stream);

  // Device-accessible allocation of expandable storage used as temorary buffers
  rmm::device_uvector<char> workspace(0, stream);

  // check if weights sum up to n_samples
  checkWeights(handle, workspace, weight.view());

  if (params.init == KMeansParams::InitMethod::Random) {
    // initializing with random samples from input dataset
    CUML_LOG_KMEANS(handle,
                    "KMeans.fit: initialize cluster centers by randomly choosing from the "
                    "input data.\n");
    initRandom<DataT, IndexT>(handle, params, data, centroidsRawData.view());
  } else if (params.init == KMeansParams::InitMethod::KMeansPlusPlus) {
    // default method to initialize is kmeans++
    CUML_LOG_KMEANS(handle, "KMeans.fit: initialize cluster centers using k-means++ algorithm.\n");
    initKMeansPlusPlus<DataT, IndexT>(handle, params, data, centroidsRawData.view(), workspace);
  } else if (params.init == KMeansParams::InitMethod::Array) {
    CUML_LOG_KMEANS(handle,
                    "KMeans.fit: initialize cluster centers from the ndarray array input "
                    "passed to init arguement.\n");

    ASSERT(centroids != nullptr,
           "centroids array is null (require a valid array of centroids for "
           "the requested initialization method)");

    raft::copy(centroidsRawData.data(), centroids, params.n_clusters * n_features, stream);
  } else {
    THROW("unknown initialization method to select initial centers");
  }
  auto weightView    = weight.view();
  auto centroidsView = centroidsRawData.view();
  auto inertiaView   = raft::make_host_scalar_view(&inertia);
  auto n_iterView    = raft::make_host_scalar_view(&n_iter);

  fit<DataT, IndexT>(handle,
                     params,
                     data,
                     weight.view(),
                     centroidsRawData.view(),
                     inertiaView,
                     n_iterView,
                     workspace);

  raft::copy(centroids, centroidsRawData.data(), params.n_clusters * n_features, stream);

  CUML_LOG_KMEANS(handle,
                  "KMeans.fit: async call returned (fit could still be running on the "
                  "device)\n");
}

};  // end namespace impl
};  // end namespace opg
};  // end namespace kmeans
};  // end namespace ML
