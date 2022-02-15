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

#include <cuml/cluster/kmeans.hpp>
#include <cuml/cluster/kmeans_mg.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/metrics/metrics.hpp>

#include <ml_cuda_utils.h>

#include <common/tensor.hpp>

#include <matrix/gather.cuh>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/random/permute.hpp>

#include <raft/comms/comms.hpp>
#include <raft/cudart_utils.h>
#include <raft/distance/fused_l2_nn.hpp>
#include <raft/linalg/add.hpp>
#include <raft/linalg/matrix_vector_op.hpp>
#include <raft/linalg/mean_squared_error.hpp>
#include <raft/linalg/reduce.hpp>
#include <raft/random/rng.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <random>

#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>

#include <ml_cuda_utils.h>

#include <common/tensor.hpp>
#include <cuml/cluster/kmeans_mg.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/metrics/metrics.hpp>

#include <fstream>
#include <numeric>
#include <random>
#include <vector>

namespace ML {

#define LOG(handle, fmt, ...)                           \
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
namespace detail {

template <typename LabelT, typename DataT>
struct FusedL2NNReduceOp {
  LabelT offset;

  FusedL2NNReduceOp(LabelT _offset) : offset(_offset){};

  typedef typename cub::KeyValuePair<LabelT, DataT> KVP;
  DI void operator()(LabelT rit, KVP* out, const KVP& other)
  {
    if (other.value < out->value) {
      out->key   = offset + other.key;
      out->value = other.value;
    }
  }

  DI void operator()(LabelT rit, DataT* out, const KVP& other)
  {
    if (other.value < *out) { *out = other.value; }
  }

  DI void init(DataT* out, DataT maxVal) { *out = maxVal; }
  DI void init(KVP* out, DataT maxVal)
  {
    out->key   = -1;
    out->value = maxVal;
  }
};

template <typename DataT>
struct SamplingOp {
  DataT* rnd;
  int* flag;
  DataT cluster_cost;
  double oversampling_factor;
  int n_clusters;

  CUB_RUNTIME_FUNCTION __forceinline__ SamplingOp(DataT c, double l, int k, DataT* rand, int* ptr)
    : cluster_cost(c), oversampling_factor(l), n_clusters(k), rnd(rand), flag(ptr)
  {
  }

  __host__ __device__ __forceinline__ bool operator()(
    const cub::KeyValuePair<ptrdiff_t, DataT>& a) const
  {
    DataT prob_threshold = (DataT)rnd[a.key];

    DataT prob_x = ((oversampling_factor * n_clusters * a.value) / cluster_cost);

    return !flag[a.key] && (prob_x > prob_threshold);
  }
};

template <typename IndexT, typename DataT>
struct KeyValueIndexOp {
  __host__ __device__ __forceinline__ IndexT
  operator()(const cub::KeyValuePair<IndexT, DataT>& a) const
  {
    return a.key;
  }
};

template <typename CountT>
CountT getDataBatchSize(const KMeansParams& params, CountT n_samples)
{
  auto minVal = std::min(params.batch_samples, n_samples);
  return (minVal == 0) ? n_samples : minVal;
}

template <typename CountT>
CountT getCentroidsBatchSize(const KMeansParams& params, CountT n_local_clusters)
{
  auto minVal = std::min(params.batch_centroids, n_local_clusters);
  return (minVal == 0) ? n_local_clusters : minVal;
}

// Computes the intensity histogram from a sequence of labels
template <typename SampleIteratorT, typename CounterT>
void countLabels(const raft::handle_t& handle,
                 SampleIteratorT labels,
                 CounterT* count,
                 int n_samples,
                 int n_clusters,
                 rmm::device_uvector<char>& workspace,
                 cudaStream_t stream)
{
  int num_levels  = n_clusters + 1;
  int lower_level = 0;
  int upper_level = n_clusters;

  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(nullptr,
                                                    temp_storage_bytes,
                                                    labels,
                                                    count,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    n_samples,
                                                    stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(workspace.data(),
                                                    temp_storage_bytes,
                                                    labels,
                                                    count,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    n_samples,
                                                    stream));
}

template <typename DataT, typename IndexT>
Tensor<DataT, 2, IndexT> sampleCentroids(const raft::handle_t& handle,
                                         Tensor<DataT, 2, IndexT>& X,
                                         Tensor<DataT, 1, IndexT>& minClusterDistance,
                                         Tensor<int, 1, IndexT>& isSampleCentroid,
                                         typename kmeans::detail::SamplingOp<DataT>& select_op,
                                         rmm::device_uvector<char>& workspace,
                                         cudaStream_t stream)
{
  int n_local_samples = X.getSize(0);
  int n_features      = X.getSize(1);

  Tensor<int, 1> nSelected({1}, stream);

  cub::ArgIndexInputIterator<DataT*> ip_itr(minClusterDistance.data());
  Tensor<cub::KeyValuePair<ptrdiff_t, DataT>, 1> sampledMinClusterDistance({n_local_samples},
                                                                           stream);
  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceSelect::If(nullptr,
                                      temp_storage_bytes,
                                      ip_itr,
                                      sampledMinClusterDistance.data(),
                                      nSelected.data(),
                                      n_local_samples,
                                      select_op,
                                      stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceSelect::If(workspace.data(),
                                      temp_storage_bytes,
                                      ip_itr,
                                      sampledMinClusterDistance.data(),
                                      nSelected.data(),
                                      n_local_samples,
                                      select_op,
                                      stream));

  int nPtsSampledInRank = 0;
  raft::copy(&nPtsSampledInRank, nSelected.data(), nSelected.numElements(), stream);
  handle.sync_stream(stream);

  int* rawPtr_isSampleCentroid = isSampleCentroid.data();
  thrust::for_each_n(handle.get_thrust_policy(),
                     sampledMinClusterDistance.begin(),
                     nPtsSampledInRank,
                     [=] __device__(cub::KeyValuePair<ptrdiff_t, DataT> val) {
                       rawPtr_isSampleCentroid[val.key] = 1;
                     });

  Tensor<DataT, 2, IndexT> inRankCp({nPtsSampledInRank, n_features}, stream);

  MLCommon::Matrix::gather(
    X.data(),
    X.getSize(1),
    X.getSize(0),
    sampledMinClusterDistance.data(),
    nPtsSampledInRank,
    inRankCp.data(),
    [=] __device__(cub::KeyValuePair<ptrdiff_t, DataT> val) {  // MapTransformOp
      return val.key;
    },
    stream);

  return inRankCp;
}

template <typename DataT, typename IndexT, typename ReductionOpT>
void computeClusterCost(const raft::handle_t& handle,
                        Tensor<DataT, 1, IndexT>& minClusterDistance,
                        rmm::device_uvector<char>& workspace,
                        DataT* clusterCost,
                        ReductionOpT reduction_op,
                        cudaStream_t stream)
{
  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(nullptr,
                                          temp_storage_bytes,
                                          minClusterDistance.data(),
                                          clusterCost,
                                          minClusterDistance.numElements(),
                                          reduction_op,
                                          DataT(),
                                          stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceReduce::Reduce(workspace.data(),
                                          temp_storage_bytes,
                                          minClusterDistance.data(),
                                          clusterCost,
                                          minClusterDistance.numElements(),
                                          reduction_op,
                                          DataT(),
                                          stream));
}

// calculate pairwise distance between 'dataset[n x d]' and 'centroids[k x d]',
// result will be stored in 'pairwiseDistance[n x k]'
template <typename DataT, typename IndexT>
void pairwise_distance(const raft::handle_t& handle,
                       Tensor<DataT, 2, IndexT>& X,
                       Tensor<DataT, 2, IndexT>& centroids,
                       Tensor<DataT, 2, IndexT>& pairwiseDistance,
                       rmm::device_uvector<char>& workspace,
                       raft::distance::DistanceType metric,
                       cudaStream_t stream)
{
  auto n_samples  = X.getSize(0);
  auto n_features = X.getSize(1);
  auto n_clusters = centroids.getSize(0);

  ASSERT(X.getSize(1) == centroids.getSize(1),
         "# features in dataset and centroids are different (must be same)");

  ML::Metrics::pairwise_distance(handle,
                                 X.data(),
                                 centroids.data(),
                                 pairwiseDistance.data(),
                                 n_samples,
                                 n_clusters,
                                 n_features,
                                 metric);
}

// Calculates a <key, value> pair for every sample in input 'X' where key is an
// index to an sample in 'centroids' (index of the nearest centroid) and 'value'
// is the distance between the sample and the 'centroid[key]'
template <typename DataT, typename IndexT>
void minClusterAndDistance(
  const raft::handle_t& handle,
  const KMeansParams& params,
  Tensor<DataT, 2, IndexT>& X,
  Tensor<DataT, 2, IndexT>& centroids,
  Tensor<cub::KeyValuePair<IndexT, DataT>, 1, IndexT>& minClusterAndDistance,
  Tensor<DataT, 1, IndexT>& L2NormX,
  rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
  rmm::device_uvector<char>& workspace,
  raft::distance::DistanceType metric,
  cudaStream_t stream)
{
  auto n_samples          = X.getSize(0);
  auto n_features         = X.getSize(1);
  auto n_clusters         = centroids.getSize(0);
  auto dataBatchSize      = kmeans::detail::getDataBatchSize(params, n_samples);
  auto centroidsBatchSize = kmeans::detail::getCentroidsBatchSize(params, n_clusters);

  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
    raft::linalg::rowNorm(L2NormBuf_OR_DistBuf.data(),
                          centroids.data(),
                          centroids.getSize(1),
                          centroids.getSize(0),
                          raft::linalg::L2Norm,
                          true,
                          stream);
  } else {
    L2NormBuf_OR_DistBuf.resize(dataBatchSize * centroidsBatchSize, stream);
  }

  // Note - pairwiseDistance and centroidsNorm share the same buffer
  // centroidsNorm [n_clusters] - tensor wrapper around centroids L2 Norm
  Tensor<DataT, 1> centroidsNorm(L2NormBuf_OR_DistBuf.data(), {n_clusters});
  // pairwiseDistance[ns x nc] - tensor wrapper around the distance buffer
  Tensor<DataT, 2, IndexT> pairwiseDistance(L2NormBuf_OR_DistBuf.data(),
                                            {dataBatchSize, centroidsBatchSize});

  cub::KeyValuePair<IndexT, DataT> initial_value(0, std::numeric_limits<DataT>::max());

  thrust::fill(handle.get_thrust_policy(),
               minClusterAndDistance.begin(),
               minClusterAndDistance.end(),
               initial_value);

  // tile over the input dataset
  for (auto dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
    // # of samples for the current batch
    auto ns = std::min(dataBatchSize, n_samples - dIdx);

    // datasetView [ns x n_features] - view representing the current batch of
    // input dataset
    auto datasetView = X.template view<2>({ns, n_features}, {dIdx, 0});

    // minClusterAndDistanceView [ns x n_clusters]
    auto minClusterAndDistanceView = minClusterAndDistance.template view<1>({ns}, {dIdx});

    auto L2NormXView = L2NormX.template view<1>({ns}, {dIdx});

    // tile over the centroids
    for (auto cIdx = 0; cIdx < n_clusters; cIdx += centroidsBatchSize) {
      // # of centroids for the current batch
      auto nc = std::min(centroidsBatchSize, n_clusters - cIdx);

      // centroidsView [nc x n_features] - view representing the current batch
      // of centroids
      auto centroidsView = centroids.template view<2>({nc, n_features}, {cIdx, 0});

      if (metric == raft::distance::DistanceType::L2Expanded ||
          metric == raft::distance::DistanceType::L2SqrtExpanded) {
        auto centroidsNormView = centroidsNorm.template view<1>({nc}, {cIdx});
        workspace.resize((sizeof(int)) * ns, stream);

        FusedL2NNReduceOp<IndexT, DataT> redOp(cIdx);
        raft::distance::KVPMinReduce<IndexT, DataT> pairRedOp;

        raft::distance::fusedL2NN<DataT, cub::KeyValuePair<IndexT, DataT>, IndexT>(
          minClusterAndDistanceView.data(),
          datasetView.data(),
          centroidsView.data(),
          L2NormXView.data(),
          centroidsNormView.data(),
          ns,
          nc,
          n_features,
          (void*)workspace.data(),
          redOp,
          pairRedOp,
          (metric == raft::distance::DistanceType::L2Expanded) ? false : true,
          false,
          stream);
      } else {
        // pairwiseDistanceView [ns x nc] - view representing the pairwise
        // distance for current batch
        auto pairwiseDistanceView = pairwiseDistance.template view<2>({ns, nc}, {0, 0});

        // calculate pairwise distance between current tile of cluster centroids
        // and input dataset
        kmeans::detail::pairwise_distance(
          handle, datasetView, centroidsView, pairwiseDistanceView, workspace, metric, stream);

        // argmin reduction returning <index, value> pair
        // calculates the closest centroid and the distance to the closest
        // centroid
        raft::linalg::coalescedReduction(
          minClusterAndDistanceView.data(),
          pairwiseDistanceView.data(),
          pairwiseDistanceView.getSize(1),
          pairwiseDistanceView.getSize(0),
          initial_value,
          stream,
          true,
          [=] __device__(const DataT val, const IndexT i) {
            cub::KeyValuePair<IndexT, DataT> pair;
            pair.key   = cIdx + i;
            pair.value = val;
            return pair;
          },
          [=] __device__(cub::KeyValuePair<IndexT, DataT> a, cub::KeyValuePair<IndexT, DataT> b) {
            return (b.value < a.value) ? b : a;
          },
          [=] __device__(cub::KeyValuePair<IndexT, DataT> pair) { return pair; });
      }
    }
  }
}

template <typename DataT, typename IndexT>
void minClusterDistance(const raft::handle_t& handle,
                        const KMeansParams& params,
                        Tensor<DataT, 2, IndexT>& X,
                        Tensor<DataT, 2, IndexT>& centroids,
                        Tensor<DataT, 1, IndexT>& minClusterDistance,
                        Tensor<DataT, 1, IndexT>& L2NormX,
                        rmm::device_uvector<DataT>& L2NormBuf_OR_DistBuf,
                        rmm::device_uvector<char>& workspace,
                        raft::distance::DistanceType metric,
                        cudaStream_t stream)
{
  auto n_samples  = X.getSize(0);
  auto n_features = X.getSize(1);
  auto n_clusters = centroids.getSize(0);

  auto dataBatchSize      = kmeans::detail::getDataBatchSize(params, n_samples);
  auto centroidsBatchSize = kmeans::detail::getCentroidsBatchSize(params, n_clusters);

  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    L2NormBuf_OR_DistBuf.resize(n_clusters, stream);
    raft::linalg::rowNorm(L2NormBuf_OR_DistBuf.data(),
                          centroids.data(),
                          centroids.getSize(1),
                          centroids.getSize(0),
                          raft::linalg::L2Norm,
                          true,
                          stream);
  } else {
    L2NormBuf_OR_DistBuf.resize(dataBatchSize * centroidsBatchSize, stream);
  }

  // Note - pairwiseDistance and centroidsNorm share the same buffer
  // centroidsNorm [n_clusters] - tensor wrapper around centroids L2 Norm
  Tensor<DataT, 1> centroidsNorm(L2NormBuf_OR_DistBuf.data(), {n_clusters});
  // pairwiseDistance[ns x nc] - tensor wrapper around the distance buffer
  Tensor<DataT, 2, IndexT> pairwiseDistance(L2NormBuf_OR_DistBuf.data(),
                                            {dataBatchSize, centroidsBatchSize});

  thrust::fill(handle.get_thrust_policy(),
               minClusterDistance.begin(),
               minClusterDistance.end(),
               std::numeric_limits<DataT>::max());

  // tile over the input data and calculate distance matrix [n_samples x
  // n_clusters]
  for (int dIdx = 0; dIdx < n_samples; dIdx += dataBatchSize) {
    // # of samples for the current batch
    auto ns = std::min(dataBatchSize, n_samples - dIdx);

    // datasetView [ns x n_features] - view representing the current batch of
    // input dataset
    auto datasetView = X.template view<2>({ns, n_features}, {dIdx, 0});

    // minClusterDistanceView [ns x n_clusters]
    auto minClusterDistanceView = minClusterDistance.template view<1>({ns}, {dIdx});

    auto L2NormXView = L2NormX.template view<1>({ns}, {dIdx});

    // tile over the centroids
    for (auto cIdx = 0; cIdx < n_clusters; cIdx += centroidsBatchSize) {
      // # of centroids for the current batch
      auto nc = std::min(centroidsBatchSize, n_clusters - cIdx);

      // centroidsView [nc x n_features] - view representing the current batch
      // of centroids
      auto centroidsView = centroids.template view<2>({nc, n_features}, {cIdx, 0});

      if (metric == raft::distance::DistanceType::L2Expanded ||
          metric == raft::distance::DistanceType::L2SqrtExpanded) {
        auto centroidsNormView = centroidsNorm.template view<1>({nc}, {cIdx});
        workspace.resize((sizeof(int)) * ns, stream);

        FusedL2NNReduceOp<IndexT, DataT> redOp(cIdx);
        raft::distance::KVPMinReduce<IndexT, DataT> pairRedOp;
        raft::distance::fusedL2NN<DataT, DataT, IndexT>(
          minClusterDistanceView.data(),
          datasetView.data(),
          centroidsView.data(),
          L2NormXView.data(),
          centroidsNormView.data(),
          ns,
          nc,
          n_features,
          (void*)workspace.data(),
          redOp,
          pairRedOp,
          (metric == raft::distance::DistanceType::L2Expanded) ? false : true,
          false,
          stream);
      } else {
        // pairwiseDistanceView [ns x nc] - view representing the pairwise
        // distance for current batch
        auto pairwiseDistanceView = pairwiseDistance.template view<2>({ns, nc}, {0, 0});

        // calculate pairwise distance between current tile of cluster centroids
        // and input dataset
        kmeans::detail::pairwise_distance(
          handle, datasetView, centroidsView, pairwiseDistanceView, workspace, metric, stream);

        raft::linalg::coalescedReduction(
          minClusterDistanceView.data(),
          pairwiseDistanceView.data(),
          pairwiseDistanceView.getSize(1),
          pairwiseDistanceView.getSize(0),
          std::numeric_limits<DataT>::max(),
          stream,
          true,
          [=] __device__(DataT val, int i) {  // MainLambda
            return val;
          },
          [=] __device__(DataT a, DataT b) {  // ReduceLambda
            return (b < a) ? b : a;
          },
          [=] __device__(DataT val) {  // FinalLambda
            return val;
          });
      }
    }
  }
}

// shuffle and randomly select 'n_samples_to_gather' from input 'in' and stores
// in 'out' does not modify the input
template <typename DataT, typename IndexT>
void shuffleAndGather(const raft::handle_t& handle,
                      const Tensor<DataT, 2, IndexT>& in,
                      Tensor<DataT, 2, IndexT>& out,
                      size_t n_samples_to_gather,
                      int seed,
                      cudaStream_t stream,
                      rmm::device_uvector<char>* workspace = nullptr)
{
  auto n_samples  = in.getSize(0);
  auto n_features = in.getSize(1);

  Tensor<IndexT, 1> indices({n_samples}, stream);

  if (workspace) {
    // shuffle indices on device using ml-prims
    raft::random::permute<DataT>(
      indices.data(), nullptr, nullptr, in.getSize(1), in.getSize(0), true, stream);
  } else {
    // shuffle indices on host and copy to device...
    std::vector<IndexT> ht_indices(n_samples);

    std::iota(ht_indices.begin(), ht_indices.end(), 0);

    std::mt19937 gen(seed);
    std::shuffle(ht_indices.begin(), ht_indices.end(), gen);

    raft::copy(indices.data(), ht_indices.data(), indices.numElements(), stream);
  }

  MLCommon::Matrix::gather(in.data(),
                           in.getSize(1),
                           in.getSize(0),
                           indices.data(),
                           n_samples_to_gather,
                           out.data(),
                           stream);
}

template <typename DataT, typename IndexT>
void countSamplesInCluster(const raft::handle_t& handle,
                           const KMeansParams& params,
                           Tensor<DataT, 2, IndexT>& X,
                           Tensor<DataT, 1, IndexT>& L2NormX,
                           Tensor<DataT, 2, IndexT>& centroids,
                           rmm::device_uvector<char>& workspace,
                           raft::distance::DistanceType metric,
                           Tensor<DataT, 1, IndexT>& sampleCountInCluster,
                           cudaStream_t stream)
{
  auto n_samples  = X.getSize(0);
  auto n_features = X.getSize(1);
  auto n_clusters = centroids.getSize(0);

  // stores (key, value) pair corresponding to each sample where
  //   - key is the index of nearest cluster
  //   - value is the distance to the nearest cluster
  Tensor<cub::KeyValuePair<IndexT, DataT>, 1, IndexT> minClusterAndDistance({n_samples}, stream);

  // temporary buffer to store distance matrix, destructor releases the resource
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

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

  // Using TransformInputIteratorT to dereference an array of cub::KeyValuePair
  // and converting them to just return the Key to be used in reduce_rows_by_key
  // prims
  kmeans::detail::KeyValueIndexOp<IndexT, DataT> conversion_op;
  cub::TransformInputIterator<IndexT,
                              kmeans::detail::KeyValueIndexOp<IndexT, DataT>,
                              cub::KeyValuePair<IndexT, DataT>*>
    itr(minClusterAndDistance.data(), conversion_op);

  // count # of samples in each cluster
  kmeans::detail::countLabels(
    handle, itr, sampleCountInCluster.data(), n_samples, n_clusters, workspace, stream);
}

/*
 * @brief Selects 'n_clusters' samples from the input X using kmeans++ algorithm.

 * @note  This is the algorithm described in
 *        "k-means++: the advantages of careful seeding". 2007, Arthur, D. and Vassilvitskii, S.
 *        ACM-SIAM symposium on Discrete algorithms.
 *
 * Scalable kmeans++ pseudocode
 * 1: C = sample a point uniformly at random from X
 * 2: while |C| < k
 * 3:   Sample x in X with probability p_x = d^2(x, C) / phi_X (C)
 * 4:   C = C U {x}
 * 5: end for
 */
template <typename DataT, typename IndexT>
void kmeansPlusPlus(const raft::handle_t& handle,
                    const KMeansParams& params,
                    Tensor<DataT, 2, IndexT>& X,
                    raft::distance::DistanceType metric,
                    rmm::device_uvector<char>& workspace,
                    rmm::device_uvector<DataT>& centroidsRawData,
                    cudaStream_t stream)
{
  auto n_samples  = X.getSize(0);
  auto n_features = X.getSize(1);
  auto n_clusters = params.n_clusters;

  // number of seeding trials for each center (except the first)
  auto n_trials = 2 + static_cast<int>(std::ceil(log(n_clusters)));

  LOG(handle,
      "Run sequential k-means++ to select %d centroids from %d input samples "
      "(%d seeding trials per iterations)",
      n_clusters,
      n_samples,
      n_trials);

  auto dataBatchSize = kmeans::detail::getDataBatchSize(params, n_samples);

  // temporary buffers
  std::vector<DataT> h_wt(n_samples);

  rmm::device_uvector<DataT> distBuffer(n_trials * n_samples, stream);

  Tensor<DataT, 2, IndexT> centroidCandidates({n_trials, n_features}, stream);

  Tensor<DataT, 1, IndexT> costPerCandidate({n_trials}, stream);

  Tensor<DataT, 1, IndexT> minClusterDistance({n_samples}, stream);

  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  rmm::device_scalar<DataT> clusterCost(stream);

  rmm::device_scalar<cub::KeyValuePair<int, DataT>> minClusterIndexAndDistance(stream);

  // L2 norm of X: ||c||^2
  Tensor<DataT, 1> L2NormX({n_samples}, stream);

  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm(
      L2NormX.data(), X.data(), X.getSize(1), X.getSize(0), raft::linalg::L2Norm, true, stream);
  }

  std::mt19937 gen(params.seed);
  std::uniform_int_distribution<> dis(0, n_samples - 1);

  // <<< Step-1 >>>: C <-- sample a point uniformly at random from X
  auto initialCentroid  = X.template view<2>({1, n_features}, {dis(gen), 0});
  int n_clusters_picked = 1;

  // reset buffer to store the chosen centroid
  centroidsRawData.resize(initialCentroid.numElements(), stream);
  raft::copy(
    centroidsRawData.begin(), initialCentroid.data(), initialCentroid.numElements(), stream);

  //  C = initial set of centroids
  Tensor<DataT, 2, IndexT> centroids(centroidsRawData.data(),
                                     {initialCentroid.getSize(0), initialCentroid.getSize(1)});
  // <<< End of Step-1 >>>

  // Calculate cluster distance, d^2(x, C), for all the points x in X to the nearest centroid
  kmeans::detail::minClusterDistance(handle,
                                     params,
                                     X,
                                     centroids,
                                     minClusterDistance,
                                     L2NormX,
                                     L2NormBuf_OR_DistBuf,
                                     workspace,
                                     metric,
                                     stream);

  LOG(handle, " k-means++ - Sampled %d/%d centroids", n_clusters_picked, n_clusters);

  // <<<< Step-2 >>> : while |C| < k
  while (n_clusters_picked < n_clusters) {
    // <<< Step-3 >>> : Sample x in X with probability p_x = d^2(x, C) / phi_X (C)
    // Choose 'n_trials' centroid candidates from X with probability proportional to the squared
    // distance to the nearest existing cluster
    raft::copy(h_wt.data(), minClusterDistance.data(), minClusterDistance.numElements(), stream);
    handle.sync_stream(stream);

    // Note - n_trials is relative small here, we don't need MLCommon::gather call
    std::discrete_distribution<> d(h_wt.begin(), h_wt.end());
    for (int cIdx = 0; cIdx < n_trials; ++cIdx) {
      auto rand_idx     = d(gen);
      auto randCentroid = X.template view<2>({1, n_features}, {rand_idx, 0});
      raft::copy(centroidCandidates.data() + cIdx * n_features,
                 randCentroid.data(),
                 randCentroid.numElements(),
                 stream);
    }

    // Calculate pairwise distance between X and the centroid candidates
    // Output - pwd [n_trails x n_samples]
    Tensor<DataT, 2, IndexT> pwd(distBuffer.data(), {n_trials, n_samples});
    kmeans::detail::pairwise_distance(
      handle, centroidCandidates, X, pwd, workspace, metric, stream);

    // Update nearest cluster distance for each centroid candidate
    // Note pwd and minDistBuf points to same buffer which currently holds pairwise distance values.
    // Outputs minDistanceBuf[m_trails x n_samples] where minDistance[i, :] contains updated
    // minClusterDistance that includes candidate-i
    Tensor<DataT, 2, IndexT> minDistBuf(distBuffer.data(), {n_trials, n_samples});
    raft::linalg::matrixVectorOp(
      minDistBuf.data(),
      pwd.data(),
      minClusterDistance.data(),
      pwd.getSize(1),
      pwd.getSize(0),
      true,
      true,
      [=] __device__(DataT mat, DataT vec) { return vec <= mat ? vec : mat; },
      stream);

    // Calculate costPerCandidate[n_trials] where costPerCandidate[i] is the cluster cost when using
    // centroid candidate-i
    raft::linalg::reduce(costPerCandidate.data(),
                         minDistBuf.data(),
                         minDistBuf.getSize(1),
                         minDistBuf.getSize(0),
                         static_cast<DataT>(0),
                         true,
                         true,
                         stream);

    // Greedy Choice - Choose the candidate that has minimum cluster cost
    // ArgMin operation below identifies the index of minimum cost in costPerCandidate
    {
      // Determine temporary device storage requirements
      size_t temp_storage_bytes = 0;
      cub::DeviceReduce::ArgMin(nullptr,
                                temp_storage_bytes,
                                costPerCandidate.data(),
                                minClusterIndexAndDistance.data(),
                                costPerCandidate.getSize(0));

      // Allocate temporary storage
      workspace.resize(temp_storage_bytes, stream);

      // Run argmin-reduction
      cub::DeviceReduce::ArgMin(workspace.data(),
                                temp_storage_bytes,
                                costPerCandidate.data(),
                                minClusterIndexAndDistance.data(),
                                costPerCandidate.getSize(0));

      int bestCandidateIdx = -1;
      raft::copy(&bestCandidateIdx, &minClusterIndexAndDistance.data()->key, 1, stream);
      /// <<< End of Step-3 >>>

      /// <<< Step-4 >>>: C = C U {x}
      // Update minimum cluster distance corresponding to the chosen centroid candidate
      raft::copy(minClusterDistance.data(),
                 minDistBuf.data() + bestCandidateIdx * n_samples,
                 n_samples,
                 stream);

      raft::copy(centroidsRawData.data() + n_clusters_picked * n_features,
                 centroidCandidates.data() + bestCandidateIdx * n_features,
                 n_features,
                 stream);

      ++n_clusters_picked;
      /// <<< End of Step-4 >>>
    }

    LOG(handle, " k-means++ - Sampled %d/%d centroids", n_clusters_picked, n_clusters);
  }  /// <<<< Step-5 >>>
}

template <typename DataT, typename IndexT>
void checkWeights(const raft::handle_t& handle,
                  rmm::device_uvector<char>& workspace,
                  Tensor<DataT, 1, IndexT>& weight,
                  cudaStream_t stream)
{
  rmm::device_scalar<DataT> wt_aggr(stream);

  int n_samples             = weight.getSize(0);
  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceReduce::Sum(
    nullptr, temp_storage_bytes, weight.data(), wt_aggr.data(), n_samples, stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceReduce::Sum(
    workspace.data(), temp_storage_bytes, weight.data(), wt_aggr.data(), n_samples, stream));

  DataT wt_sum = 0;
  raft::copy(&wt_sum, wt_aggr.data(), 1, stream);
  handle.sync_stream(stream);

  if (wt_sum != n_samples) {
    LOG(handle,
        "[Warning!] KMeans: normalizing the user provided sample weights to "
        "sum up to %d samples",
        n_samples);

    DataT scale = n_samples / wt_sum;
    raft::linalg::unaryOp(
      weight.data(),
      weight.data(),
      weight.numElements(),
      [=] __device__(const DataT& wt) { return wt * scale; },
      stream);
  }
}
};  // namespace detail
};  // namespace kmeans
};  // namespace ML
